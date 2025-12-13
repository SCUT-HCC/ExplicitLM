#!/usr/bin/env python3
"""
Hydra-Zen + DeepSpeed 预训练入口
"""
import os
import json
import time
import gc

# 设置进程名称（在 nvidia-smi 中显示的名称）
try:
    import setproctitle
    process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
    setproctitle.setproctitle(process_name)
except ImportError:
    # 如果没有 setproctitle，尝试使用 prctl (Linux only)
    try:
        import prctl
        process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
        prctl.set_name(process_name.encode('utf-8'))
    except (ImportError, AttributeError):
        # 如果都不可用，跳过（不影响训练）
        pass

from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import set_seed
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import argparse
from hydra_zen import launch, zen, instantiate
from config import store,_main_cfg_func          # 触发配置注册
from utils.logger import Logger
from utils.pretrain_datasets import create_pretrain_dataloader, create_validation_dataloader
from utils.train_loop_pretrain import train_epoch
from utils.model_initializer import init_model
from hydra.utils import get_original_cwd
from pathlib import Path
from omegaconf import OmegaConf

try:
    import swanlab
except ImportError:
    swanlab = None


def to_primitive(cfg):
    """将配置转换为基本数据类型，处理 Path 对象"""
    if isinstance(cfg, Path):
        return str(cfg)
    if isinstance(cfg, (dict, list, tuple)): # 处理普通容器
        if isinstance(cfg, dict):
             return {k: to_primitive(v) for k, v in cfg.items()}
        return type(cfg)(to_primitive(v) for v in cfg)
    if OmegaConf.is_config(cfg): # 处理 OmegaConf 对象
        return to_primitive(OmegaConf.to_container(cfg, resolve=True))
    return cfg


def main(cfg):
    """cfg 就是 Hydra-Zen 注入的五大配置节点"""
    m_cfg  = cfg.model
    d_cfg  = cfg.dataset
    l_cfg  = cfg.logging
    tr_cfg = cfg.training      # 所有训练/DeepSpeed 参数都在这儿
    proj_root = Path(get_original_cwd())
    # 补全路径
    m_cfg.cache_path = proj_root / m_cfg.cache_path
    m_cfg.database_init_path = proj_root / m_cfg.database_init_path
    # 配置 DDP 参数：允许未使用的参数（用于部分模型组件可能不参与梯度计算的情况）
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ds_plugin = DeepSpeedPlugin(zero_stage=tr_cfg.zero_stage)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=ds_plugin,
        # mixed_precision=tr_cfg.mixed_precision,
    )
    set_seed(tr_cfg.seed + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(l_cfg.out_dir, exist_ok=True)
        os.makedirs(l_cfg.save_dir, exist_ok=True)

    swanlab_run = None
    Logger(f"accelerator.is_main_process: {accelerator.is_main_process}", accelerator)
    Logger(f"swanlab is not None: {swanlab is not None}", accelerator)
    Logger(f"l_cfg.swanlab_online: {l_cfg.swanlab_online}", accelerator)
    if l_cfg.swanlab_online == False and accelerator.is_main_process and swanlab is not None:
        mode = "cloud" if l_cfg.swanlab_online else "offline"
        Logger(f"SwanLab 模式：{mode}", accelerator)
        Logger(f"SwanLab 运行中...", accelerator)
        # 从环境变量获取API key，如果未设置则使用项目默认值
        api_key = os.environ.get("SWANLAB_API_KEY", "GtiI1qjU5lco6MKKSrRmN")
        swanlab_run = swanlab.init(
            project=l_cfg.swanlab_project,
            experiment_name=f"ExplicitLM-Pretrain-{tr_cfg.epochs}e-{tr_cfg.batch_size}b",
            config=to_primitive(cfg),  # 使用转换后的配置，处理 Path 对象
            mode=mode,
            api_key=api_key  # 使用项目特定的API key
        )

    # ------------------------------------------------------------------
    # 4. 模型 / 优化器 / 调度器 / 数据
    # ------------------------------------------------------------------
    # 输出当前目录
    try:
        model, tokenizer = init_model(m_cfg, accelerator)          # 你原来的函数，直接吃 dict
    except Exception as e:
        Logger(f"警告：模型初始化失败({e})，使用默认模型", accelerator)
    
    # 参数过滤：只训练 requires_grad=True 的参数
    try:
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in optimizer_params)
        total = sum(p.numel() for p in model.parameters())
        Logger(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", accelerator)
    except Exception as e:
        Logger(f"警告：无法统计参数({e})，使用所有参数", accelerator)
        optimizer_params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=tr_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )



    train_loader = create_pretrain_dataloader(
        proj_root / d_cfg.dataset_path, tokenizer,
        tr_cfg.batch_size, m_cfg.max_seq_len,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader   = create_validation_dataloader(
        proj_root / d_cfg.val_dataset_path, tokenizer,
        tr_cfg.batch_size, m_cfg.max_seq_len,num_samples=200
    )
    steps_per_epoch = len(train_loader) // tr_cfg.accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * steps_per_epoch * tr_cfg.epochs),
        num_training_steps=steps_per_epoch * tr_cfg.epochs,
    )

    # ------------------------------------------------------------------
    # 5. Accelerator.prepare
    # ------------------------------------------------------------------
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    Logger(instantiate(cfg), accelerator)

    # ================== [修改] 断点续训加载逻辑 ==================
    start_epoch = 0
    resume_step = 0  # 新增：用于记录需要从当前epoch的第几个step开始
    
    resume_path = getattr(tr_cfg, 'resume_from_checkpoint', None)
    if resume_path is not None and os.path.exists(resume_path):
        Logger(f"正在从断点恢复: {resume_path}", accelerator)
        accelerator.load_state(resume_path)
        
        training_state_path = os.path.join(resume_path, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            start_epoch = training_state.get("epoch", 0)
            resume_step = training_state.get("step", 0) + 1 # 从断点的下一以步开始
            Logger(f"已恢复至 Epoch {start_epoch}, Step {resume_step}", accelerator)
        else:
            Logger("警告：未找到 training_state.json，将从 Epoch 0 开始", accelerator)
    # ==========================================================

    # Logger(instantiate(cfg).model,accelerator)
    # ------------------------------------------------------------------
    # 6. 训练循环
    # ------------------------------------------------------------------
    # 计算当前 epoch 需要跳过的步数：
    for epoch in range(start_epoch, tr_cfg.epochs):
        # 只有在起始 epoch 才需要跳过 resume_step，后续 epoch 都从 0 开始
        current_resume_step = resume_step if epoch == start_epoch else 0
        train_epoch(
            epoch=epoch,
            accelerator=accelerator,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=instantiate(cfg),   # 兼容你原来的 args 用法
            overall_start_time=time.time(),
            swanlab_run=swanlab_run,
            tokenizer=tokenizer,
            val_loader=val_loader,
            resume_step=current_resume_step  # [新增] 传递 resume_step
        )
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 7. 收尾
    # ------------------------------------------------------------------
    if accelerator.is_main_process and swanlab_run:
        # 获取SwanLab实验URL
        if l_cfg.swanlab_online:                       # 云版
            exp_url = str(swanlab_run.public.cloud.experiment_url)
        else:                                          # 本地版
            exp_url = 'local-mode'                     # 或者 swanlab_run.path

        # 写入临时文件供脚本读取
        with open('.swanlab_url', 'w') as f:
            f.write(exp_url)

        Logger(f"SwanLab URL已保存: {exp_url}", accelerator)

    #########################################################
    # 第十阶段：关闭SwanLab
    #########################################################

    if l_cfg.use_swanlab and accelerator.is_main_process:
        if swanlab_run is not None:
            swanlab_run.finish()
            Logger("SwanLab运行已结束", accelerator)

    Logger("训练完成！", accelerator)


if __name__ == "__main__":
    import sys
    # Extract command line arguments for config overrides, skipping script name
    # We accept key=value arguments directly from command line
    overrides = []
    for arg in sys.argv[1:]:
        if '=' in arg and not arg.startswith('--'):
            overrides.append(arg)
    
    # Launch with command line overrides
    launch(_main_cfg_func, main, overrides=overrides)
