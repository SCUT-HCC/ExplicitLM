#!/usr/bin/env python3
"""
Hydra-Zen + DeepSpeed 预训练入口
"""
import os
import time
import gc
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import set_seed
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from hydra_zen import launch, zen, instantiate
from config import store,_main_cfg_func          # 触发配置注册
from utils.logger import Logger
from utils.pretrain_datasets import create_pretrain_dataloader, create_validation_dataloader
from utils.train_loop import train_epoch
from utils.model_initializer import init_model
from hydra.utils import get_original_cwd
from pathlib import Path

try:
    import swanlab
except ImportError:
    swanlab = None


def main(cfg):
    """cfg 就是 Hydra-Zen 注入的五大配置节点"""
    # ------------------------------------------------------------------
    # 1. 解构配置
    # ------------------------------------------------------------------
    m_cfg  = cfg.model
    d_cfg  = cfg.dataset
    l_cfg  = cfg.logging
    tr_cfg = cfg.training      # 所有训练/DeepSpeed 参数都在这儿
    proj_root = Path(get_original_cwd())
    # ------------------------------------------------------------------
    # 2. Accelerator + DeepSpeed
    # ------------------------------------------------------------------
    ds_plugin = DeepSpeedPlugin(zero_stage=tr_cfg.zero_stage)
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        deepspeed_plugin=ds_plugin,
        # mixed_precision=tr_cfg.mixed_precision,
    )
    set_seed(tr_cfg.seed + accelerator.process_index)

    # ------------------------------------------------------------------
    # 3. 目录 & SwanLab
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        os.makedirs(l_cfg.out_dir, exist_ok=True)
        os.makedirs(l_cfg.save_dir, exist_ok=True)

    swanlab_run = None
    if l_cfg.use_swanlab and accelerator.is_main_process and swanlab is not None:
        mode = "cloud" if l_cfg.swanlab_online else "local"
        swanlab_run = swanlab.init(
            project=l_cfg.swanlab_project,
            experiment_name=f"MiniMind-Pretrain-{tr_cfg.epochs}e-{tr_cfg.batch_size}b",
            config=instantiate(cfg),   # 把完整配置 flatten 上传
            mode=mode
        )

    # ------------------------------------------------------------------
    # 4. 模型 / 优化器 / 调度器 / 数据
    # ------------------------------------------------------------------
    # 输出当前目录

    model, tokenizer = init_model(m_cfg)          # 你原来的函数，直接吃 dict
        # 1. 参数过滤（EMA 逻辑保持原样）
    try:
        model_config = model.module.config if hasattr(model, 'module') else model.config
        use_ema = getattr(model_config, 'use_ema_update', False)
        if use_ema:
            optimizer_params = [p for p in model.parameters() if p.requires_grad]
            trainable = sum(p.numel() for p in optimizer_params)
            total = sum(p.numel() for p in model.parameters())
            Logger(f"EMA 模式：可训练参数 {trainable:,} / {total:,}", accelerator)
        else:
            optimizer_params = model.parameters()
            Logger("传统模式：所有参数参与优化", accelerator)
    except Exception as e:
        Logger(f"警告：无法读取配置({e})，默认所有参数参与优化", accelerator)
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
        num_workers=4,
        pin_memory=True,
    )
    val_loader   = create_validation_dataloader(
        proj_root / d_cfg.val_dataset_path, tokenizer,
        tr_cfg.batch_size, m_cfg.max_seq_len,num_samples=200
    )
    Logger(proj_root / d_cfg.val_dataset_path, accelerator)
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
    # Logger(instantiate(cfg).model,accelerator)
    # ------------------------------------------------------------------
    # 6. 训练循环
    # ------------------------------------------------------------------
    for epoch in range(tr_cfg.epochs):
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
        )
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 7. 收尾
    # ------------------------------------------------------------------
    if accelerator.is_main_process and swanlab_run:
        swanlab_run.finish()


if __name__ == "__main__":
    # 注意：train 函数就是位置参数
    launch(_main_cfg_func,main)
