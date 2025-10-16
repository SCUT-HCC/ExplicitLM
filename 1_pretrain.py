#!/usr/bin/env python3

"""
预训练主程序

功能：
- 初始化分布式训练环境（Accelerator + DeepSpeed）
- 配置混合精度训练
- 设置随机种子保证可复现性
"""

import os
import time
import gc
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

from utils.config_utils import setup_config
from utils.logger import Logger
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import set_seed
from utils.pretrain_datasets import create_pretrain_dataloader, create_validation_dataloader
from utils.train_loop import train_epoch

try:
    import swanlab
except ImportError:
    swanlab = None


def main():
    """
    预训练主函数

    实现流程：
    1. 加载超参数配置
    2. 初始化分布式训练环境
    3. 设置随机种子
    4. 创建输出目录
    5. TODO: 模型初始化
    6. TODO: 数据加载器初始化
    7. TODO: 训练循环
    """

    #########################################################
    # 第一阶段：配置初始化
    #########################################################
    args = setup_config()

    #########################################################
    # 第二阶段：初始化 Accelerator 和 DeepSpeed
    #########################################################
    # 配置 DDP 参数：允许未使用的参数（用于部分模型组件可能不参与梯度计算的情况）
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # 配置 DeepSpeed 插件
    # 注意：大部分配置在 ds_config.json 中已定义，这里只设置关键参数
    # ds_config.json 中已配置：
    #   - ZeRO-2 优化
    #   - CPU offload（优化器和参数）
    #   - bf16 混合精度
    #   - gradient clipping 和 accumulation（设为 auto）
    ds_plugin = DeepSpeedPlugin(
        zero_stage=2,  # 使用 ZeRO-2 优化，与 ds_config.json 保持一致
    )

    # 初始化 Accelerator
    # 混合精度、梯度累积等参数由 ds_config.json 控制
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=ds_plugin,
    )

    #########################################################
    # 第三阶段：设置随机种子
    #########################################################
    # 为每个进程设置不同的随机种子，确保数据采样的多样性
    set_seed(1337 + accelerator.process_index)

    #########################################################
    # 第四阶段：创建输出目录
    #########################################################
    # 配置保存路径：args.save_dir 作为模型检查点和训练输出的根目录
    args.save_dir = args.out_dir

    # 仅主进程负责创建目录，避免多进程竞争
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)

        # 打印配置信息
        print(f"初始化完成:")
        print(f"  - 进程数: {accelerator.num_processes}")
        print(f"  - 混合精度: {accelerator.mixed_precision}")
        print(f"  - DeepSpeed ZeRO Stage: 2")
        print(f"  - 设备: {accelerator.device}")
        print(f"  - 输出目录: {args.out_dir}")

    #########################################################
    # 第五阶段：配置SwanLab实验追踪
    #########################################################
    swanlab_run: Optional[Any] = None

    if hasattr(args, 'use_swanlab') and args.use_swanlab and accelerator.is_main_process:
        if swanlab is None:
            Logger("警告: SwanLab未安装，跳过实验追踪配置", accelerator)
        else:
            # 配置实验运行名称
            args.swanlab_run_name = (
                f"MiniMind-Pretrain-Epoch-{args.epochs}-"
                f"BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
            )

            # 创建配置字典用于SwanLab追踪
            config_dict: Dict[str, Any] = vars(args).copy()

            # 根据配置选择在线或离线模式
            mode = "online" if getattr(args, 'swanlab_online', False) else "offline"

            # 初始化SwanLab实验实例
            swanlab_run = swanlab.init(
                project=getattr(args, 'swanlab_project', 'MiniMind'),
                experiment_name=args.swanlab_run_name,
                description="MiniMind预训练实验追踪",
                config=config_dict,
                mode=mode
            )

            Logger(f"SwanLab初始化完成 (模式: {mode})", accelerator)
            Logger(f"  - 项目: {getattr(args, 'swanlab_project', 'MiniMind')}", accelerator)
            Logger(f"  - 实验名称: {args.swanlab_run_name}", accelerator)

    #########################################################
    # 第六阶段：模型初始化
    #########################################################
    from utils.model_initializer import init_model

    Logger("开始初始化模型...", accelerator)

    # 初始化模型和tokenizer
    model, tokenizer = init_model(args)

    Logger(f"模型初始化完成，准备使用Accelerator...", accelerator)
    # =========================================================
    # 第七阶段：模型初始化
    # =========================================================
    from utils.model_initializer import init_model

    Logger("开始初始化模型...", accelerator)
    model, tokenizer = init_model(args)
    Logger("模型初始化完成", accelerator)

    # =========================================================
    # 第八阶段：优化器 & 调度器（prepare 之前必须创建好）
    # =========================================================
    Logger("开始创建优化器/调度器...", accelerator)

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

    # 2. 优化器
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    Logger(f"AdamW 优化器已创建  lr={args.learning_rate}", accelerator)
    Logger("准备数据加载器...", accelerator)
    train_dataloader = create_pretrain_dataloader(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = create_validation_dataloader(
        val_data_path=args.val_dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        num_samples=200,
    )

    # 3. 调度器
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    Logger(f"Cosine 调度器  warmup={warmup_steps}  total={total_steps}", accelerator)

    # =========================================================
    # 第九阶段：数据加载器 + 模型 + 优化器 + 调度器 一起 prepare
    # =========================================================

    Logger("Accelerator prepare 中（DeepSpeed ZeRO-2）...", accelerator)
    # 关键：一次性把 5 个对象都交给 accelerator，保证 DeepSpeed 能拿到合法 optimizer
    model, optimizer, scheduler, train_dataloader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_loader
    )

    Logger("Accelerator 准备完毕", accelerator)
    #########################################################
    # 第九阶段：训练循环
    #########################################################
    Logger("开始训练循环...", accelerator)

    # 记录整体训练开始时间
    overall_start_time = time.time()

    # Epoch循环
    for epoch in range(args.epochs):
        Logger(f"开始第{epoch+1}轮训练", accelerator)

        train_epoch(
            epoch=epoch,
            accelerator=accelerator,
            model=model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            overall_start_time=overall_start_time,
            swanlab_run=swanlab_run,
            tokenizer=tokenizer,
            val_loader=val_loader
        )

        # 每个epoch结束后进行内存清理
        Logger(f"第{epoch+1}轮训练完成，进行内存清理", accelerator)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #########################################################
    # 导出SwanLab URL（训练完成后，关闭之前）
    #########################################################
    if accelerator.is_main_process and swanlab_run:
        # 获取SwanLab实验URL
        exp_url = swanlab_run.url if hasattr(swanlab_run, 'url') else "N/A"

        # 写入临时文件供脚本读取
        with open('.swanlab_url', 'w') as f:
            f.write(exp_url)

        Logger(f"SwanLab URL已保存: {exp_url}", accelerator)

    #########################################################
    # 第十阶段：关闭SwanLab
    #########################################################
    if hasattr(args, 'use_swanlab') and args.use_swanlab and accelerator.is_main_process:
        if swanlab_run is not None:
            swanlab_run.finish()
            Logger("SwanLab运行已结束", accelerator)

    Logger("训练完成！", accelerator)


if __name__ == '__main__':
    main()