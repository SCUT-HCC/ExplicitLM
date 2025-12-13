#!/usr/bin/env python3
"""
阶段2：训练知识融合组件（GatedMemoryFusion + memory_norm）

训练目标：
- 加载预训练的 MemoryGate（冻结）
- 只训练 GatedMemoryFusion 和 memory_norm
- 冻结 Backbone 和 MemoryGate
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import swanlab
from pathlib import Path

from utils.model_initializer import init_model, load_pretrained_memory_gate
from utils.pretrain_datasets import create_pretrain_dataloader, create_validation_dataloader
from utils.logger import Logger


def freeze_parameters_for_fusion_training(model, accelerator):
    """
    冻结参数：只训练 gated_memory_fusion 和 memory_norm
    冻结：backbone, memory_gate
    """
    Logger("🔒 设置参数冻结策略（阶段2：只训练 Fusion）", accelerator)
    
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        # 可训练：gated_memory_fusion, memory_norm
        is_fusion_component = any(keyword in name for keyword in [
            "gated_memory_fusion",
            "memory_norm",
        ])
        
        # 冻结：memory_gate, backbone, memory_bank
        is_memory_gate = "memory_gate" in name
        is_memory_bank = "memory_bank" in name
        
        if is_memory_bank:
            param.requires_grad = False
            frozen_params += param.numel()
        elif is_fusion_component:
            param.requires_grad = True
            trainable_params += param.numel()
        elif is_memory_gate:
            # 冻结 MemoryGate（已预训练）
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            # 冻结所有其他参数（backbone）
            param.requires_grad = False
            frozen_params += param.numel()
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"参数冻结完成: 冻结 {frozen_params / 1e6:.3f}M, 可训练 {trainable_params / 1e6:.3f}M", accelerator)
    
    return frozen_params, trainable_params


def main():
    parser = argparse.ArgumentParser(description="阶段2：训练知识融合组件")
    
    # 模型配置
    parser.add_argument("--qwen3_model_path", type=str, required=True, help="Qwen3 模型路径")
    parser.add_argument("--pretrained_memory_gate_path", type=str, required=True, help="预训练 MemoryGate 权重路径")
    parser.add_argument("--knowledge_num", type=int, default=1024*1024, help="记忆库条目数")
    parser.add_argument("--knowledge_length", type=int, default=16, help="每个记忆条目的 token 数")
    parser.add_argument("--knowledge_dim", type=int, default=128, help="记忆嵌入维度")
    parser.add_argument("--num_candidates", type=int, default=8, help="候选记忆数")
    parser.add_argument("--num_selected", type=int, default=1, help="选中的记忆数")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0, help="Gumbel-Softmax 温度")
    
    # 数据配置
    parser.add_argument("--dataset_path", type=str, required=True, help="训练数据路径（JSONL格式）")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="验证数据路径（可选）")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup 步数")
    
    # Loss 配置
    parser.add_argument("--ce_loss_coef", type=float, default=1.0, help="CE Loss 系数")
    parser.add_argument("--similarity_loss_coef", type=float, default=0.0, help="Similarity Loss 系数（阶段2默认0）")
    parser.add_argument("--diversity_loss_coef", type=float, default=0.0, help="Diversity Loss 系数（阶段2默认0）")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="checkpoints/fusion", help="输出目录")
    parser.add_argument("--save_interval", type=int, default=500, help="保存间隔（步数）")
    parser.add_argument("--swanlab_project", type=str, default="explicitlm-fusion", help="SwanLab 项目名")
    parser.add_argument("--swanlab_online", action="store_true", help="使用 SwanLab 在线模式")
    
    args = parser.parse_args()
    
    # 初始化 Accelerator
    accelerator = Accelerator()
    
    # 初始化 SwanLab
    if accelerator.is_main_process:
        swanlab.init(
            project=args.swanlab_project,
            config=vars(args),
            mode="cloud" if args.swanlab_online else "offline"
        )
    
    Logger("=" * 60, accelerator)
    Logger("阶段2：训练知识融合组件", accelerator)
    Logger("=" * 60, accelerator)
    
    # 初始化模型
    model_args = {
        "qwen3_model_path": args.qwen3_model_path,
        "knowledge_num": args.knowledge_num,
        "knowledge_length": args.knowledge_length,
        "knowledge_dim": args.knowledge_dim,
        "num_candidates": args.num_candidates,
        "num_selected": args.num_selected,
        "gumbel_temperature": args.gumbel_temperature,
        "use_moe": False,
        "dropout": 0.0,
        "cache_path": None,  # 不使用 cache
        "recompute_cache": False,
    }
    
    model, tokenizer = init_model(model_args, accelerator)
    Logger("模型初始化完成", accelerator)
    
    # 加载预训练的 MemoryGate
    load_pretrained_memory_gate(model, args.pretrained_memory_gate_path, accelerator)
    
    # 设置参数冻结（只训练 Fusion）
    freeze_parameters_for_fusion_training(model, accelerator)
    
    # 准备数据
    train_loader = create_pretrain_dataloader(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = None
    if args.val_dataset_path and os.path.exists(args.val_dataset_path):
        val_loader = create_validation_dataloader(
            data_path=args.val_dataset_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=4,
        )
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader) // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # 准备
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # 损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_sim_loss_sum = 0.0
        epoch_div_loss_sum = 0.0
        steps_in_epoch = 0
        
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        
        for step, (X, Y, loss_mask) in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # 前向传播
                res = model(X)
                
                # 计算 CE Loss
                ce_loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
                
                # 处理辅助损失
                similarity_loss = torch.tensor(0.0, device=ce_loss.device)
                diversity_loss = torch.tensor(0.0, device=ce_loss.device)
                
                if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                    aux_loss = res.aux_loss
                    if isinstance(aux_loss, dict):
                        similarity_loss = aux_loss.get('similarity_loss', torch.tensor(0.0, device=ce_loss.device))
                        diversity_loss = aux_loss.get('diversity_loss', torch.tensor(0.0, device=ce_loss.device))
                        
                        if isinstance(similarity_loss, torch.Tensor):
                            similarity_loss = accelerator.gather(similarity_loss).mean()
                        if isinstance(diversity_loss, torch.Tensor):
                            diversity_loss = accelerator.gather(diversity_loss).mean()
                
                # 总损失
                total_loss = (
                    args.ce_loss_coef * ce_loss +
                    args.similarity_loss_coef * similarity_loss +
                    args.diversity_loss_coef * diversity_loss
                ) / args.accumulation_steps
                
                # 反向传播
                accelerator.backward(total_loss)
                
                if (step + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # 累积统计
                epoch_loss_sum += total_loss.item() * args.accumulation_steps
                epoch_ce_loss_sum += ce_loss.item()
                epoch_sim_loss_sum += similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else 0.0
                epoch_div_loss_sum += diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0.0
                steps_in_epoch += 1
                
                # 日志
                if global_step % 10 == 0 and accelerator.is_main_process:
                    running_loss = epoch_loss_sum / steps_in_epoch
                    running_ce = epoch_ce_loss_sum / steps_in_epoch
                    running_sim = epoch_sim_loss_sum / steps_in_epoch
                    running_div = epoch_div_loss_sum / steps_in_epoch
                    
                    swanlab.log({
                        "train/step_loss": total_loss.item() * args.accumulation_steps,
                        "train/running_loss": running_loss,
                        "train/ce_loss": running_ce,
                        "train/similarity_loss": running_sim,
                        "train/diversity_loss": running_div,
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)
                    
                    progress_bar.set_description(
                        f"Epoch {epoch} | Loss: {running_loss:.4f} | "
                        f"CE: {running_ce:.4f} | Sim: {running_sim:.4f} | Div: {running_div:.4f}"
                    )
                
                # 保存 checkpoint
                if global_step % args.save_interval == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path)
                    Logger(f"Checkpoint saved: {save_path}", accelerator)
        
        # Epoch 结束：计算平均损失
        avg_loss = epoch_loss_sum / steps_in_epoch
        avg_ce = epoch_ce_loss_sum / steps_in_epoch
        avg_sim = epoch_sim_loss_sum / steps_in_epoch
        avg_div = epoch_div_loss_sum / steps_in_epoch
        
        Logger(f"Epoch {epoch} 完成: Loss={avg_loss:.4f}, CE={avg_ce:.4f}, Sim={avg_sim:.4f}, Div={avg_div:.4f}", accelerator)
        
        if accelerator.is_main_process:
            swanlab.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_ce_loss": avg_ce,
                "train/epoch_similarity_loss": avg_sim,
                "train/epoch_diversity_loss": avg_div,
            }, step=global_step)
        
        # 验证
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for X, Y, loss_mask in val_loader:
                    res = model(X)
                    ce_loss = loss_fct(
                        res.logits.view(-1, res.logits.size(-1)),
                        Y.view(-1)
                    ).view(Y.size())
                    ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
                    val_loss_sum += ce_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
            Logger(f"验证损失: {avg_val_loss:.4f}", accelerator)
            
            if accelerator.is_main_process:
                swanlab.log({"val/loss": avg_val_loss}, step=global_step)
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_path = os.path.join(args.output_dir, "fusion_best.pth")
                    torch.save(unwrapped_model.state_dict(), best_path)
                    Logger(f"最佳模型已保存: {best_path} (val_loss={best_val_loss:.4f})", accelerator)
        
        # 保存 epoch checkpoint
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            epoch_path = os.path.join(args.output_dir, f"fusion_epoch_{epoch}.pth")
            torch.save(unwrapped_model.state_dict(), epoch_path)
            Logger(f"Epoch checkpoint saved: {epoch_path}", accelerator)
    
    # 训练完成
    accelerator.end_training()
    if accelerator.is_main_process:
        swanlab.finish()
        Logger("阶段2训练完成！", accelerator)


if __name__ == "__main__":
    main()

