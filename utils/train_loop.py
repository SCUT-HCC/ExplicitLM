"""
训练循环模块

功能：
- train_epoch: 单个epoch的训练循环
- 支持梯度累积、验证评估、模型保存
- 集成SwanLab实验追踪
"""

import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from utils.logger import logger
from utils.train_utils import validate_model, format_time

try:
    import swanlab
except ImportError:
    swanlab = None


def train_epoch(
    epoch: int,
    accelerator: Accelerator,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    args: Any,
    overall_start_time: float,
    swanlab_run: Optional[Any],
    tokenizer: Any,
    val_loader: Optional[DataLoader] = None
) -> None:
    """
    单个epoch的训练循环

    参数：
        epoch: 当前epoch编号（从0开始）
        accelerator: Accelerator实例
        model: 训练模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 配置参数
        overall_start_time: 整体训练开始时间
        swanlab_run: SwanLab运行实例
        tokenizer: Tokenizer实例
        val_loader: 验证数据加载器（可选）

    说明：
        混合精度由DeepSpeed配置文件（ds_config.json）自动控制
        - bf16已启用，无需手动创建autocast上下文
        - 梯度累积由DeepSpeed自动处理
        - 梯度裁剪由DeepSpeed自动处理
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    epoch_start_time = time.time()
    total_steps_in_epoch = len(train_loader)
    total_training_steps = args.epochs * total_steps_in_epoch
    moe_path = '_moe' if args.use_moe else ''
    best_loss = float('inf')

    # 记录初始状态
    last_log_time = epoch_start_time

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 前向传播（DeepSpeed自动处理bf16混合精度）
        # 第一个epoch的embedding冻结处理
        if step == 0 and args.embeddings_epoch == epoch:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.freeze_embedding = True
            Logger(f"设置freeze_embedding=True (epoch {epoch}, step {step})", accelerator)

        res = model(X, step=step)

        # 计算主要损失（交叉熵损失）
        ce_loss = loss_fct(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()

        # 处理辅助损失
        similarity_loss = 0
        diversity_loss = 0

        if hasattr(res, 'aux_loss') and res.aux_loss is not None:
            aux_loss = res.aux_loss
            if isinstance(aux_loss, dict):
                # 三损失结构
                similarity_loss = aux_loss.get('similarity_loss', 0)
                diversity_loss = aux_loss.get('diversity_loss', 0)

                # 分布式训练中的损失聚合
                if isinstance(similarity_loss, torch.Tensor):
                    similarity_loss = accelerator.gather(similarity_loss).mean()
                if isinstance(diversity_loss, torch.Tensor):
                    diversity_loss = accelerator.gather(diversity_loss).mean()

        # 三损失系统：CE + Similarity + Diversity
        similarity_coef = getattr(args, 'similarity_loss_coef', 0.1)
        diversity_coef = getattr(args, 'diversity_loss_coef', 0.05)

        total_loss = (
            ce_loss +
            similarity_coef * similarity_loss +
            diversity_coef * diversity_loss
        )
        loss = total_loss / args.accumulation_steps

        # 反向传播
        accelerator.backward(loss)

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()

        # VQ-VAE风格的EMA更新（仅在启用时执行）
        if hasattr(res, 'ema_stats') and res.ema_stats is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'apply_ema_update'):
                ema_update_stats = unwrapped_model.apply_ema_update(res.ema_stats)

                # 记录EMA更新统计信息
                if (step + 1) % args.log_interval == 0 and accelerator.is_main_process:
                    if ema_update_stats.get('ema_update_applied', False):
                        total_memories = args.knowledge_num
                        Logger(
                            f"EMA更新 - Step: {ema_update_stats['ema_step']}, "
                            f"更新记忆数: {ema_update_stats['updated_memories']}/{total_memories} "
                            f"({ema_update_stats['update_ratio']:.4f}), "
                            f"覆盖率: {ema_update_stats['selected_memory_coverage']:.4f}",
                            accelerator
                        )

        # 验证评估和日志记录（仅主进程）
        if (step + 1) % args.log_interval == 0 and accelerator.is_main_process:
            current_time = time.time()

            # 计算当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 计算时间估算
            epoch_elapsed_time = current_time - epoch_start_time
            epoch_steps_done = step + 1
            epoch_avg_step_time = epoch_elapsed_time / epoch_steps_done
            epoch_remaining_time = epoch_avg_step_time * (total_steps_in_epoch - epoch_steps_done)

            total_elapsed_time = current_time - overall_start_time
            total_steps_done = epoch * total_steps_in_epoch + epoch_steps_done
            total_avg_step_time = total_elapsed_time / total_steps_done if total_steps_done > 0 else 0
            total_remaining_time = total_avg_step_time * (total_training_steps - total_steps_done) if total_steps_done > 0 else 0

            # 计算训练速度
            interval_elapsed_time = current_time - last_log_time
            tokens_processed_interval = args.log_interval * args.batch_size * args.max_seq_len
            tokens_per_sec = tokens_processed_interval / interval_elapsed_time if interval_elapsed_time > 0 else 0
            last_log_time = current_time

            # 执行验证评估
            val_loss = None
            if val_loader is not None:
                try:
                    val_loss = validate_model(model, val_loader, loss_fct, accelerator)
                    Logger(f"验证损失: {val_loss:.4f}", accelerator)
                except Exception as e:
                    Logger(f"验证评估失败: {e}", accelerator)
                    val_loss = None

            # 获取记忆库更新统计（如果模型支持）
            memory_update_stats = {}
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'get_memory_update_stats'):
                try:
                    memory_update_stats = unwrapped_model.get_memory_update_stats()
                except Exception as e:
                    Logger(f"获取记忆更新统计失败: {e}", accelerator)

            # 获取余弦相似度统计
            avg_selected_similarity = 0.0
            if hasattr(res, 'cosine_stats') and res.cosine_stats is not None:
                cosine_stats = res.cosine_stats
                selected_similarities = [
                    v for k, v in cosine_stats.items()
                    if k.endswith('_selected_avg_similarity')
                ]
                if selected_similarities:
                    import numpy as np
                    avg_selected_similarity = np.mean(selected_similarities)

            # 构建日志字典
            log_dict = {
                "epoch": epoch + 1,
                "step": step + 1,
                "total_steps_in_epoch": total_steps_in_epoch,
                "train/loss_ce": ce_loss.item(),
                "train/loss_similarity": similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else similarity_loss,
                "train/loss_diversity": diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
                "train/loss_total": total_loss.item(),
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "epoch_time_left_seconds": epoch_remaining_time,
                "total_time_left_seconds": total_remaining_time,
                "train/avg_selected_similarity": avg_selected_similarity,
            }

            # 添加验证损失
            if val_loss is not None:
                log_dict["val/loss"] = val_loss

            # 添加记忆库更新统计
            log_dict.update(memory_update_stats)

            # 控制台输出
            Logger(
                f"Epoch {epoch+1}/{args.epochs}, Step {step+1}/{total_steps_in_epoch}, "
                f"CE: {log_dict['train/loss_ce']:.4f}, "
                f"Sim: {log_dict['train/loss_similarity']:.4f}, "
                f"Div: {log_dict['train/loss_diversity']:.4f}, "
                f"Total: {log_dict['train/loss_total']:.4f}, "
                f"Val: {log_dict.get('val/loss', 'N/A')}, "
                f"LR: {log_dict['lr']:.6f}, "
                f"Speed: {log_dict['tokens_per_sec']:.2f} tokens/sec | "
                f"Sel.Sim: {avg_selected_similarity:.4f} | "
                f"Epoch剩余: {format_time(epoch_remaining_time)} | "
                f"总剩余: {format_time(total_remaining_time)}",
                accelerator
            )

            # SwanLab日志记录
            if args.use_swanlab and swanlab_run:
                swanlab_run.log(log_dict)

        # 模型保存（仅主进程）
        if accelerator.is_main_process:
            loss_total = loss.item() * args.accumulation_steps
            if best_loss > loss_total:
                best_loss = loss_total
                ckp = f'{args.save_dir}/pretrain_{args.dim}{moe_path}.pth'

                # 获取解包后的模型
                unwrapped_model = accelerator.unwrap_model(model)

                # 保存模型参数
                accelerator.save(unwrapped_model.state_dict(), ckp)
                Logger(f"模型已保存至 {ckp}", accelerator)
