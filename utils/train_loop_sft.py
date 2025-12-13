"""
SFT训练循环模块

功能：
- train_epoch_sft: SFT阶段的单个epoch训练循环
- eval_model_sft: 生成式评估函数（文本生成+准确率计算）
- judger: 文本匹配评估器
- 支持梯度累积、验证评估、模型保存
- 集成SwanLab实验追踪
"""

import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import numpy as np

from utils.logger import Logger
from utils.train_utils import format_time

try:
    import swanlab
except ImportError:
    swanlab = None


def judger(
    generated_text: str,
    std_output: str,
    judger_mode: str
) -> bool:
    """
    评估生成文本与标准答案的匹配程度

    参数：
        generated_text: 生成的文本
        std_output: 标准答案
        judger_mode: 判断模式
            - exact: 精确匹配
            - contains: 标准答案包含在生成文本中
            - startswith: 生成文本以标准答案开头
            - endswith: 生成文本以标准答案结尾

    返回：
        是否匹配成功
    """
    gen_stripped = generated_text.strip()
    std_stripped = std_output.strip()

    if judger_mode == 'exact':
        return gen_stripped == std_stripped
    elif judger_mode == 'contains':
        return std_stripped in gen_stripped
    elif judger_mode == 'startswith':
        return gen_stripped.startswith(std_stripped)
    elif judger_mode == 'endswith':
        return gen_stripped.endswith(std_stripped)
    return False


def eval_model_sft(
    model: nn.Module,
    eval_loader: DataLoader,
    tokenizer: Any,
    accelerator: Accelerator,
    args: Any,
    judger_mode: str = "startswith"
) -> dict:
    """
    SFT阶段的生成式评估函数

    参数：
        model: 待评估模型（wrapped model）
        eval_loader: 评估数据加载器
        tokenizer: Tokenizer实例
        accelerator: Accelerator实例
        args: 配置参数
        judger_mode: 判断模式

    返回：
        性能字典，包含每个样本的结果和整体准确率

    说明：
        - 模型生成完整回复
        - 与标准答案对比（startswith/contains等）
        - 计算准确率指标
    """
    # 验证 tokenizer 配置
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 缺少 eos_token_id 配置")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token_id 配置")

    # 设置评估模式（在 wrapped model 上操作）
    model.eval()
    performance = {}
    total_correct = 0
    total_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            try:
                if isinstance(batch, tuple) and len(batch) == 2:
                    prompt_input, std_output = batch
                    
                    if isinstance(prompt_input, (list, tuple)):
                        if len(prompt_input) > 0:
                            prompt_input = prompt_input[0]
                        else:
                            Logger(f"警告: prompt_input 为空，跳过 step {step}", accelerator)
                            continue
                    
                    if isinstance(std_output, (list, tuple)):
                        if len(std_output) > 0:
                            std_output = std_output[0]
                        else:
                            Logger(f"警告: std_output 为空，跳过 step {step}", accelerator)
                            continue
                    
                    if not isinstance(prompt_input, str):
                        Logger(f"警告: prompt_input 不是字符串类型 (type: {type(prompt_input)}), 跳过 step {step}", accelerator)
                        continue
                        
                    if not isinstance(std_output, str):
                        Logger(f"警告: std_output 不是字符串类型 (type: {type(std_output)}), 跳过 step {step}", accelerator)
                        continue
                else:
                    Logger(f"警告: 评估数据格式错误 (batch type: {type(batch)}, len: {len(batch) if hasattr(batch, '__len__') else 'N/A'}), 跳过 step {step}", accelerator)
                    continue
            except Exception as e:
                Logger(f"警告: 处理batch时出错 (step {step}): {e}, 跳过", accelerator)
                continue

            try:
                x = torch.tensor(
                    tokenizer(prompt_input)['input_ids'],
                    device=accelerator.device
                ).unsqueeze(0)
            except Exception as e:
                Logger(f"警告: Tokenization 失败 (step {step}): {e}", accelerator)
                continue

            unwrapped_model = accelerator.unwrap_model(model)

            try:
                generated = unwrapped_model.generate(
                    x,
                    max_new_tokens=args.training.max_new_tokens,
                    max_length=args.model.max_seq_len + args.training.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            except Exception as e:
                Logger(f"警告: 生成失败 (step {step}): {e}", accelerator)
                continue

            try:
                generated_text = tokenizer.decode(
                    generated.squeeze()[x.shape[1]:].tolist(),
                    skip_special_tokens=True
                )
            except Exception as e:
                Logger(f"警告: 解码失败 (step {step}): {e}", accelerator)
                continue

            generated_text = generated_text.strip()
            flag = judger(generated_text, std_output, judger_mode)

            performance[step] = {
                'prompt': prompt_input,
                'std_output': std_output,
                'generated_text': generated_text,
                'result': flag,
                'judger_mode': judger_mode
            }

            total_correct += int(flag)
            total_steps += 1

            if accelerator.is_main_process and step % max(1, len(eval_loader) // args.training.show_eval_res) == 0:
                Logger(
                    f"评估样例 Step {step}: "
                    f"生成='{generated_text[:30]}...' | "
                    f"标准='{std_output[:30]}...' | "
                    f"匹配={flag}",
                    accelerator
                )

    accuracy = total_correct / total_steps if total_steps > 0 else 0.0

    performance['overall'] = {
        'total_steps': total_steps,
        'total_correct': total_correct,
        'accuracy': accuracy
    }

    if accelerator.is_main_process:
        Logger(
            f"评估结果: 准确率={accuracy:.4f} ({total_correct}/{total_steps})",
            accelerator
        )

    model.train()

    return performance


def train_epoch_sft(
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
    eval_loader: Optional[DataLoader] = None
) -> None:
    """
    SFT阶段的单个epoch训练循环

    参数：
        epoch: 当前epoch编号（从0开始）
        accelerator: Accelerator实例
        model: 训练模型（wrapped model）
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 配置参数（hydra-zen结构：args.training/model/logging）
        overall_start_time: 整体训练开始时间
        swanlab_run: SwanLab运行实例
        tokenizer: Tokenizer实例
        eval_loader: 评估数据加载器（可选）

    说明：
        混合精度由DeepSpeed配置文件（ds_config.json）自动控制
        - bf16已启用，无需手动创建autocast上下文
        - 梯度累积由DeepSpeed自动处理
        - 梯度裁剪由DeepSpeed自动处理

        评估方式：
        - 使用eval_model_sft进行生成式评估
        - 计算文本生成准确率
        - 适用于SFT阶段的对话质量评估
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    epoch_start_time = time.time()
    total_steps_in_epoch = len(train_loader)
    total_training_steps = args.training.epochs * total_steps_in_epoch
    moe_path = '_moe' if args.model.use_moe else ''
    unwrapped_model = accelerator.unwrap_model(model)
    hidden_size = getattr(unwrapped_model.config, 'hidden_size', 2560)
    best_loss = float('inf')
    best_accuracy = 0.0
    last_log_time = epoch_start_time

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if scheduler is not None:
            scheduler.step()

        res = model(X, step=step)

        ce_loss = loss_fct(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()

        aux_loss = res.aux_loss
        similarity_loss = aux_loss['similarity_loss']
        diversity_loss = aux_loss['diversity_loss']

        similarity_coef = args.training.similarity_loss_coef
        diversity_coef = args.training.diversity_loss_coef

        total_loss = (
            ce_loss +
            similarity_coef * similarity_loss +
            diversity_coef * diversity_loss
        )
        loss = total_loss / args.training.accumulation_steps

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % args.training.accumulation_steps == 0:
            try:
                from deepspeed import get_accelerator
                get_accelerator().empty_cache()
            except (ImportError, AttributeError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(accelerator, 'sync'):
                        accelerator.sync()

        if (step + 1) % args.logging.log_interval == 0 and accelerator.is_main_process:
            current_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_elapsed_time = current_time - epoch_start_time
            epoch_steps_done = step + 1
            epoch_avg_step_time = epoch_elapsed_time / epoch_steps_done
            epoch_remaining_time = epoch_avg_step_time * (total_steps_in_epoch - epoch_steps_done)

            total_elapsed_time = current_time - overall_start_time
            total_steps_done = epoch * total_steps_in_epoch + epoch_steps_done
            total_avg_step_time = total_elapsed_time / total_steps_done if total_steps_done > 0 else 0
            total_remaining_time = total_avg_step_time * (total_training_steps - total_steps_done) if total_steps_done > 0 else 0
            interval_elapsed_time = current_time - last_log_time
            tokens_processed_interval = args.logging.log_interval * args.training.batch_size * args.model.max_seq_len
            tokens_per_sec = tokens_processed_interval / interval_elapsed_time if interval_elapsed_time > 0 else 0
            last_log_time = current_time

            unwrapped_model = accelerator.unwrap_model(model)
            try:
                memory_update_stats = unwrapped_model.get_memory_update_stats()
            except Exception as e:
                Logger(f"获取记忆更新统计失败: {e}", accelerator)
                memory_update_stats = {}

            cosine_stats = res.cosine_stats
            selected_similarities = [
                v for k, v in cosine_stats.items()
                if k.endswith('_selected_avg_similarity')
            ]
            avg_selected_similarity = np.mean(selected_similarities) if selected_similarities else 0.0

            similarity_loss_log = accelerator.gather(similarity_loss).mean().item() if accelerator.num_processes > 1 else similarity_loss.item()
            diversity_loss_log = accelerator.gather(diversity_loss).mean().item() if accelerator.num_processes > 1 else diversity_loss.item()
            
            log_dict = {
                "epoch": epoch + 1,
                "step": step + 1,
                "total_steps_in_epoch": total_steps_in_epoch,
                "train/loss_ce": ce_loss.item(),
                "train/loss_similarity": similarity_loss_log,
                "train/loss_diversity": diversity_loss_log,
                "train/loss_total": total_loss.item(),
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "epoch_time_left_seconds": epoch_remaining_time,
                "total_time_left_seconds": total_remaining_time,
                "train/avg_selected_similarity": avg_selected_similarity,
            }

            log_dict.update(memory_update_stats)

            Logger(
                f"Epoch {epoch+1}/{args.training.epochs}, Step {step+1}/{total_steps_in_epoch} | "
                f"Loss: {log_dict['train/loss_total']:.4f} (CE:{log_dict['train/loss_ce']:.4f} "
                f"Sim:{log_dict['train/loss_similarity']:.4f} Div:{log_dict['train/loss_diversity']:.4f}) | "
                f"LR: {log_dict['lr']:.6f} | Speed: {log_dict['tokens_per_sec']:.0f} tok/s",
                accelerator
            )

            if args.logging.use_swanlab and swanlab_run:
                swanlab_run.log(log_dict)

        eval_interval = args.training.eval_interval
        start_eval = args.training.start_eval

        if (step + 1) % eval_interval == 0 and (step + 1) >= start_eval and eval_loader is not None:
            if accelerator.is_main_process:
                Logger(f"开始评估...", accelerator)

            performance = eval_model_sft(
                model=model,
                eval_loader=eval_loader,
                tokenizer=tokenizer,
                accelerator=accelerator,
                args=args,
                judger_mode=args.training.judger_mode
            )

            # 记录评估指标到SwanLab
            if accelerator.is_main_process and args.logging.use_swanlab and swanlab_run:
                swanlab_run.log({
                    "val/eval_accuracy": performance['overall']['accuracy'],
                    "val/eval_total_steps": performance['overall']['total_steps'],
                    "val/eval_total_correct": performance['overall']['total_correct']
                })

            # 基于准确率保存最佳模型
            current_accuracy = performance['overall']['accuracy']
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                if accelerator.is_main_process:
                    ckp = f'{args.logging.save_dir}/sft_best_acc_{hidden_size}{moe_path}.pth'
                    # 使用 DeepSpeed/Accelerate 方式获取模型状态字典（自动处理分片参数）
                    try:
                        # accelerator.get_state_dict 会自动收集 DeepSpeed 分片的参数
                        state_dict = accelerator.get_state_dict(model, unwrap=False)
                        # 提取模型权重（排除 optimizer 等）
                        model_state_dict = {}
                        for k, v in state_dict.items():
                            # 跳过 optimizer 和 scheduler 相关的键
                            if not any(x in k for x in ['optimizer', 'scheduler', 'lr_scheduler']):
                                # 移除可能的 'module.' 前缀
                                model_key = k[7:] if k.startswith('module.') else k
                                model_state_dict[model_key] = v.cpu() if hasattr(v, 'cpu') else v
                        
                        torch.save(model_state_dict, ckp)
                        Logger(f"最佳准确率模型已保存: {ckp} (acc={best_accuracy:.4f})", accelerator)
                    except Exception as e:
                        Logger(f"保存模型失败: {e}", accelerator)

        # 基于损失保存模型（仅主进程）
        if accelerator.is_main_process:
            loss_total = loss.item() * args.training.accumulation_steps
            if best_loss > loss_total:
                best_loss = loss_total
                ckp = f'{args.logging.save_dir}/sft_{hidden_size}{moe_path}.pth'

                # 使用 DeepSpeed/Accelerate 方式获取模型状态字典（自动处理分片参数）
                try:
                    # accelerator.get_state_dict 会自动收集 DeepSpeed 分片的参数
                    state_dict = accelerator.get_state_dict(model, unwrap=False)
                    # 提取模型权重（排除 optimizer 等）
                    model_state_dict = {}
                    for k, v in state_dict.items():
                        # 跳过 optimizer 和 scheduler 相关的键
                        if not any(x in k for x in ['optimizer', 'scheduler', 'lr_scheduler']):
                            # 移除可能的 'module.' 前缀
                            model_key = k[7:] if k.startswith('module.') else k
                            model_state_dict[model_key] = v.cpu() if hasattr(v, 'cpu') else v
                    
                    torch.save(model_state_dict, ckp)
                    Logger(f"最佳损失模型已保存: {ckp} (loss={best_loss:.4f})", accelerator)
                except Exception as e:
                    Logger(f"保存模型失败: {e}", accelerator)
