#!/usr/bin/env python3
"""
记忆组件训练入口（Memory Components Training）

功能：
- 基于 Qwen3-4B 预训练模型，只训练记忆相关组件
- 冻结 Qwen3 backbone，只训练 MemoryGate、GatedMemoryFusion 和 MemoryNorm
- 使用对话格式数据训练
- 支持分布式训练和混合精度
"""
import os
import time
import gc
from typing import Optional, Any

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
import json
import sys
from config import get_default_config, merge_config
from utils.logger import Logger
from utils.sft_datasets import create_sft_dataloader, create_sft_eval_dataloader
from utils.train_loop_sft import train_epoch_sft
from utils.model_initializer import init_model, load_pretrained_memory_gate, load_pretrained_fusion
from pathlib import Path

try:
    import swanlab
except ImportError:
    swanlab = None


class ConfigDict:
    """配置字典包装类，支持点号访问（如 cfg.model.qwen3_model_path）"""
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigDict(value))
                else:
                    setattr(self, key, value)
        else:
            raise ValueError("ConfigDict 只能从字典创建")
    
    def get(self, key, default=None):
        """获取属性，如果不存在返回默认值"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """支持字典式访问"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """支持字典式设置"""
        if isinstance(value, dict):
            setattr(self, key, ConfigDict(value))
        else:
            setattr(self, key, value)
    
    def items(self):
        """返回 (key, value) 对，兼容字典接口"""
        result = []
        for k in dir(self):
            if not k.startswith('_') and not callable(getattr(self, k)):
                result.append((k, getattr(self, k)))
        return result
    
    def keys(self):
        """返回所有键，兼容字典接口"""
        return [k for k in dir(self) if not k.startswith('_') and not callable(getattr(self, k))]
    
    def values(self):
        """返回所有值，兼容字典接口"""
        return [getattr(self, k) for k in dir(self) if not k.startswith('_') and not callable(getattr(self, k))]
    
    def __iter__(self):
        """支持迭代"""
        return iter(self.keys())
    
    def to_dict(self):
        """转换为普通字典"""
        result = {}
        for k in self.keys():
            v = getattr(self, k)
            if isinstance(v, ConfigDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def main(cfg):
    """cfg 是配置对象（ConfigDict），包含 model, dataset, logging, training 四个子配置"""
    m_cfg = cfg.model
    d_cfg = cfg.dataset
    l_cfg = cfg.logging
    tr_cfg = cfg.training
    m_cfg = {k: getattr(m_cfg, k) for k in dir(m_cfg) if not k.startswith('_') and not callable(getattr(m_cfg, k))}
    d_cfg = {k: getattr(d_cfg, k) for k in dir(d_cfg) if not k.startswith('_') and not callable(getattr(d_cfg, k))}
    l_cfg = {k: getattr(l_cfg, k) for k in dir(l_cfg) if not k.startswith('_') and not callable(getattr(l_cfg, k))}
    tr_cfg = {k: getattr(tr_cfg, k) for k in dir(tr_cfg) if not k.startswith('_') and not callable(getattr(tr_cfg, k))}
    
    proj_root = Path(__file__).parent.resolve()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ds_plugin = DeepSpeedPlugin(zero_stage=tr_cfg["zero_stage"])
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=ds_plugin,
    )
    set_seed(tr_cfg["seed"] + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(l_cfg["out_dir"], exist_ok=True)
        os.makedirs(l_cfg["save_dir"], exist_ok=True)

    swanlab_run: Optional[Any] = None
    if l_cfg["use_swanlab"] and accelerator.is_main_process and swanlab is not None:
        mode = "cloud" if l_cfg["swanlab_online"] else "offline"
        Logger(f"SwanLab 模式：{mode}", accelerator)
        Logger(f"SwanLab 运行中...", accelerator)
        api_key = os.environ.get("SWANLAB_API_KEY", "GtiI1qjU5lco6MKKSrRmN")
        flat_config = {}
        for section in cfg.keys():
            values = getattr(cfg, section)
            if isinstance(values, ConfigDict):
                for key in values.keys():
                    value = getattr(values, key)
                    flat_config[f"{section}.{key}"] = value
            else:
                for key, value in values.items():
                    flat_config[f"{section}.{key}"] = value
        swanlab_run = swanlab.init(
            project=l_cfg["swanlab_project"],
            experiment_name=f"ExplicitLM-SFT-{tr_cfg['epochs']}e-{tr_cfg['batch_size']}b-{tr_cfg['learning_rate']}lr",
            config=flat_config,
            mode=mode,
            api_key=api_key
        )

    model, tokenizer = init_model(m_cfg)
    Logger("模型架构初始化完成", accelerator)

    if d_cfg.get('pretrained_router_path'):
        router_path = proj_root / d_cfg['pretrained_router_path']
        try:
            load_pretrained_memory_gate(model, str(router_path), accelerator)
            Logger("✓ Router 权重加载完成", accelerator)
        except (FileNotFoundError, Exception) as e:
            Logger(f"警告: 加载 Router 权重失败: {e}，将跳过加载", accelerator)

    if d_cfg.get('pretrained_fusion_path'):
        fusion_path = proj_root / d_cfg['pretrained_fusion_path']
        try:
            load_pretrained_fusion(model, str(fusion_path), accelerator)
        except (FileNotFoundError, Exception) as e:
            Logger(f"警告: 加载 Fusion 权重失败: {e}，将跳过加载", accelerator)

    Logger("🔒 设置参数冻结策略（只训练 memory 组件）", accelerator)
    frozen_params = 0
    trainable_params = 0
    memory_bank_params = 0
    keys_params = 0
    
    for name, param in model.named_parameters():
        is_keys = "keys" in name and "memory_gate" in name
        is_memory_bank = "memory_bank" in name
        is_memory_component = any(keyword in name for keyword in [
            "memory_gate", "gated_memory_fusion", "memory_norm"
        ]) and not is_keys
        
        if is_memory_bank:
            param.requires_grad = False
            memory_bank_params += param.numel()
            frozen_params += param.numel()
        elif is_keys:
            param.requires_grad = False
            keys_params += param.numel()
            frozen_params += param.numel()
        elif is_memory_component:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    Logger(f"参数冻结: 冻结 {frozen_params / 1e6:.3f}M, 可训练 {trainable_params / 1e6:.3f}M, Memory bank {memory_bank_params / 1e6:.3f}M, Keys {keys_params / 1e6:.3f}M", accelerator)

    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in optimizer_params)
    total_count = sum(p.numel() for p in model.parameters())
    Logger(f"优化器参数: {trainable_count / 1e6:.3f}M / {total_count / 1e6:.3f}M ({trainable_count / total_count * 100:.2f}%)", accelerator)
    
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    Logger(f"  - 可训练参数模块: {len(trainable_param_names)} 个", accelerator)
    if len(trainable_param_names) <= 10:
        for name in trainable_param_names:
            Logger(f"    * {name}", accelerator)
    else:
        Logger(f"    * {trainable_param_names[0]} ... (共{len(trainable_param_names)}个)", accelerator)

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=tr_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    model = model.cpu()
    Logger("模型准备完成（ZeRO Stage 2）", accelerator)

    train_loader = create_sft_dataloader(
        data_path=str(proj_root / d_cfg["sft_dataset_path"]),
        tokenizer=tokenizer,
        batch_size=tr_cfg["batch_size"],
        max_length=m_cfg["max_seq_len"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = create_sft_eval_dataloader(
        eval_data_path=str(proj_root / d_cfg["sft_val_dataset_path"]),
        batch_size=1,
        max_samples=tr_cfg["eval_num_samples"],
    )

    steps_per_epoch = len(train_loader) // tr_cfg["accumulation_steps"]
    epochs = tr_cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * steps_per_epoch * epochs),
        num_training_steps=steps_per_epoch * epochs,
    )

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    if isinstance(cfg, ConfigDict):
        cfg_dict = cfg.to_dict()
    else:
        cfg_dict = cfg
    Logger(f"配置信息: {json.dumps(cfg_dict, indent=2, default=str)}", accelerator)

    overall_start_time = time.time()

    for epoch in range(epochs):
        train_epoch_sft(
            epoch=epoch,
            accelerator=accelerator,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=ConfigDict(cfg) if isinstance(cfg, dict) else cfg,  # 传递配置对象以支持点号访问
            overall_start_time=overall_start_time,
            swanlab_run=swanlab_run,
            tokenizer=tokenizer,
            eval_loader=val_loader,
        )
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process and swanlab_run:
        # 获取 SwanLab 实验 URL
        exp_url = str(swanlab_run.public.cloud.experiment_url) if l_cfg["swanlab_online"] else 'local-mode'
        with open('.swanlab_url', 'w') as f:
            f.write(exp_url)
        Logger(f"SwanLab URL 已保存: {exp_url}", accelerator)

    if l_cfg["use_swanlab"] and accelerator.is_main_process:
        if swanlab_run is not None:
            swanlab_run.finish()
            Logger("SwanLab 运行已结束", accelerator)

    Logger("记忆组件训练完成！", accelerator)


def parse_args():
    """解析命令行参数"""
    # 支持 key=value 格式的参数
    overrides = {}
    
    def convert_value(value):
        """智能类型转换"""
        # 布尔值
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        # 尝试转换为整数
        try:
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
        except:
            pass
        
        # 尝试转换为浮点数（包括科学计数法）
        try:
            return float(value)
        except ValueError:
            pass
        
        # 如果都失败，返回原始字符串
        return value
    
    # 解析命令行参数
    for arg in sys.argv[1:]:
        if '=' in arg and not arg.startswith('--'):
            key, value = arg.split('=', 1)
            overrides[key] = convert_value(value)
    
    return overrides


if __name__ == "__main__":
    import sys
    cfg = get_default_config()
    overrides = parse_args()
    cfg = merge_config(cfg, overrides)
    
    if not cfg["model"].get("qwen3_model_path"):
        raise ValueError("必须指定 model.qwen3_model_path 参数（通过命令行: model.qwen3_model_path=/path/to/model）")
    
    cfg_obj = ConfigDict(cfg)
    main(cfg_obj)
