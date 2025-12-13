# 旧代码归档

本目录包含已不再使用的旧版本代码文件。

## 文件说明

- `1_pretrain.py` - 旧版本的预训练脚本（使用小模型配置，已被新的训练流程替代）
- `LMConfig.py` - 旧版本的小模型配置类（dim=512, n_layers=8等，已被 Qwen3-4B 配置替代）
- `config_utils.py` - 使用旧配置类的工具函数（不再使用）

## 当前项目状态

当前项目基于 **Qwen3-4B**，使用以下训练脚本：
- `train_router.py` - MemoryGate 训练
- `train_fusion.py` - Fusion 组件训练
- `train_joint.py` - 联合微调
- `train_memory.py` - 记忆组件训练（原 `2_sft.py`）

所有模型架构参数从 Qwen3-4B 的 `config.json` 加载，不再使用小模型配置。

