# Git 提交指南

## 快速开始

### 1. 初始化仓库（如果还没有）

```bash
cd /data2/zengzheni/lvchangwei/new_repo
git init
```

### 2. 一次性提交所有代码

```bash
# 添加所有文件（.gitignore会自动过滤）
git add .

# 查看将要提交的文件
git status

# 提交
git commit -m "feat: 初始化ExplicitLM项目

- 集成Qwen3模型支持
- 实现预训练和SFT训练流程
- 配置DeepSpeed Stage 3
- 添加数据处理和验证工具
- 完善文档和使用说明"
```

### 3. 连接到远程仓库（可选）

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## 提交信息规范

建议使用以下格式：

```
<type>: <subject>

<body>
```

### Type 类型：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 示例：

```bash
# 新功能
git commit -m "feat: 添加DeepSpeed Stage 3支持"

# 修复bug
git commit -m "fix: 修复验证数据集格式问题"

# 文档更新
git commit -m "docs: 更新SFT训练文档"
```

## 当前项目主要文件

### 核心代码
- `train_router.py` - MemoryGate 训练脚本
- `train_fusion.py` - Fusion 组件训练脚本
- `train_joint.py` - 联合微调脚本
- `train_memory.py` - 记忆组件训练脚本（只训练记忆模块，冻结 Qwen3 backbone）
- `config/` - 配置模块（使用字典格式，支持命令行参数覆盖）
- `utils/` - 工具函数

### 脚本
- `scripts/convert_omcq_to_sft.py` - 数据转换
- `scripts/test_sft_data.py` - 数据测试
- `scripts/run_sft_gpu67.sh` - 训练启动脚本

### 配置
- `pyproject.toml` - 项目依赖
- `accelerate_config.yaml` - Accelerate配置
- `ds_config.json` - DeepSpeed配置

### 文档
- `README.md` - 项目说明
- `docs/` - 详细文档

