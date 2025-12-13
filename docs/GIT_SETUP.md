# Git 仓库设置指南

## 1. 初始化 Git 仓库

```bash
cd /data2/zengzheni/lvchangwei/new_repo

# 初始化git仓库（如果还没有）
git init

# 或者如果已经有远程仓库
git remote add origin <your-repo-url>
```

## 2. 配置 Git（首次使用）

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## 3. 检查需要提交的文件

```bash
# 查看所有文件状态
git status

# 查看会被忽略的文件
git status --ignored
```

## 4. 添加文件到暂存区

### 推荐：分阶段提交

```bash
# 第一阶段：核心代码和配置
git add ExplicitLM/*.py
git add ExplicitLM/config/
git add ExplicitLM/utils/
git add ExplicitLM/scripts/*.py
git add ExplicitLM/scripts/*.sh
git add ExplicitLM/pyproject.toml
git add ExplicitLM/accelerate_config.yaml
git add ExplicitLM/ds_config.json
git add ExplicitLM/README.md
git add ExplicitLM/docs/
git add .gitignore
git add ExplicitLM/.gitignore

# 提交核心代码
git commit -m "feat: 添加ExplicitLM核心代码和配置

- 添加Qwen3模型集成
- 添加SFT训练脚本和数据集处理
- 配置DeepSpeed Stage 3支持
- 添加数据处理工具和脚本"

# 第二阶段：文档和示例（如果需要）
git add ExplicitLM/examples/
git add README.md
git commit -m "docs: 添加使用文档和示例代码"
```

### 或者：一次性提交所有

```bash
# 添加所有文件（.gitignore会自动排除不需要的文件）
git add .

# 查看将要提交的文件
git status

# 提交
git commit -m "feat: 初始化ExplicitLM项目

- 添加Qwen3模型集成
- 添加预训练和SFT训练脚本
- 配置DeepSpeed Stage 3支持
- 添加数据处理和验证工具
- 添加完整的使用文档"
```

## 5. 查看提交历史

```bash
git log --oneline
```

## 6. 推送到远程仓库（如果有）

```bash
# 设置远程仓库
git remote add origin <your-repo-url>

# 推送主分支
git push -u origin main

# 或者如果使用master分支
git branch -M main  # 重命名为main
git push -u origin main
```

## 7. 创建标签（可选）

```bash
# 创建版本标签
git tag -a v0.1.0 -m "初始版本：Qwen3集成和SFT训练支持"
git push origin v0.1.0
```

## 8. 日常开发工作流

```bash
# 查看修改
git status
git diff

# 添加修改
git add <file>

# 提交
git commit -m "feat: 描述你的修改"

# 推送
git push
```

## 注意事项

1. **不要提交的文件**（已在.gitignore中）：
   - 模型权重文件（*.pth, *.pt）
   - 训练日志（*.log）
   - 缓存文件（__pycache__, *.pyc）
   - 虚拟环境（.venv/, venv/）
   - 训练输出（out/, outputs/）

2. **大文件处理**：
   - 如果必须提交大文件，考虑使用Git LFS
   - 或者使用DVC管理数据和模型

3. **敏感信息**：
   - 不要提交包含API密钥、密码等敏感信息的文件
   - 使用环境变量或配置文件模板

