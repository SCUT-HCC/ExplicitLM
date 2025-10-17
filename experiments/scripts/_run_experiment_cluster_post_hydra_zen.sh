#!/bin/bash
################################################################################
# ExplicitLM集群实验后处理脚本 - Hydra-Zen版
# 用途：在登陆节点执行后处理任务
#
# 调用方式：由用户手动调用
# 执行环境：登陆节点（有网络，可访问Git/DVC）
################################################################################

set -e
set -o pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 验证参数
if [ $# -ne 1 ]; then
    log_error "用法: $0 <exp_id>"
    echo "例如: $0 exp_001_hydra_zen"
    exit 1
fi

EXP_ID="$1"

log_info "========================================="
log_info "【集群模式 - 后处理阶段 - Hydra-Zen版】"
log_info "实验ID: $EXP_ID"
log_info "========================================="

# 路径定义
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"
RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}.json"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url_${EXP_ID}"
META_FILE="${PROJECT_ROOT}/.experiment_meta_${EXP_ID}"

################################################################################
# 加载状态文件
################################################################################
log_info "加载状态文件..."

if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    log_success "状态已加载: $STATE_FILE"
    log_info "  - 实验ID: $EXP_ID"
    log_info "  - 代码版本: ${CODE_COMMIT:0:8}"
    log_info "  - Hydra-Zen配置覆盖: $TRAIN_ARGS"
else
    log_error "未找到状态文件: $STATE_FILE"
    exit 1
fi

################################################################################
# 追踪模型权重
################################################################################
log_info "步骤1/4: 追踪模型权重到DVC..."

# 检查checkpoint目录
if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 列出生成的文件
log_info "生成的checkpoint文件:"
ls -lh "$CHECKPOINT_DIR"

# DVC追踪
dvc add "$CHECKPOINT_DIR"

# 获取DVC文件路径
CHECKPOINT_DVC="${CHECKPOINT_DIR}.dvc"

# 读取DVC文件的MD5哈希（作为权重版本标识）
if [ -f "$CHECKPOINT_DVC" ]; then
    CHECKPOINT_HASH=$(grep "md5:" "$CHECKPOINT_DVC" | awk '{print $2}')
    log_success "DVC追踪完成 (Hash: ${CHECKPOINT_HASH:0:8})"
else
    log_error "DVC文件生成失败: $CHECKPOINT_DVC"
    exit 1
fi

################################################################################
# 生成实验记录文件
################################################################################
log_info "步骤2/4: 生成实验记录文件..."

# 读取临时元数据
if [ -f "$META_FILE" ]; then
    EXPERIMENT_META=$(cat "$META_FILE")
else
    log_error "找不到元数据文件: $META_FILE"
    exit 1
fi

# Extract hyperparameters from TRAIN_ARGS (convert hydra_zen format to JSON)
PARAMS_JSON=$(python3 -c "
import sys, json
import re

# Parse hydra_zen style arguments (key=value format)
args_str = '$TRAIN_ARGS'
pairs = args_str.split()

params = {}
for pair in pairs:
    if '=' in pair:
        key, value = pair.split('=', 1)
        # Try to convert to appropriate type
        try:
            # Check if it's a numeric value first
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Try boolean values
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            # Keep as string otherwise
            else:
                pass
        params[key] = value

print(json.dumps(params, indent=2))
" 2>/dev/null || echo "{}")

# 获取环境信息
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "N/A")
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")

# 生成完整记录文件
cat > "$RECORD_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")",
    "script": "run_experiment_hydra_zen.sh",
    "command": "python 1_pretrain.py $TRAIN_ARGS"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "code_commit_short": "${CODE_COMMIT:0:8}",
    "data": {
      "dataset_commit": "$DATABASE_COMMIT",
      "dataset_commit_short": "${DATABASE_COMMIT:0:8}",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "database_init_commit_short": "${DATABASE_INIT_COMMIT:-N/A}",
      "cache_commit": "${CACHE_COMMIT:-N/A}",
      "cache_commit_short": "${CACHE_COMMIT:0:8}"
    },
    "checkpoint_dvc": "$CHECKPOINT_DVC",
    "checkpoint_hash": "$CHECKPOINT_HASH",
    "checkpoint_hash_short": "${CHECKPOINT_HASH:0:8}"
  },
  "hyperparameters": $PARAMS_JSON,
  "results": {
    "swanlab_url": "$(cat $SWANLAB_URL_FILE 2>/dev/null || echo "N/A")",
    "checkpoint_dir": "$CHECKPOINT_DIR"
  },
  "environment": {
    "python_version": "$PYTHON_VERSION",
    "cuda_version": "$CUDA_VERSION",
    "num_gpus": $NUM_GPUS
  },
  "reproduction": {
    "code_checkout": "git checkout $CODE_COMMIT",
    "data_checkout_steps": [
      "git checkout $DATABASE_COMMIT && dvc checkout data/database.dvc && git checkout -"
    ],
    "checkpoint_pull": "dvc pull ${CHECKPOINT_DVC}",
    "full_command": "# 1. 恢复代码版本\\\\ngit checkout $CODE_COMMIT\\\\n\\\\n# 2. 恢复数据集版本\\\\ngit checkout $DATABASE_COMMIT && dvc checkout data/database.dvc && git checkout -\\\\n\\\\n# 3. 运行训练\\\\npython 1_pretrain.py $TRAIN_ARGS"
  }
}
EOF

log_success "实验记录已生成: $RECORD_FILE"

# 显示记录文件内容
echo ""
log_info "========== 实验记录内容 =========="
cat "$RECORD_FILE" | python3 -m json.tool 2>/dev/null || cat "$RECORD_FILE"
log_info "=================================="
echo ""

################################################################################
# Git提交所有变更
################################################################################
log_info "步骤3/4: 提交所有变更到Git..."

# 显示将要提交的变更
echo ""
log_info "将要提交的变更："
git status --short
echo ""

# 添加所有变更
git add -A

# 提交（使用实验ID和描述）
git commit -m "exp: ${EXP_ID} - Hydra-Zen后处理提交"

log_success "所有变更已提交到Git"
log_info "Commit包含："
log_info "  - 记录文件: $RECORD_FILE"
log_info "  - DVC元文件: ${CHECKPOINT_DVC}"
log_info "  - 其他代码变更 (如有)"

################################################################################
# 清理临时文件
################################################################################
log_info "步骤4/4: 清理临时文件..."

rm -f "$SWANLAB_URL_FILE"
rm -f "$META_FILE"

log_success "清理完成"

################################################################################
# 实验总结
################################################################################
echo ""
log_success "========================================="
log_success "   后处理阶段完成！"
log_success "========================================="
echo ""
log_info "📋 记录文件: $RECORD_FILE"
log_info "🔬 SwanLab URL: $(cat $SWANLAB_URL_FILE 2>/dev/null || echo "N/A")"
log_info "💾 Checkpoint: $CHECKPOINT_DIR"
log_info "🏷️  代码版本: ${CODE_COMMIT:0:8}"
log_info " 权重哈希: ${CHECKPOINT_HASH:0:8}"
echo ""
log_success "========================================="