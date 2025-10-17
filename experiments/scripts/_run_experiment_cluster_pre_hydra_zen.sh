#!/bin/bash
################################################################################
# ExplicitLM集群实验预处理脚本 - Hydra-Zen版
# 用途：准备集群实验，生成状态文件
#
# 调用方式：由实验脚本source调用
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

# 验证必需变量
if [ -z "$EXP_ID" ] || [ -z "$EXP_DESC" ] || [ -z "$TRAIN_ARGS" ]; then
    log_error "缺少必需变量！"
    echo "需要在调用脚本中定义："
    echo "  EXP_ID=\"实验ID\""
    echo "  EXP_DESC=\"实验描述\""
    echo "  DATASET_VERSION=\"训练数据集版本(可选)\""
    echo "  VAL_DATASET_VERSION=\"验证数据集版本(可选)\""
    echo "  EMBEDDING_VERSION=\"预训练嵌入版本(可选)\""
    echo "  DATABASE_VERSION=\"知识库初始化版本(可选)\""
    echo "  CACHE_VERSION=\"缓存版本(可选)\""
    echo "  TRAIN_ARGS=\"Hydra-Zen配置覆盖参数，格式: 'param=value param2=value2'\""
    exit 1
fi

log_info "========================================="
log_info "【集群模式 - 预处理阶段 - Hydra-Zen版】"
log_info "实验ID: $EXP_ID"
log_info "实验描述: $EXP_DESC"
log_info "Hydra-Zen配置覆盖: $TRAIN_ARGS"
log_info "========================================="

# 路径定义
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"

################################################################################
# 前置检查
################################################################################
log_info "步骤1/3: 前置检查..."

# 检查是否在Git仓库中
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    log_error "当前不在Git仓库中！"
    exit 1
fi

# 检查DVC是否初始化
if [ ! -d "${PROJECT_ROOT}/.dvc" ]; then
    log_error "DVC未初始化！请先运行: dvc init"
    exit 1
fi

# 检查实验ID是否已存在
if [ -f "${PROJECT_ROOT}/experiments/records/${EXP_ID}.json" ]; then
    log_error "实验ID ${EXP_ID} 已存在！"
    log_info "现有记录文件: ${PROJECT_ROOT}/experiments/records/${EXP_ID}.json"
    read -p "是否覆盖？(y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "取消实验"
        exit 0
    fi
fi

# 创建必要目录
mkdir -p "${PROJECT_ROOT}/experiments/records"
mkdir -p "$CHECKPOINT_DIR"

log_success "前置检查通过"

################################################################################
# 记录代码版本
################################################################################
log_info "步骤2/3: 记录代码版本..."

# 记录当前HEAD的commit hash
CODE_COMMIT=$(git rev-parse HEAD)

log_success "代码版本已记录: ${CODE_COMMIT:0:8}"

# 检查是否有未提交的变更
if ! git diff --quiet || ! git diff --cached --quiet; then
    log_warning "检测到未提交的变更"
    git status --short
fi

################################################################################
# 生成状态文件
################################################################################
log_info "步骤3/3: 生成状态文件..."

# 生成状态文件，包含所有必需的变量
cat > "$STATE_FILE" <<EOF
# Cluster Experiment State File - Hydra-Zen Version
# Generated at $(date -u +'%Y-%m-%dT%H:%M:%SZ')

# 实验信息
export EXP_ID="$EXP_ID"
export EXP_DESC="$EXP_DESC"

# 代码版本
export CODE_COMMIT="$CODE_COMMIT"

# 数据版本
export DATASET_VERSION="$DATASET_VERSION"
export VAL_DATASET_VERSION="$VAL_DATASET_VERSION"
export EMBEDDING_VERSION="$EMBEDDING_VERSION"
export DATABASE_VERSION="$DATABASE_VERSION"
export CACHE_VERSION="$CACHE_VERSION"

# Hydra-Zen配置覆盖参数
export TRAIN_ARGS="$TRAIN_ARGS"

# 输出路径
export CHECKPOINT_DIR="$CHECKPOINT_DIR"
EOF

log_success "状态文件已生成: $STATE_FILE"

# 记录到实验元数据文件（用于后续处理）
cat > "${PROJECT_ROOT}/.experiment_meta_${EXP_ID}" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "script": "run_experiment_pre_hydra_zen.sh"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "data": {
      "dataset_commit": "N/A",
      "val_dataset_commit": "N/A",
      "embedding_commit": "N/A",
      "database_init_commit": "N/A",
      "cache_commit": "N/A"
    },
    "command": "python 1_pretrain.py $TRAIN_ARGS"
  }
}
EOF

echo ""
log_success "========================================="
log_success "   预处理阶段完成！"
log_success "========================================="
echo ""
log_info "📋 状态文件: $STATE_FILE"
log_info "🏷️  代码版本: ${CODE_COMMIT:0:8}"
log_info "🔧 准备就绪，可以提交到计算节点运行"
echo ""
log_info "下一步操作："
log_info "1. 提交作业到计算节点执行训练 (使用 _run_experiment_cluster_train_hydra_zen.sh)"
log_info "2. 训练完成后运行后处理脚本 (使用 _run_experiment_cluster_post_hydra_zen.sh)"
echo ""
log_success "========================================="