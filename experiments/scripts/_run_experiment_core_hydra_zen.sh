#!/bin/bash
################################################################################
# ExplicitLM核心实验运行脚本 - Hydra-Zen版
# 用途：被各实验脚本调用的核心逻辑，使用hydra_zen配置
#
# 调用方式：由实验脚本source调用，需要预先定义以下变量：
#   - EXP_ID: 实验ID
#   - EXP_DESC: 实验描述
#   - DATASET_VERSION: 训练数据集版本（Git commit hash，可选）
#   - EMBEDDING_VERSION: 预训练嵌入版本（可选）
#   - DATABASE_VERSION: 知识库初始化版本（可选）
#   - CACHE_VERSION: 缓存数据版本（可选）
#   - TRAIN_ARGS: Hydra-Zen配置覆盖参数 (格式: "param=value param2=value2")
#
# 示例：
#   EXP_ID="exp_001"
#   EXP_DESC="基线实验 Hydra-Zen配置版"
#   DATASET_VERSION=""
#   VAL_DATASET_VERSION=""
#   EMBEDDING_VERSION=""
#   DATABASE_VERSION=""
#   CACHE_VERSION=""
#   TRAIN_ARGS="training.epochs=10 model.knowledge_num=1048576"
#   source experiments/scripts/_run_experiment_core_hydra_zen.sh
################################################################################

set -e  # 遇到错误立即退出
set -o pipefail  # 管道命令中任何一个失败都返回失败

################################################################################
# 颜色定义
################################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# 日志函数
################################################################################
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

################################################################################
# 验证必需变量
################################################################################
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
log_info "实验ID: $EXP_ID"
log_info "实验描述: $EXP_DESC"
log_info "训练数据集版本: ${DATASET_VERSION:-当前版本}"
log_info "Hydra-Zen配置覆盖: $TRAIN_ARGS"
log_info "========================================="

################################################################################
# 目录和文件路径定义
################################################################################
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"
RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}.json"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url"
META_FILE="${PROJECT_ROOT}/.experiment_meta_${EXP_ID}"

################################################################################
# 前置检查
################################################################################
check_prerequisites() {
    log_info "步骤1/9: 前置检查..."

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
    if [ -f "$RECORD_FILE" ]; then
        log_error "实验ID ${EXP_ID} 已存在！"
        log_info "现有记录文件: $RECORD_FILE"
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
}

################################################################################
# 记录代码版本（训练前）
################################################################################
record_code_version() {
    log_info "步骤2/9: 记录代码版本..."

    # 记录当前HEAD的commit hash（训练前的代码状态）
    CODE_COMMIT=$(git rev-parse HEAD)

    log_success "代码版本已记录: ${CODE_COMMIT:0:8}"

    # 显示当前工作区状态
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_warning "检测到未提交的变更，将在训练后一起提交"
        git status --short
    fi
}

################################################################################
# 数据版本切换和同步（细粒度）
################################################################################
sync_data() {
    log_info "步骤3/9: 数据版本切换和同步（细粒度）..."

    # 保存当前分支
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # 同步函数：切换单个数据集到指定版本
    sync_dataset() {
        local dataset_name=$1
        local target_version=$2
        local dvc_file="data/${dataset_name}.dvc"

        if [ -n "$target_version" ]; then
            log_info "  - ${dataset_name}: 切换到版本 ${target_version:0:8}"

            # 切换到指定commit
            git checkout "$target_version" --quiet

            # 仅checkout该数据集
            dvc checkout "$dvc_file"

            # 记录该数据集版本
            eval "${dataset_name^^}_COMMIT=\"$target_version\""

            # 切回当前分支
            git checkout "$CURRENT_BRANCH" --quiet
        else
            log_info "  - ${dataset_name}: 使用当前版本"

            # 仅checkout该数据集
            dvc checkout "$dvc_file" 2>/dev/null || true

            # 获取该数据集对应的Git commit
            local commit=$(git log -1 --format="%H" -- "$dvc_file" 2>/dev/null || echo "$CODE_COMMIT")
            eval "${dataset_name^^}_COMMIT=\"$commit\""
        fi
    }

    # 同步训练数据集
    sync_dataset "database" "$DATASET_VERSION"         # data/database.dvc (训练数据集)

    # 可选数据集（仅在项目使用且指定了版本时同步）
    [ -n "$EMBEDDING_VERSION" ] && sync_dataset "embeddings" "$EMBEDDING_VERSION"      # data/embeddings.dvc (如果存在)
    [ -n "$DATABASE_VERSION" ] && sync_dataset "database_init" "$DATABASE_VERSION"    # data/database_init.dvc (如果存在)
    [ -n "$CACHE_VERSION" ] && sync_dataset "cache" "$CACHE_VERSION"                   # cache.dvc (如果存在)

    log_success "数据集同步完成"
    log_info "  - Database (训练数据): ${DATABASE_COMMIT:0:8}"
    # [ -n "$EMBEDDING_VERSION" ] && log_info "  - Embeddings (预训练权重): ${EMBEDDINGS_COMMIT:0:8}"
    # [ -n "$DATABASE_VERSION" ] && log_info "  - Database Init (知识库): ${DATABASE_INIT_COMMIT:0:8}"
    # [ -n "$CACHE_VERSION" ] && log_info "  - Cache: ${CACHE_COMMIT:0:8}"
}

################################################################################
# 记录实验元数据（训练前）
################################################################################
record_pre_training_meta() {
    log_info "步骤4/9: 记录训练前元数据..."

    # 生成时间戳
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # 记录到临时文件
    cat > "$META_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$TIMESTAMP",
    "script": "run_experiment_hydra_zen.sh"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "data": {
      "dataset_commit": "$DATABASE_COMMIT",
      "val_dataset_commit": "$BENCHMARKS_COMMIT",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "cache_commit": "${CACHE_COMMIT:-N/A}"
    }
  },
  "command": "python 1_pretrain.py $TRAIN_ARGS"
}
EOF

    log_success "元数据已记录到临时文件: $META_FILE"
}

################################################################################
# 运行训练
################################################################################
run_training() {
    log_info "步骤5/9: 开始训练..."

    # 清理旧的SwanLab URL文件
    rm -f "$SWANLAB_URL_FILE"

    # 构建训练命令 - 使用hydra_zen格式的参数
    TRAIN_CMD="python 1_pretrain.py $TRAIN_ARGS"

    log_info "执行命令: $TRAIN_CMD"
    echo ""

    # 运行训练（不捕获输出，直接显示）
    eval $TRAIN_CMD

    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        log_error "训练失败！"
        exit 1
    fi

    log_success "训练完成"
}

################################################################################
# 读取SwanLab URL
################################################################################
get_swanlab_url() {
    log_info "步骤6/9: 获取SwanLab实验URL..."

    # 从临时文件读取（需要1_pretrain.py配合写入）
    if [ -f "$SWANLAB_URL_FILE" ]; then
        SWANLAB_URL=$(cat "$SWANLAB_URL_FILE")
        log_success "SwanLab URL: $SWANLAB_URL"
    else
        SWANLAB_URL="N/A"
        log_warning "未找到SwanLab URL文件，可能未启用SwanLab或训练脚本未写入"
    fi
}

################################################################################
# 追踪模型权重
################################################################################
track_checkpoint() {
    log_info "步骤7/9: 追踪模型权重到DVC..."

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
}

################################################################################
# 生成实验记录文件
################################################################################
generate_record() {
    log_info "步骤8/9: 生成实验记录文件..."

    # 读取临时元数据
    EXPERIMENT_META=$(cat "$META_FILE")

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
      "embedding_commit_short": "${EMBEDDINGS_COMMIT:0:8}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "database_init_commit_short": "${DATABASE_INIT_COMMIT:0:8}",
      "cache_commit": "${CACHE_COMMIT:-N/A}",
      "cache_commit_short": "${CACHE_COMMIT:0:8}"
    },
    "checkpoint_dvc": "$CHECKPOINT_DVC",
    "checkpoint_hash": "$CHECKPOINT_HASH",
    "checkpoint_hash_short": "${CHECKPOINT_HASH:0:8}"
  },
  "hyperparameters": $PARAMS_JSON,
  "results": {
    "swanlab_url": "$SWANLAB_URL",
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
}

################################################################################
# Git提交所有变更（一次性提交）
################################################################################
commit_all_changes() {
    log_info "步骤9/9: 提交所有变更到Git..."

    # 显示将要提交的变更
    echo ""
    log_info "将要提交的变更："
    git status --short
    echo ""

    # 添加所有变更
    git add -A

    # 提交（使用实验ID和描述）
    git commit -m "exp: ${EXP_ID} - ${EXP_DESC}"

    log_success "所有变更已提交到Git"
    log_info "Commit包含："
    log_info "  - 实验脚本 (如有新增/修改)"
    log_info "  - 记录文件: $RECORD_FILE"
    log_info "  - DVC元文件: ${CHECKPOINT_DVC}"
    log_info "  - 其他代码变更 (如有)"
}

################################################################################
# 清理临时文件
################################################################################
cleanup() {
    log_info "清理临时文件..."
    rm -f "$SWANLAB_URL_FILE"
    rm -f "$META_FILE"
    log_success "清理完成"
}

################################################################################
# 实验总结
################################################################################
print_summary() {
    echo ""
    log_success "========================================="
    log_success "   实验 ${EXP_ID} 执行完成！"
    log_success "========================================="
    echo ""
    log_info "📋 记录文件: $RECORD_FILE"
    log_info "🔬 SwanLab URL: $SWANLAB_URL"
    log_info "💾 Checkpoint: $CHECKPOINT_DIR"
    log_info "🏷️  代码版本: ${CODE_COMMIT:0:8}"
    log_info "📊 训练数据集版本: ${DATABASE_COMMIT:0:8}"
    log_info " 权重哈希: ${CHECKPOINT_HASH:0:8}"
    echo ""
    log_info "复现命令（详见记录文件的reproduction字段）:"
    echo "  1. 恢复代码: git checkout $CODE_COMMIT"
    echo "  2. 恢复数据: 使用记录文件中的data_checkout_steps"
    echo "  3. 拉取权重: dvc pull ${CHECKPOINT_DVC}"
    echo ""
    log_success "========================================="
}

################################################################################
# 主流程
################################################################################
main() {
    check_prerequisites
    record_code_version
    sync_data
    record_pre_training_meta
    run_training
    get_swanlab_url
    track_checkpoint
    generate_record
    commit_all_changes
    cleanup
    print_summary
}

# 执行主流程
main