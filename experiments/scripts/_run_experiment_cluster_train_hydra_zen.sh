#!/bin/bash
################################################################################
# ExplicitLM集群实验训练脚本（计算节点执行）- Hydra-Zen版
# 用途：在计算节点执行训练，使用hydra_zen配置
#
# 调用方式：由实验脚本source调用
# 执行环境：计算节点（无网络，有GPU）
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
if [ -z "$EXP_ID" ]; then
    log_error "缺少实验ID！"
    echo "需要先定义 EXP_ID 或加载状态文件"
    exit 1
fi

log_info "========================================="
log_info "【集群模式 - 训练阶段 - Hydra-Zen版】"
log_info "实验ID: $EXP_ID"
log_info "========================================="

# 路径定义
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url_${EXP_ID}"

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
    log_warning "未找到状态文件: $STATE_FILE"
    log_info "使用当前环境变量（确保已正确设置）"

    # 设置默认路径
    CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"

    if [ -z "$TRAIN_ARGS" ]; then
        log_error "缺少训练参数 TRAIN_ARGS"
        exit 1
    fi
fi

################################################################################
# 运行训练
################################################################################
log_info "开始训练..."

# 清理旧的SwanLab URL文件
rm -f "$SWANLAB_URL_FILE"
rm -f "${PROJECT_ROOT}/.swanlab_url"

# 构建训练命令 - 使用hydra_zen格式的参数
TRAIN_CMD="python 1_pretrain.py $TRAIN_ARGS"

log_info "执行命令: $TRAIN_CMD"
echo ""

# 运行训练
eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    log_error "训练失败！"
    exit 1
fi

log_success "训练完成"

################################################################################
# 处理SwanLab URL（如果存在）
################################################################################
log_info "处理SwanLab URL..."

# 检查两个可能的URL文件位置
if [ -f "${PROJECT_ROOT}/.swanlab_url" ]; then
    SWANLAB_URL=$(cat "${PROJECT_ROOT}/.swanlab_url")
    # 复制到实验专用文件
    echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
    log_success "SwanLab URL已保存: $SWANLAB_URL"
elif [ -f "$SWANLAB_URL_FILE" ]; then
    SWANLAB_URL=$(cat "$SWANLAB_URL_FILE")
    log_success "SwanLab URL: $SWANLAB_URL"
else
    SWANLAB_URL="N/A"
    echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
    log_warning "未找到SwanLab URL"
fi

################################################################################
# 检查checkpoint生成
################################################################################
log_info "检查checkpoint生成..."

if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

log_info "生成的checkpoint文件:"
ls -lh "$CHECKPOINT_DIR"
log_success "Checkpoint生成完成"

echo ""
log_success "========================================="
log_success "   训练阶段完成！"
log_success "========================================="
echo ""
log_info "📋 下一步操作："
log_info "1. 将checkpoint同步回登陆节点（如需要）"
log_info "2. 在登陆节点运行后续脚本："
log_info "   source ${PROJECT_ROOT}/experiments/scripts/${EXP_ID}_post.sh"
echo ""
log_info "📝 训练结果已保存，等待后续处理"