#!/bin/bash
# Router 训练启动脚本

# ========== 配置区域 ==========
# 设置GPU可见设备（使用双卡：5, 6）
export CUDA_VISIBLE_DEVICES=5,6

# 设置PyTorch内存分配配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置SwanLab API Key（如果 .env 文件中没有设置）
if [ -z "$SWANLAB_API_KEY" ]; then
    export SWANLAB_API_KEY=dKWI69mdEndjB2P9YdY8f
fi

# 进入项目目录
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

# 显示GPU信息
echo "=========================================="
echo "使用GPU: 5, 6"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | grep -E "^[56],"

# 优先使用 uv
if command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ 使用 uv 运行训练"
    PYTHON_CMD="uv run python"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 已激活虚拟环境: $(which python)"
    PYTHON_CMD="python"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 已激活venv虚拟环境: $(which python)"
    PYTHON_CMD="python"
else
    echo "⚠️  未找到虚拟环境，使用系统Python: $(which python)"
    PYTHON_CMD="python"
fi

# ========== 训练配置 ==========
DATA_PATH="data/train_labeled.jsonl"
MODEL_NAME= "/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b"
OUTPUT_DIR="checkpoints/router"
BATCH_SIZE=1
LR=1e-4
EPOCHS=3
KNOWLEDGE_NUM=65536
KNOWLEDGE_DIM=2048
NUM_CANDIDATES=32
MAX_LENGTH=128
TEMPERATURE=0.5
SWANLAB_PROJECT="explicitlm-router"

# ========== 运行训练 ==========
echo "=========================================="
echo "开始 Router 训练"
echo "=========================================="

$PYTHON_CMD train_router.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --knowledge_num $KNOWLEDGE_NUM \
    --knowledge_dim $KNOWLEDGE_DIM \
    --num_candidates $NUM_CANDIDATES \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --swanlab_project "$SWANLAB_PROJECT"

echo "=========================================="
echo "训练完成"
echo "=========================================="

