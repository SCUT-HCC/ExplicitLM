#!/bin/bash
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "torch",
# ]
# ///

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Path configuration
DATA_DIR="experiments/dnn_cosine_sim"
META_PATH="$DATA_DIR/meta.json"
DATA_PATH="$DATA_DIR/train_with_labels.jsonl"

# Check if metadata exists
if [ ! -f "$META_PATH" ]; then
    echo "Error: $META_PATH not found. Please run prepare_data.py first."
    exit 1
fi

# Extract knowledge_num
KNOWLEDGE_NUM=$(python3 -c "import json; print(json.load(open('$META_PATH'))['knowledge_num'])")
echo "Detected Knowledge Num: $KNOWLEDGE_NUM"

# Launch training
echo "Starting training..."
accelerate launch experiments/dnn_cosine_sim/train_router.py \
    --data_path "$DATA_PATH" \
    --knowledge_num "$KNOWLEDGE_NUM" \
    --batch_size 8 \
    --epochs 3 \
    --lr 1e-4 \
    --output_dir "$DATA_DIR/checkpoints" \
    --wandb_project "dnn-cosine-sim"
