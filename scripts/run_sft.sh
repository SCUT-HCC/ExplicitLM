#!/bin/bash
# 记忆组件训练启动脚本（Memory Components Training）
# 只训练 MemoryGate、Fusion、MemoryNorm，Qwen3 backbone 完全冻结

# ========== 配置区域 ==========
# 设置GPU可见设备（平衡显存）
export CUDA_VISIBLE_DEVICES=4,5

# 设置PyTorch内存分配配置
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 设置SwanLab API Key
export SWANLAB_API_KEY=GtiI1qjU5lco6MKKSrRmN

# 进入项目目录
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

# 设置进程显示名称（在 nvidia-smi 中显示的名称）
export PYTHON_PROCESS_NAME="llama-env"

# 优先使用 uv
if command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ 使用 uv 运行训练"
    ACCELERATE_CMD="uv run accelerate launch --config_file accelerate_config.yaml"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 已激活虚拟环境: $(which python)"
    ACCELERATE_CMD="accelerate launch --config_file accelerate_config.yaml"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 已激活venv虚拟环境: $(which python)"
    ACCELERATE_CMD="accelerate launch --config_file accelerate_config.yaml"
else
    echo "⚠️  未找到虚拟环境，使用系统Python: $(which python)"
    ACCELERATE_CMD="accelerate launch --config_file accelerate_config.yaml"
fi

# 显示GPU信息
echo "=========================================="
echo "使用GPU"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | grep -E "^[67],"

# ========== 训练配置 ==========

QWEN3_MODEL_PATH="/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b" 
CACHE_PATH="data/cache/knowledge_cache.pt"
PRETRAINED_ROUTER_PATH=""  # Router 预训练权重路径（可选）
PRETRAINED_FUSION_PATH=""  # Fusion 预训练权重路径（可选）
SFT_DATASET_PATH="sft_data/omcq_trex_sft.jsonl"
SFT_VAL_DATASET_PATH="data/benchmarks/eval_data.json"

# 训练超参数（优化内存使用）
LEARNING_RATE=5e-5
BATCH_SIZE=1  # 进一步减小批次大小：2 -> 1，避免OOM（Qwen3-4B hidden_size=2560，内存消耗大）
ACCUMULATION_STEPS=128  # 相应增加梯度累积：64 -> 128，保持有效批次大小
EPOCHS=3
MAX_SEQ_LEN=256  # 保持256，进一步减小可能影响训练效果

echo ""
echo "=========================================="
echo "🚀 启动记忆组件训练（Memory Components Training）"
echo "=========================================="
echo ""
echo "配置："
echo "  - Qwen3 模型: $QWEN3_MODEL_PATH"
echo "  - Cache 路径: $CACHE_PATH"
echo "  - Router 权重: $PRETRAINED_ROUTER_PATH"
echo "  - Fusion 权重: $PRETRAINED_FUSION_PATH"
echo "  - 训练数据: $SFT_DATASET_PATH"
echo "  - 验证数据: $SFT_VAL_DATASET_PATH"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 梯度累积: $ACCUMULATION_STEPS"
    echo "  - 训练轮数: $EPOCHS"
    echo "  - 最大序列长度: $MAX_SEQ_LEN"
    echo "  - DeepSpeed Stage: 2 (ZeRO-2)"
    echo "  - Checkpoint 保存目录: out/"
    echo ""

# 检查必要文件
if [ ! -f "$SFT_DATASET_PATH" ]; then
    echo "❌ 错误: SFT 训练数据不存在: $SFT_DATASET_PATH"
    echo "请先运行数据转换脚本:"
    echo "  python3 scripts/convert_omcq_to_sft.py --input sft_data/omcq_trex_data.json --output $SFT_DATASET_PATH"
    exit 1
fi

# 确保输出目录存在
mkdir -p out

# 启动训练
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

$ACCELERATE_CMD train_memory.py \
    model.qwen3_model_path="$QWEN3_MODEL_PATH" \
    model.cache_path="$CACHE_PATH" \
    model.recompute_cache=False \
    model.database_init_path="" \
    model.knowledge_num=1048576 \
    model.knowledge_dim=1536 \
    model.max_seq_len="$MAX_SEQ_LEN" \
    training.num_candidates=16 \
    dataset.sft_dataset_path="$SFT_DATASET_PATH" \
    dataset.pretrained_router_path="$PRETRAINED_ROUTER_PATH" \
    dataset.pretrained_fusion_path="$PRETRAINED_FUSION_PATH" \
    dataset.sft_val_dataset_path="$SFT_VAL_DATASET_PATH" \
    training.learning_rate="$LEARNING_RATE" \
    training.batch_size="$BATCH_SIZE" \
    training.accumulation_steps="$ACCUMULATION_STEPS" \
    training.epochs="$EPOCHS" \
    training.zero_stage=2 \
    model.keys_path="data/keys.pt" \
    model.gate_rank=128 \
    model.fusion_rank=128 \
    logging.out_dir="out" \
    logging.save_dir="out"

echo ""
echo "=========================================="
echo "✅ 训练完成"
echo "=========================================="

