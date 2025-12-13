"""
Qwen3-4B 模型参数配置
从 /data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b/config.json 提取
这些参数是固定的，不能修改（用于加载预训练权重）
"""

# Qwen3-4B 固定架构参数（从预训练模型config.json读取）
Qwen34BConf = {
    # ===== 模型架构参数（固定，不可修改）=====
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    
    # 核心维度参数
    "hidden_size": 2560,              # 隐藏层维度
    "num_hidden_layers": 36,           # Transformer层数
    "num_attention_heads": 32,         # 注意力头数
    "num_key_value_heads": 8,          # Key-Value头数（GQA）
    "head_dim": 80,                    # 每个注意力头的维度 (hidden_size / num_attention_heads = 2560 / 32 = 80)
    "intermediate_size": 9728,         # MLP中间层维度 (hidden_size * 3.8)
    
    # 词汇表和位置编码
    "vocab_size": 151936,              # 词汇表大小
    "max_position_embeddings": 40960,  # 最大位置编码长度
    "rope_theta": 1000000,             # RoPE旋转位置编码的theta参数
    
    # 归一化和激活函数
    "rms_norm_eps": 1e-06,             # RMSNorm的epsilon
    "hidden_act": "silu",              # 激活函数类型
    
    # 注意力机制
    "attention_bias": False,           # 是否使用注意力偏置
    "attention_dropout": 0.0,          # 注意力dropout率
    "use_sliding_window": False,       # 是否使用滑动窗口注意力
    "sliding_window": None,            # 滑动窗口大小（None表示不使用）
    "max_window_layers": 36,           # 最大窗口层数
    
    # RoPE缩放
    "rope_scaling": None,              # RoPE缩放配置（None表示不使用）
    
    # 权重初始化
    "initializer_range": 0.02,         # 权重初始化范围
    
    # 词嵌入
    "tie_word_embeddings": True,       # 是否共享输入和输出词嵌入权重
    
    # 其他配置
    "use_cache": True,                 # 是否使用KV缓存
    "torch_dtype": "bfloat16",         # 模型数据类型
    "transformers_version": "4.51.0",  # Transformers版本
    
    # Tokenizer相关
    "bos_token_id": 151643,            # 开始token ID
    "eos_token_id": 151645,            # 结束token ID
    "pad_token_id": 151643,            # Padding token ID（通常使用bos_token_id）
}

