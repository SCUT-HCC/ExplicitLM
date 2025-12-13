# 模型配置（普通字典）
ModelConf = {
    # 模型类型（基于 Qwen3-4B，架构参数从 qwen3_model_path 加载）
    "model_type": "ExplicitLM",
    "model_variant": "model_memory",
    
    # Qwen3 模型路径（必需，用于加载预训练权重和配置）
    "qwen3_model_path": "",  # 必须通过命令行或配置文件指定
    
    # 序列长度
    "max_seq_len": 256,  # 最大序列长度
    
    # Memory / Knowledge 配置
    "use_token_memory": True,
    "knowledge_dim": 1536,  # 与 router 权重匹配
    "knowledge_length": 16,  # 每个记忆条目的 token 数
    "knowledge_num": 1024*1024,  # 1048576 个记忆条目
    "keys_path": "data/keys.pt",  # Product Key Memory 的 keys 路径
    "cache_path": "data/cache/knowledge_cache.pt",  # 记忆库缓存路径
    "recompute_cache": False,  # 是否重新计算缓存
    "disable_db": False,  # 是否禁用数据库功能
    "database_init_path": "data/knowledge_base/sentence_trex_data.json",  # 知识库初始化数据
    
    # LoRA 配置（用于参数高效训练）
    "gate_rank": 128,  # MemoryGate 低秩分解的 rank，None 表示使用原版本，>0 表示使用 LoRA 版本
    "fusion_rank": 128,  # 融合模块低秩分解的 rank，None 表示使用原版本，>0 表示使用 LoRA 版本
    
    # MoE 配置（当前未使用）
    "use_moe": False,
    "n_routed_experts": 4,
    "n_shared_experts": True,
    "num_experts_per_tok": 2,
    "aux_loss_alpha": 0.1,
    "gumbel_temperature": 1.0,
    "norm_topk_prob": True,
    "scoring_func": "softmax",
}