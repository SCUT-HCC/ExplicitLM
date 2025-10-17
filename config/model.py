from hydra_zen import builds

ModelConf = builds(
    dict,
    # 网络规模
    model_type="minimind",
    model_variant="model_memory",
    dim=512,
    n_layers=8,
    n_heads=16,
    n_kv_heads=8,
    vocab_size=6400,
    max_seq_len=512,
    dropout=0.0,
    norm_eps=1.0e-5,
    rope_theta=1_000_000.0,
    flash_attn=True,
    multiple_of=64,
    # memory / knowledge
    use_token_memory=True,
    knowledge_dim=128,
    knowledge_length=16,
    knowledge_num=1_048_576,
    cache_path="cache/knowledge_cache.pt",
    recompute_cache=False,
    disable_db=False,
    database_init_path=None,
    freeze_ratio=0.2,
    # MoE
    use_moe=False,
    n_routed_experts=4,
    n_shared_experts=True,
    num_experts_per_tok=2,
    aux_loss_alpha=0.1,
    gumbel_temperature=1.0,
    norm_topk_prob=True,
    scoring_func="softmax",
    # EMA
    use_ema_update=True,
    ema_decay=0.9,
    ema_update_freq=5,
    populate_full_signature=False,
)