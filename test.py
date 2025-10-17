# config.py
"""
hydra-zen 版 MiniMind 配置
运行：
    python config.py +model=model_memory training.epochs=5
"""

from hydra_zen import (
    builds,
    store,
    zen,
    make_config,
    launch,
    instantiate,
)

# ------------------------------------------------------------------
# 1. 分块定义配置（与 YAML 一一对应）
# ------------------------------------------------------------------
ModelConf = builds(
    dict,
    model_type="minimind",
    model_variant="model_memory",
    dim=512,
    hidden_dim=None,
    n_layers=8,
    n_heads=16,
    n_kv_heads=8,
    vocab_size=6400,
    max_seq_len=512,
    dropout=0.0,
    norm_eps=1.0e-05,
    rope_theta=1_000_000.0,
    flash_attn=True,
    multiple_of=64,
    # memory & knowledge
    use_token_memory=True,
    knowledge_dim=128,
    knowledge_length=16,
    knowledge_num=1_048_576,
    cache_path="cache/knowledge_cache.pt",
    recompute_cache=False,
    disable_db=False,
    database_init_path=None,
    # moe
    use_moe=False,
    n_routed_experts=4,
    n_shared_experts=True,
    num_experts_per_tok=2,
    aux_loss_alpha=0.1,
    gumbel_temperature=1.0,
    norm_topk_prob=True,
    scoring_func="softmax",
    # ema
    use_ema_update=True,
    ema_decay=0.9,
    ema_update_freq=5,
    populate_full_signature=False,
)

LoggingConf = builds(
    dict,
    use_swanlab=True,
    swanlab_online=True,
    swanlab_project="MiniMind",
    log_interval=10,
    out_dir="out",
    save_dir="out",
    populate_full_signature=False,
)

DatasetConf = builds(
    dict,
    dataset_path="data/database/merged_pretrain.jsonl",
    val_dataset_path="data/benchmarks/eval_data.json",
    max_subject_len=8,
    max_predicate_len=4,
    max_object_len=8,
    populate_full_signature=False,
)

TrainingConf = builds(
    dict,
    batch_size=48,
    accumulation_steps=16,
    epochs=3,
    embeddings_epoch=2,
    learning_rate=2.0e-4,
    freeze_ratio=0.2,
    seq_aux=True,
    num_candidates=16,
    num_selected=1,
    transformers_version="4.57.0",
    populate_full_signature=False,
)

# ------------------------------------------------------------------
# 2. 注册到 Hydra-Zen 存储
# ------------------------------------------------------------------
store(
    make_config(
        model=ModelConf,
        logging=LoggingConf,
        dataset=DatasetConf,
        training=TrainingConf,
    ),
    name="lmconfig",
    package="_global_",
)

# ------------------------------------------------------------------
# 3. 示例训练函数（仅打印配置，可替换成真实训练逻辑）
# ------------------------------------------------------------------
def train_fn(cfg):
    """用户自定义训练入口"""
    print("Final merged config:")
    print(cfg)
    # 实例化任意需要对象
    # model = instantiate(cfg.model)
    # ...

# ------------------------------------------------------------------
# 4. 启动入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    launch(
        zen(train_fn),
        config_name="lmconfig",
        version_base="1.2",
    )