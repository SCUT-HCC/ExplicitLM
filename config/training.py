from hydra_zen import builds

# --------------- 原 training 超参 ---------------
TrainingConf = builds(
    dict,
    # ---- 优化器 / 学习率 ----
    batch_size=4,
    accumulation_steps=16,
    epochs=3,
    embeddings_epoch=2,
    learning_rate=2.0e-4,
    seq_aux=True,
    num_candidates=16,
    num_selected=1,
    transformers_version="4.57.0",
    # ---- 原 trainer 字段 ----
    # DeepSpeed / Accelerator
    zero_stage=2,                # ZeRO-2
    mixed_precision="bf16",      # 混合精度
    seed=1337,                   # 随机种子
    devices="auto",              # 多卡
    strategy="deepspeed_stage_2",
    log_interval=10,
    populate_full_signature=False,
)