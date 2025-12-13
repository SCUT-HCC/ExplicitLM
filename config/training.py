# 训练配置（普通字典）
TrainingConf = {
    # ---- 优化器 / 学习率 ----
    "batch_size": 4,
    "accumulation_steps": 32,
    "epochs": 3,
    "embeddings_epoch": 2,
    "learning_rate": 2.0e-4,
    "seq_aux": True,
    "num_candidates": 16,
    "num_selected": 1,
    "transformers_version": "4.57.0",
    # ---- 原 trainer 字段 ----
    # DeepSpeed / Accelerator
    "zero_stage": 2,                # ZeRO-2
    "mixed_precision": "bf16",      # 混合精度
    "seed": 1337,                   # 随机种子
    "devices": "auto",              # 多卡
    "strategy": "deepspeed_stage_3",
    "log_interval": 10,

    # ---- SFT 特定参数 ----
    "max_new_tokens": 50,           # 生成的最大token数
    "eval_interval": 500,           # 评估间隔（步数）
    "start_eval": 100,              # 开始评估的步数
    "judger_mode": "startswith",    # 判断模式
    "show_eval_res": 5,             # 评估样例展示数量
    "similarity_loss_coef": 0.1,    # 相似度损失系数
    "diversity_loss_coef": 0.05,    # 多样性损失系数
    "eval_num_samples": 200,        # 评估样本数量
}