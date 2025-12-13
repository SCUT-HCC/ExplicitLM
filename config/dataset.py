# 数据集配置（普通字典）
DatasetConf = {
    "dataset_path": "data/database/merged_pretrain.jsonl",  # 预训练数据集路径
    "val_dataset_path": "data/benchmarks/eval_data.jsonl",  # 验证数据集路径
    "max_subject_len": 8,
    "max_predicate_len": 4,
    "max_object_len": 8,
    # ---- sft 相关字段 ----
    "pretrained_router_path": "",  # Router 预训练权重路径（可选，如 router_only.pt）
    "pretrained_fusion_path": "",  # Fusion 预训练权重路径（可选）
    "sft_dataset_path": "data/database/merged_sft.jsonl",  # SFT 数据集路径
    "sft_val_dataset_path": "data/benchmarks/eval_data.json",  # SFT 验证数据集路径
    "system_message": "You are a helpful assistant.",  # 系统消息
}