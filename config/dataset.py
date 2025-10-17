from hydra_zen import builds

DatasetConf = builds(
    dict,
    dataset_path="data/database/merged_pretrain.jsonl",
    val_dataset_path="data/benchmarks/eval_data.jsonl",
    max_subject_len=8,
    max_predicate_len=4,
    max_object_len=8,
    populate_full_signature=False,
)