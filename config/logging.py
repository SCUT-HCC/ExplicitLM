from hydra_zen import builds

LoggingConf = builds(
    dict,
    use_swanlab=True,
    swanlab_online=True,
    swanlab_project="MiniMind",
    log_interval=1,
    out_dir="out",
    save_dir="out",
    populate_full_signature=False,
)