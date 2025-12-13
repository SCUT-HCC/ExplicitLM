from .model import ModelConf
from .dataset import DatasetConf
from .logging import LoggingConf
from .training import TrainingConf


def get_default_config():
    """
    获取默认配置字典
    
    返回:
        dict: 包含 model, dataset, logging, training 四个子配置的字典
    """
    return {
        "model": ModelConf.copy(),
        "dataset": DatasetConf.copy(),
        "logging": LoggingConf.copy(),
        "training": TrainingConf.copy(),
    }


def merge_config(default_cfg, override_dict):
    """
    合并配置，支持嵌套字典的深度合并
    
    参数:
        default_cfg: 默认配置字典
        override_dict: 覆盖配置字典（格式：{"model.qwen3_model_path": "path", ...} 或嵌套字典）
    
    返回:
        dict: 合并后的配置字典
    """
    import copy
    merged = copy.deepcopy(default_cfg)

    for key, value in override_dict.items():
        if "." in key:
            # 分割键路径
            keys = key.split(".")
            # 遍历到目标位置
            target = merged
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            # 设置值
            target[keys[-1]] = value
        else:
            # 直接设置顶层键
            merged[key] = value
    
    return merged