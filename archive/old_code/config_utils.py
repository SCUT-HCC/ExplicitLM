"""
配置工具模块
用于自动化配置管理，从LMConfig提取参数并与命令行参数合并
"""

import argparse
import inspect
import typing
from models.configs.LMConfig import LMConfig


def _add_argument_by_type(parser: argparse.ArgumentParser, param_name: str,
                          param_type, default_value):
    """
    根据参数类型添加argparse参数

    Args:
        parser: argparse解析器
        param_name: 参数名
        param_type: 参数类型注解
        default_value: 默认值
    """
    # 处理bool类型（特殊处理）
    if param_type == bool:
        if default_value is False:
            # 默认False，添加--xxx来启用
            parser.add_argument(f'--{param_name}', action='store_true',
                              help=f'启用{param_name}')
        else:
            # 默认True，添加--no-xxx来禁用
            parser.add_argument(f'--no-{param_name}', action='store_true',
                              dest=param_name + '_disable',
                              help=f'禁用{param_name}')
        return

    # 处理typing泛型类型（如List[int]）
    origin = typing.get_origin(param_type)
    if origin is list:
        args = typing.get_args(param_type)
        element_type = args[0] if args else str
        parser.add_argument(f'--{param_name}', type=element_type, nargs='+',
                          help=f'{param_name} (列表类型)')
        return

    # 处理基本类型
    type_mapping = {
        int: int,
        float: float,
        str: str,
    }

    if param_type in type_mapping:
        parser.add_argument(f'--{param_name}', type=type_mapping[param_type],
                          help=f'{param_name} (默认: {default_value})')
    else:
        # 其他类型默认为字符串
        parser.add_argument(f'--{param_name}', type=str,
                          help=f'{param_name} (默认: {default_value})')


def setup_config() -> LMConfig:
    """
    一站式配置设置函数

    自动从LMConfig提取所有参数创建argparse解析器，
    解析命令行参数后与LMConfig默认值合并

    优先级：命令行参数 > LMConfig默认值

    Returns:
        LMConfig: 配置好的超参数对象

    使用示例：
        在main.py中:
        ```python
        from utils.config_utils import setup_config
        config = setup_config()
        ```
    """
    # 1. 自省LMConfig.__init__获取所有参数
    sig = inspect.signature(LMConfig.__init__)

    # 2. 创建argparse解析器（不设置default，遵循规则1）
    parser = argparse.ArgumentParser(description='模型超参数配置')

    for param_name, param in sig.parameters.items():
        # 跳过self和**kwargs
        if param_name in ['self', 'kwargs']:
            continue

        # 获取参数类型和默认值
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        default_value = param.default if param.default != inspect.Parameter.empty else None

        # 添加参数到解析器
        _add_argument_by_type(parser, param_name, param_type, default_value)

    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 创建LMConfig实例（使用默认值）
    config = LMConfig()

    # 5. 用命令行参数覆盖默认值
    args_dict = vars(args)
    for key, value in args_dict.items():
        # 处理bool类型的禁用标志
        if key.endswith('_disable'):
            original_key = key.replace('_disable', '')
            if value is True:  # 用户使用了--no-xxx
                setattr(config, original_key, False)
        elif value is not None:  # 用户提供了该参数
            setattr(config, key, value)

    return config


def print_config(config: LMConfig):
    """
    打印配置信息（按类别分组显示）

    Args:
        config: LMConfig实例
    """
    print("=" * 80)
    print("模型配置信息")
    print("=" * 80)

    # 基本模型架构参数
    print("\n【基本模型架构参数】")
    print(f"  dim: {config.dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_kv_heads: {config.n_kv_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  multiple_of: {config.multiple_of}")
    print(f"  norm_eps: {config.norm_eps}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  rope_theta: {config.rope_theta}")
    print(f"  dropout: {config.dropout}")
    print(f"  flash_attn: {config.flash_attn}")
    print(f"  embeddings_epoch: {config.embeddings_epoch}")

    # DB相关配置
    print("\n【DB相关配置】")
    print(f"  disable_db: {config.disable_db}")

    # MOE相关配置
    print("\n【MOE相关配置】")
    print(f"  use_moe: {config.use_moe}")
    if config.use_moe:
        print(f"  num_experts_per_tok: {config.num_experts_per_tok}")
        print(f"  n_routed_experts: {config.n_routed_experts}")
        print(f"  n_shared_experts: {config.n_shared_experts}")
        print(f"  scoring_func: {config.scoring_func}")
        print(f"  aux_loss_alpha: {config.aux_loss_alpha}")
        print(f"  seq_aux: {config.seq_aux}")
        print(f"  norm_topk_prob: {config.norm_topk_prob}")

    # 知识库相关配置
    print("\n【知识库相关配置】")
    print(f"  knowledge_num: {config.knowledge_num}")
    print(f"  knowledge_length: {config.knowledge_length}")
    print(f"  knowledge_dim: {config.knowledge_dim}")

    # 记忆更新相关配置
    print("\n【记忆更新相关配置】")
    print(f"  use_token_memory: {config.use_token_memory}")
    print(f"  freeze_ratio: {config.freeze_ratio}")
    print("  注意：记忆库在训练时固定，推理时通过 LLMLingua 更新")

    # 实验1.4.10相关配置
    print("\n【实验1.4.10: Gumbel-Softmax相关配置】")
    print(f"  num_candidates: {config.num_candidates}")
    print(f"  num_selected: {config.num_selected}")
    print(f"  gumbel_temperature: {config.gumbel_temperature}")

    # 三元组提取相关配置
    print("\n【三元组提取相关配置】")
    print(f"  max_subject_len: {config.max_subject_len}")
    print(f"  max_predicate_len: {config.max_predicate_len}")
    print(f"  max_object_len: {config.max_object_len}")

    # 模型初始化相关配置
    print("\n【模型初始化相关配置】")
    print(f"  model_variant: {config.model_variant}")
    print(f"  pretrained_embedding_path: {config.pretrained_embedding_path}")
    print(f"  database_init_path: {config.database_init_path}")
    print(f"  cache_path: {config.cache_path}")
    print(f"  recompute_cache: {config.recompute_cache}")

    print("\n" + "=" * 80)
