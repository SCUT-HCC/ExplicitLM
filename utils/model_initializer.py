"""
模型初始化工具模块

提供统一的模型初始化接口，支持多种模型类型和初始化策略。
包含权重初始化、预训练嵌入加载、知识数据库处理等功能。

兼容新版 dict 配置系统。
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hydra.utils import get_original_cwd
from pathlib import Path
from models.configs.LMConfig import LMConfig
from utils.logger import Logger


# ---------- 以下代码完全不变，仅把 args 当成 dict 使用 ----------
class ModelTypeConfig:
    """模型类型配置映射"""
    SUPPORTED_TYPES = {
        "model": {
            "module_path": "model.core.ExplicitLM",
            "class_name": "MiniMindLM",
            "requires_weight_init": True,
            "database_attribute": "knowledge_dataset.knowledge_dataset",
        },
        "model_original": {
            "module_path": "model.model_original",
            "class_name": "MiniMindLM",
            "requires_weight_init": False,
            "database_attribute": None,
        },
        "model_no_feed": {
            "module_path": "model.model_no_feed",
            "class_name": "MiniMindLM",
            "requires_weight_init": True,
            "database_attribute": "knowledge_dataset.knowledge_dataset",
        },
        "model_memory": {
            "module_path": "models.core.ExplicitLM",
            "class_name": "MiniMindLM",
            "requires_weight_init": True,
            "database_attribute": "memory_bank",
            "memory_optimization": True,
        },
    }

    @classmethod
    def get_config(cls, model_type: str) -> Dict[str, Any]:
        if model_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的模型类型: {model_type}。"
                f"支持的类型: {list(cls.SUPPORTED_TYPES.keys())}"
            )
        return cls.SUPPORTED_TYPES[model_type]


class WeightInitializer:
    """权重初始化器"""

    @staticmethod
    def initialize_model_weights(model: nn.Module, model_type: str) -> None:
        Logger("执行模型权重初始化...")
        RMSNorm = WeightInitializer._import_rmsnorm(model_type)
        WeightInitializer._init_embeddings(model)
        WeightInitializer._init_layers(model, RMSNorm)
        WeightInitializer._init_knowledge_components(model)
        Logger("模型权重初始化完成")

    @staticmethod
    def _import_rmsnorm(model_type: str):
        try:
            config = ModelTypeConfig.get_config(model_type)
            module = __import__(config["module_path"], fromlist=["RMSNorm"])
            return module.RMSNorm
        except (ImportError, AttributeError):
            Logger("警告: 无法导入RMSNorm，跳过RMSNorm初始化")
            return None

    @staticmethod
    def _init_embeddings(model: nn.Module) -> None:
        if hasattr(model, "tok_embeddings"):
            nn.init.normal_(model.tok_embeddings.weight, mean=0.0, std=0.02)
        if hasattr(model, "output"):
            is_shared = (
                hasattr(model, "tok_embeddings")
                and hasattr(model.tok_embeddings, "weight")
                and model.output.weight is model.tok_embeddings.weight
            )
            if not is_shared:
                nn.init.normal_(model.output.weight, mean=0.0, std=0.02)

    @staticmethod
    def _init_layers(model: nn.Module, RMSNorm) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif RMSNorm and isinstance(module, RMSNorm):
                if hasattr(module, "weight"):
                    nn.init.ones_(module.weight)

    @staticmethod
    def _init_knowledge_components(model: nn.Module) -> None:
        if hasattr(model, "knowledge_dataset") and hasattr(model.knowledge_dataset, "keys"):
            nn.init.normal_(model.knowledge_dataset.keys, mean=0.0, std=0.02)


class EmbeddingLoader:
    @staticmethod
    def load_pretrained_embeddings(model: nn.Module, embedding_path: str) -> None:
        Logger(f"加载预训练嵌入权重: {embedding_path}")
        pretrained_embeddings = torch.load(embedding_path)
        if hasattr(model, "tok_embeddings"):
            model.tok_embeddings.weight.data.copy_(pretrained_embeddings)
        if hasattr(model, "output"):
            model.output.weight.data.copy_(pretrained_embeddings)
        Logger("预训练嵌入权重加载完成")


class DatabaseProcessor:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 以下所有方法完全不变，仅把 args 当 dict 使用
    def load_or_process_database(
        self,
        database_path: str,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
        recompute: bool = False,
    ) -> torch.Tensor:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        processed_tensor = self._try_load_cache(
            cache_path, knowledge_num, knowledge_length, recompute
        )
        if processed_tensor is None:
            processed_tensor = self._process_database(
                database_path, cache_path, knowledge_num, knowledge_length
            )
        return processed_tensor

    def _try_load_cache(
        self,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
        recompute: bool,
    ) -> Optional[torch.Tensor]:
        if recompute or not os.path.exists(cache_path):
            return None
        try:
            Logger(f"加载缓存文件: {cache_path}")
            processed_tensor = torch.load(cache_path)
            cached_num, cached_length = processed_tensor.shape
            if cached_length != knowledge_length:
                Logger("缓存 knowledge_length 不匹配，重新计算...")
                return None
            if cached_num < knowledge_num:
                Logger("缓存 knowledge_num 不足，重新计算...")
                return None
            if cached_num > knowledge_num:
                processed_tensor = processed_tensor[:knowledge_num, :]
            Logger(f"成功加载缓存数据，形状: {processed_tensor.shape}")
            return processed_tensor
        except Exception as e:
            Logger(f"加载缓存失败: {e}，重新计算...")
            return None

    def _process_database(
        self,
        database_path: str,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
    ) -> torch.Tensor:
        Logger(f"加载数据库文件: {database_path}")
        with open(database_path, "r", encoding="utf-8") as f:
            database_data = json.load(f)
        sentences_data = self._extract_sentences(database_data)
        Logger(f"从数据库加载了 {len(sentences_data)} 条句子")
        processed_tensor, database_mapping = self._process_sentences(
            sentences_data, knowledge_num, knowledge_length
        )
        self._save_cache_and_mapping(
            processed_tensor, database_mapping, cache_path, database_path
        )
        return processed_tensor

    def _extract_sentences(self, database_data: List[Dict]) -> List[Dict[str, str]]:
        sentences_data = []
        for data in database_data:
            if "target" in data and len(data["target"]) > 0:
                target = data["target"][0]
                sentences_data.append(
                    {
                        "sentence": target.get("sentence", ""),
                        "uuid": target.get("uuid", ""),
                        "subject": target.get("subject", ""),
                        "predicate": target.get("predicate", ""),
                        "object": target.get("object", ""),
                    }
                )
        return sentences_data

    def _process_sentences(
        self,
        sentences_data: List[Dict[str, str]],
        knowledge_num: int,
        knowledge_length: int,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        Logger("处理句子数据...")
        processed_rows = []
        database_mapping = []
        num_to_process = min(knowledge_num, len(sentences_data))
        total_sentences = 0
        truncated_sentences = 0
        for i in range(num_to_process):
            sentence_data = sentences_data[i]
            sentence = sentence_data["sentence"]
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            original_length = len(sentence_tokens)
            total_sentences += 1
            if len(sentence_tokens) > knowledge_length:
                truncated_sentences += 1
                sentence_tokens = sentence_tokens[:knowledge_length]
            elif len(sentence_tokens) < knowledge_length:
                sentence_tokens.extend([self.pad_token_id] * (knowledge_length - len(sentence_tokens)))
            processed_rows.append(sentence_tokens)
            database_mapping.append(
                {
                    "database_index": i,
                    "uuid": sentence_data["uuid"],
                    "sentence": sentence,
                    "subject": sentence_data.get("subject", ""),
                    "predicate": sentence_data.get("predicate", ""),
                    "object": sentence_data.get("object", ""),
                    "token_count": len(sentence_tokens),
                    "is_truncated": original_length > knowledge_length,
                }
            )
        while len(processed_rows) < knowledge_num:
            processed_rows.append([self.pad_token_id] * knowledge_length)
        processed_tensor = torch.tensor(processed_rows, dtype=torch.long)
        self._log_statistics(
            total_sentences, truncated_sentences, num_to_process,
            knowledge_num, knowledge_length, processed_tensor.shape
        )
        return processed_tensor, database_mapping

    def _log_statistics(
        self,
        total_sentences: int,
        truncated_sentences: int,
        num_processed: int,
        knowledge_num: int,
        knowledge_length: int,
        final_shape: torch.Size,
    ) -> None:
        truncation_ratio = truncated_sentences / total_sentences if total_sentences > 0 else 0.0
        Logger(f"截断句子统计:")
        Logger(f"  - 总句子数: {total_sentences}")
        Logger(f"  - 截断句子数: {truncated_sentences}")
        Logger(f"  - 截断占比: {truncation_ratio:.4f} ({truncation_ratio*100:.2f}%)")
        Logger(f"数据处理完成:")
        Logger(f"  - 处理句子数: {num_processed}")
        Logger(f"  - 添加空条目数: {knowledge_num - num_processed}")
        Logger(f"  - 最终形状: {final_shape}")
        Logger(f"  - 期望形状: ({knowledge_num}, {knowledge_length})")

    def _save_cache_and_mapping(
        self,
        processed_tensor: torch.Tensor,
        database_mapping: List[Dict],
        cache_path: str,
        database_path: str,
    ) -> None:
        try:
            torch.save(processed_tensor, cache_path)
            Logger(f"处理结果已保存到: {cache_path}")
        except Exception as e:
            Logger(f"保存处理结果失败: {e}")
        try:
            mapping_file_path = cache_path.replace(".pt", "_mapping.json")
            mapping_data = {
                "metadata": {
                    "total_entries": len(database_mapping),
                    "knowledge_num": processed_tensor.shape[0],
                    "knowledge_length": processed_tensor.shape[1],
                    "source_file": database_path,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "mappings": database_mapping,
            }
            with open(mapping_file_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            Logger(f"数据库映射已保存到: {mapping_file_path}")
        except Exception as e:
            Logger(f"保存数据库映射失败: {e}")


class MemoryBankProcessor:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def process_memory_bank(
        self,
        database_path: str,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
    ) -> torch.Tensor:
        if os.path.exists(cache_path):
            Logger(f"从缓存加载memory_bank初始化数据: {cache_path}")
            processed_tensor = torch.load(cache_path)
            Logger(f"加载的memory_bank数据形状: {processed_tensor.shape}")
            return processed_tensor
        Logger(f"处理文本数据用于memory_bank初始化: {database_path}")
        with open(database_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        Logger(f"从 {database_path} 加载了 {len(data)} 条句子")
        processed_tensor, database_mapping = self._process_memory_sentences(
            data, knowledge_num, knowledge_length
        )
        self._save_memory_cache(
            processed_tensor, database_mapping, cache_path, database_path
        )
        return processed_tensor

    def _process_memory_sentences(
        self,
        data: List,
        knowledge_num: int,
        knowledge_length: int,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        processed_rows = []
        database_mapping = []
        total_sentences = len(data)
        truncated_sentences = 0
        num_to_process = min(len(data), knowledge_num)
        Logger(f"处理 {num_to_process}/{total_sentences} 条句子")
        for idx, item in enumerate(data[:num_to_process]):
            if idx % 1000 == 0:
                Logger(f"处理句子 {idx+1}/{num_to_process}")
            sentence_info = self._extract_sentence_info(item)
            sentence = sentence_info["sentence"]
            try:
                tokens_result = self.tokenizer(
                    sentence,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=len(sentence),
                    padding=False,
                    return_tensors="pt",
                )
                tokens = tokens_result["input_ids"].squeeze().tolist()
                if not isinstance(tokens, list):
                    tokens = [tokens]
                original_length = len(tokens)
                if len(tokens) > knowledge_length:
                    tokens = tokens[:knowledge_length]
                    truncated_sentences += 1
                elif len(tokens) < knowledge_length:
                    tokens.extend([self.pad_token_id] * (knowledge_length - len(tokens)))
                processed_rows.append(tokens)
                database_mapping.append(
                    {
                        "database_index": idx,
                        "uuid": sentence_info["uuid"],
                        "sentence": sentence,
                        "subject": sentence_info["subject"],
                        "predicate": sentence_info["predicate"],
                        "object": sentence_info["object"],
                        "token_count": len(tokens),
                        "is_truncated": original_length > knowledge_length,
                    }
                )
            except Exception as e:
                Logger(f"处理句子 {idx} 时出错: {e}")
                empty_tokens = [self.pad_token_id] * knowledge_length
                processed_rows.append(empty_tokens)
                database_mapping.append(
                    {
                        "database_index": idx,
                        "uuid": sentence_info["uuid"],
                        "sentence": sentence,
                        "subject": sentence_info["subject"],
                        "predicate": sentence_info["predicate"],
                        "object": sentence_info["object"],
                        "token_count": knowledge_length,
                        "is_truncated": False,
                        "processing_error": str(e),
                    }
                )
        while len(processed_rows) < knowledge_num:
            processed_rows.append([self.pad_token_id] * knowledge_length)
        processed_tensor = torch.tensor(processed_rows, dtype=torch.long)
        self._log_memory_statistics(
            total_sentences, truncated_sentences, num_to_process,
            knowledge_num, knowledge_length, processed_tensor.shape
        )
        return processed_tensor, database_mapping

    def _extract_sentence_info(self, item: Any) -> Dict[str, str]:
        if isinstance(item, dict):
            if "target" in item and len(item["target"]) > 0:
                target = item["target"][0]
                return {
                    "sentence": target.get("sentence", ""),
                    "uuid": target.get("uuid", ""),
                    "subject": target.get("subject", ""),
                    "predicate": target.get("predicate", ""),
                    "object": target.get("object", ""),
                }
            else:
                return {
                    "sentence": item.get("sentence", "") or item.get("text", "") or str(item),
                    "uuid": item.get("uuid", ""),
                    "subject": item.get("subject", ""),
                    "predicate": item.get("predicate", ""),
                    "object": item.get("object", ""),
                }
        else:
            return {
                "sentence": str(item),
                "uuid": "",
                "subject": "",
                "predicate": "",
                "object": "",
            }

    def _log_memory_statistics(
        self,
        total_sentences: int,
        truncated_sentences: int,
        num_processed: int,
        knowledge_num: int,
        knowledge_length: int,
        final_shape: torch.Size,
    ) -> None:
        truncation_ratio = truncated_sentences / total_sentences if total_sentences > 0 else 0.0
        Logger(f"截断句子统计:")
        Logger(f"  - 总句子数: {total_sentences}")
        Logger(f"  - 截断句子数: {truncated_sentences}")
        Logger(f"  - 截断占比: {truncation_ratio:.4f} ({truncation_ratio*100:.2f}%)")
        Logger(f"Memory_bank数据处理完成:")
        Logger(f"  - 处理句子数: {num_processed}")
        Logger(f"  - 添加空条目数: {knowledge_num - num_processed}")
        Logger(f"  - 最终形状: {final_shape}")
        Logger(f"  - 期望形状: ({knowledge_num}, {knowledge_length})")

    def _save_memory_cache(
        self,
        processed_tensor: torch.Tensor,
        database_mapping: List[Dict],
        cache_path: str,
        database_path: str,
    ) -> None:
        try:
            torch.save(processed_tensor, cache_path)
            Logger(f"处理结果已保存到: {cache_path}")
        except Exception as e:
            Logger(f"保存处理结果失败: {e}")
        try:
            mapping_file_path = cache_path.replace(".pt", "_mapping.json")
            mapping_data = {
                "metadata": {
                    "total_entries": len(database_mapping),
                    "knowledge_num": processed_tensor.shape[0],
                    "knowledge_length": processed_tensor.shape[1],
                    "source_file": database_path,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "mappings": database_mapping,
            }
            with open(mapping_file_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            Logger(f"数据库映射已保存到: {mapping_file_path}")
        except Exception as e:
            Logger(f"保存数据库映射失败: {e}")


# ------------------------------------------------------------------
# 统一入口函数：只改参数读取方式，其余不动
# ------------------------------------------------------------------
def init_model(args: dict):
    """
    统一的模型初始化接口（直接使用 dict 配置）

    Args:
        args: 配置字典，含全部超参

    Returns:
        (model, tokenizer) tuple
    """
    model_type = args.get("model_variant", "model_memory")
    pretrained_embedding_path = args.get("pretrained_embedding_path", None)
    database_init_path = args.get("database_init_path", None)
    cache_path = args.get("cache_path", "cache/knowledge_cache.pt")
    recompute_cache = args.get("recompute_cache", False)

    Logger(f"使用模型类型: {model_type}")
    type_config = ModelTypeConfig.get_config(model_type)

    # 动态导入模型类
    module = __import__(type_config["module_path"], fromlist=[type_config["class_name"]])
    MiniMindLM = getattr(module, type_config["class_name"])
    # 输出当前目录
    Logger(f"当前目录: {os.getcwd()}")
    # 加载 tokenizer
    tokenizer_dir = Path(get_original_cwd()) / "models" / "minimind_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    # 构造 LMConfig 对象（仅用于满足旧构造函数签名）
    # lm_config = LMConfig(
    #     dim=args["dim"],
    #     n_layers=args["n_layers"],
    #     n_heads=args["n_heads"],
    #     n_kv_heads=args["n_kv_heads"],
    #     vocab_size=args["vocab_size"],
    #     max_seq_len=args["max_seq_len"],
    #     knowledge_num=args["knowledge_num"],
    #     knowledge_length=args["knowledge_length"],
    #     knowledge_dim=args["knowledge_dim"],
    #     model_variant=model_type,
    #     pretrained_embedding_path=pretrained_embedding_path,
    #     database_init_path=database_init_path,
    #     cache_path=cache_path,
    #     recompute_cache=recompute_cache,
    #     use_moe=args.get("use_moe", False),
    #     flash_attn=args.get("flash_attn", True),
    #     dropout=args.get("dropout", 0.0),
    # )

    # 创建模型
    model = MiniMindLM(args)


    # 权重初始化
    if type_config["requires_weight_init"]:
        WeightInitializer.initialize_model_weights(model, model_type)

    if type_config.get("memory_optimization"):
        Logger("✅ 显存优化策略：候选项减少(32→16) + DeepSpeed参数offload")

    # 预训练嵌入
    if pretrained_embedding_path:
        EmbeddingLoader.load_pretrained_embeddings(model, pretrained_embedding_path)

    # 数据库 / 记忆库初始化
    if database_init_path and type_config["database_attribute"]:
        _initialize_database(
            model=model,
            tokenizer=tokenizer,
            database_path=database_init_path,
            cache_path=cache_path,
            knowledge_num=lm_config.knowledge_num,
            knowledge_length=lm_config.knowledge_length,
            recompute=recompute_cache,
            model_type=model_type,
            database_attribute=type_config["database_attribute"],
        )

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"LLM总参数量：{total_params:.3f} 百万")

    return model, tokenizer


# ---------- 以下辅助函数完全不变 ----------
def _initialize_database(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    database_path: str,
    cache_path: str,
    knowledge_num: int,
    knowledge_length: int,
    recompute: bool,
    model_type: str,
    database_attribute: str,
) -> None:
    if model_type == "model_memory":
        processor = MemoryBankProcessor(tokenizer)
        if not cache_path or cache_path == "cache/knowledge_cache.pt":
            cache_path = f"cache/memory_bank_init_{knowledge_num}_{knowledge_length}.pt"
        processed_tensor = processor.process_memory_bank(
            database_path=database_path,
            cache_path=cache_path,
            knowledge_num=knowledge_num,
            knowledge_length=knowledge_length,
        )
    else:
        processor = DatabaseProcessor(tokenizer)
        processed_tensor = processor.load_or_process_database(
            database_path=database_path,
            cache_path=cache_path,
            knowledge_num=knowledge_num,
            knowledge_length=knowledge_length,
            recompute=recompute,
        )
    _set_database_attribute(model, database_attribute, processed_tensor)
    Logger("数据库嵌入和句子已存储到模型")


def _set_database_attribute(model: nn.Module, attribute_path: str, data: torch.Tensor) -> None:
    attributes = attribute_path.split(".")
    target = model
    for attr in attributes[:-1]:
        if not hasattr(target, attr):
            Logger(f"警告: 找不到属性 {attr}，无法初始化数据库")
            return
        target = getattr(target, attr)
    final_attr = attributes[-1]
    if hasattr(target, final_attr):
        getattr(target, final_attr).data.copy_(data)
        Logger(f"成功初始化 model.{attribute_path} 使用处理后的数据")
    else:
        Logger(f"警告: 找不到 model.{attribute_path} 进行初始化")
        globals()["processed_database"] = data