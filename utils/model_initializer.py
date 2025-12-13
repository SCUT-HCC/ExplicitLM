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
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
# 移除 hydra 依赖，使用标准库
from pathlib import Path
from utils.logger import Logger


# ---------- 以下代码用于处理记忆库数据 ----------
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
        recompute: bool = False,
    ) -> torch.Tensor:
        if not recompute and os.path.exists(cache_path):
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
def init_model(args: dict, accelerator=None):
    """
    统一的模型初始化接口（仅支持Qwen3架构）

    Args:
        args: 配置字典，需包含：
            - qwen3_model_path: Qwen3预训练模型路径（必需）
            - 记忆库相关配置（knowledge_num, knowledge_dim等）
        accelerator: Accelerator 对象，用于分布式训练时的日志输出

    Returns:
        (model, tokenizer) tuple
    """
    # 只支持 Qwen3 架构
    qwen3_model_path = args.get("qwen3_model_path", None)
    if qwen3_model_path is None:
        raise ValueError("必须指定qwen3_model_path参数，指向Qwen3-4B预训练模型路径")
    
    return _init_qwen3_model(args, accelerator)


def _init_qwen3_model(args: dict, accelerator=None):
    """Qwen3 架构模型初始化"""
    qwen3_model_path = args.get("qwen3_model_path", None)
    if qwen3_model_path is None:
        raise ValueError("必须指定qwen3_model_path参数，指向Qwen3-4B预训练模型路径")
    
    database_init_path = args.get("database_init_path", None)
    cache_path = args.get("cache_path", "cache/knowledge_cache.pt")
    recompute_cache = args.get("recompute_cache", False)

    Logger("开始模型初始化流程（Qwen3架构）", accelerator)
    
    qwen3_config = Qwen3Config.from_pretrained(qwen3_model_path)
    
    # 提取记忆库配置
    memory_cfg = {
        "knowledge_num": args.get("knowledge_num", 1024 * 1024),
        "knowledge_length": args.get("knowledge_length", 16),
        "knowledge_dim": args.get("knowledge_dim", 128),
        "num_candidates": args.get("num_candidates", 16),
        "num_selected": args.get("num_selected", 1),
        "gumbel_temperature": args.get("gumbel_temperature", 1.0),
        "use_moe": args.get("use_moe", False),
        "dropout": args.get("dropout", 0.0),
        # LoRA 配置
        "gate_rank": args.get("gate_rank", None),
        "fusion_rank": args.get("fusion_rank", None),
    }
    
    # 如果配置中指定了 keys_path，添加到 memory_cfg
    if "keys_path" in args and args.get("keys_path"):
        # 处理相对路径
        keys_path = args["keys_path"]
        if not os.path.isabs(keys_path):
            # 使用当前工作目录（不再依赖 hydra）
            original_cwd = os.getcwd()
            keys_path = os.path.join(original_cwd, keys_path)
        memory_cfg["keys_path"] = keys_path
    from models.core.ExplicitLM import ExplicitLM
    
    original_cwd = os.getcwd()
    local_tokenizer_path = Path(original_cwd) / "models" / "qwen_tokenizer"
    if local_tokenizer_path.exists() and (local_tokenizer_path / "tokenizer.json").exists():
        Logger(f"  - 使用本地tokenizer: {local_tokenizer_path}", accelerator)
        tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path), trust_remote_code=True)
    else:
        Logger(f"  - 从Qwen模型路径加载: {qwen3_model_path}", accelerator)
        tokenizer = AutoTokenizer.from_pretrained(qwen3_model_path, trust_remote_code=True)
    
    # 确保Qwen tokenizer的特殊token配置正确
    if tokenizer.pad_token is None:
        # Qwen tokenizer可能没有pad_token，使用eos_token作为pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        Logger("警告: Qwen tokenizer没有pad_token，使用eos_token作为pad_token", accelerator)

    # 创建模型（确保在CPU上初始化，DeepSpeed ZeRO-3会在prepare时处理）
    # 显式指定设备为CPU，避免模型初始化时占用GPU显存
    model = ExplicitLM(qwen3_config=qwen3_config, memory_cfg=memory_cfg)
    # 确保模型所有参数和buffers都在CPU上
    model = model.cpu()

    # 从Qwen3预训练模型加载权重
    try:
        from transformers import Qwen3ForCausalLM
        pretrained_model = Qwen3ForCausalLM.from_pretrained(
            qwen3_model_path,
            torch_dtype=torch.float32,  # 先加载为float32，后续可以转换
            device_map="cpu",
        )
        
        # 加载匹配的权重
        model_state_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        
        def map_pretrained_key(pretrained_key: str) -> str:
            """将预训练模型的层名称映射到我们的模型层名称"""
            # 移除 "model." 前缀
            if pretrained_key.startswith("model."):
                key = pretrained_key[6:]  # 移除 "model." 前缀
            else:
                key = pretrained_key
            
            # 映射层名称
            if key.startswith("layers."):
                # 将 layers.X.xxx 映射为 layers.X.qwen3_decoder.xxx
                parts = key.split(".", 2)
                if len(parts) >= 3:
                    layer_idx = parts[1]
                    rest = parts[2]
                    return f"layers.{layer_idx}.qwen3_decoder.{rest}"
                else:
                    return key
            else:
                # embed_tokens, norm, lm_head 等直接使用
                return key
        
        loaded_keys = []
        missing_keys = []
        shape_mismatches = []
        
        for key in model_state_dict.keys():
            # 跳过新增的参数
            if key.startswith("memory_bank") or key.startswith("tok_embeddings") or \
               "memory_gate" in key or "gated_memory_fusion" in key or "memory_norm" in key:
                continue
            
            # 尝试直接匹配
            if key in pretrained_state_dict:
                if model_state_dict[key].shape == pretrained_state_dict[key].shape:
                    model_state_dict[key] = pretrained_state_dict[key]
                    loaded_keys.append(key)
                else:
                    shape_mismatches.append(f"{key} (shape: {model_state_dict[key].shape} vs {pretrained_state_dict[key].shape})")
            else:
                # 尝试通过映射找到对应的预训练权重
                found = False
                for pretrained_key in pretrained_state_dict.keys():
                    mapped_key = map_pretrained_key(pretrained_key)
                    if mapped_key == key:
                        if model_state_dict[key].shape == pretrained_state_dict[pretrained_key].shape:
                            model_state_dict[key] = pretrained_state_dict[pretrained_key]
                            loaded_keys.append(key)
                            found = True
                            break
                        else:
                            shape_mismatches.append(f"{key} (shape: {model_state_dict[key].shape} vs {pretrained_state_dict[pretrained_key].shape})")
                            found = True
                            break
                
                if not found:
                    missing_keys.append(key)
        
        model.load_state_dict(model_state_dict, strict=False)
        Logger(f"权重加载完成: {len(loaded_keys)}个参数", accelerator)
        if missing_keys:
            Logger(f"警告: {len(missing_keys)}个参数未加载", accelerator)
            if len(missing_keys) <= 10:
                for key in missing_keys[:10]:
                    Logger(f"    - {key}", accelerator)
        if shape_mismatches:
            Logger(f"警告: {len(shape_mismatches)}个参数形状不匹配", accelerator)
            if len(shape_mismatches) <= 5:
                for key in shape_mismatches[:5]:
                    Logger(f"    - {key}", accelerator)
        
        # 清理临时模型
        del pretrained_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 强制同步，确保显存释放
            torch.cuda.synchronize()
        
    except Exception as e:
        Logger(f"警告: 从预训练模型加载权重失败: {e}", accelerator)
        Logger("将使用随机初始化的权重", accelerator)

    # 数据库 / 记忆库初始化（MOE 模式下跳过）
    use_moe = memory_cfg.get("use_moe", False)
    if use_moe:
        Logger("MOE 模式：跳过记忆库初始化", accelerator)
    elif database_init_path:
        Logger(f"  - 数据库路径: {database_init_path}", accelerator)
        Logger(f"  - 缓存路径: {cache_path}", accelerator)
        Logger(f"  - 知识库大小: {memory_cfg['knowledge_num']}", accelerator)
        Logger(f"  - 知识条目长度: {memory_cfg['knowledge_length']}", accelerator)
        
        _initialize_database(
            model=model,
            tokenizer=tokenizer,
            database_path=database_init_path,
            cache_path=cache_path,
            knowledge_num=memory_cfg["knowledge_num"],
            knowledge_length=memory_cfg["knowledge_length"],
            recompute=recompute_cache,
            model_type="qwen3_explicitlm",
            database_attribute="memory_bank",
            accelerator=accelerator,
        )
    else:
        Logger("警告: 未指定数据库初始化路径，记忆库将使用随机初始化", accelerator)

    # 冻结Qwen主模型参数，只保留记忆库相关参数可训练
    Logger("🔒 冻结Qwen主模型参数...", accelerator)
    frozen_params = 0
    trainable_params = 0
    
    # 冻结所有Qwen基础组件
    for name, param in model.named_parameters():
        # 保留可训练的参数：记忆库相关组件
        is_memory_component = any(keyword in name for keyword in [
            "memory_gate",  # 记忆门控模块
            "gated_memory_fusion",  # 记忆融合模块
            "memory_norm",  # 记忆归一化层
        ])
        
        # memory_bank存储的是token IDs（int64），训练时固定，推理时通过LLMLingua更新
        # 所以始终设置为不可训练
        is_memory_bank = "memory_bank" in name
        if is_memory_bank:
            param.requires_grad = False
            frozen_params += param.numel()
        elif is_memory_component:
            # 其他记忆相关组件始终可训练
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            # 冻结所有其他参数（Qwen主模型）
            param.requires_grad = False
            frozen_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"参数冻结完成: 冻结 {frozen_params / 1e6:.3f}M, 可训练 {trainable_params / 1e6:.3f}M", accelerator)

    return model, tokenizer


def load_pretrained_memory_gate(model: nn.Module, memory_gate_path: str, accelerator=None):
    """
    加载预训练的 MemoryGate 权重到 ExplicitLM 的所有层
    
    支持两种格式：
    1. 直接的 state_dict (OrderedDict)
    2. router_only.pt 格式: {'head': OrderedDict, 'memory_gate_cfg': dict}
    
    Args:
        model: ExplicitLM 模型实例
        memory_gate_path: MemoryGate 权重文件路径
        accelerator: Accelerator 实例（可选，用于日志）
    """
    from utils.logger import Logger
    
    Logger(f"加载预训练 MemoryGate 权重: {memory_gate_path}", accelerator)
    
    if not os.path.exists(memory_gate_path):
        raise FileNotFoundError(f"MemoryGate 权重文件不存在: {memory_gate_path}")
    
    # 加载权重
    loaded_data = torch.load(memory_gate_path, map_location='cpu')
    
    # 处理不同的文件格式
    if isinstance(loaded_data, dict):
        # 检查是否是 router_only.pt 格式
        if 'head' in loaded_data:
            Logger("检测到 router_only.pt 格式，提取 head 权重", accelerator)
            memory_gate_state = loaded_data['head']
            if 'memory_gate_cfg' in loaded_data:
                Logger(f"Router 配置: {loaded_data['memory_gate_cfg']}", accelerator)
        else:
            # 直接是 state_dict
            memory_gate_state = loaded_data
    else:
        # 直接是 OrderedDict
        memory_gate_state = loaded_data
    
    # 统计加载情况
    loaded_layers = 0
    total_params = 0
    missing_keys = []
    unexpected_keys = []
    
    # 遍历所有层，加载 MemoryGate 权重
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'memory_gate') and layer.memory_gate is not None:
            try:
                # 尝试加载权重
                missing, unexpected = layer.memory_gate.load_state_dict(memory_gate_state, strict=False)
                
                if missing:
                    missing_keys.extend([f"layer_{layer_idx}.{k}" for k in missing])
                if unexpected:
                    unexpected_keys.extend([f"layer_{layer_idx}.{k}" for k in unexpected])
                
                loaded_layers += 1
                total_params += sum(p.numel() for p in layer.memory_gate.parameters())
            except Exception as e:
                Logger(f"警告: 层 {layer_idx} 加载 MemoryGate 失败: {e}", accelerator)
    
    if loaded_layers == 0:
        raise ValueError("未找到任何 MemoryGate 模块，请检查模型结构")
    
    Logger(f"✓ MemoryGate 加载完成: {loaded_layers} 层, {total_params / 1e6:.3f}M 参数", accelerator)
    
    if missing_keys:
        Logger(f"警告: {len(missing_keys)} 个参数未找到（前5个）: {missing_keys[:5]}", accelerator)
    if unexpected_keys:
        Logger(f"警告: {len(unexpected_keys)} 个意外参数（前5个）: {unexpected_keys[:5]}", accelerator)
    
    return loaded_layers


def load_pretrained_fusion(model: nn.Module, fusion_path: str, accelerator=None):
    """
    加载预训练的知识融合组件权重（GatedMemoryFusion + memory_norm）
    
    Args:
        model: ExplicitLM 模型实例
        fusion_path: Fusion 权重文件路径
        accelerator: Accelerator 实例（可选，用于日志）
    """
    from utils.logger import Logger
    
    Logger(f"加载预训练 Fusion 权重: {fusion_path}", accelerator)
    
    if not os.path.exists(fusion_path):
        Logger(f"警告: Fusion 权重文件不存在: {fusion_path}，将跳过加载", accelerator)
        return False
    
    checkpoint = torch.load(fusion_path, map_location='cpu')
    
    # 只加载 fusion 相关的权重
    fusion_keys = [k for k in checkpoint.keys() if 'gated_memory_fusion' in k or 'memory_norm' in k]
    
    if not fusion_keys:
        Logger("警告: 检查点中未找到 Fusion 相关权重", accelerator)
        return False
    
    # 加载权重到所有层
    model_state_dict = model.state_dict()
    loaded_keys = []
    
    for key in fusion_keys:
        if key in model_state_dict:
            if model_state_dict[key].shape == checkpoint[key].shape:
                model_state_dict[key] = checkpoint[key]
                loaded_keys.append(key)
    
    model.load_state_dict(model_state_dict, strict=False)
    Logger(f"✓ Fusion 权重加载完成: {len(loaded_keys)} 个参数", accelerator)
    
    return True


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
    accelerator=None,
) -> None:
    if not os.path.exists(database_path):
        Logger(f"错误: 数据库文件不存在: {database_path}", accelerator)
        raise FileNotFoundError(f"数据库文件不存在: {database_path}")
    
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
        processor = MemoryBankProcessor(tokenizer)
        if not cache_path or cache_path == "cache/knowledge_cache.pt":
            cache_path = f"cache/memory_bank_init_{knowledge_num}_{knowledge_length}.pt"
        
        start_time = time.time()
        processed_tensor = processor.process_memory_bank(
            database_path=database_path,
            cache_path=cache_path,
            knowledge_num=knowledge_num,
            knowledge_length=knowledge_length,
            recompute=recompute,
        )
    
    if processed_tensor is None:
        raise ValueError("数据处理失败，返回的张量为None")
    
    _set_database_attribute(model, database_attribute, processed_tensor, accelerator)
    
    total_time = time.time() - start_time
    Logger(f"记忆库初始化完成: {processed_tensor.shape[0]}条目, 耗时{total_time:.2f}秒", accelerator)


def _set_database_attribute(model: nn.Module, attribute_path: str, data: torch.Tensor, accelerator=None) -> None:
    attributes = attribute_path.split(".")
    target = model
    for attr in attributes[:-1]:
        if not hasattr(target, attr):
            Logger(f"警告: 找不到属性 {attr}", accelerator)
            return
        target = getattr(target, attr)
    final_attr = attributes[-1]
    if hasattr(target, final_attr):
        # memory_bank 存储在 CPU 上以节省 GPU 显存
        if final_attr == "memory_bank":
            data = data.cpu()
        getattr(target, final_attr).data.copy_(data)
    else:
        Logger(f"警告: 找不到 model.{attribute_path}", accelerator)
        globals()["processed_database"] = data
