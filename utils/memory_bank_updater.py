"""
记忆库更新器：将提取的事实更新到 memory_bank
"""
import os
import json
import time
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path

from utils.fact_extractor import FactExtractor
from utils.logger import Logger


class MemoryBankUpdater:
    """记忆库更新器，支持多种更新策略"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        fact_extractor: Optional[FactExtractor] = None,
        update_strategy: str = "fifo",  # "fifo", "lru", "random", "similarity", "importance"
        max_entries: Optional[int] = None,
        similarity_threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        """
        初始化记忆库更新器
        
        Args:
            model: ExplicitLM 模型实例
            tokenizer: tokenizer 实例
            fact_extractor: 事实提取器（如果为None，将自动创建）
            update_strategy: 更新策略
                - "fifo": 先进先出，替换最旧的条目
                - "lru": 最近最少使用，替换最久未使用的条目
                - "random": 随机替换
                - "similarity": 基于相似度，替换最相似的条目
                - "importance": 基于重要性评分，替换最不重要的条目
            max_entries: 最大条目数（如果为None，使用memory_bank的当前大小）
            similarity_threshold: 相似度阈值（用于similarity策略）
            device: 运行设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.update_strategy = update_strategy
        self.similarity_threshold = similarity_threshold
        self.device = device or next(model.parameters()).device
        
        # 初始化事实提取器
        if fact_extractor is None:
            self.fact_extractor = FactExtractor(device=self.device)
        else:
            self.fact_extractor = fact_extractor
        
        # 获取 memory_bank 信息
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            self.memory_bank = model.memory_bank
            self.knowledge_num = model.memory_bank.shape[0]
            self.knowledge_length = model.memory_bank.shape[1]
        else:
            raise ValueError("模型没有 memory_bank 属性或 memory_bank 为 None")
        
        self.max_entries = max_entries or self.knowledge_num
        
        # 初始化使用统计（用于 LRU 策略）
        self.usage_stats = {
            'last_used': torch.zeros(self.knowledge_num, dtype=torch.long),
            'access_count': torch.zeros(self.knowledge_num, dtype=torch.long),
        }
        self.access_counter = 0
        
        # 初始化重要性评分（用于 importance 策略）
        self.importance_scores = torch.ones(self.knowledge_num, dtype=torch.float32)
        
        Logger(f"记忆库更新器初始化完成: 策略={update_strategy}, 大小={self.knowledge_num}x{self.knowledge_length}")
    
    def update_with_facts(
        self,
        facts: List[str],
        update_indices: Optional[List[int]] = None,
    ) -> Dict[str, any]:
        """
        使用提取的事实更新 memory_bank
        
        Args:
            facts: 提取的事实列表（字符串列表）
            update_indices: 要更新的索引列表（如果为None，根据策略自动选择）
        
        Returns:
            更新统计信息
        """
        if not facts:
            return {"updated_count": 0, "message": "没有事实需要更新"}
        
        # 将事实转换为 token IDs
        fact_token_ids = []
        valid_facts = []
        
        for fact in facts:
            if not fact or not fact.strip():
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(
                fact,
                add_special_tokens=False,
                max_length=self.knowledge_length,
                truncation=True,
                padding=False,
            )
            
            # 填充或截断到 knowledge_length
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            if len(tokens) < self.knowledge_length:
                tokens.extend([pad_token_id] * (self.knowledge_length - len(tokens)))
            else:
                tokens = tokens[:self.knowledge_length]
            
            fact_token_ids.append(tokens)
            valid_facts.append(fact)
        
        if not fact_token_ids:
            return {"updated_count": 0, "message": "没有有效的事实"}
        
        # 转换为 tensor
        fact_tensor = torch.tensor(fact_token_ids, dtype=torch.int64, device=self.device)
        num_facts = fact_tensor.shape[0]
        
        # 确定要更新的索引
        if update_indices is None:
            update_indices = self._select_update_indices(num_facts)
        
        # 确保索引数量匹配
        if len(update_indices) != num_facts:
            if len(update_indices) > num_facts:
                update_indices = update_indices[:num_facts]
            else:
                # 如果索引不足，补充更多索引
                additional_indices = self._select_update_indices(
                    num_facts - len(update_indices),
                    exclude_indices=update_indices
                )
                update_indices.extend(additional_indices)
        
        # 更新 memory_bank
        with torch.no_grad():
            for i, idx in enumerate(update_indices):
                if 0 <= idx < self.knowledge_num:
                    self.memory_bank[idx] = fact_tensor[i]
                    # 更新使用统计
                    self.usage_stats['last_used'][idx] = self.access_counter
                    self.usage_stats['access_count'][idx] += 1
                    self.access_counter += 1
        
        return {
            "updated_count": len(update_indices),
            "update_indices": update_indices[:10],  # 只返回前10个
            "facts_count": num_facts,
            "strategy": self.update_strategy,
        }
    
    def _select_update_indices(
        self,
        num_indices: int,
        exclude_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """根据策略选择要更新的索引"""
        exclude_set = set(exclude_indices or [])
        available_indices = [i for i in range(self.knowledge_num) if i not in exclude_set]
        
        if self.update_strategy == "fifo":
            # 先进先出：选择最旧的条目（使用统计中最小的 last_used）
            sorted_indices = sorted(
                available_indices,
                key=lambda i: self.usage_stats['last_used'][i].item()
            )
            return sorted_indices[:num_indices]
        
        elif self.update_strategy == "lru":
            # 最近最少使用：选择最久未使用的条目
            sorted_indices = sorted(
                available_indices,
                key=lambda i: (
                    self.usage_stats['last_used'][i].item(),
                    -self.usage_stats['access_count'][i].item()  # 访问次数少的优先
                )
            )
            return sorted_indices[:num_indices]
        
        elif self.update_strategy == "random":
            # 随机选择
            import random
            return random.sample(available_indices, min(num_indices, len(available_indices)))
        
        elif self.update_strategy == "similarity":
            # 基于相似度：选择最相似的条目（需要计算embedding相似度）
            # 这里简化实现，使用随机选择
            import random
            return random.sample(available_indices, min(num_indices, len(available_indices)))
        
        elif self.update_strategy == "importance":
            # 基于重要性：选择重要性评分最低的条目
            sorted_indices = sorted(
                available_indices,
                key=lambda i: self.importance_scores[i].item()
            )
            return sorted_indices[:num_indices]
        
        else:
            # 默认使用 FIFO
            sorted_indices = sorted(
                available_indices,
                key=lambda i: self.usage_stats['last_used'][i].item()
            )
            return sorted_indices[:num_indices]
    
    def update_from_text(
        self,
        text: str,
        compression_rate: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        从原始文本提取事实并更新 memory_bank
        
        Args:
            text: 原始文本
            compression_rate: 压缩率（如果为None，使用fact_extractor的默认值）
        
        Returns:
            更新统计信息
        """
        # 提取事实
        if compression_rate is not None:
            original_rate = self.fact_extractor.compression_rate
            self.fact_extractor.compression_rate = compression_rate
        
        fact_result = self.fact_extractor.extract_facts(text, return_annotations=False)
        compressed_text = fact_result['compressed_text']
        
        if compression_rate is not None:
            self.fact_extractor.compression_rate = original_rate
        
        # 将压缩后的文本分割成多个事实（按句子分割）
        facts = self._split_into_facts(compressed_text)
        
        # 更新 memory_bank
        return self.update_with_facts(facts)
    
    def _split_into_facts(self, text: str) -> List[str]:
        """将文本分割成多个事实（按句子分割）"""
        import re
        # 按句号、问号、感叹号分割
        sentences = re.split(r'[.!?]\s+', text)
        # 过滤空句子和过短的句子
        facts = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return facts
    
    def get_statistics(self) -> Dict[str, any]:
        """获取记忆库使用统计"""
        return {
            "total_entries": self.knowledge_num,
            "max_entries": self.max_entries,
            "update_strategy": self.update_strategy,
            "access_counter": self.access_counter,
            "usage_stats": {
                "min_last_used": self.usage_stats['last_used'].min().item(),
                "max_last_used": self.usage_stats['last_used'].max().item(),
                "min_access_count": self.usage_stats['access_count'].min().item(),
                "max_access_count": self.usage_stats['access_count'].max().item(),
            },
            "importance_stats": {
                "min_score": self.importance_scores.min().item(),
                "max_score": self.importance_scores.max().item(),
                "mean_score": self.importance_scores.mean().item(),
            },
        }

