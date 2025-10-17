"""
ExplicitLMBlock: 基于记忆增强的Transformer块

该模块实现了一个替代传统FFN的记忆增强架构，通过以下机制提升模型性能：
- 自注意力机制处理序列上下文
- 记忆门控生成多个候选记忆项
- Gumbel-Softmax实现可微分的离散选择
- 多样性损失和相似度损失优化记忆选择质量
- 门控融合机制整合记忆信息
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.configs.LMConfig import LMConfig
from models.layers.Attention import Attention
from models.layers.RMSNorm import RMSNorm
from models.memory_bank.MemoryGate import MemoryGate
from models.memory_bank.GatedMemoryFusion import GatedMemoryFusion


class ExplicitLMBlock(nn.Module):
    """
    Transformer块，使用基于记忆的交叉注意力机制替代传统FFN

    该模块实现了以下核心功能：
    1. 自注意力机制处理序列信息
    2. 记忆门控生成候选记忆项
    3. Gumbel-Softmax进行可微分的离散选择
    4. 计算相似度损失和多样性损失以优化记忆选择
    """

    def __init__(self, layer_id: int, cfg: dict) -> None:
        """
        初始化ExplicitLMBlock

        Args:
            layer_id: 当前层的ID索引
            cfg: 模型配置字典，含模型维度、头数等参数
        """
        super().__init__()
        self.cfg = cfg  # 保存cfg引用
        self.n_heads = cfg["n_heads"]
        self.dim = cfg["dim"]
        self.head_dim = cfg["dim"] // cfg["n_heads"]
        self.attention = Attention(cfg)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(cfg["dim"], eps=cfg["norm_eps"])
        self.memory_norm = RMSNorm(cfg["dim"], eps=cfg["norm_eps"])

        # 记忆相关模块
        self.memory_gate = MemoryGate(cfg)
        self.gated_memory_fusion = GatedMemoryFusion(cfg)

        # Gumbel-Softmax参数
        self.gumbel_temperature = cfg.get("gumbel_temperature", 1.0)

    # ---------------- 以下所有函数仅把 self.config 换成 self.cfg ----------------
    def gumbel_softmax_selection(
        self,
        similarity_scores: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity_scores) + 1e-20) + 1e-20)
        logits = (similarity_scores + gumbel_noise) / temperature
        soft_weights = F.softmax(logits, dim=-1)
        if hard:
            _, max_indices = soft_weights.max(dim=-1, keepdim=True)
            hard_weights = torch.zeros_like(soft_weights).scatter_(-1, max_indices, 1.0)
            selection_weights = hard_weights - soft_weights.detach() + soft_weights
            selected_indices = max_indices.squeeze(-1)
        else:
            selection_weights = soft_weights
            selected_indices = torch.argmax(soft_weights, dim=-1)
        return selection_weights, selected_indices

    def compute_diversity_loss(self, candidate_memories: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_candidates, dim = candidate_memories.shape
        normalized_memories = F.normalize(candidate_memories, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized_memories, normalized_memories.transpose(-2, -1))
        mask = torch.eye(num_candidates, device=candidate_memories.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, -1, -1)
        off_diagonal_similarities = similarity_matrix.masked_select(~mask)
        avg_similarity = off_diagonal_similarities.mean()
        diversity_loss = avg_similarity
        return diversity_loss

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
        collect_ema_stats: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]], Dict[str, Union[torch.Tensor, float]]],
    ]:
        # ===== 第一阶段：自注意力处理 =====
        h_attn = self.attention(self.attention_norm(x), pos_cis)
        h = x + h_attn

        # ===== 第二阶段：记忆查询准备 =====
        h_for_memory = self.memory_norm(h_attn)

        # ===== 第三阶段：候选记忆生成 =====
        candidate_indices, _candidate_scores = self.memory_gate(h_for_memory)
        bsz, seq_len, num_candidates = candidate_indices.shape

        # ===== 第四阶段：候选记忆解码 =====
        candidate_indices_flat = candidate_indices.view(-1)
        candidate_token_ids = memory_bank[candidate_indices_flat]
        candidate_embeddings = tok_embeddings(candidate_token_ids)
        candidate_memories = candidate_embeddings.mean(dim=1)
        candidate_memories = candidate_memories.view(bsz, seq_len, num_candidates, self.dim)

        # ===== 第五阶段：相似度计算 =====
        h_expanded = h_for_memory.unsqueeze(2).expand(-1, -1, num_candidates, -1)
        similarity_scores = F.cosine_similarity(h_expanded, candidate_memories, dim=-1)

        # ===== 第六阶段：Gumbel-Softmax选择 =====
        selection_weights, selected_indices = self.gumbel_softmax_selection(
            similarity_scores, temperature=self.gumbel_temperature, hard=True
        )

        # ===== 第七阶段：损失计算 =====
        selected_similarities = (similarity_scores * selection_weights).sum(dim=-1)
        similarity_loss = -selected_similarities.mean()
        diversity_loss = self.compute_diversity_loss(candidate_memories)

        # ===== 第八阶段：记忆融合 =====
        selected_memory = (candidate_memories * selection_weights.unsqueeze(-1)).sum(dim=2)
        memory_output = self.gated_memory_fusion(h_for_memory, selected_memory)

        # ===== 第九阶段：输出生成 =====
        out = h + memory_output

        # ===== 第十阶段：统计信息收集 =====
        layer_stats = self._compute_selection_stats(candidate_indices, selection_weights)
        cosine_stats = {
            "similarity_scores": similarity_scores,
            "selected_similarities": selected_similarities,
            "avg_similarity": similarity_scores.mean().item(),
            "max_similarity": similarity_scores.max().item(),
            "min_similarity": similarity_scores.min().item(),
            "selected_avg_similarity": selected_similarities.mean().item(),
            "selection_entropy": -torch.sum(
                selection_weights * torch.log(selection_weights + 1e-10), dim=-1
            ).mean().item(),
        }

        ema_stats = None
        if collect_ema_stats and self.training:
            selected_memory_indices = candidate_indices.gather(2, selected_indices.unsqueeze(-1))
            ema_stats = {
                "memory_indices": selected_memory_indices,
                "memory_scores": torch.ones_like(selected_memory_indices.float()),
                "h_for_memory": h_for_memory,
                "selected_memory": selected_memory.unsqueeze(2),
            }

        if collect_ema_stats:
            return out, similarity_loss, diversity_loss, layer_stats, ema_stats, cosine_stats
        else:
            return out, similarity_loss, diversity_loss, layer_stats, cosine_stats

    def _compute_selection_stats(
        self,
        candidate_indices: torch.Tensor,
        selection_weights: torch.Tensor,
    ) -> Dict[str, float]:
        device = candidate_indices.device
        flat_indices = candidate_indices.view(-1)
        flat_weights = selection_weights.view(-1)
        memory_counts = torch.zeros(self.cfg["knowledge_num"], device=device)
        memory_counts.scatter_add_(0, flat_indices, flat_weights)
        with torch.no_grad():
            coverage_rate = (memory_counts > 0.01).float().mean().item()
            top10_threshold = torch.quantile(memory_counts, 0.9)
            hot_memories = (memory_counts >= top10_threshold).sum().item()
            dead_memories = (memory_counts < 0.01).sum().item()
            selection_variance = memory_counts.var().item()
            stats = {
                "coverage_rate": coverage_rate,
                "hot_memories": hot_memories,
                "dead_memories": dead_memories,
                "selection_variance": selection_variance,
                "max_selections": memory_counts.max().item(),
                "min_selections": memory_counts.min().item(),
            }
        return stats