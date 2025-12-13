"""
Qwen3ExplicitLMBlock: 基于Qwen3DecoderLayer的记忆增强Transformer块

在Qwen3DecoderLayer基础上添加记忆检索和融合机制。
"""

from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Config,
    Cache,
)
from transformers.utils import TransformersKwargs
from typing import Unpack

from models.memory_bank.MemoryGate import MemoryGate
from models.memory_bank.MemoryGateLoRA import MemoryGateLoRA
from models.memory_bank.GatedMemoryFusion import GatedMemoryFusion
from models.memory_bank.GatedMemoryFusionLoRA import GatedMemoryFusionLoRA
from models.layers.RMSNorm import RMSNorm

logger = logging.getLogger(__name__)


class Qwen3ExplicitLMBlock(nn.Module):
    """基于Qwen3DecoderLayer的记忆增强Transformer块"""

    def __init__(self, config: Qwen3Config, layer_idx: int, memory_cfg: dict) -> None:
        """
        Args:
            config: Qwen3Config配置对象
            layer_idx: 当前层的ID索引
            memory_cfg: 记忆库相关配置字典
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.memory_cfg = memory_cfg
        self.hidden_size = config.hidden_size
        
        self.qwen3_decoder = Qwen3DecoderLayer(config, layer_idx)
        
        use_moe = memory_cfg.get("use_moe", False)
        if not use_moe:
            self.memory_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
            memory_cfg_with_dim = memory_cfg.copy()
            memory_cfg_with_dim["dim"] = config.hidden_size
            
            # 根据 gate_rank 选择 MemoryGate 版本
            gate_rank = memory_cfg.get("gate_rank", None)
            if gate_rank is None or gate_rank == 0:
                self.memory_gate = MemoryGate(memory_cfg_with_dim)
            else:
                memory_cfg_with_dim["gate_rank"] = gate_rank
                self.memory_gate = MemoryGateLoRA(memory_cfg_with_dim)
            
            # 根据 fusion_rank 选择融合模块版本
            fusion_rank = memory_cfg.get("fusion_rank", None)
            if fusion_rank is None or fusion_rank == 0:
                self.gated_memory_fusion = GatedMemoryFusion(memory_cfg_with_dim)
            else:
                memory_cfg_with_dim["fusion_rank"] = fusion_rank
                self.gated_memory_fusion = GatedMemoryFusionLoRA(memory_cfg_with_dim)
            
            self.gumbel_temperature = memory_cfg.get("gumbel_temperature", 1.0)
        else:
            self.memory_norm = None
            self.memory_gate = None
            self.gated_memory_fusion = None

    def gumbel_softmax_selection(
        self,
        similarity_scores: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-Softmax选择机制"""
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
        """计算多样性损失"""
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        memory_bank: Optional[torch.Tensor] = None,
        tok_embeddings: Optional[nn.Embedding] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]]]:
        """
        Args:
            hidden_states: 输入隐藏状态
            memory_bank: 记忆库 [knowledge_num, knowledge_length]
            tok_embeddings: token嵌入层
        Returns:
            (output, sim_loss, div_loss, layer_stats, cosine_stats)
        """
        use_moe = self.memory_cfg.get("use_moe", False)
        
        hidden_states = self.qwen3_decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        if use_moe:
            similarity_loss = torch.tensor(0.0, device=hidden_states.device)
            diversity_loss = torch.tensor(0.0, device=hidden_states.device)
            layer_stats = {}
            cosine_stats = {}
            return hidden_states, similarity_loss, diversity_loss, layer_stats, cosine_stats
        else:
            h_for_memory = self.memory_norm(hidden_states)
            
            if attention_mask is not None:
                ## 这个地方数据定下来后要改一下，我不确定数据最后传进来是什么形状，就先这么写了
                if attention_mask.dim() == 5:
                    bsz, _, seq_len, _, _ = attention_mask.shape
                    causal_matrix = attention_mask[:, 0, :, :, 0]
                    diag_indices = torch.arange(seq_len, device=causal_matrix.device)
                    seq_mask = causal_matrix[:, diag_indices, diag_indices].float()
                elif attention_mask.dim() == 4:
                    bsz = attention_mask.shape[0]
                    seq_len = attention_mask.shape[-1]
                    causal_matrix = attention_mask[:, 0, :, :]
                    diag_indices = torch.arange(seq_len, device=causal_matrix.device)
                    seq_mask = causal_matrix[:, diag_indices, diag_indices].float()
                elif attention_mask.dim() == 2:
                    seq_mask = attention_mask.float()
                else:
                    seq_mask = attention_mask.squeeze().float()
                    if seq_mask.dim() == 2 and seq_mask.shape[0] == seq_mask.shape[1]:
                        bsz = 1 if seq_mask.shape[0] == seq_mask.shape[1] else seq_mask.shape[0]
                        seq_len = seq_mask.shape[-1]
                        diag_indices = torch.arange(seq_len, device=seq_mask.device)
                        seq_mask = seq_mask[diag_indices, diag_indices].unsqueeze(0).float()
                    else:
                        while seq_mask.dim() > 2:
                            if seq_mask.shape[1] == 1:
                                seq_mask = seq_mask[:, 0]
                            else:
                                seq_mask = seq_mask[..., 0]
                        if seq_mask.dim() == 1:
                            seq_mask = seq_mask.unsqueeze(0)
                ### 到这里结束

                ## 均值embd，可以确认一下这里
                seq_mask = seq_mask.to(dtype=h_for_memory.dtype)
                input_mask_expanded = seq_mask.unsqueeze(-1).expand(h_for_memory.size())
                sum_embeddings = torch.sum(h_for_memory * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                mean_embeddings = mean_embeddings.to(dtype=h_for_memory.dtype)
            else:
                mean_embeddings = h_for_memory.mean(dim=1)
            
            query_embeddings = mean_embeddings.unsqueeze(1)
            
            scores_1, scores_2 = self.memory_gate.compute_sub_scores(query_embeddings)
            candidate_indices, candidate_scores = self.memory_gate.generate_candidates(scores_1, scores_2)
            bsz, _, num_candidates = candidate_indices.shape
            seq_len = h_for_memory.shape[1]
            
            candidate_indices = candidate_indices.expand(bsz, seq_len, num_candidates)
            candidate_scores = candidate_scores.expand(bsz, seq_len, num_candidates)
            
            candidate_indices_flat = candidate_indices.reshape(-1)
            # memory_bank 存储在 CPU 上，使用时临时传输到 GPU
            if memory_bank.device.type != 'cuda':
                memory_bank = memory_bank.to(candidate_indices_flat.device, non_blocking=True)
            candidate_token_ids = memory_bank[candidate_indices_flat]
            
            batch_size_embed = 32
            num_total_candidates = candidate_token_ids.shape[0]
            candidate_embeddings_list = []
            
            for i in range(0, num_total_candidates, batch_size_embed):
                end_idx = min(i + batch_size_embed, num_total_candidates)
                batch_token_ids = candidate_token_ids[i:end_idx]
                batch_embeddings = tok_embeddings(batch_token_ids)
                batch_memories = batch_embeddings.mean(dim=1)
                candidate_embeddings_list.append(batch_memories)
                del batch_embeddings, batch_token_ids
                torch.cuda.empty_cache()
            
            candidate_memories_flat = torch.cat(candidate_embeddings_list, dim=0)
            candidate_memories = candidate_memories_flat.reshape(bsz, seq_len, num_candidates, self.hidden_size)
            del candidate_embeddings_list, candidate_memories_flat, candidate_token_ids, candidate_indices_flat
            
            # 优化：归一化后立即计算相似度，减少中间张量
            h_normalized = F.normalize(h_for_memory, p=2, dim=-1)
            candidate_normalized = F.normalize(candidate_memories, p=2, dim=-1)
            similarity_scores = torch.einsum('bsd,bsnd->bsn', h_normalized, candidate_normalized)
            del h_normalized, candidate_normalized
            
            selection_weights, selected_indices = self.gumbel_softmax_selection(
                similarity_scores, temperature=self.gumbel_temperature, hard=True
            )
            
            selected_similarities = (similarity_scores * selection_weights).sum(dim=-1)
            similarity_loss = -selected_similarities.mean()
            diversity_loss = self.compute_diversity_loss(candidate_memories)
            
            selected_memory = (candidate_memories * selection_weights.unsqueeze(-1)).sum(dim=2)
            # 计算完成后立即释放 candidate_memories 以节省显存
            del candidate_memories
            torch.cuda.empty_cache()
            memory_output = self.gated_memory_fusion(
                h_for_memory, 
                selected_memory,
                similarity_scores=selected_similarities
            )
            
            out = hidden_states + memory_output
            
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
            
            return out, similarity_loss, diversity_loss, layer_stats, cosine_stats

    def _compute_selection_stats(
        self,
        candidate_indices: torch.Tensor,
        selection_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """计算选择统计信息"""
        device = candidate_indices.device
        flat_indices = candidate_indices.reshape(-1)
        flat_weights = selection_weights.reshape(-1)
        knowledge_num = self.memory_cfg["knowledge_num"]
        memory_counts = torch.zeros(knowledge_num, device=device, dtype=flat_weights.dtype)
        memory_counts.scatter_add_(0, flat_indices, flat_weights)
        with torch.no_grad():
            memory_counts_fp32 = memory_counts.float()
            coverage_rate = (memory_counts_fp32 > 0.01).float().mean().item()
            top10_threshold = torch.quantile(memory_counts_fp32, 0.9)
            hot_memories = (memory_counts_fp32 >= top10_threshold).sum().item()
            dead_memories = (memory_counts_fp32 < 0.01).sum().item()
            selection_variance = memory_counts_fp32.var().item()
            stats = {
                "coverage_rate": coverage_rate,
                "hot_memories": hot_memories,
                "dead_memories": dead_memories,
                "selection_variance": selection_variance,
                "max_selections": memory_counts_fp32.max().item(),
                "min_selections": memory_counts_fp32.min().item(),
            }
        return stats

