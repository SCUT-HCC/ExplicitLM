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

    def __init__(self, layer_id: int, config: LMConfig) -> None:
        """
        初始化ExplicitLMBlock

        Args:
            layer_id: 当前层的ID索引
            config: 语言模型配置对象，包含模型维度、头数等参数
        """
        super().__init__()
        self.config = config  # 保存config引用
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.memory_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # 记忆相关模块
        self.memory_gate = MemoryGate(config)
        self.gated_memory_fusion = GatedMemoryFusion(config)
        
        # Gumbel-Softmax参数
        self.gumbel_temperature = getattr(config, 'gumbel_temperature', 1.0)

        # self.attentionpool = nn.Linear(config.dim, 1)
    
    def gumbel_softmax_selection(
        self,
        similarity_scores: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Gumbel-Softmax进行可微分的离散选择

        该方法实现了Gumbel-Softmax技巧，允许在保持可微分性的同时进行离散选择。
        通过添加Gumbel噪声并应用softmax，可以实现平滑的离散采样。
        在硬模式下，使用straight-through estimator保持梯度流动。

        Args:
            similarity_scores: 形状为[batch_size, seq_len, num_candidates]的相似度分数张量
            temperature: Gumbel-Softmax温度参数，控制分布的平滑程度（默认1.0）
            hard: 是否使用硬选择模式生成one-hot向量（默认True）

        Returns:
            selection_weights: 形状为[batch_size, seq_len, num_candidates]的选择权重
            selected_indices: 形状为[batch_size, seq_len]的选中索引，用于统计分析
        """
        # 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity_scores) + 1e-20) + 1e-20)
        logits = (similarity_scores + gumbel_noise) / temperature
        
        # Softmax
        soft_weights = F.softmax(logits, dim=-1)
        
        if hard:
	            # 硬选择：创建one-hot向量
            _, max_indices = soft_weights.max(dim=-1, keepdim=True)
            hard_weights = torch.zeros_like(soft_weights).scatter_(-1, max_indices, 1.0)
            # 使用straight-through estimator
            selection_weights = hard_weights - soft_weights.detach() + soft_weights
            selected_indices = max_indices.squeeze(-1)  # [batch_size, seq_len]
        else:
            # 软选择
            selection_weights = soft_weights
            selected_indices = torch.argmax(soft_weights, dim=-1)
            
        return selection_weights, selected_indices
    
    def compute_diversity_loss(self, candidate_memories: torch.Tensor) -> torch.Tensor:
        """
        计算候选集内部多样性损失，鼓励候选项之间的差异性

        该方法通过计算候选记忆项之间的余弦相似度矩阵，评估候选集的多样性。
        相似度越高表示候选项越相似，多样性越低；损失函数鼓励降低候选项之间的相似度。

        Args:
            candidate_memories: 形状为[batch_size, seq_len, num_candidates, dim]的候选记忆张量

        Returns:
            diversity_loss: 标量张量，表示候选集的平均相似度（作为损失）
        """
        bsz, seq_len, num_candidates, dim = candidate_memories.shape
        
        # 计算候选项之间的相似度矩阵
        # 归一化候选记忆用于计算余弦相似度
        normalized_memories = F.normalize(candidate_memories, p=2, dim=-1)  # [batch, seq_len, num_candidates, dim]
        
        # 计算相似度矩阵: [batch, seq_len, num_candidates, num_candidates]
        similarity_matrix = torch.matmul(normalized_memories, normalized_memories.transpose(-2, -1))
        
	        # 移除对角线（自相似度=1）
        mask = torch.eye(num_candidates, device=candidate_memories.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, -1, -1)
        
        # 计算非对角线元素的平均相似度（希望越小越好，表示越多样）
        off_diagonal_similarities = similarity_matrix.masked_select(~mask)
        avg_similarity = off_diagonal_similarities.mean()
        
        # 多样性损失：相似度越高，损失越大
        diversity_loss = avg_similarity
        
        return diversity_loss

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
        collect_ema_stats: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]], Dict[str, Union[torch.Tensor, float]]]
    ]:
        """
        前向传播：基于Gumbel-Softmax的记忆选择机制

        该方法实现了完整的记忆增强Transformer块的前向传播流程：
        1. 自注意力处理输入序列
        2. 记忆门控生成多个候选记忆项
        3. 使用Gumbel-Softmax进行可微分的记忆选择
        4. 计算相似度损失和多样性损失
        5. 通过门控融合机制整合选中的记忆

        Args:
            x: 形状为[batch_size, seq_len, dim]的输入张量
            pos_cis: 位置编码张量，用于自注意力机制
            memory_bank: 形状为[knowledge_num, knowledge_length]的共享记忆库，存储token IDs
            tok_embeddings: token嵌入层，用于解码记忆库中的token序列
            collect_ema_stats: 是否收集EMA更新统计信息（默认False）

        Returns:
            out: 形状为[batch_size, seq_len, dim]的输出张量
            similarity_loss: 相似度损失标量（可微分）
            diversity_loss: 多样性损失标量
            layer_stats: 该层的监控统计信息字典
            ema_stats: EMA更新统计信息（仅当collect_ema_stats=True时返回）
            cosine_stats: 余弦相似度统计信息字典
        """
        # ===== 第一阶段：自注意力处理 =====
        h_attn = self.attention(self.attention_norm(x), pos_cis)
        h = x + h_attn  # 残差连接

        # ===== 第二阶段：记忆查询准备 =====
        # 使用自注意力输出作为记忆查询的基础（核心设计：利用上下文信息）
        h_for_memory = self.memory_norm(h_attn)

        # ===== 第三阶段：候选记忆生成 =====
        # 通过记忆门控生成多个候选记忆项索引和分数
        candidate_indices, _candidate_scores = self.memory_gate(h_for_memory)
        # candidate_indices: 形状[batch, seq_len, num_candidates]，候选记忆在库中的索引
        # _candidate_scores: 形状[batch, seq_len, num_candidates]，候选记忆的初始分数（当前未使用，保留以保持接口一致性）
        
        bsz, seq_len, num_candidates = candidate_indices.shape

        # ===== 第四阶段：候选记忆解码 =====
        # 将候选索引展平以批量查询记忆库
        candidate_indices_flat = candidate_indices.view(-1)  # [batch * seq_len * num_candidates]
        candidate_token_ids = memory_bank[candidate_indices_flat]  # [batch * seq_len * num_candidates, knowledge_length]

        # 将token序列转换为嵌入向量并通过平均池化获得记忆表示
        candidate_embeddings = tok_embeddings(candidate_token_ids)  # [batch * seq_len * num_candidates, knowledge_length, dim]
        candidate_memories = candidate_embeddings.mean(dim=1)  # 平均池化: [batch * seq_len * num_candidates, dim]
        candidate_memories = candidate_memories.view(bsz, seq_len, num_candidates, self.dim)  # 重塑: [batch, seq_len, num_candidates, dim]

        # ===== 第五阶段：相似度计算（可微分） =====
        # 计算查询向量与所有候选记忆的余弦相似度
        h_expanded = h_for_memory.unsqueeze(2).expand(-1, -1, num_candidates, -1)  # [batch, seq_len, num_candidates, dim]
        similarity_scores = F.cosine_similarity(h_expanded, candidate_memories, dim=-1)  # [batch, seq_len, num_candidates]

        # ===== 第六阶段：Gumbel-Softmax选择 =====
        # 使用Gumbel-Softmax进行可微分的离散选择
        selection_weights, selected_indices = self.gumbel_softmax_selection(
            similarity_scores,
            temperature=self.gumbel_temperature,
            hard=True
        )
        # selection_weights: [batch, seq_len, num_candidates]，选择权重（支持梯度）
        # selected_indices: [batch, seq_len]，选中的候选索引（用于统计）
        
        # ===== 第七阶段：损失计算 =====
        # 相似度损失：鼓励选中的记忆与查询向量高度相似
        selected_similarities = (similarity_scores * selection_weights).sum(dim=-1)  # [batch, seq_len]
        similarity_loss = -selected_similarities.mean()  # 负号：相似度越高，损失越小

        # 多样性损失：鼓励候选集内部的多样性
        diversity_loss = self.compute_diversity_loss(candidate_memories)

        # ===== 第八阶段：记忆融合 =====
        # 使用选择权重对候选记忆进行加权求和，得到最终选中的记忆
        selected_memory = (candidate_memories * selection_weights.unsqueeze(-1)).sum(dim=2)  # [batch, seq_len, dim]

        # 通过门控MLP融合机制将选中的记忆与查询向量融合
        memory_output = self.gated_memory_fusion(h_for_memory, selected_memory)

        # ===== 第九阶段：输出生成 =====
        # 最终残差连接，整合原始表示和记忆增强表示
        out = h + memory_output
        
        # ===== 第十阶段：统计信息收集 =====
        # 计算选择统计信息（覆盖率、热门记忆等）
        layer_stats = self._compute_selection_stats(candidate_indices, selection_weights)

        # 计算详细的余弦相似度统计信息
        cosine_stats = {
            'similarity_scores': similarity_scores,  # [batch, seq_len, num_candidates]
            'selected_similarities': selected_similarities,  # [batch, seq_len]
            'avg_similarity': similarity_scores.mean().item(),  # 所有候选的平均相似度
            'max_similarity': similarity_scores.max().item(),  # 最大相似度
            'min_similarity': similarity_scores.min().item(),  # 最小相似度
            'selected_avg_similarity': selected_similarities.mean().item(),  # 选中记忆的平均相似度
            'selection_entropy': -torch.sum(
                selection_weights * torch.log(selection_weights + 1e-10), dim=-1
            ).mean().item()  # 选择分布的熵（衡量选择的不确定性）
        }

        # 收集EMA更新统计信息（仅在训练模式且需要时）
        ema_stats = None
        if collect_ema_stats and self.training:
            # 扩展选中的索引以匹配EMA更新的期望格式
            selected_memory_indices = candidate_indices.gather(2, selected_indices.unsqueeze(-1))  # [batch, seq_len, 1]
            ema_stats = {
                'memory_indices': selected_memory_indices,          # [batch, seq_len, 1]
                'memory_scores': torch.ones_like(selected_memory_indices.float()),   # [batch, seq_len, 1] - 选中的权重为1
                'h_for_memory': h_for_memory,                       # [batch, seq_len, dim]
                'selected_memory': selected_memory.unsqueeze(2),    # [batch, seq_len, 1, dim]
            }
        
        if collect_ema_stats:
            return out, similarity_loss, diversity_loss, layer_stats, ema_stats, cosine_stats
        else:
            return out, similarity_loss, diversity_loss, layer_stats, cosine_stats
    
    def _compute_selection_stats(
        self,
        candidate_indices: torch.Tensor,
        selection_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算基于候选项选择的统计信息

        该方法分析记忆选择的模式，包括记忆覆盖率、热门记忆数量、
        死亡记忆数量以及选择方差等指标，用于监控记忆库的使用情况。

        Args:
            candidate_indices: 形状为[batch_size, seq_len, num_candidates]的候选索引张量
            selection_weights: 形状为[batch_size, seq_len, num_candidates]的Gumbel-Softmax权重

        Returns:
            stats: 包含多种选择统计指标的字典
        """
        device = candidate_indices.device

        # 将索引和权重展平以便统计
        flat_indices = candidate_indices.view(-1)  # [batch * seq_len * num_candidates]
        flat_weights = selection_weights.view(-1)  # [batch * seq_len * num_candidates]

        # 统计每个记忆条目被选中的加权次数
        # 使用scatter_add累加每个记忆索引对应的选择权重
        memory_counts = torch.zeros(self.config.knowledge_num, device=device)
        memory_counts.scatter_add_(0, flat_indices, flat_weights)

        # 计算各种统计指标
        with torch.no_grad():
            # 覆盖率：被选中概率>1%的记忆条目比例
            coverage_rate = (memory_counts > 0.01).float().mean().item()

            # 热门记忆：前10%使用频率的记忆条目数量
            top10_threshold = torch.quantile(memory_counts, 0.9)
            hot_memories = (memory_counts >= top10_threshold).sum().item()

            # 死亡记忆：几乎从未被选中（<1%）的记忆条目数量
            dead_memories = (memory_counts < 0.01).sum().item()

            # 选择方差：衡量记忆使用分布的不均匀程度
            selection_variance = memory_counts.var().item()

            # 构建统计字典
            stats = {
                'coverage_rate': coverage_rate,  # 记忆覆盖率
                'hot_memories': hot_memories,  # 热门记忆数量
                'dead_memories': dead_memories,  # 死亡记忆数量
                'selection_variance': selection_variance,  # 选择方差
                'max_selections': memory_counts.max().item(),  # 最大选择次数
                'min_selections': memory_counts.min().item(),  # 最小选择次数
            }

        return stats