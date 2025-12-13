import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedMemoryFusion(nn.Module):
    """
    门控记忆融合模块（带Shortcut机制）
    
    使用SwiGLU门控MLP融合记忆，通过相似度动态控制memory贡献。
    公式：out = hidden_states + alpha * memory_output
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 模型配置字典
        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg["dim"]
        self.knowledge_dim = cfg["knowledge_dim"]
        self.num_selected = cfg.get("num_selected", 1)

        concat_dim = self.dim + self.num_selected * self.dim

        self.gate_proj = nn.Linear(concat_dim, self.dim, bias=False)
        self.up_proj = nn.Linear(concat_dim, self.dim, bias=False)
        self.down_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.similarity_gate = nn.Sequential(
            nn.Linear(1, self.dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(self.dim // 4, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.memory_weight_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(
        self,   
        h_attn: torch.Tensor,
        selected_memory: torch.Tensor,
        similarity_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_attn: 自注意力输出 [batch_size, seq_len, dim]
            selected_memory: 选中的记忆 [batch_size, seq_len, dim]
            similarity_scores: 相似度分数 [batch_size, seq_len]（可选）
        Returns:
            memory_output: 记忆融合输出 [batch_size, seq_len, dim]
        """
        bsz, seq_len, _ = h_attn.shape

        concat_input = torch.cat([h_attn, selected_memory], dim=-1)
        gate = F.silu(self.gate_proj(concat_input))
        up = self.up_proj(concat_input)
        fusion_output = gate * up
        memory_output = self.dropout(self.down_proj(fusion_output))

        if similarity_scores is not None:
            avg_similarity = similarity_scores.mean(dim=-1, keepdim=True)
        else:
            avg_similarity = torch.full(
                (bsz, 1), 0.5, device=h_attn.device, dtype=h_attn.dtype
            )
        
        alpha = self.similarity_gate(avg_similarity) + self.memory_weight_bias
        alpha = torch.clamp(alpha, 0.0, 1.0)
        alpha_expanded = alpha.unsqueeze(1).expand(-1, seq_len, -1)
        weighted_memory_output = alpha_expanded * memory_output

        return weighted_memory_output