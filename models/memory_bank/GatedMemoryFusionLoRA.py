import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedMemoryFusionLoRA(nn.Module):
    """
    低秩分解版本的门控记忆融合模块（带Shortcut机制）
    
    使用低秩分解的 SwiGLU 门控 MLP 融合记忆，通过相似度动态控制 memory 贡献。
    公式：out = hidden_states + alpha * memory_output
    
    相比原版本：
    - gate_proj, up_proj, down_proj 使用低秩分解（LoRA-style）
    - 移除 similarity_gate，直接使用 similarity_scores 作为 alpha
    - 参数量减少约 92%，计算量减少约 50%
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 模型配置字典
                - dim: 隐藏层维度
                - knowledge_dim: 知识维度
                - num_selected: 选中的记忆数量
                - fusion_rank: 低秩分解的 rank（默认 128）
                - dropout: Dropout 比率
        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg["dim"]
        self.knowledge_dim = cfg["knowledge_dim"]
        self.num_selected = cfg.get("num_selected", 1)
        self.fusion_rank = cfg.get("fusion_rank", 128)

        concat_dim = self.dim + self.num_selected * self.dim

        # 低秩分解的 gate_proj: Linear(concat_dim, rank) + Linear(rank, dim)
        self.gate_proj_lora_a = nn.Linear(concat_dim, self.fusion_rank, bias=False)
        self.gate_proj_lora_b = nn.Linear(self.fusion_rank, self.dim, bias=False)

        # 低秩分解的 up_proj: Linear(concat_dim, rank) + Linear(rank, dim)
        self.up_proj_lora_a = nn.Linear(concat_dim, self.fusion_rank, bias=False)
        self.up_proj_lora_b = nn.Linear(self.fusion_rank, self.dim, bias=False)

        # 低秩分解的 down_proj: Linear(dim, rank) + Linear(rank, dim)
        self.down_proj_lora_a = nn.Linear(self.dim, self.fusion_rank, bias=False)
        self.down_proj_lora_b = nn.Linear(self.fusion_rank, self.dim, bias=False)

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

        # 拼接输入
        concat_input = torch.cat([h_attn, selected_memory], dim=-1)

        # 低秩分解的 gate_proj: gate = SiLU(LoRA(concat_input))
        gate_intermediate = self.gate_proj_lora_a(concat_input)
        gate = F.silu(self.gate_proj_lora_b(gate_intermediate))

        # 低秩分解的 up_proj: up = LoRA(concat_input)
        up_intermediate = self.up_proj_lora_a(concat_input)
        up = self.up_proj_lora_b(up_intermediate)

        # SwiGLU 融合
        fusion_output = gate * up

        # 低秩分解的 down_proj: memory_output = Dropout(LoRA(fusion_output))
        memory_intermediate = self.down_proj_lora_a(fusion_output)
        memory_output = self.dropout(self.down_proj_lora_b(memory_intermediate))

        # 直接使用 similarity_scores 作为 alpha（移除 similarity_gate）
        if similarity_scores is not None:
            # 使用平均相似度作为权重，并扩展到所有维度
            alpha = similarity_scores.mean(dim=-1, keepdim=True)  # [bsz, 1]
            alpha = torch.clamp(alpha, 0.0, 1.0)
            alpha_expanded = alpha.unsqueeze(1).expand(-1, seq_len, -1)  # [bsz, seq_len, 1]
        else:
            # 如果没有 similarity_scores，使用默认值 0.5
            alpha_expanded = torch.full(
                (bsz, seq_len, 1), 0.5, device=h_attn.device, dtype=h_attn.dtype
            )

        weighted_memory_output = alpha_expanded * memory_output

        return weighted_memory_output

