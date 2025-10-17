import torch
import torch.nn as nn
import torch.nn.functional as F

from models.configs import LMConfig


class GatedMemoryFusion(nn.Module):
    """
    门控记忆融合模块

    使用类似SwiGLU的门控MLP结构，将自注意力输出与选中的记忆进行融合。
    通过门控机制动态调节记忆信息的贡献，实现自适应的信息融合。

    核心思想：
    - 将自注意力输出和选中记忆拼接作为输入
    - 使用门控投影(gate_proj)生成门控信号
    - 使用上投影(up_proj)生成内容信号
    - 通过element-wise乘法实现门控融合
    - 使用下投影(down_proj)映射回原始维度

    门控机制类似SwiGLU：
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        cfg: 模型配置字典，需包含以下字段：
            - dim: 隐藏层维度
            - knowledge_dim: 记忆键的维度
            - num_selected: 选中的记忆数量（默认为1）
            - dropout: Dropout概率
    """

    def __init__(self, cfg: dict) -> None:
        """
        初始化门控记忆融合模块

        Args:
            cfg: 模型配置字典
        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg["dim"]
        self.knowledge_dim = cfg["knowledge_dim"]
        self.num_selected = cfg.get("num_selected", 1)  # 选择的最佳记忆数量

        # 计算拼接后的输入维度
        # 输入由两部分组成：
        # 1. h_attn: [batch, seq, dim]
        # 2. selected_memory: [batch, seq, num_selected * dim]
        concat_dim = self.dim + self.num_selected * self.dim

        # 门控MLP结构（类似SwiGLU）
        self.gate_proj = nn.Linear(concat_dim, self.dim, bias=False)  # 门控投影
        self.up_proj = nn.Linear(concat_dim, self.dim, bias=False)    # 上投影
        self.down_proj = nn.Linear(self.dim, self.dim, bias=False)    # 下投影

        # Dropout层用于正则化
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(
        self,
        h_attn: torch.Tensor,
        selected_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：融合自注意力输出和选中的记忆

        处理流程：
        1. 拼接自注意力输出和选中记忆
        2. 通过门控投影和激活函数生成门控信号
        3. 通过上投影生成内容信号
        4. 门控信号与内容信号逐元素相乘
        5. 通过下投影映射回原始维度
        6. 应用dropout正则化

        Args:
            h_attn: 自注意力输出，形状为 [batch_size, seq_len, dim]
            selected_memory: 选中的记忆，形状为 [batch_size, seq_len, dim]
                           （当num_selected=1时，表示单个最佳记忆）

        Returns:
            融合后的输出，形状为 [batch_size, seq_len, dim]

        Note:
            门控机制公式：output = dropout(down_proj(silu(gate_proj(x)) * up_proj(x)))
        """
        bsz, seq_len, _ = h_attn.shape

        # 步骤1: 拼接自注意力输出和选中记忆
        # [batch, seq, dim] + [batch, seq, dim] → [batch, seq, 2*dim]
        concat_input = torch.cat([h_attn, selected_memory], dim=-1)

        # 步骤2-4: 门控MLP处理（类似SwiGLU）
        # 门控分支：通过SiLU激活函数生成门控信号
        gate = F.silu(self.gate_proj(concat_input))  # [batch, seq_len, dim]

        # 内容分支：生成待门控的内容
        up = self.up_proj(concat_input)  # [batch, seq_len, dim]

        # 门控融合：逐元素相乘
        fusion_output = gate * up  # [batch, seq_len, dim]

        # 步骤5-6: 输出投影和正则化
        output = self.down_proj(fusion_output)  # [batch, seq_len, dim]
        output = self.dropout(output)

        return output