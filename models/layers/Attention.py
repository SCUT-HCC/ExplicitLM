import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.configs import LMConfig
from models.layers.pos_cis import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值（KV）张量以支持分组查询注意力（Grouped Query Attention, GQA）

    在GQA中，多个查询头共享同一组键值头，此函数通过重复键值头来匹配查询头的数量。
    等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)，但实现更高效。

    Args:
        x: 键或值张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep: 每个键值头需要重复的次数

    Returns:
        重复后的张量，形状为 (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头自注意力模块（无KV缓存）

    实现了标准的多头自注意力机制，支持：
    - 分组查询注意力（GQA）：多个查询头可以共享键值头
    - 旋转位置编码（RoPE）：通过旋转嵌入注入位置信息
    - Flash Attention优化：当可用时自动使用高效的注意力实现
    - 因果掩码：确保只能关注当前及之前的位置

    注意：此版本完全去除了KV cache相关代码，适用于训练场景
    """

    def __init__(self, args: LMConfig):
        """
        初始化注意力模块

        Args:
            args: 模型配置对象
        """
        super().__init__()

        # 键值头数量配置
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        assert (
            args.n_heads % self.n_kv_heads == 0
        ), f"查询头数量({args.n_heads})必须能被键值头数量({self.n_kv_heads})整除"

        # 注意力头配置
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 查询、键、值和输出的线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # Flash Attention支持检测
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attn
        )

        # 因果注意力掩码（上三角为负无穷）
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor) -> torch.Tensor:
        """
        前向传播（无KV缓存）

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            pos_cis: 旋转位置编码，形状为 (seq_len, head_dim // 2)

        Returns:
            注意力输出张量，形状为 (batch_size, seq_len, dim)
        """
        bsz, seq_len, _ = x.shape

        # 计算查询、键、值
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑为多头形式
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 注意：完全去除了KV cache相关代码

        # 调整维度顺序并扩展键值以匹配查询头数量
        xq, xk, xv = (
            xq.transpose(1, 2),  # (batch, n_heads, seq_len, head_dim)
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 计算注意力
        if self.flash and seq_len != 1:
            # 使用Flash Attention优化
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # 标准注意力计算
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 合并多头输出并应用输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output