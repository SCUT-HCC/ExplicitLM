import torch


def precompute_pos_cis(cfg: dict, end: int = int(32 * 1024)) -> torch.Tensor:
    """
    预计算旋转位置编码（RoPE）的复数表示

    RoPE通过旋转矩阵为每个位置生成独特的编码，使模型能够感知序列中token的相对位置。
    使用复数表示可以高效地实现二维旋转变换。

    Args:
        cfg: 模型配置字典，必须包含 "dim" 和 "rope_theta"
        end: 最大序列长度

    Returns:
        复数形式的位置编码张量，形状为 (end, dim//2)
    """
    dim = cfg["dim"] // cfg["n_heads"]  # head_dim
    theta = cfg["rope_theta"]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, pos_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将旋转位置编码应用到查询（Q）和键（K）张量上

    通过复数乘法实现旋转变换，为注意力机制注入位置信息。

    Args:
        xq: 查询张量，形状为 (batch, seqlen, n_heads, head_dim)
        xk: 键张量，形状为 (batch, seqlen, n_heads, head_dim)
        pos_cis: 预计算的位置编码，形状为 (seqlen, head_dim//2)

    Returns:
        应用位置编码后的 (xq, xk) 元组
    """

    def unite_shape(pos_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        assert 1 < ndim, f"输入张量维度必须大于1，当前为{ndim}"
        assert pos_cis.shape == (
            x.shape[1],
            x.shape[-1],
        ), f"位置编码形状{pos_cis.shape}与输入形状不匹配，期望({x.shape[1]}, {x.shape[-1]})"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)