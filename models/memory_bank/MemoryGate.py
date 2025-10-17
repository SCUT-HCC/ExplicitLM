import torch
import torch.nn as nn
import torch.nn.functional as F

from models.configs import LMConfig


class MemoryGate(nn.Module):
    """
    基于Product Key Memory的记忆选择门控机制

    使用Product Key Memory算法实现高效的记忆库检索。通过将查询向量分解为两个子查询，
    并分别与两组独立的键进行匹配，最终通过笛卡尔积组合得到候选记忆索引。

    核心思想：
    - 将N个记忆拆分为√N × √N的Product Key结构
    - 查询向量分为两部分，分别匹配两组键
    - 通过top-k选择和笛卡尔积生成候选记忆
    - 使用softmax归一化得到候选分数

    Args:
        cfg: 模型配置字典，需包含以下字段：
            - dim: 输入特征维度
            - knowledge_num: 记忆库大小（必须是完全平方数）
            - knowledge_dim: 记忆键的维度
            - num_candidates: 生成的候选记忆数量
            - dropout: Dropout概率
    """

    def __init__(self, cfg: dict) -> None:
        """
        初始化记忆门控模块

        Args:
            cfg: 模型配置字典
        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg["dim"]
        self.knowledge_num = cfg["knowledge_num"]
        self.knowledge_dim = cfg["knowledge_dim"]

        # 候选记忆配置
        self.num_candidates = cfg.get("num_candidates", 32)  # 生成的候选数量
        self.num_selected = cfg.get("num_selected", 1)  # 后续选择的最终数量

        # 验证知识库数量必须是完全平方数（Product Key Memory的要求）
        assert int(self.knowledge_num ** 0.5) ** 2 == self.knowledge_num, \
            f"记忆库大小({self.knowledge_num})必须是完全平方数以支持Product Key Memory"

        self.num_keys = int(self.knowledge_num ** 0.5)

        # 查询投影层：将输入维度映射到knowledge_dim
        # 输出会被分割为两部分，分别用于两个Product Key
        self.gate_proj = nn.Linear(self.dim, self.knowledge_dim, bias=False)

        # Product Key Memory: 两个独立的键集合
        # 形状: [2, √knowledge_num, knowledge_dim // 2]
        self.keys = nn.Parameter(torch.randn(2, self.num_keys, self.knowledge_dim // 2))

        # Dropout层用于正则化
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：通过Product Key Memory选择候选记忆

        处理流程：
        1. 将输入投影到查询空间并分割为两部分
        2. 分别与两组键计算相似度得分
        3. 对每组键选择top-k候选
        4. 通过笛卡尔积组合两组候选
        5. 选择最终的top-k候选记忆
        6. 归一化分数并应用dropout

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]

        Returns:
            candidate_indices: 候选记忆索引，形状为 [batch_size, seq_len, num_candidates]
            candidate_scores: 候选记忆分数，形状为 [batch_size, seq_len, num_candidates]

        Note:
            返回的候选项会在后续模块（如MiniMindBlock）中进行相似度选择和多样性损失计算
        """
        bsz, seq_len, _ = x.shape

        # 步骤1: 生成查询向量
        queries = self.gate_proj(x)  # [batch, seq_len, knowledge_dim]

        # 步骤2: 分割查询向量为两部分，用于Product Key
        q1 = queries[:, :, : self.knowledge_dim // 2]  # 前半部分
        q2 = queries[:, :, self.knowledge_dim // 2 :]  # 后半部分

        # 步骤3: 计算与两个键集合的相似度分数
        # einsum 'bsd,kd->bsk': (batch, seq, dim) × (keys, dim) → (batch, seq, keys)
        scores_1 = torch.einsum("bsd,kd->bsk", q1, self.keys[0])  # [batch, seq_len, num_keys]
        scores_2 = torch.einsum("bsd,kd->bsk", q2, self.keys[1])  # [batch, seq_len, num_keys]

        # 步骤4: 对每个键集合选择top-k候选
        topk_scores_1, topk_indices_1 = scores_1.topk(self.num_candidates, dim=-1)
        topk_scores_2, topk_indices_2 = scores_2.topk(self.num_candidates, dim=-1)

        # 步骤5: 通过笛卡尔积组合两组候选
        # 分数相加：[batch, seq, num_candidates, 1] + [batch, seq, 1, num_candidates]
        #         → [batch, seq, num_candidates, num_candidates]
        combined_scores = topk_scores_1.unsqueeze(-1) + topk_scores_2.unsqueeze(-2)

        # 索引组合：index1 * num_keys + index2
        # 这样可以将二维索引(i, j)映射到一维记忆索引(i * √N + j)
        combined_indices = (
            topk_indices_1.unsqueeze(-1) * self.num_keys + topk_indices_2.unsqueeze(-2)
        )

        # 步骤6: 展平并选择最终的top-k候选
        # 展平笛卡尔积结果: [batch, seq, num_candidates * num_candidates]
        combined_scores = combined_scores.view(bsz, seq_len, -1)
        combined_indices = combined_indices.view(bsz, seq_len, -1)

        # 选择分数最高的num_candidates个候选
        candidate_scores, candidate_pk_indices = combined_scores.topk(
            self.num_candidates, dim=-1
        )
        candidate_indices = combined_indices.gather(-1, candidate_pk_indices)

        # 步骤7: 归一化候选分数并应用dropout
        candidate_scores = F.softmax(candidate_scores, dim=-1)
        candidate_scores = self.dropout(candidate_scores)

        return candidate_indices, candidate_scores