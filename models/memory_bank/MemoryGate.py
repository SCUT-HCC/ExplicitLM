import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Transformer-style Residual Block (Pre-Norm FFN)
    Structure: x + Linear(GELU(Linear(LayerNorm(x))))
    """
    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


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
            - num_candidates_internal: 内部检索数量 (Default: 128)
            - dropout: Dropout概率
            - keys_path: (Optional) Path to pre-computed keys file
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
        self.num_candidates = cfg.get("num_candidates", 32)  # 最终输出的候选数量
        self.num_candidates_internal = cfg.get("num_candidates_internal", 128)  # 内部检索数量 (扩大搜索空间)
        self.num_selected = cfg.get("num_selected", 1)  # 后续选择的最终数量

        # 验证知识库数量必须是完全平方数（Product Key Memory的要求）
        assert int(self.knowledge_num ** 0.5) ** 2 == self.knowledge_num, \
            f"记忆库大小({self.knowledge_num})必须是完全平方数以支持Product Key Memory"

        self.num_keys = int(self.knowledge_num ** 0.5)

        # 查询投影层：升级为 ResNet 结构
        # 1. Input Projection: Align features
        # 2. ResNet Stack: Deep feature extraction
        # 3. Output Projection: Map to knowledge space
        self.gate_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),  # Input Projection
            ResidualBlock(self.dim, expansion_factor=2, dropout=cfg["dropout"]),  # ResBlock 1
            ResidualBlock(self.dim, expansion_factor=2, dropout=cfg["dropout"]),  # ResBlock 2
            nn.LayerNorm(self.dim),  # Final Norm before output projection
            nn.Linear(self.dim, self.knowledge_dim, bias=False)  # Output Projection
        )

        # Product Key Memory: 两个独立的键集合
        # 形状: [2, √knowledge_num, knowledge_dim // 2]
        if "keys_path" in cfg and cfg["keys_path"] and os.path.exists(cfg["keys_path"]):
            print(f"Loading frozen keys from {cfg['keys_path']}...")
            loaded_keys = torch.load(cfg["keys_path"])
            # Ensure shape matches
            expected_shape = (2, self.num_keys, self.knowledge_dim // 2)
            assert loaded_keys.shape == expected_shape, \
                f"Keys shape mismatch. Expected {expected_shape}, got {loaded_keys.shape}"
            
            self.keys = nn.Parameter(loaded_keys)
            self.keys.requires_grad = False  # Freeze keys
            print("Keys loaded and FROZEN.")
        else:
            print("Initializing random keys (Training from scratch)...")
            self.keys = nn.Parameter(torch.randn(2, self.num_keys, self.knowledge_dim // 2))

        # 学习温度系数 (Logit Scale)，初始化为 1/0.07 ≈ 14.3
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Dropout层用于正则化
        self.dropout = nn.Dropout(cfg["dropout"])

    def compute_sub_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算输入与两组键的相似度分数 (Cosine Similarity)

        Args:
            x: 输入张量 [batch, seq, dim]

        Returns:
            scores_1: 第一组键的分数 [batch, seq, num_keys]
            scores_2: 第二组键的分数 [batch, seq, num_keys]
        """
        # 步骤1: 生成查询向量
        queries = self.gate_proj(x)  # [batch, seq_len, knowledge_dim]

        # 步骤2: 分割查询向量为两部分，用于Product Key
        q1 = queries[:, :, : self.knowledge_dim // 2]  # 前半部分
        q2 = queries[:, :, self.knowledge_dim // 2 :]  # 后半部分

        # Normalize queries and keys (Cosine Similarity)
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # Ensure keys are in the same dtype as input (handle fp16/bf16 mismatch)
        keys = self.keys.to(dtype=x.dtype)
        k1 = F.normalize(keys[0], p=2, dim=-1)
        k2 = F.normalize(keys[1], p=2, dim=-1)

        # Clamp logit scale to prevent instability
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # 步骤3: 计算与两个键集合的相似度分数
        # einsum 'bsd,kd->bsk': (batch, seq, dim) × (keys, dim) → (batch, seq, keys)
        scores_1 = torch.einsum("bsd,kd->bsk", q1, k1) * logit_scale
        scores_2 = torch.einsum("bsd,kd->bsk", q2, k2) * logit_scale
        
        return scores_1, scores_2

    def generate_candidates(self, scores_1: torch.Tensor, scores_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据子键分数生成最终候选记忆

        Args:
            scores_1: [batch, seq, num_keys]
            scores_2: [batch, seq, num_keys]

        Returns:
            candidate_indices: [batch, seq, num_candidates]
            candidate_scores: [batch, seq, num_candidates]
        """
        bsz, seq_len, _ = scores_1.shape

        # 步骤4: 对每个键集合选择top-k候选
        # 使用 num_candidates_internal (128) 进行内部检索，扩大搜索空间
        topk_scores_1, topk_indices_1 = scores_1.topk(self.num_candidates_internal, dim=-1)
        topk_scores_2, topk_indices_2 = scores_2.topk(self.num_candidates_internal, dim=-1)

        # 步骤5: 通过笛卡尔积组合两组候选
        # 分数相加：[batch, seq, K_int, 1] + [batch, seq, 1, K_int]
        #         → [batch, seq, K_int, K_int]
        combined_scores = topk_scores_1.unsqueeze(-1) + topk_scores_2.unsqueeze(-2)

        # 索引组合：index1 * num_keys + index2
        # 这样可以将二维索引(i, j)映射到一维记忆索引(i * √N + j)
        combined_indices = (
            topk_indices_1.unsqueeze(-1) * self.num_keys + topk_indices_2.unsqueeze(-2)
        )

        # 步骤6: 展平并选择最终的top-k候选
        # 展平笛卡尔积结果: [batch, seq, K_int * K_int]
        combined_scores = combined_scores.view(bsz, seq_len, -1)
        combined_indices = combined_indices.view(bsz, seq_len, -1)

        # 选择分数最高的num_candidates (32) 个候选作为最终输出
        candidate_scores, candidate_pk_indices = combined_scores.topk(
            self.num_candidates, dim=-1
        )
        candidate_indices = combined_indices.gather(-1, candidate_pk_indices)

        # 步骤7: 归一化候选分数并应用dropout
        candidate_scores = F.softmax(candidate_scores, dim=-1)
        candidate_scores = self.dropout(candidate_scores)

        return candidate_indices, candidate_scores

    def compute_loss(self, scores_1: torch.Tensor, scores_2: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        计算训练损失

        Args:
            scores_1: [batch, seq, num_keys]
            scores_2: [batch, seq, num_keys]
            target_indices: [batch, seq, k]

        Returns:
            loss: scalar
        """
        raise NotImplementedError("Please call compute_loss_soft with target_scores")

    def compute_loss_soft(self, scores_1: torch.Tensor, scores_2: torch.Tensor, 
                         target_indices: torch.Tensor, target_scores: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Compute Soft Label Loss (KL Divergence)
        
        Args:
            scores_1: [batch, seq, num_keys]
            scores_2: [batch, seq, num_keys]
            target_indices: [batch, seq, k] - Indices of top-k targets
            target_scores: [batch, seq, k] - Cosine similarity scores of top-k targets
            temperature: Temperature for smoothing target distribution
        """
        # 1. Compute Log Probabilities of Model
        log_probs_1 = F.log_softmax(scores_1, dim=-1)  # [batch, seq, num_keys]
        log_probs_2 = F.log_softmax(scores_2, dim=-1)  # [batch, seq, num_keys]
        
        # 2. Decompose target indices
        target_u = target_indices // self.num_keys
        target_v = target_indices % self.num_keys
        
        # 3. Gather Model Log Probs for the specific targets
        # log P_model(k) = log P1(u) + log P2(v)
        target_log_probs_1 = log_probs_1.gather(-1, target_u)  # [batch, seq, k]
        target_log_probs_2 = log_probs_2.gather(-1, target_v)  # [batch, seq, k]
        target_log_probs = target_log_probs_1 + target_log_probs_2  # [batch, seq, k]
        
        # 4. Compute Target Distribution P_target
        # We use softmax over the top-k scores to get a probability distribution over the k candidates
        # Note: This assumes the rest of the universe has 0 probability, which is a reasonable approximation for top-32
        target_probs = F.softmax(target_scores / temperature, dim=-1)  # [batch, seq, k]
        
        # 5. Compute Cross Entropy Loss: - sum(P_target * log P_model)
        # Sum over k candidates
        loss_per_sample = - (target_probs * target_log_probs).sum(dim=-1)  # [batch, seq]
        
        return loss_per_sample.mean()

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
            返回的候选项会在后续模块（如ExplicitLMBlock）中进行相似度选择和多样性损失计算
        """
        scores_1, scores_2 = self.compute_sub_scores(x)
        return self.generate_candidates(scores_1, scores_2)

    def forward_loss(self, x: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training the router.
        """
        scores_1, scores_2 = self.compute_sub_scores(x)
        return self.compute_loss(scores_1, scores_2, target_indices)
