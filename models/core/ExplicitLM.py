"""
ExplicitLM: 基于显式记忆增强的语言模型

该模型实现了一个创新的Transformer架构，使用显式记忆库替代传统的FFN层：
- 共享记忆库存储可学习的token序列
- EMA更新机制实现类似VQ-VAE的codebook更新
- 支持记忆冻结策略以保护重要知识
- 多损失优化系统（相似度损失+多样性损失）
- 无KV缓存的流式生成能力
"""

from typing import Dict, List, Optional, Union, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.configs.LMConfig import LMConfig
from models.core.ExplicitLMBlock import ExplicitLMBlock
from models.layers.RMSNorm import RMSNorm
from models.layers.pos_cis import precompute_pos_cis


class MiniMindLM(PreTrainedModel):
    """
    基于显式记忆增强的因果语言模型

    该模型通过共享记忆库增强Transformer架构，记忆库存储token序列并通过
    EMA机制动态更新，实现了更高效的知识存储和检索机制。
    """

    config_class = LMConfig

    def __init__(self, params: Optional[LMConfig] = None) -> None:
        """
        初始化ExplicitLM模型

        Args:
            params: 模型配置对象，包含所有超参数设置
        """
        self.params = params
        super().__init__(self.params)

        # ===== 基础架构组件 =====
        self.vocab_size: int = params.vocab_size
        self.n_layers: int = params.n_layers

        # Token嵌入层和输出层（权重共享）
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight  # 权重绑定

        # Transformer层堆叠
        self.layers = nn.ModuleList([ExplicitLMBlock(l, params) for l in range(self.n_layers)])

        # 最终归一化层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # 位置编码预计算（RoPE）
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
            persistent=False
        )

        # ===== 共享记忆库初始化 =====
        # 存储token_id序列而非特征向量（类似VQ-VAE的codebook设计）
        # 形状: [knowledge_num, knowledge_length]
        if params.use_ema_update:
            # EMA模式：禁用梯度更新，通过指数移动平均更新记忆
            self.memory_bank = nn.Parameter(
                torch.randint(0, params.vocab_size, (params.knowledge_num, params.knowledge_length)),
                requires_grad=False
            )
        else:
            # 梯度模式：传统端到端梯度更新
            self.memory_bank = nn.Parameter(
                torch.randint(0, params.vocab_size, (params.knowledge_num, params.knowledge_length)),
                requires_grad=True
            )

        # ===== EMA更新相关缓冲区 =====
        if params.use_ema_update:
            # 记录每个记忆条目的更新次数
            self.register_buffer(
                'ema_update_count',
                torch.zeros(params.knowledge_num),
                persistent=False
            )
            # EMA全局步数计数器
            self.register_buffer(
                'ema_step_counter',
                torch.zeros(1, dtype=torch.long),
                persistent=False
            )

        # 记录上一步的记忆库状态，用于计算更新统计
        self.register_buffer(
            'prev_memory_bank',
            torch.zeros_like(self.memory_bank),
            persistent=False
        )

        # ===== 记忆冻结机制 =====
        # 标记哪些记忆条目被冻结以保护重要知识
        if params.freeze_ratio > 0.0:
            freeze_num = int(params.knowledge_num * params.freeze_ratio)
            freeze_mask = torch.zeros(params.knowledge_num, dtype=torch.bool)
            freeze_mask[:freeze_num] = True  # 冻结前N个条目
            self.register_buffer('freeze_mask', freeze_mask, persistent=False)
            print(
                f"🔥 Memory bank freezing enabled: {freeze_num}/{params.knowledge_num} "
                f"entries ({params.freeze_ratio*100:.1f}%) frozen",
                flush=True
            )
        else:
            self.register_buffer(
                'freeze_mask',
                torch.zeros(params.knowledge_num, dtype=torch.bool),
                persistent=False
            )
            print("🔥 Memory bank freezing disabled: all entries can be updated", flush=True)

        # 输出容器
        self.OUT = CausalLMOutputWithPast()
    
    def get_memory_update_stats(self) -> Dict[str, float]:
        """
        计算记忆库更新统计信息

        该方法通过比较当前记忆库和上一步的记忆库状态，计算各种更新指标，
        包括L2距离变化、余弦相似度和更新率等。

        Returns:
            update_stats: 包含以下键的统计字典：
                - memory_avg_l2_change: 平均L2距离变化
                - memory_max_l2_change: 最大L2距离变化
                - memory_cosine_similarity: 整体余弦相似度
                - memory_update_rate: 更新率（变化显著的记忆比例）
                - memory_updated_count: 更新的记忆条目数量
        """
        with torch.no_grad():
            if hasattr(self, 'prev_memory_bank') and self.prev_memory_bank.numel() > 0:
                # 计算L2距离变化
                l2_distance = torch.norm(self.memory_bank - self.prev_memory_bank, p=2, dim=-1)
                avg_l2_distance = l2_distance.mean().item()
                max_l2_distance = l2_distance.max().item()
                
                # 计算余弦相似度
                cos_sim = F.cosine_similarity(
                    self.memory_bank.view(-1), 
                    self.prev_memory_bank.view(-1), 
                    dim=0
                ).item()
                
                # 计算更新率（发生显著变化的记忆条目比例）
                threshold = 0.01  # 更新阈值
                updated_memories = (l2_distance > threshold).sum().item()
                update_rate = updated_memories / self.memory_bank.size(0)
                
                update_stats = {
                    'memory_avg_l2_change': avg_l2_distance,
                    'memory_max_l2_change': max_l2_distance,
                    'memory_cosine_similarity': cos_sim,
                    'memory_update_rate': update_rate,
                    'memory_updated_count': updated_memories
                }
            else:
                # 第一次调用时的默认值
                update_stats = {
                    'memory_avg_l2_change': 0.0,
                    'memory_max_l2_change': 0.0,
                    'memory_cosine_similarity': 1.0,
                    'memory_update_rate': 0.0,
                    'memory_updated_count': 0
                }
            
            # 更新prev_memory_bank
            self.prev_memory_bank.copy_(self.memory_bank)
            
            return update_stats

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        **args
    ) -> CausalLMOutputWithPast:
        """
        前向传播（不支持KV缓存）

        该方法实现完整的前向传播流程，包括：
        1. Token嵌入和位置编码
        2. 通过所有Transformer层（含记忆增强机制）
        3. 收集各层的损失和统计信息
        4. 最终归一化和输出投影

        Args:
            input_ids: 形状为[batch_size, seq_len]的输入token ID张量
            **args: 其他参数，支持：
                - start_pos: 起始位置（默认0）
                - collect_ema_stats: 是否收集EMA统计信息

        Returns:
            CausalLMOutputWithPast: 包含以下字段的输出对象：
                - logits: 语言模型的预测logits
                - last_hidden_state: 最后一层的隐藏状态
                - aux_loss: 辅助损失字典（相似度损失+多样性损失）
                - layer_stats: 各层的统计信息
                - ema_stats: EMA更新统计（如果collect_ema_stats=True）
                - cosine_stats: 余弦相似度统计
                - past_key_values: None（不支持KV缓存）
        """
        # 提取参数
        start_pos: int = args.get('start_pos', 0)
        collect_ema_stats: bool = args.get('collect_ema_stats', self.params.use_ema_update and self.training)

        # ===== 第一阶段：嵌入和位置编码 =====
        h = self.dropout(self.tok_embeddings(input_ids))  # [batch_size, seq_len, dim]
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]

        # ===== 第二阶段：Transformer层处理 =====
        # 收集所有层的损失和统计信息（双损失系统：相似度+多样性）
        total_similarity_loss = torch.tensor(0.0, device=h.device)
        total_diversity_loss = torch.tensor(0.0, device=h.device)
        all_layer_stats: Dict[str, float] = {}
        all_ema_stats: Dict[str, Dict] = {}
        all_cosine_stats: Dict[str, Union[torch.Tensor, float]] = {}

        for layer_idx, layer in enumerate(self.layers):
            if collect_ema_stats:
                # 训练模式：收集EMA更新所需的统计信息
                h, similarity_loss, diversity_loss, layer_stats, ema_stats, cosine_stats = layer(
                    h, pos_cis, self.memory_bank, self.tok_embeddings, collect_ema_stats=True
                )
                all_ema_stats[f'layer_{layer_idx}'] = ema_stats
            else:
                # 推理模式：不收集EMA统计
                h, similarity_loss, diversity_loss, layer_stats, cosine_stats = layer(
                    h, pos_cis, self.memory_bank, self.tok_embeddings, collect_ema_stats=False
                )

            # 累加双损失
            total_similarity_loss += similarity_loss
            total_diversity_loss += diversity_loss

            # 收集各层统计信息（添加层级前缀）
            for key, value in layer_stats.items():
                all_layer_stats[f'layer_{layer_idx}_{key}'] = value

            # 收集余弦相似度统计
            for key, value in cosine_stats.items():
                all_cosine_stats[f'layer_{layer_idx}_{key}'] = value

        # ===== 第三阶段：输出投影 =====
        logits = self.output(self.norm(h))  # [batch_size, seq_len, vocab_size]

        # ===== 第四阶段：构建输出 =====
        # 计算平均辅助损失
        n_layers = len(self.layers)
        aux_loss = {
            'similarity_loss': total_similarity_loss / n_layers,
            'diversity_loss': total_diversity_loss / n_layers,
        }

        # 填充输出容器
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('layer_stats', all_layer_stats)
        self.OUT.__setitem__('ema_stats', all_ema_stats if collect_ema_stats else None)
        self.OUT.__setitem__('cosine_stats', all_cosine_stats)
        self.OUT.__setitem__('past_key_values', None)  # 不支持KV缓存

        return self.OUT

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.90,
        stream: bool = False,
        rp: float = 1.,
        pad_token_id: int = 0,
        num_return_sequences: int = 1,
        **args
    ) -> torch.Tensor:
        """
        文本生成（不支持KV缓存）

        该方法支持流式和非流式两种生成模式，使用top-p采样和重复惩罚机制。
        由于不支持KV缓存，每一步都需要重新计算整个序列。

        Args:
            input_ids: 输入token序列，形状[batch_size, seq_len]
            eos_token_id: 结束符token ID（默认2）
            max_new_tokens: 最大生成token数量（默认1024）
            temperature: 采样温度，控制输出多样性（默认0.75）
            top_p: nucleus采样阈值（默认0.90）
            stream: 是否使用流式生成（默认False）
            rp: 重复惩罚系数（默认1.0，无惩罚）
            pad_token_id: 填充token ID（默认0）
            num_return_sequences: 每个输入生成的序列数量（默认1）
            **args: 其他传递给forward的参数

        Returns:
            生成的token序列，形状[batch_size * num_return_sequences, total_seq_len]
        """
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            for _ in range(num_return_sequences):
                out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, **args)
                tokens_list = [tokens[:, -1:] for tokens in out]
                gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
                full_sequence = torch.cat([non_pad, gen], dim=-1)
                generated.append(full_sequence)

        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        output = torch.cat(generated, dim=0)
        res = output.view(input_ids.size(0) * num_return_sequences, -1)
        return res

    def _stream(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        rp: float,
        **args
    ) -> Iterator[torch.Tensor]:
        """
        流式生成（不支持KV缓存）

        该方法实现流式token生成，每次迭代返回新生成的token序列。
        由于不支持KV缓存，每一步都需要重新计算整个序列的表示。

        Args:
            input_ids: 输入token序列，形状[1, seq_len]
            eos_token_id: 结束符token ID
            max_new_tokens: 最大生成token数量
            temperature: 采样温度
            top_p: nucleus采样阈值
            rp: 重复惩罚系数
            **args: 其他传递给forward的参数

        Yields:
            每次生成后新增的token序列，形状[1, generated_len]
        """
        start = input_ids.shape[1]
        while input_ids.shape[1] < start + max_new_tokens:
	            # 每次都重新计算整个序列（因为没有KV cache）
            out = self(input_ids, **args)
            logits = out.logits[:, -1, :]
            
            # 重复惩罚
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            
            # Top-p采样
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
    
    def apply_ema_update(self, ema_stats: Dict[str, Dict]) -> Dict[str, Union[bool, int, float]]:
        """
        应用基于EMA的记忆库更新（批量化优化版本）

        该方法实现了类似VQ-VAE的EMA更新机制，通过以下步骤更新记忆库：
        1. 收集所有层选中的记忆索引和对应的查询特征
        2. 对每个被选中的记忆计算平均查询特征
        3. 使用EMA公式更新记忆的特征表示
        4. 将更新后的特征重新编码为token序列
        5. 应用冻结mask保护重要记忆

        Args:
            ema_stats: 从forward传播收集的EMA统计信息字典，格式为：
                {
                    'layer_0': {
                        'memory_indices': torch.Tensor,  # [batch, seq_len, num_selected]
                        'h_for_memory': torch.Tensor,    # [batch, seq_len, dim]
                        ...
                    },
                    'layer_1': {...},
                    ...
                }

        Returns:
            update_stats: 更新统计字典，包含：
                - ema_update_applied: 是否成功应用更新
                - ema_step: 当前EMA步数
                - total_selections: 总选择次数
                - total_layers: 参与更新的层数
                - updated_memories: 实际更新的记忆条目数
                - update_ratio: 更新比例
                - frozen_memories: 冻结的记忆数量
                - frozen_ratio: 冻结比例
                - ema_decay: EMA衰减系数
                - selected_memory_coverage: 记忆覆盖率
        """
        if not self.params.use_ema_update:
            return {}

        # ===== 第一阶段：更新频率检查 =====
        self.ema_step_counter += 1

        if self.ema_step_counter % self.params.ema_update_freq != 0:
            return {'ema_update_applied': False, 'reason': 'frequency_check_failed'}

        with torch.no_grad():
            device = self.memory_bank.device
            knowledge_num, knowledge_length = self.memory_bank.shape
            dim = self.params.dim

            # ===== 第二阶段：数据收集 =====
            # 批量收集所有层的选择信息，避免频繁的字典操作
            all_indices: List[torch.Tensor] = []
            all_features: List[torch.Tensor] = []
            total_selections = 0
            total_layers = 0
            
            # 遍历所有层的EMA统计信息
            for layer_ema_stats in ema_stats.values():
                if layer_ema_stats is None:
                    continue

                total_layers += 1
                memory_indices = layer_ema_stats['memory_indices']  # [batch, seq_len, num_selected]
                h_for_memory = layer_ema_stats['h_for_memory']      # [batch, seq_len, dim]

                bsz, seq_len, num_selected = memory_indices.shape
                total_selections += bsz * seq_len * num_selected

                # 展平索引用于批量处理
                flat_indices = memory_indices.view(-1)  # [batch * seq_len * num_selected]

                # 为每个选择位置复制对应的查询特征
                h_expanded = h_for_memory.unsqueeze(2).expand(-1, -1, num_selected, -1)  # [batch, seq_len, num_selected, dim]
                flat_h = h_expanded.reshape(-1, dim)  # [batch * seq_len * num_selected, dim]

                all_indices.append(flat_indices)
                all_features.append(flat_h)

            if not all_indices:
                return {'ema_update_applied': False, 'reason': 'no_ema_stats'}

            # ===== 第三阶段：数据合并和聚合 =====
            # 合并所有层的数据
            all_indices = torch.cat(all_indices, dim=0)  # [total_selections]
            all_features = torch.cat(all_features, dim=0)  # [total_selections, dim]

            # 计算每个唯一记忆索引的平均查询特征（批量化避免循环）
            unique_indices, inverse_indices = torch.unique(all_indices, return_inverse=True)

            # 使用scatter_add进行批量聚合
            aggregated_features = torch.zeros(unique_indices.size(0), dim, device=device, dtype=all_features.dtype)
            count_per_memory = torch.zeros(unique_indices.size(0), device=device, dtype=all_features.dtype)

            aggregated_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, dim), all_features)
            count_per_memory.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=all_features.dtype))

            # 计算每个记忆的平均查询特征
            avg_features = aggregated_features / count_per_memory.unsqueeze(1)  # [unique_count, dim]

            # ===== 第四阶段：分批EMA更新 =====
            # 分批处理以控制显存使用
            batch_size = 4096  # 每批处理4096个记忆条目
            updated_memories = 0

            for i in range(0, unique_indices.size(0), batch_size):
                end_i = min(i + batch_size, unique_indices.size(0))
                batch_indices = unique_indices[i:end_i]
                batch_avg_features = avg_features[i:end_i]

                # 解码当前记忆的token序列为特征向量
                current_tokens_batch = self.memory_bank[batch_indices]  # [batch_size, knowledge_length]
                current_embeddings_batch = self.tok_embeddings(current_tokens_batch.view(-1)).view(
                    batch_indices.size(0), knowledge_length, dim
                )  # [batch_size, knowledge_length, dim]

                # 准备EMA更新的特征
                old_features_batch = current_embeddings_batch.view(batch_indices.size(0), -1)  # [batch_size, knowledge_length * dim]
                expanded_new_features = batch_avg_features.repeat(1, knowledge_length)  # [batch_size, knowledge_length * dim]

                # EMA更新公式：new = γ * old + (1-γ) * new_avg
                updated_features_batch = (
                    self.params.ema_decay * old_features_batch +
                    (1 - self.params.ema_decay) * expanded_new_features
                )

                # 将更新后的特征重新编码为token ID
                updated_reshaped = updated_features_batch.view(-1, dim)  # [batch_size * knowledge_length, dim]
                logits_batch = self.output(updated_reshaped)  # [batch_size * knowledge_length, vocab_size]
                new_token_ids_batch = torch.argmax(logits_batch, dim=-1).view(batch_indices.size(0), knowledge_length)

                # ===== 第五阶段：应用冻结mask =====
                # 只更新未被冻结的记忆条目
                unfrozen_mask_batch = ~self.freeze_mask[batch_indices]  # [batch_size]

                if unfrozen_mask_batch.any():
                    unfrozen_indices = batch_indices[unfrozen_mask_batch]
                    unfrozen_tokens = new_token_ids_batch[unfrozen_mask_batch]
                    self.memory_bank[unfrozen_indices] = unfrozen_tokens
                    updated_memories += unfrozen_indices.size(0)
            
            # ===== 第六阶段：统计信息收集 =====
            update_ratio = updated_memories / knowledge_num

            # 计算冻结相关统计
            frozen_count = self.freeze_mask.sum().item()
            total_memories = knowledge_num

            # 构建更新统计字典
            update_stats = {
                'ema_update_applied': True,
                'ema_step': self.ema_step_counter.item(),
                'total_selections': total_selections,
                'total_layers': total_layers,
                'updated_memories': updated_memories,
                'update_ratio': update_ratio,
                'frozen_memories': frozen_count,
                'frozen_ratio': frozen_count / total_memories,
                'ema_decay': self.params.ema_decay,
                'selected_memory_coverage': updated_memories / knowledge_num,
            }

            return update_stats