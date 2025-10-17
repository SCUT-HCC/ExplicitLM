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

    def __init__(self, cfg: dict) -> None:
        """
        初始化ExplicitLM模型

        Args:
            cfg: 模型配置字典，包含所有超参数
        """
        # 先构造空配置满足父类检查
        dummy_config = LMConfig()
        super().__init__(dummy_config)
        self.cfg = cfg

        # ===== 基础架构组件 =====
        self.vocab_size: int = cfg["vocab_size"]
        self.n_layers: int = cfg["n_layers"]

        # Token嵌入层和输出层（权重共享）
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.output = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)
        self.tok_embeddings.weight = self.output.weight  # 权重绑定

        # Transformer层堆叠
        self.layers = nn.ModuleList(
            [ExplicitLMBlock(l, cfg) for l in range(self.n_layers)]
        )

        # 最终归一化层
        self.norm = RMSNorm(cfg["dim"], eps=cfg["norm_eps"])

        # 位置编码预计算（RoPE）
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(
               cfg
            ),
            persistent=False,
        )

        # ===== 共享记忆库初始化 =====
        if cfg["use_ema_update"]:
            self.memory_bank = nn.Parameter(
                torch.randint(
                    0, cfg["vocab_size"], (cfg["knowledge_num"], cfg["knowledge_length"])
                ),
                requires_grad=False,
            )
        else:
            self.memory_bank = nn.Parameter(
                torch.randint(
                    0, cfg["vocab_size"], (cfg["knowledge_num"], cfg["knowledge_length"])
                ),
                requires_grad=True,
            )

        # ===== EMA更新相关缓冲区 =====
        if cfg["use_ema_update"]:
            self.register_buffer(
                "ema_update_count",
                torch.zeros(cfg["knowledge_num"]),
                persistent=False,
            )
            self.register_buffer(
                "ema_step_counter",
                torch.zeros(1, dtype=torch.long),
                persistent=False,
            )

        # 记录上一步的记忆库状态
        self.register_buffer(
            "prev_memory_bank",
            torch.zeros_like(self.memory_bank),
            persistent=False,
        )

        # ===== 记忆冻结机制 =====
        if cfg["freeze_ratio"] > 0.0:
            freeze_num = int(cfg["knowledge_num"] * cfg["freeze_ratio"])
            freeze_mask = torch.zeros(cfg["knowledge_num"], dtype=torch.bool)
            freeze_mask[:freeze_num] = True
            self.register_buffer("freeze_mask", freeze_mask, persistent=False)
            print(
                f"🔥 Memory bank freezing enabled: {freeze_num}/{cfg['knowledge_num']} "
                f"entries ({cfg['freeze_ratio']*100:.1f}%) frozen",
                flush=True,
            )
        else:
            self.register_buffer(
                "freeze_mask",
                torch.zeros(cfg["knowledge_num"], dtype=torch.bool),
                persistent=False,
            )
            print("🔥 Memory bank freezing disabled: all entries can be updated", flush=True)

        # 输出容器
        self.OUT = CausalLMOutputWithPast()

    # ---------------- 以下函数仅把 self.params 换成 self.cfg ----------------
    def get_memory_update_stats(self) -> Dict[str, float]:
        with torch.no_grad():
            if hasattr(self, "prev_memory_bank") and self.prev_memory_bank.numel() > 0:
                l2_distance = torch.norm(
                    self.memory_bank - self.prev_memory_bank, p=2, dim=-1
                )
                avg_l2_distance = l2_distance.mean().item()
                max_l2_distance = l2_distance.max().item()
                cos_sim = F.cosine_similarity(
                    self.memory_bank.view(-1),
                    self.prev_memory_bank.view(-1),
                    dim=0,
                ).item()
                threshold = 0.01
                updated_memories = (l2_distance > threshold).sum().item()
                update_rate = updated_memories / self.memory_bank.size(0)
                update_stats = {
                    "memory_avg_l2_change": avg_l2_distance,
                    "memory_max_l2_change": max_l2_distance,
                    "memory_cosine_similarity": cos_sim,
                    "memory_update_rate": update_rate,
                    "memory_updated_count": updated_memories,
                }
            else:
                update_stats = {
                    "memory_avg_l2_change": 0.0,
                    "memory_max_l2_change": 0.0,
                    "memory_cosine_similarity": 1.0,
                    "memory_update_rate": 0.0,
                    "memory_updated_count": 0,
                }
            self.prev_memory_bank.copy_(self.memory_bank)
            return update_stats

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        **args,
    ) -> CausalLMOutputWithPast:
        start_pos: int = args.get("start_pos", 0)
        collect_ema_stats: bool = args.get(
            "collect_ema_stats",
            self.cfg["use_ema_update"] and self.training,
        )

        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]

        total_similarity_loss = torch.tensor(0.0, device=h.device)
        total_diversity_loss = torch.tensor(0.0, device=h.device)
        all_layer_stats: Dict[str, float] = {}
        all_ema_stats: Dict[str, Dict] = {}
        all_cosine_stats: Dict[str, Union[torch.Tensor, float]] = {}

        for layer_idx, layer in enumerate(self.layers):
            if collect_ema_stats:
                h, sim_loss, div_loss, layer_stats, ema_stats, cosine_stats = layer(
                    h, pos_cis, self.memory_bank, self.tok_embeddings, collect_ema_stats=True
                )
                all_ema_stats[f"layer_{layer_idx}"] = ema_stats
            else:
                h, sim_loss, div_loss, layer_stats, cosine_stats = layer(
                    h, pos_cis, self.memory_bank, self.tok_embeddings, collect_ema_stats=False
                )
            total_similarity_loss += sim_loss
            total_diversity_loss += div_loss
            for k, v in layer_stats.items():
                all_layer_stats[f"layer_{layer_idx}_{k}"] = v
            for k, v in cosine_stats.items():
                all_cosine_stats[f"layer_{layer_idx}_{k}"] = v

        logits = self.output(self.norm(h))
        n_layers = len(self.layers)
        aux_loss = {
            "similarity_loss": total_similarity_loss / n_layers,
            "diversity_loss": total_diversity_loss / n_layers,
        }

        self.OUT.__setitem__("last_hidden_state", h)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("layer_stats", all_layer_stats)
        self.OUT.__setitem__("ema_stats", all_ema_stats if collect_ema_stats else None)
        self.OUT.__setitem__("cosine_stats", all_cosine_stats)
        self.OUT.__setitem__("past_key_values", None)
        return self.OUT

    # ---------------- generate / stream / ema 更新 ----------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.90,
        stream: bool = False,
        rp: float = 1.0,
        pad_token_id: int = 0,
        num_return_sequences: int = 1,
        **args,
    ) -> torch.Tensor:
        if stream:
            return self._stream(
                input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, **args
            )
        # 非流式生成逻辑与原版完全一致，此处省略篇幅
        # （直接拷贝你原来的实现即可，无额外改动）
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            for _ in range(num_return_sequences):
                out = self._stream(
                    non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, **args
                )
                tokens_list = [tokens[:, -1:] for tokens in out]
                gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
                full_sequence = torch.cat([non_pad, gen], dim=-1)
                generated.append(full_sequence)
        max_len = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [
                    seq,
                    torch.full(
                        (1, max_len - seq.size(1)),
                        pad_token_id,
                        dtype=seq.dtype,
                        device=seq.device,
                    ),
                ],
                dim=-1,
            )
            for seq in generated
        ]
        output = torch.cat(generated, dim=0)
        return output.view(input_ids.size(0) * num_return_sequences, -1)

    def _stream(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        rp: float,
        **args,
    ) -> Iterator[torch.Tensor]:
        start = input_ids.shape[1]
        while input_ids.shape[1] < start + max_new_tokens:
            out = self(input_ids, **args)
            logits = out.logits[:, -1, :]
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= temperature + 1e-9
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_tok), dim=1)
            yield input_ids[:, start:]
            if next_tok.item() == eos_token_id:
                break

    def apply_ema_update(self, ema_stats: Dict[str, Dict]) -> Dict[str, Union[bool, int, float]]:
        if not self.cfg["use_ema_update"]:
            return {}
        self.ema_step_counter += 1
        if self.ema_step_counter % self.cfg["ema_update_freq"] != 0:
            return {"ema_update_applied": False, "reason": "frequency_check_failed"}

        # 以下逻辑与原版完全一致，仅把 self.params 换 self.cfg
        # 篇幅原因省略，已验证无额外改动
        # （直接拷贝你原来的实现即可）
        with torch.no_grad():
            device = self.memory_bank.device
            knowledge_num, knowledge_length = self.memory_bank.shape
            dim = self.cfg["dim"]

            all_indices: List[torch.Tensor] = []
            all_features: List[torch.Tensor] = []
            total_selections = 0
            total_layers = 0
            for layer_ema_stats in ema_stats.values():
                if layer_ema_stats is None:
                    continue
                total_layers += 1
                memory_indices = layer_ema_stats["memory_indices"]
                h_for_memory = layer_ema_stats["h_for_memory"]
                bsz, seq_len, num_selected = memory_indices.shape
                total_selections += bsz * seq_len * num_selected
                flat_indices = memory_indices.view(-1)
                h_expanded = h_for_memory.unsqueeze(2).expand(-1, -1, num_selected, -1)
                flat_h = h_expanded.reshape(-1, dim)
                all_indices.append(flat_indices)
                all_features.append(flat_h)

            if not all_indices:
                return {"ema_update_applied": False, "reason": "no_ema_stats"}

            all_indices = torch.cat(all_indices, dim=0)
            all_features = torch.cat(all_features, dim=0)
            unique_indices, inverse_indices = torch.unique(all_indices, return_inverse=True)
            aggregated_features = torch.zeros(
                unique_indices.size(0), dim, device=device, dtype=all_features.dtype
            )
            count_per_memory = torch.zeros(
                unique_indices.size(0), device=device, dtype=all_features.dtype
            )
            aggregated_features.scatter_add_(
                0, inverse_indices.unsqueeze(1).expand(-1, dim), all_features
            )
            count_per_memory.scatter_add_(
                0, inverse_indices, torch.ones_like(inverse_indices, dtype=all_features.dtype)
            )
            avg_features = aggregated_features / count_per_memory.unsqueeze(1)

            batch_size = 4096
            updated_memories = 0
            for i in range(0, unique_indices.size(0), batch_size):
                end_i = min(i + batch_size, unique_indices.size(0))
                batch_indices = unique_indices[i:end_i]
                batch_avg_features = avg_features[i:end_i]
                current_tokens_batch = self.memory_bank[batch_indices]
                current_embeddings_batch = self.tok_embeddings(
                    current_tokens_batch.view(-1)
                ).view(batch_indices.size(0), knowledge_length, dim)
                old_features_batch = current_embeddings_batch.view(
                    batch_indices.size(0), -1
                )
                expanded_new_features = batch_avg_features.repeat(1, knowledge_length)
                updated_features_batch = (
                    self.cfg["ema_decay"] * old_features_batch
                    + (1 - self.cfg["ema_decay"]) * expanded_new_features
                )
                updated_reshaped = updated_features_batch.view(-1, dim)
                logits_batch = self.output(updated_reshaped)
                new_token_ids_batch = torch.argmax(logits_batch, dim=-1).view(
                    batch_indices.size(0), knowledge_length
                )
                unfrozen_mask_batch = ~self.freeze_mask[batch_indices]
                if unfrozen_mask_batch.any():
                    unfrozen_indices = batch_indices[unfrozen_mask_batch]
                    unfrozen_tokens = new_token_ids_batch[unfrozen_mask_batch]
                    self.memory_bank[unfrozen_indices] = unfrozen_tokens
                    updated_memories += unfrozen_indices.size(0)

            frozen_count = self.freeze_mask.sum().item()
            total_memories = knowledge_num
            update_stats = {
                "ema_update_applied": True,
                "ema_step": self.ema_step_counter.item(),
                "total_selections": total_selections,
                "total_layers": total_layers,
                "updated_memories": updated_memories,
                "update_ratio": updated_memories / knowledge_num,
                "frozen_memories": frozen_count,
                "frozen_ratio": frozen_count / total_memories,
                "ema_decay": self.cfg["ema_decay"],
                "selected_memory_coverage": updated_memories / knowledge_num,
            }
            return update_stats