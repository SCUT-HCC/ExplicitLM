from transformers import PretrainedConfig
from typing import List


class LMConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            ####################################################
            # 基本模型架构参数
            ####################################################
            dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6400,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            rope_theta: int = 1e6,
            dropout: float = 0.0,
            flash_attn: bool = True,
            embeddings_epoch: int = 2,
            ####################################################
            # DB相关配置
            ####################################################
            disable_db: bool = False,  # 特殊模式：禁用数据库功能
            ####################################################
            # MOE相关配置
            # 当use_moe为false时，以下参数无效
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: bool = True,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            ####################################################
            # 知识库相关配置
            ####################################################
            knowledge_num: int = 1024*1024,
            knowledge_length: int = 16,
            knowledge_dim: int = 128,
            ####################################################
            # EMA更新相关配置（灵感来自VQ-VAE）
            ####################################################
            use_ema_update: bool = True,   # 是否使用EMA更新memory_bank
            ema_decay: float = 0.9,        # 🔥 1.4.6: 进一步降低衰减率，允许更激进更新 (0.999 → 0.8)
            ema_update_freq: int = 5,      # 🔥 1.4.6: 进一步提高更新频率 (1 → 5)
            use_token_memory: bool = True,  # 🔥 1.4.6: 新增token-based memory flag
            freeze_ratio: float = 0.2,     # 🔥 新增: memory_bank冻结率 (0.0表示不冻结，0.2表示20%条目不更新)
            ####################################################
            # 实验1.4.10: 优化的Gumbel-Softmax + 多样性损失
            ####################################################
            num_candidates: int = 16,       # 🔥 实验1.4.10优化: 候选记忆条目数量 (32→16 减少50%显存)
            num_selected: int = 1,          # 🔥 实验1.4.10: 选中的记忆条目数量 (现在只选1个最佳)
            gumbel_temperature: float = 1.0, # 🔥 实验1.4.10: Gumbel-Softmax温度参数
            ####################################################
            # 三元组提取相关配置
            ####################################################
            max_subject_len: int = 8,
            max_predicate_len: int = 4,
            max_object_len: int = 8,
            ####################################################
            # SwanLab实验追踪相关配置
            ####################################################
            use_swanlab: bool = False,
            swanlab_online: bool = False,
            swanlab_project: str = "MiniMind",
            ####################################################
            # 模型初始化相关配置
            ####################################################
            model_variant: str = "model_memory",  # 模型变体类型：model, model_original, model_no_feed, model_memory
            pretrained_embedding_path: str = None,  # 预训练嵌入权重文件路径
            database_init_path: str = None,  # 知识库/记忆库初始化数据文件路径
            cache_path: str = "cache/knowledge_cache.pt",  # 处理后数据的缓存路径
            recompute_cache: bool = False,  # 是否强制重新计算缓存
            ####################################################
            # 训练相关配置
            ####################################################
            dataset_path: str = "data/database/merged_pretrain.jsonl",  # 预训练数据集路径
            val_dataset_path: str = "data/benchmarks/eval_data.json",  # 验证数据集路径
            batch_size: int = 48,  # 训练批次大小
            accumulation_steps: int = 16,  # 梯度累积步数
            epochs: int = 3,  # 训练轮数
            learning_rate: float = 2e-4,  # 学习率
            out_dir: str = "out",  # 输出目录
            ####################################################
            # log相关配置
            ####################################################
            log_interval:int = 10,
            **kwargs
    ):
        ####################################################
        # 基本模型架构参数
        ####################################################
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.embeddings_epoch = embeddings_epoch
        ####################################################
        # DB相关配置
        ####################################################
        self.disable_db = disable_db  # 设置是否禁用数据库
        ####################################################
        # MOE相关配置
        # 当use_moe为false时，以下参数无效
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        ####################################################
        # 知识库相关配置
        ####################################################
        self.knowledge_num = knowledge_num
        self.knowledge_length = knowledge_length
        self.knowledge_dim = knowledge_dim
        ####################################################
        # EMA更新相关配置（灵感来自VQ-VAE）
        ####################################################
        self.use_ema_update = use_ema_update
        self.ema_decay = ema_decay
        self.ema_update_freq = ema_update_freq
        self.use_token_memory = use_token_memory  # 🔥 1.4.6: token-based memory flag
        self.freeze_ratio = freeze_ratio  # 🔥 新增: memory_bank冻结率
        ####################################################
        # 实验1.4.10: 优化的Gumbel-Softmax + 多样性损失
        ####################################################
        self.num_candidates = num_candidates
        self.num_selected = num_selected
        self.gumbel_temperature = gumbel_temperature
        ####################################################
        # 三元组提取相关配置
        ####################################################
        self.max_subject_len = max_subject_len
        self.max_predicate_len = max_predicate_len
        self.max_object_len = max_object_len
        ####################################################
        # SwanLab实验追踪相关配置
        ####################################################
        self.use_swanlab = use_swanlab
        self.swanlab_online = swanlab_online
        self.swanlab_project = swanlab_project
        ####################################################
        # 模型初始化相关配置
        ####################################################
        self.model_variant = model_variant
        self.pretrained_embedding_path = pretrained_embedding_path
        self.database_init_path = database_init_path
        self.cache_path = cache_path
        self.recompute_cache = recompute_cache
        ####################################################
        # 训练相关配置
        ####################################################
        self.dataset_path = dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.out_dir = out_dir
        ####################################################
        # Log相关配置
        ####################################################
        self.log_interval = log_interval
        super().__init__(**kwargs)
