"""
预训练数据集模块

功能：
- PretrainDataset: 基础预训练数据集类
- create_pretrain_dataloader: 数据加载器工厂函数
- 支持数据过滤、验证和批处理
"""

import json
import os
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# 启用tokenizer并行化
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _format_text_for_qwen_tokenizer(tokenizer: Any, text: str) -> str:
    """
    根据Qwen tokenizer的特殊token格式化文本。
    Qwen通常使用 <|im_start|>user\n{text}<|im_end|> 格式。
    """
    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"

    # 检查tokenizer是否包含这些特殊token
    if im_start_token in tokenizer.get_added_tokens_decoder().values() and \
       im_end_token in tokenizer.get_added_tokens_decoder().values():
        # 使用Qwen的对话格式
        formatted_text = f"{im_start_token}user\n{text}{im_end_token}"
    else:
        # 回退到标准的bos_token/eos_token（如果存在）
        if tokenizer.bos_token is not None and not text.startswith(tokenizer.bos_token):
            text = f"{tokenizer.bos_token}{text}"
        if tokenizer.eos_token is not None and not text.endswith(tokenizer.eos_token):
            text = f"{text}{tokenizer.eos_token}"
        formatted_text = text
    return formatted_text


class PretrainDataset(Dataset):
    """
    预训练数据集类

    功能：
    - 从JSONL文件加载文本数据
    - 自动添加特殊token（bos_token/eos_token）
    - 生成input_ids和loss_mask用于训练

    数据格式：
    输入：JSONL文件，每行为一个JSON对象，包含'text'字段
    输出：(X, Y, loss_mask)元组
        - X: 输入序列 (input_ids[:-1])
        - Y: 目标序列 (input_ids[1:])
        - loss_mask: 损失计算掩码 (排除padding位置)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512
    ):
        """
        初始化预训练数据集

        Args:
            data_path: JSONL数据文件路径
            tokenizer: Tokenizer实例，需包含bos_token、eos_token、pad_token_id
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据

        Args:
            path: JSONL文件路径

        Returns:
            样本列表，每个样本为包含'text'字段的字典
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                data = json.loads(line)
                samples.append(data)
        return samples

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            index: 样本索引

        Returns:
            (X, Y, loss_mask)元组
            - X: 输入序列张量 [max_length-1]
            - Y: 目标序列张量 [max_length-1]
            - loss_mask: 损失掩码张量 [max_length-1]
        """
        sample = self.samples[index]
        text = str(sample['text'])

        # 使用Qwen tokenizer的特殊格式（如果适用）
        text = _format_text_for_qwen_tokenizer(self.tokenizer, text)

        # Tokenization with padding and truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 提取input_ids并生成loss_mask
        input_ids = encoding.input_ids.squeeze()  # [max_length]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # [max_length]

        # 生成训练对：X = input_ids[:-1], Y = input_ids[1:]
        X = input_ids[:-1].clone()  # [max_length-1]
        Y = input_ids[1:].clone()   # [max_length-1]
        loss_mask = loss_mask[1:]   # [max_length-1]

        return X, Y, loss_mask


def create_pretrain_dataloader(
    data_path: str,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建预训练数据加载器的工厂函数

    Args:
        data_path: JSONL数据文件路径
        tokenizer: Tokenizer实例
        batch_size: 批次大小
        max_length: 最大序列长度，默认512
        shuffle: 是否打乱数据，默认True
        num_workers: 数据加载进程数，默认4
        pin_memory: 是否使用pin_memory加速GPU传输，默认True

    Returns:
        DataLoader实例

    使用示例：
        ```python
        from utils.pretrain_datasets import create_pretrain_dataloader

        train_loader = create_pretrain_dataloader(
            data_path='data/train.jsonl',
            tokenizer=tokenizer,
            batch_size=32,
            max_length=512
        )

        for batch_idx, (X, Y, loss_mask) in enumerate(train_loader):
            # X, Y, loss_mask: [batch_size, max_length-1]
            ...
        ```
    """
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后一个不完整的batch，保证batch_size一致
    )

    return dataloader


def validate_dataset(
    data_path: str,
    tokenizer: Any,
    max_samples: int = 10
) -> Dict[str, Any]:
    """
    验证数据集完整性和统计信息

    Args:
        data_path: JSONL数据文件路径
        tokenizer: Tokenizer实例
        max_samples: 打印的最大样本数，默认10

    Returns:
        包含统计信息的字典：
        - total_samples: 总样本数
        - avg_text_length: 平均文本长度
        - sample_examples: 示例样本列表
    """
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512
    )

    total_samples = len(dataset)
    text_lengths = []
    sample_examples = []

    for i in range(min(max_samples, total_samples)):
        X, Y, loss_mask = dataset[i]
        sample_info = {
            'index': i,
            'X_shape': X.shape,
            'Y_shape': Y.shape,
            'loss_mask_shape': loss_mask.shape,
            'num_valid_tokens': loss_mask.sum().item(),
            'text_preview': dataset.samples[i]['text'][:100]  # 前100个字符
        }
        sample_examples.append(sample_info)
        text_lengths.append(len(dataset.samples[i]['text']))

    stats = {
        'total_samples': total_samples,
        'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'sample_examples': sample_examples
    }

    return stats


class ValidationDataset(Dataset):
    """
    验证数据集类

    功能：
    - 从JSON/JSONL文件加载验证数据
    - 支持限制样本数量以加快验证
    - 兼容PretrainDataset的数据格式

    数据格式：
    输入：JSON/JSONL文件，每行包含'text'字段
    输出：与PretrainDataset相同的(X, Y, loss_mask)元组
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512,
        num_samples: int = 200
    ):
        """
        初始化验证数据集

        Args:
            data_path: JSON/JSONL验证数据文件路径
            tokenizer: Tokenizer实例
            max_length: 最大序列长度
            num_samples: 验证样本数量限制，用于加快验证速度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.samples = self._load_validation_data(data_path)

    def _load_validation_data(self, path: str) -> List[Dict[str, Any]]:
        """
        从JSON/JSONL文件加载验证数据

        Args:
            path: 验证数据文件路径

        Returns:
            样本列表
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"验证数据文件不存在: {path}")

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.num_samples:  # 限制验证样本数量
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    # 提取text字段
                    if 'text' in data:
                        samples.append({'text': data['text']})
                except json.JSONDecodeError:
                    continue

        return samples

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本，与PretrainDataset保持相同的接口

        Args:
            index: 样本索引

        Returns:
            (X, Y, loss_mask)元组
            - X: 输入序列张量 [max_length-1]
            - Y: 目标序列张量 [max_length-1]
            - loss_mask: 损失掩码张量 [max_length-1]
        """
        sample = self.samples[index]
        text = str(sample['text'])

        # 使用Qwen tokenizer的特殊格式（如果适用）
        text = _format_text_for_qwen_tokenizer(self.tokenizer, text)

        # Tokenization with padding and truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 提取input_ids并生成loss_mask
        input_ids = encoding.input_ids.squeeze()  # [max_length]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # [max_length]

        # 生成训练对：X = input_ids[:-1], Y = input_ids[1:]
        X = input_ids[:-1].clone()  # [max_length-1]
        Y = input_ids[1:].clone()   # [max_length-1]
        loss_mask = loss_mask[1:]   # [max_length-1]

        return X, Y, loss_mask


def create_validation_dataloader(
    val_data_path: str,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 512,
    num_samples: int = 200,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建验证数据加载器的工厂函数

    Args:
        val_data_path: 验证数据文件路径
        tokenizer: Tokenizer实例
        batch_size: 批次大小
        max_length: 最大序列长度，默认512
        num_samples: 验证样本数量，默认200
        num_workers: 数据加载进程数，默认4
        pin_memory: 是否使用pin_memory加速GPU传输，默认True

    Returns:
        DataLoader实例，如果文件不存在返回None

    使用示例：
        ```python
        from utils.pretrain_datasets import create_validation_dataloader

        val_loader = create_validation_dataloader(
            val_data_path='data/benchmarks/eval_data.json',
            tokenizer=tokenizer,
            batch_size=32,
            max_length=512,
            num_samples=200
        )

        if val_loader is not None:
            for batch_idx, (X, Y, loss_mask) in enumerate(val_loader):
                # X, Y, loss_mask: [batch_size, max_length-1]
                ...
        ```
    """
    if not os.path.exists(val_data_path):
        return None

    dataset = ValidationDataset(
        data_path=val_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # 验证集保留所有样本
    )

    return dataloader
