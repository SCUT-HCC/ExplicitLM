"""
数据集处理模块

本模块用于将生成的三种题型数据进行UUID过滤、格式转换和多规模采样。
主要功能：
1. 从cluster_tokens_single_mapping.json加载合法UUID白名单（前20%样本）
2. 将判断题、宾语MCQ、谓词MCQ数据转换为对话格式
3. 支持UUID后缀编码（判断题yes/no、宾语MCQ、谓词MCQ各有标识）
4. 生成固定验证集和测试集（所有训练规模共享）
5. 支持多规模训练集采样（如100k、50k、10k）
6. 输出为JSONL格式的训练/验证/测试文件

数据流水线：
输入:
├─ judgment_trex_data.json (judgment_generator.py生成)
├─ omcq_trex_data.json (object_mcq_generator.py生成)
├─ pmcq_trex_data.json (predicate_mcq_generator.py生成)
└─ cluster_tokens_single_mapping.json (UUID白名单)

输出:
splits/
├─ judgement-{size}k/ (train.jsonl, valid.jsonl, test.jsonl)
├─ omcq-{size}k/ (train.jsonl, valid.jsonl, test.jsonl)
└─ pmcq-{size}k/ (train.jsonl, valid.jsonl, test.jsonl)

对话格式：
{
    "conversations": [
        {"role": "user", "content": "问题内容", "uuid": "xxx_a1"},
        {"role": "assistant", "content": "答案", "uuid": "xxx_a1"}
    ]
}

UUID后缀规则：
- 判断题yes答案: _a1
- 判断题no答案: _a4
- 宾语MCQ: _a2
- 谓词MCQ: _a3
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Set

from tqdm import tqdm


class DatasetProcessor:
    """
    数据集处理器

    负责加载原始题型数据，进行UUID过滤、格式转换、采样划分。
    支持三种数据类型：判断题(judgement)、宾语MCQ(omcq)、谓词MCQ(pmcq)。
    """

    # UUID后缀映射常量
    UUID_SUFFIX_YES: str = "_a1"
    UUID_SUFFIX_OMCQ: str = "_a2"
    UUID_SUFFIX_PMCQ: str = "_a3"
    UUID_SUFFIX_NO: str = "_a4"

    # 冻结比例常量（仅使用前20%样本）
    FREEZE_RATIO: float = 0.2

    def __init__(
        self,
        train_sizes: List[int],
        valid_size: int,
        test_size: int,
        data_dir: str,
        parent_folder: str = "splits",
        random_seed: int = 42
    ) -> None:
        """
        初始化数据集处理器

        参数:
            train_sizes: 训练集规模列表（如[100000, 50000, 10000]）
            valid_size: 验证集固定大小
            test_size: 测试集固定大小
            data_dir: 数据文件所在目录
            parent_folder: 输出目录名称
            random_seed: 随机种子（保证可复现）
        """
        self.data_dir = data_dir
        self.parent_folder = parent_folder
        self.train_sizes = sorted(train_sizes, reverse=True)
        self.valid_size = valid_size
        self.test_size = test_size
        self.random_seed = random_seed

        # 第一阶段：定义数据文件映射
        self.data_files: Dict[str, str] = {
            "judgement": "judgment_trex_data.json",
            "omcq": "omcq_trex_data.json",
            "pmcq": "pmcq_trex_data.json",
        }
        self.data_types: List[str] = list(self.data_files.keys())

        # 第二阶段：加载合法UUID白名单
        self.legal_uuids: Set[str] = self._load_legal_uuids()

        # 第三阶段：设置随机种子
        random.seed(self.random_seed)

    def _load_legal_uuids(self) -> Set[str]:
        """
        加载合法UUID集合

        从cluster_tokens_single_mapping.json中加载UUID白名单，
        仅保留前FREEZE_RATIO（20%）的样本。

        返回:
            合法UUID集合

        异常:
            RuntimeError: 映射文件不存在或格式错误
        """
        mapping_path: str = os.path.join(self.data_dir, "cluster_tokens_single_mapping.json")
        mapping_data: Optional[dict] = self._read_json_file(mapping_path)

        if mapping_data is None:
            raise RuntimeError(
                f"cluster_tokens_single_mapping.json 不存在或格式错误: {mapping_path}"
            )

        # 第一阶段：获取元数据
        total_entries: int = mapping_data["metadata"]["total_entries"]
        freeze_threshold: int = int(total_entries * self.FREEZE_RATIO)

        # 第二阶段：提取合法UUID（前20%）
        legal_uuids: Set[str] = {
            item["uuid"]
            for item in mapping_data["mappings"]
            if item["database_index"] < freeze_threshold
        }

        print(
            f"[INFO] 加载了 {len(legal_uuids):,} 个合法UUID "
            f"(来自前 {self.FREEZE_RATIO*100:.0f}% 条目)"
        )

        return legal_uuids

    @staticmethod
    def _read_json_file(file_path: str) -> Optional[dict]:
        """
        读取JSON文件

        参数:
            file_path: JSON文件路径

        返回:
            解析后的字典，失败时返回None
        """
        if not os.path.isfile(file_path):
            print(f"[WARN] 文件不存在: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] JSON格式错误或编码错误: {file_path} ({e})")
            return None

    @staticmethod
    def _write_jsonl_file(file_path: str, data: List[dict]) -> None:
        """
        写入JSONL文件

        每行写入一个JSON对象，适合流式处理大规模数据。

        参数:
            file_path: JSONL文件路径
            data: 待写入的字典列表
        """
        # 第一阶段：创建输出目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 第二阶段：逐行写入JSON
        with open(file_path, "w", encoding="utf-8") as f:
            for item in tqdm(
                data,
                desc=f"写入 {os.path.basename(file_path)}",
                colour="blue"
            ):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _convert_entry_to_conversation(self, entry: dict, data_type: str) -> dict:
        """
        将单个条目转换为对话格式

        根据数据类型将题目和答案转换为统一的对话格式，
        并添加对应的UUID后缀标识。

        参数:
            entry: 原始条目字典（包含question、answer等字段）
            data_type: 数据类型（judgement/omcq/pmcq）

        返回:
            对话格式字典

        异常:
            ValueError: 条目缺少必需字段
        """
        # 第一阶段：构建问题内容
        content: str = entry["question"]
        if data_type in ("omcq", "pmcq") and "options" in entry:
            content += f" {entry['options']}"

        # 第二阶段：提取答案
        answer: Optional[str] = entry.get("correct_answer") or entry.get("answer")
        if not answer:
            raise ValueError(
                f"条目缺少 'answer' 或 'correct_answer' 字段: {entry}"
            )

        # 第三阶段：构建对话结构
        conversation: dict = {
            "conversations": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": answer},
            ]
        }

        # 第四阶段：添加UUID后缀
        base_uuid: str = entry["uuid"]
        if data_type == "judgement":
            suffix = self.UUID_SUFFIX_YES if answer.lower() == "yes" else self.UUID_SUFFIX_NO
        elif data_type == "omcq":
            suffix = self.UUID_SUFFIX_OMCQ
        else:  # pmcq
            suffix = self.UUID_SUFFIX_PMCQ

        full_uuid: str = base_uuid + suffix
        conversation["conversations"][0]["uuid"] = full_uuid
        conversation["conversations"][1]["uuid"] = full_uuid

        return conversation

    def _process_single_data_type(self, data_type: str) -> None:
        """
        处理单个数据类型

        完整流程：
        1. 加载原始JSON数据
        2. UUID过滤和格式转换
        3. 去重（基于原始UUID）
        4. 划分测试集和验证集（固定）
        5. 为每个训练规模从训练池采样
        6. 写入JSONL文件

        参数:
            data_type: 数据类型（judgement/omcq/pmcq）
        """
        # 第一阶段：加载原始数据
        file_path: str = os.path.join(self.data_dir, self.data_files[data_type])
        raw_data: Optional[dict] = self._read_json_file(file_path)

        if raw_data is None:
            print(f"[SKIP] {data_type} 由于文件缺失或无效")
            return

        # 第二阶段：过滤、转换、去重
        seen_uuids: Set[str] = set()
        conversations: List[dict] = []

        for item in tqdm(
            raw_data,
            desc=f"过滤并转换 {data_type}",
            colour="green"
        ):
            for entry in item.get("target", []):
                try:
                    # 转换为对话格式
                    conv: dict = self._convert_entry_to_conversation(entry, data_type)
                except (KeyError, ValueError) as e:
                    print(f"[WARN] 跳过格式错误的条目 ({data_type}): {e}")
                    continue

                # UUID白名单过滤
                base_uuid: str = entry["uuid"]
                if base_uuid not in self.legal_uuids:
                    continue

                # 去重（基于原始UUID，不带后缀）
                if base_uuid in seen_uuids:
                    continue
                seen_uuids.add(base_uuid)

                conversations.append(conv)

        if not conversations:
            print(f"[{data_type}] 没有命中任何合法UUID，跳过")
            return

        print(f"[{data_type}] 去重并过滤后样本总数: {len(conversations)}")

        # 第三阶段：验证样本充足性
        total_needed: int = self.test_size + self.valid_size
        if len(conversations) < total_needed:
            raise ValueError(
                f"[{data_type}] 样本不足 ({len(conversations)}) "
                f"无法抽取 {self.test_size} 测试集 + {self.valid_size} 验证集"
            )

        # 第四阶段：一次性抽样测试集和验证集
        sampled_indices: Set[int] = set(
            random.sample(range(len(conversations)), total_needed)
        )
        sorted_indices: List[int] = sorted(list(sampled_indices))

        test_pool: List[dict] = [
            conversations[i] for i in sorted_indices[:self.test_size]
        ]
        valid_pool: List[dict] = [
            conversations[i] for i in sorted_indices[self.test_size:]
        ]

        # 第五阶段：构建训练池（剩余样本）
        train_pool: List[dict] = [
            conversations[i] for i in range(len(conversations))
            if i not in sampled_indices
        ]

        print(
            f"[{data_type}] 数据池 -> 训练: {len(train_pool)}, "
            f"验证: {len(valid_pool)}, 测试: {len(test_pool)}"
        )

        # 第六阶段：为每个训练规模采样并写入
        for train_size in self.train_sizes:
            if train_size == 0:
                continue

            # 从训练池有放回采样
            if not train_pool:
                train_data: List[dict] = []
            else:
                # 如果训练规模小于等于训练池，使用无放回采样
                if train_size <= len(train_pool):
                    train_data = random.sample(train_pool, k=train_size)
                else:
                    train_data = random.choices(train_pool, k=train_size)

            # 构建输出目录
            suffix: str = f"{train_size // 1000}k"
            output_dir: str = os.path.join(
                self.data_dir, self.parent_folder, f"{data_type}-{suffix}"
            )

            print(f"写入 {data_type}-{suffix} 数据到 {output_dir}")

            # 写入三个JSONL文件
            self._write_jsonl_file(
                os.path.join(output_dir, "train.jsonl"), train_data
            )
            self._write_jsonl_file(
                os.path.join(output_dir, "valid.jsonl"), valid_pool
            )
            self._write_jsonl_file(
                os.path.join(output_dir, "test.jsonl"), test_pool
            )

            print(
                f"[{data_type}-{suffix}] 完成 -> "
                f"训练 {len(train_data)} | 验证 {len(valid_pool)} | 测试 {len(test_pool)}"
            )

    def run(self) -> None:
        """
        运行数据集处理流程

        验证所有必需文件存在后，依次处理三种数据类型。
        """
        # 第一阶段：检查必需文件
        missing_files: List[str] = [
            f for f in self.data_files.values()
            if not os.path.exists(os.path.join(self.data_dir, f))
        ]

        if missing_files:
            print("错误: 缺少必需文件:")
            for missing in missing_files:
                print(f"   - {missing}")
            return

        # 第二阶段：处理每种数据类型
        for data_type in self.data_types:
            self._process_single_data_type(data_type)

        print("所有数据集处理完成")


def main() -> None:
    """
    主函数：解析命令行参数并执行数据集处理

    支持的参数：
    --train-sizes: 训练集规模列表（如 100000 50000 10000）
    --valid: 验证集大小（固定）
    --test: 测试集大小（固定）
    --data-dir: 数据文件目录
    --parent-folder: 输出目录名称
    --seed: 随机种子
    """
    parser = argparse.ArgumentParser(
        description="处理数据集：UUID过滤、格式转换、多规模采样"
    )

    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs='+',
        default=[100_000, 50_000, 10_000],
        help="训练集规模列表（例如: --train-sizes 100000 50000 10000）"
    )
    parser.add_argument(
        "--valid",
        type=int,
        default=100,
        help="验证集大小（对所有训练规模固定）"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=20_000,
        help="测试集大小（对所有训练规模固定）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="包含JSON文件和映射文件的目录"
    )
    parser.add_argument(
        "--parent-folder",
        type=str,
        default="splits",
        help="输出目录名称（默认: 'splits'）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保证可复现，默认: 42）"
    )

    args = argparse.parse_args()

    # 创建处理器并运行
    processor = DatasetProcessor(
        train_sizes=args.train_sizes,
        valid_size=args.valid,
        test_size=args.test,
        data_dir=args.data_dir,
        parent_folder=args.parent_folder,
        random_seed=args.seed
    )
    processor.run()


if __name__ == "__main__":
    main()
