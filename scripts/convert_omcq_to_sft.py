#!/usr/bin/env python3
"""
将 OMCQ 数据转换为 SFT 对话格式

输入格式：
{
  "target": [
    {
      "question": "...",
      "options": "A:...,B:...,C:...",
      "correct_answer": "A:...",
      "uuid": "..."
    }
  ]
}

输出格式（JSONL）：
{
  "conversations": [
    {"role": "user", "content": "问题\n选项\n请选择正确答案。"},
    {"role": "assistant", "content": "正确答案"}
  ]
}
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_omcq_to_conversation(omcq_item: dict) -> dict:
    """
    将单个 OMCQ 样本转换为对话格式
    
    Args:
        omcq_item: OMCQ 格式的数据项
        
    Returns:
        对话格式的数据项，如果转换失败返回 None
    """
    try:
        # 提取 target 列表
        targets = omcq_item.get("target", [])
        if not targets or len(targets) == 0:
            return None
        
        # 取第一个 target（通常只有一个）
        target = targets[0]
        
        question = target.get("question", "").strip()
        options = target.get("options", "").strip()
        correct_answer = target.get("correct_answer", "").strip()
        
        # 验证必要字段
        if not question or not options or not correct_answer:
            return None
        
        # 构建用户输入：问题 + 选项 + 提示
        user_content = f"{question}\n{options}\n请选择正确答案。"
        
        # 构建助手回复：正确答案
        assistant_content = correct_answer
        
        # 构建对话格式
        conversation_item = {
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
        return conversation_item
        
    except Exception as e:
        print(f"转换失败: {e}")
        return None


def convert_file(input_path: str, output_path: str, max_samples: int = None):
    """
    转换整个文件
    
    Args:
        input_path: 输入的 OMCQ JSON 文件路径
        output_path: 输出的 JSONL 文件路径
        max_samples: 最大转换样本数（None 表示全部）
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 读取输入文件: {input_path}")
    print(f"💾 输出文件: {output_path}")
    
    # 读取输入 JSON 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 总样本数: {len(data)}")
    
    # 转换数据
    converted_count = 0
    failed_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="转换中"):
            if max_samples and converted_count >= max_samples:
                break
                
            converted_item = convert_omcq_to_conversation(item)
            
            if converted_item is not None:
                f_out.write(json.dumps(converted_item, ensure_ascii=False) + '\n')
                converted_count += 1
            else:
                failed_count += 1
    
    print(f"\n✅ 转换完成!")
    print(f"  - 成功转换: {converted_count} 条")
    print(f"  - 失败/跳过: {failed_count} 条")
    print(f"  - 输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将 OMCQ 数据转换为 SFT 对话格式")
    parser.add_argument(
        "--input",
        type=str,
        default="sft_data/omcq_trex_data.json",
        help="输入的 OMCQ JSON 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sft_data/omcq_trex_sft.jsonl",
        help="输出的 JSONL 文件路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大转换样本数（用于测试，None 表示全部）"
    )
    
    args = parser.parse_args()
    
    convert_file(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    main()

