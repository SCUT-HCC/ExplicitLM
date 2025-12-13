"""
ExplicitLM 快速开始示例

展示如何使用cache知识库进行模型初始化和文本生成

使用前请配置：
1. QWEN3_MODEL_PATH: Qwen3-4B 模型路径
2. LLMLINGUA_MODEL_PATH: LLMLingua-2-BERT 模型路径（如果启用事实提取）
"""
import sys
import torch
sys.path.insert(0, '.')

from utils.model_initializer import init_model
from utils.dual_path_inference import DualPathInference
# from utils.fact_extractor import FactExtractor  # 如果启用事实提取，取消注释


def main():
    print('='*60)
    print('ExplicitLM 快速开始示例')
    print('='*60)
    
    QWEN3_MODEL_PATH = '/path/to/Qwen3-4b'  # 请替换为实际的Qwen3模型路径
    LLMLINGUA_MODEL_PATH = '/path/to/llmlingua-2-bert'  # 请替换为实际的LLMLingua模型路径
    
    args = {
        'qwen3_model_path': QWEN3_MODEL_PATH,
        'knowledge_num': 1024 * 1024,  # 1048576 个记忆条目
        'knowledge_length': 16,        # 每个条目16个token
        'knowledge_dim': 128,          # 记忆嵌入维度
        'use_moe': False,
        'num_candidates': 8,
        'num_selected': 1,
        # 使用预处理的cache知识库
        'cache_path': 'data/cache/knowledge_cache.pt',
        'recompute_cache': False,
    }
    
    print('\n📦 初始化模型...')
    model, tokenizer = init_model(args, accelerator=None)
    model.eval()
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print('模型初始化完成')
    print(f'  - Memory bank形状: {model.memory_bank.shape if hasattr(model, "memory_bank") else "N/A"}')
    
    print('\n初始化双路推理包装器...')
    
    # 如果需要启用事实提取，需要初始化FactExtractor
    # from utils.fact_extractor import FactExtractor
    # fact_extractor = FactExtractor(
    #     model_path=LLMLINGUA_MODEL_PATH,
    #     compression_rate=0.4,
    # )
    
    dual_path = DualPathInference(
        model=model,
        tokenizer=tokenizer,
        # fact_extractor=fact_extractor,  # 如果启用事实提取，取消注释
        enable_fact_extraction=False,  # 先禁用事实提取，只测试生成
        fact_update_frequency=1,
        update_strategy='fifo',
    )
    print('双路推理初始化完成')
    
    test_cases = [
        "什么是人工智能？",
        "请介绍一下机器学习的基本概念。",
    ]
    
    print('\n' + '='*60)
    print('开始生成测试')
    print('='*60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f'\n--- 测试 {i}/{len(test_cases)} ---')
        print(f'输入: {test_text}')
        
        try:
            input_ids = tokenizer.encode(test_text, return_tensors='pt').to(model.device)
            
            with torch.no_grad():
                result = dual_path.generate(
                    input_ids,
                    input_text=test_text,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                )
            
            print(f'\n生成结果:')
            print(result['generated_text'])
            
        except Exception as e:
            print(f'错误: 生成失败: {e}')
            import traceback
            traceback.print_exc()
    
    print('\n' + '='*60)
    print('测试完成')
    print('='*60)


if __name__ == '__main__':
    main()

