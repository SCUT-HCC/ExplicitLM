# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "tqdm",
#     "swanlab",
#     "numpy",
# ]
# ///

import os
import argparse
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import swanlab

# 设置进程名称（在 nvidia-smi 中显示的名称）
try:
    import setproctitle
    process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
    setproctitle.setproctitle(process_name)
except ImportError:
    # 如果没有 setproctitle，尝试使用 prctl (Linux only)
    try:
        import prctl
        process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
        prctl.set_name(process_name.encode('utf-8'))
    except (ImportError, AttributeError):
        # 如果都不可用，跳过（不影响训练）
        pass

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时忽略

from experiments.dnn_similarity.model import QwenMemoryRouter

class RouterDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if len(self.data) > 0:
            sample = self.data[0]
            if 'query' not in sample or 'target_indices' not in sample:
                raise ValueError(f"Expected 'query' format with 'target_indices' field, got keys: {sample.keys()}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        target_indices = item['target_indices']
        target_scores = item.get('target_scores', [1.0] * len(target_indices))
        
        encodings = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids.squeeze(0),
            "attention_mask": encodings.attention_mask.squeeze(0),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "target_scores": torch.tensor(target_scores, dtype=torch.float)
        }

def calculate_metrics(pred_indices, target_indices, target_scores=None):
    batch_size = pred_indices.size(0)
    recall_sum = 0
    soft_recall_sum = 0
    hit_sum = 0
    correct_count_sum = 0
    
    pred_indices_cpu = pred_indices.detach().cpu().numpy()
    target_indices_cpu = target_indices.detach().cpu().numpy()
    
    if target_scores is not None:
        target_scores_cpu = target_scores.detach().cpu().numpy()
    
    for i in range(batch_size):
        targets = set(target_indices_cpu[i])
        preds = set(pred_indices_cpu[i])
        
        intersection = targets.intersection(preds)
        num_correct = len(intersection)
        
        correct_count_sum += num_correct
        recall_sum += num_correct / len(targets) if len(targets) > 0 else 0
        hit_sum += 1 if num_correct > 0 else 0
        
        if target_scores is not None:
            total_target_score = target_scores_cpu[i].sum()
            retrieved_score = 0
            t_indices = target_indices_cpu[i]
            t_scores = target_scores_cpu[i]
            target_score_map = {idx: score for idx, score in zip(t_indices, t_scores)}
            
            for pred_idx in preds:
                if pred_idx in target_score_map:
                    retrieved_score += target_score_map[pred_idx]
            
            soft_recall_sum += retrieved_score / total_target_score if total_target_score > 0 else 0

    return {
        "recall_sum": recall_sum,
        "soft_recall_sum": soft_recall_sum,
        "hit_sum": hit_sum,
        "correct_count_sum": correct_count_sum,
        "num_samples": batch_size,
        "total_candidates": batch_size * pred_indices.size(1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train_labeled.jsonl", help="训练数据路径（query格式，需包含target_indices和target_scores）")
    parser.add_argument("--model_name", type=str, default="/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b", help="Qwen3 模型路径")
    parser.add_argument("--output_dir", type=str, default="checkpoints/router", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (默认 4，可根据 GPU 内存调整)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--knowledge_num", type=int, default=65536, help="Total number of items in knowledge base（需为完全平方数）")
    parser.add_argument("--knowledge_dim", type=int, default=2048)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for soft label loss")
    parser.add_argument("--swanlab_project", type=str, default="explicitlm-router")
    parser.add_argument("--swanlab_api_key", type=str, default='dKWI69mdEndjB2P9YdY8f', help="SwanLab API Key（也可通过环境变量 SWANLAB_API_KEY 设置）")
    parser.add_argument("--gpu_id", type=str, default="5,6")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.swanlab_api_key:
        os.environ["SWANLAB_API_KEY"] = args.swanlab_api_key
    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        swanlab.init(project=args.swanlab_project, config=vars(args))
    
    meta_path = os.path.join(os.path.dirname(args.data_path), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            knowledge_num = meta.get("knowledge_num", args.knowledge_num)
            knowledge_dim = meta.get("embedding_dim", args.knowledge_dim)
            keys_path = meta.get("keys_path", None)
    else:
        knowledge_num = args.knowledge_num
        knowledge_dim = args.knowledge_dim
        keys_path = None

    sqrt_num = math.ceil(math.sqrt(knowledge_num))
    padded_knowledge_num = sqrt_num ** 2
    if padded_knowledge_num != knowledge_num:
        accelerator.print(f"Padding knowledge_num from {knowledge_num} to {padded_knowledge_num}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = RouterDataset(
        args.data_path, 
        tokenizer, 
        args.max_length
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    accelerator.print(f"Dataset loaded: {len(dataset)} samples")
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    hidden_size = config.hidden_size
    
    memory_gate_cfg = {
        "dim": hidden_size,
        "knowledge_num": padded_knowledge_num,
        "knowledge_dim": knowledge_dim,
        "num_candidates": args.num_candidates,
        "dropout": 0.1,
        "keys_path": keys_path
    }
    
    model = QwenMemoryRouter(args.model_name, memory_gate_cfg, freeze_backbone=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        
        epoch_loss_sum = 0.0
        epoch_recall_sum = 0.0
        epoch_soft_recall_sum = 0.0
        epoch_hit_sum = 0.0
        epoch_correct_count_sum = 0.0
        epoch_total_samples = 0
        epoch_total_candidates = 0
        steps_in_epoch = 0
        
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                loss, pred_indices = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_indices=batch["target_indices"],
                    target_scores=batch["target_scores"],
                    temperature=args.temperature
                )
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                batch_metrics = calculate_metrics(
                    pred_indices.squeeze(1), 
                    batch["target_indices"],
                    batch["target_scores"]
                )
                
                epoch_loss_sum += loss.item()
                epoch_recall_sum += batch_metrics["recall_sum"]
                epoch_soft_recall_sum += batch_metrics["soft_recall_sum"]
                epoch_hit_sum += batch_metrics["hit_sum"]
                epoch_correct_count_sum += batch_metrics["correct_count_sum"]
                epoch_total_samples += batch_metrics["num_samples"]
                epoch_total_candidates += batch_metrics["total_candidates"]
                steps_in_epoch += 1
                
                if global_step % 10 == 0 and accelerator.is_main_process:
                    running_avg_loss = epoch_loss_sum / steps_in_epoch
                    running_avg_recall = epoch_recall_sum / epoch_total_samples if epoch_total_samples > 0 else 0
                    running_avg_soft_recall = epoch_soft_recall_sum / epoch_total_samples if epoch_total_samples > 0 else 0
                    running_avg_hit_rate = epoch_hit_sum / epoch_total_samples if epoch_total_samples > 0 else 0
                    
                    swanlab.log({
                        "train/step_loss": loss.item(),
                        "train/running_avg_loss": running_avg_loss,
                        "train/running_avg_recall": running_avg_recall,
                        "train/running_avg_soft_recall": running_avg_soft_recall,
                        "train/running_avg_hit_rate": running_avg_hit_rate,
                        "lr": scheduler.get_last_lr()[0]
                    }, step=global_step)
                
                global_step += 1
        
        avg_loss = epoch_loss_sum / len(dataloader)
        avg_recall = epoch_recall_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_soft_recall = epoch_soft_recall_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_hit_rate = epoch_hit_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_correct = epoch_correct_count_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        precision = epoch_correct_count_sum / epoch_total_candidates if epoch_total_candidates > 0 else 0
        
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Recall={avg_recall:.4f}, HitRate={avg_hit_rate:.4f}")
            
            swanlab.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_recall": avg_recall,
                "train/epoch_soft_recall": avg_soft_recall,
                "train/epoch_hit_rate": avg_hit_rate,
                "train/epoch_avg_correct": avg_correct,
                "train/epoch_precision": precision
            }, step=global_step)
            
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.save_state(save_path)
            
            unwrapped_model = accelerator.unwrap_model(model)
            memory_gate_state = {}
            for k, v in unwrapped_model.head.state_dict().items():
                memory_gate_state[k] = v.cpu()
            
            memory_gate_path = os.path.join(args.output_dir, f"memory_gate_epoch_{epoch}.pth")
            torch.save(memory_gate_state, memory_gate_path)

    accelerator.end_training()
    if accelerator.is_main_process:
        swanlab.finish()

if __name__ == "__main__":
    main()