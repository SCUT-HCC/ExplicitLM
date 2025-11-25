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

from experiments.dnn_cosine_sim.model import QwenMemoryRouter

class RouterDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        target_indices = item['target_indices'] # List of ints
        
        # Tokenize query
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
            "target_scores": torch.tensor(item['target_scores'], dtype=torch.float)
        }

def calculate_metrics(pred_indices, target_indices, target_scores=None):
    """
    Calculate Recall@K, Hit Rate, and Soft Recall
    pred_indices: [batch, k_pred]
    target_indices: [batch, k_target]
    target_scores: [batch, k_target] (Optional, for Soft Recall)
    
    Returns:
        dict with raw sums and counts for accumulation
    """
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
            # Calculate Soft Recall
            # Sum of scores of retrieved targets / Sum of all target scores
            total_target_score = target_scores_cpu[i].sum()
            
            # Find scores of retrieved targets
            retrieved_score = 0
            # We need to map target index to its score. 
            # Since target_indices and target_scores are aligned:
            t_indices = target_indices_cpu[i]
            t_scores = target_scores_cpu[i]
            
            # Create a lookup for this sample
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
    parser.add_argument("--data_path", type=str, default="experiments/dnn_cosine_sim/train_with_labels.jsonl")
    parser.add_argument("--model_name", type=str, default="/home/pci/ycz/Code/ExplicitLM/Qwen")
    parser.add_argument("--output_dir", type=str, default="experiments/dnn_cosine_sim/checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--knowledge_num", type=int, default=1024*1024, help="Total number of items in knowledge base")
    parser.add_argument("--knowledge_dim", type=int, default=2048)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for soft label loss")
    parser.add_argument("--swanlab_project", type=str, default="dnn-cosine-sim")
    
    args = parser.parse_args()
    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        swanlab.init(project=args.swanlab_project, config=vars(args))
    
    # Calculate perfect square for knowledge_num
    # MemoryGate requires knowledge_num to be a perfect square
    sqrt_num = math.ceil(math.sqrt(args.knowledge_num))
    padded_knowledge_num = sqrt_num ** 2
    if padded_knowledge_num != args.knowledge_num:
        accelerator.print(f"Padding knowledge_num from {args.knowledge_num} to {padded_knowledge_num} (square of {sqrt_num})")
    
    # Load Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Dataset
    dataset = RouterDataset(args.data_path, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Load Model Config to get dim
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    hidden_size = config.hidden_size
    
    memory_gate_cfg = {
        "dim": hidden_size,
        "knowledge_num": padded_knowledge_num,
        "knowledge_dim": args.knowledge_dim,
        "num_candidates": args.num_candidates,
        "dropout": 0.1
    }
    
    model = QwenMemoryRouter(args.model_name, memory_gate_cfg, freeze_backbone=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    num_training_steps = args.epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )
    
    # Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Train
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        
        # Epoch metric accumulators
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
                # Forward pass returns loss and predicted indices
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
                
                # Accumulate metrics
                # pred_indices: [batch, 1, k_pred] -> [batch, k_pred]
                batch_metrics = calculate_metrics(
                    pred_indices.squeeze(1), 
                    batch["target_indices"],
                    batch["target_scores"]
                )
                
                # Sync metrics across devices for accurate logging if distributed
                # For simplicity, we'll compute local metrics and average them if needed, 
                # but standard practice is to just log main process or all-reduce. 
                # Here we stick to local accumulation and assume dataloader is sharded properly.
                
                epoch_loss_sum += loss.item()
                epoch_recall_sum += batch_metrics["recall_sum"]
                epoch_soft_recall_sum += batch_metrics["soft_recall_sum"]
                epoch_hit_sum += batch_metrics["hit_sum"]
                epoch_correct_count_sum += batch_metrics["correct_count_sum"]
                epoch_total_samples += batch_metrics["num_samples"]
                epoch_total_candidates += batch_metrics["total_candidates"]
                steps_in_epoch += 1
                
                # Log running average metrics
                if global_step % 10 == 0:
                     if accelerator.is_main_process:
                         # Calculate running averages
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
                         
                         print(f"Step {global_step}: Loss {loss.item():.4f} | Avg Loss {running_avg_loss:.4f} | Avg Recall {running_avg_recall:.4f} | Avg SoftRecall {running_avg_soft_recall:.4f} | Avg HitRate {running_avg_hit_rate:.4f}")
                
                global_step += 1
        
        # Calculate and log epoch metrics
        avg_loss = epoch_loss_sum / len(dataloader) # Approximation since batch size varies, but close enough
        avg_recall = epoch_recall_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_soft_recall = epoch_soft_recall_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_hit_rate = epoch_hit_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_correct = epoch_correct_count_sum / epoch_total_samples if epoch_total_samples > 0 else 0
        precision = epoch_correct_count_sum / epoch_total_candidates if epoch_total_candidates > 0 else 0
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} finished.")
            print(f"Avg Loss: {avg_loss:.4f}")
            print(f"Recall: {avg_recall:.4f}")
            print(f"Soft Recall: {avg_soft_recall:.4f}")
            print(f"Hit Rate: {avg_hit_rate:.4f}")
            print(f"Avg Correct: {avg_correct:.4f}")
            print(f"Precision: {precision:.4f}")
            
            swanlab.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_recall": avg_recall,
                "train/epoch_soft_recall": avg_soft_recall,
                "train/epoch_hit_rate": avg_hit_rate,
                "train/epoch_avg_correct": avg_correct,
                "train/epoch_precision": precision
            }, step=global_step)
        
        # Save checkpoint
        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.save_state(save_path)
            print(f"Saved checkpoint to {save_path}")

    accelerator.end_training()
    if accelerator.is_main_process:
        swanlab.finish()

if __name__ == "__main__":
    main()
