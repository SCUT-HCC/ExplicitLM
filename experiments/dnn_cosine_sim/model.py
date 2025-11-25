
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from models.memory_bank.MemoryGate import MemoryGate

class QwenMemoryRouter(nn.Module):
    def __init__(self, model_name_or_path: str, memory_gate_cfg: dict, freeze_backbone: bool = True):
        super().__init__()
        print(f"Loading backbone: {model_name_or_path}")
        self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        print("Initializing MemoryGate...")
        self.head = MemoryGate(memory_gate_cfg)
        
    def forward(self, input_ids, attention_mask=None, target_indices=None, target_scores=None, temperature=0.1):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            target_indices: [batch, k] (Optional, for training)
            target_scores: [batch, k] (Optional, for soft label training)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state # [batch, seq, dim]
        
        bsz, seq_len, dim = last_hidden_state.shape
        
        # Select last token representation
        if attention_mask is not None:
            # Find the index of the last non-padding token
            last_token_indices = attention_mask.sum(dim=1) - 1
            # Clamp to ensure no negative indices (though shouldn't happen with valid mask)
            last_token_indices = last_token_indices.clamp(min=0)
            last_token_state = last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), last_token_indices]
        else:
            last_token_state = last_hidden_state[:, -1, :]
            
        # Reshape to [batch, 1, dim] for MemoryGate
        last_token_state = last_token_state.unsqueeze(1)
        
        # Compute scores once
        scores_1, scores_2 = self.head.compute_sub_scores(last_token_state)
        
        if target_indices is not None:
            # Training mode
            # target_indices: [batch, k] -> [batch, 1, k]
            target_indices_reshaped = target_indices.unsqueeze(1)
            
            if target_scores is not None:
                # Soft Label Training
                target_scores_reshaped = target_scores.unsqueeze(1)
                loss = self.head.compute_loss_soft(scores_1, scores_2, target_indices_reshaped, target_scores_reshaped, temperature)
            else:
                # Hard Label Training (Legacy)
                loss = self.head.compute_loss(scores_1, scores_2, target_indices_reshaped)
            
            # Also compute predictions for accuracy metrics
            indices, scores = self.head.generate_candidates(scores_1, scores_2)
            
            return loss, indices
        else:
            # Inference mode
            indices, scores = self.head.generate_candidates(scores_1, scores_2)
            return indices, scores
