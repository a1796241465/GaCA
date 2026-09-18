import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, values, mask):
        scores = self.scorer(values).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)

def masked_mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, values):
        return self.layers(values)

class GaCA(nn.Module):
    def __init__(self, num_classes, hidden_dim, dropout):
        super().__init__()
        self.seq_pool = AttentionPooling(1280)
        self.str_pool = AttentionPooling(512)
        self.seq_proj = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.str_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = ClassificationHead(hidden_dim, num_classes, dropout)

    def forward_with_gate(self, sequence, structure, seq_mask, str_mask):
        seq_hidden = self.seq_proj(self.seq_pool(sequence, seq_mask))
        str_hidden = self.str_proj(self.str_pool(structure, str_mask))
        gate = self.gate(torch.cat([seq_hidden, str_hidden], dim=1))
        fused = gate * str_hidden + (1.0 - gate) * seq_hidden
        return self.classifier(fused), gate

    def forward(self, sequence, structure, seq_mask, str_mask):
        logits, _ = self.forward_with_gate(sequence, structure, seq_mask, str_mask)
        return logits

