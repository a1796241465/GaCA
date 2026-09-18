import torch
import torch.nn as nn
from gaca.model import AttentionPooling, ClassificationHead, masked_mean

class SingleModality(nn.Module):
    def __init__(self, num_classes, hidden_dim, dropout, modality):
        super().__init__()
        self.modality = modality
        input_dim = 1280 if modality == "sequence" else 512
        self.pool = AttentionPooling(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.classifier = ClassificationHead(hidden_dim, num_classes, dropout)

    def forward(self, sequence, structure, seq_mask, str_mask):
        if self.modality == "sequence":
            hidden = self.projection(self.pool(sequence, seq_mask))
        else:
            hidden = self.projection(self.pool(structure, str_mask))
        return self.classifier(hidden)

class MeanVectorGate(nn.Module):
    def __init__(self, num_classes, hidden_dim, dropout):
        super().__init__()
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

    def forward(self, sequence, structure, seq_mask, str_mask):
        seq_hidden = self.seq_proj(masked_mean(sequence, seq_mask))
        str_hidden = self.str_proj(masked_mean(structure, str_mask))
        gate = self.gate(torch.cat([seq_hidden, str_hidden], dim=1))
        return self.classifier(gate * str_hidden + (1.0 - gate) * seq_hidden)

class AttentionConcat(nn.Module):
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
        self.classifier = ClassificationHead(hidden_dim * 2, num_classes, dropout)

    def forward(self, sequence, structure, seq_mask, str_mask):
        seq_hidden = self.seq_proj(self.seq_pool(sequence, seq_mask))
        str_hidden = self.str_proj(self.str_pool(structure, str_mask))
        return self.classifier(torch.cat([seq_hidden, str_hidden], dim=1))

class CrossAttention(nn.Module):
    def __init__(self, num_classes, hidden_dim, dropout, heads):
        super().__init__()
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
        self.seq_to_str = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.str_to_seq = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.seq_norm = nn.LayerNorm(hidden_dim)
        self.str_norm = nn.LayerNorm(hidden_dim)
        self.seq_pool = AttentionPooling(hidden_dim)
        self.str_pool = AttentionPooling(hidden_dim)
        self.classifier = ClassificationHead(hidden_dim * 2, num_classes, dropout)

    def forward(self, sequence, structure, seq_mask, str_mask):
        sequence = self.seq_proj(sequence)
        structure = self.str_proj(structure)
        seq_update, _ = self.seq_to_str(
            query=sequence,
            key=structure,
            value=structure,
            key_padding_mask=~str_mask,
            need_weights=False,
        )
        str_update, _ = self.str_to_seq(
            query=structure,
            key=sequence,
            value=sequence,
            key_padding_mask=~seq_mask,
            need_weights=False,
        )
        sequence = self.seq_norm(sequence + seq_update)
        structure = self.str_norm(structure + str_update)
        pooled = torch.cat(
            [
                self.seq_pool(sequence, seq_mask),
                self.str_pool(structure, str_mask),
            ],
            dim=1,
        )
        return self.classifier(pooled)

class MeanConcat(nn.Module):


    def __init__(self, num_classes, dropout):
        super().__init__()
        self.classifier = ClassificationHead(
            1280 + 512, num_classes, dropout
        )

    def forward(self, sequence, structure, seq_mask, str_mask):
        sequence = masked_mean(sequence, seq_mask)
        structure = masked_mean(structure, str_mask)
        return self.classifier(torch.cat([sequence, structure], dim=1))
