import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


DATA_ROOT = Path(r"D:\EC")
DEFAULT_OUTPUT = Path(r"D:\EC\new_str\Modify\journal_rerun_20260727")
DEFAULT_FEATURE_ROOT = DEFAULT_OUTPUT / "feature_store"

CSV_PATHS = {
    "train": DATA_ROOT / "train_cleaned_with_structure.csv",
    "lt30": DATA_ROOT / "test_30_cleaned_with_structure.csv",
    "30_50": DATA_ROOT / "test_30_50_clean.csv",
}

STORE_NAMES = {
    "train": ("seq_train", "str_train"),
    "lt30": ("seq_lt30", "str_lt30"),
    "30_50": ("seq_30_50", "str_30_50"),
}

MODEL_NAMES = (
    "gaca",
    "seq_only",
    "str_only",
    "mean_vector_gate",
    "attention_concat",
    "cross_attention",
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MemmapFeatureStore:
    def __init__(self, directory):
        directory = Path(directory)
        self.values = np.load(directory / "values.npy", mmap_mode="r")
        index = pd.read_csv(directory / "index.csv", dtype={"protein_id": str})
        self.locations = {
            row.protein_id: (int(row.offset), int(row.length))
            for row in index.itertuples(index=False)
        }

    def __contains__(self, protein_id):
        return str(protein_id) in self.locations

    def get(self, protein_id):
        offset, length = self.locations[str(protein_id)]
        return np.asarray(
            self.values[offset : offset + length], dtype=np.float32
        )


class ProteinDataset(Dataset):
    def __init__(
        self,
        csv_path,
        seq_store,
        str_store,
        label_encoder=None,
        fit_label_encoder=False,
    ):
        frame = pd.read_csv(csv_path, dtype={"Entry": str})
        frame["Entry"] = frame["Entry"].astype(str)
        rows = frame[
            frame["Entry"].map(
                lambda protein_id: protein_id in seq_store and protein_id in str_store
            )
        ].copy()
        rows["EC number"] = rows["EC number"].astype(str)

        if fit_label_encoder:
            label_encoder = LabelEncoder()
            label_encoder.fit(rows["EC number"])
        else:
            known = set(label_encoder.classes_)
            rows = rows[rows["EC number"].isin(known)].copy()

        self.frame = rows.reset_index(drop=True)
        self.ids = self.frame["Entry"].tolist()
        self.labels_text = self.frame["EC number"].tolist()
        self.labels = label_encoder.transform(self.labels_text)
        self.label_encoder = label_encoder
        self.seq_store = seq_store
        self.str_store = str_store

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        protein_id = self.ids[index]
        sequence = torch.from_numpy(self.seq_store.get(protein_id))
        structure = torch.from_numpy(self.str_store.get(protein_id))
        return sequence, structure, int(self.labels[index]), protein_id


def collate_batch(batch):
    sequences, structures, labels, protein_ids = zip(*batch)
    seq_lengths = torch.tensor([item.shape[0] for item in sequences])
    str_lengths = torch.tensor([item.shape[0] for item in structures])
    seq_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    str_padded = pad_sequence(structures, batch_first=True, padding_value=0.0)
    seq_mask = (
        torch.arange(seq_padded.shape[1]).unsqueeze(0) < seq_lengths.unsqueeze(1)
    )
    str_mask = (
        torch.arange(str_padded.shape[1]).unsqueeze(0) < str_lengths.unsqueeze(1)
    )
    return (
        seq_padded,
        str_padded,
        seq_mask,
        str_mask,
        torch.tensor(labels, dtype=torch.long),
        list(protein_ids),
    )


def make_label_protected_split(labels, validation_ratio, seed):
    labels = np.asarray(labels)
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)

    rng = np.random.default_rng(seed)
    eligible = []
    protected = []
    for indices in groups.values():
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)
        protected.append(int(indices[0]))
        eligible.extend(indices[1:].tolist())

    target_validation_size = math.ceil(len(labels) * validation_ratio)
    if target_validation_size > len(eligible):
        raise ValueError("Not enough non-protected examples for validation")
    eligible = np.asarray(eligible, dtype=int)
    rng.shuffle(eligible)
    validation = np.sort(eligible[:target_validation_size])
    validation_set = set(validation.tolist())
    fitting = np.asarray(
        [index for index in range(len(labels)) if index not in validation_set],
        dtype=int,
    )
    return fitting, validation, np.asarray(protected, dtype=int)


def save_or_load_split(dataset, output_dir, validation_ratio, seed):
    split_path = output_dir / "fixed_fit_validation_split.csv"
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"protein_id": str})
        id_to_index = {protein_id: index for index, protein_id in enumerate(dataset.ids)}
        fitting = np.asarray(
            [id_to_index[item] for item in split.loc[split["split"] == "fit", "protein_id"]],
            dtype=int,
        )
        validation = np.asarray(
            [
                id_to_index[item]
                for item in split.loc[split["split"] == "validation", "protein_id"]
            ],
            dtype=int,
        )
        return fitting, validation

    fitting, validation, protected = make_label_protected_split(
        dataset.labels, validation_ratio, seed
    )
    protected_set = set(protected.tolist())
    validation_set = set(validation.tolist())
    records = []
    for index, protein_id in enumerate(dataset.ids):
        records.append(
            {
                "protein_id": protein_id,
                "ec_label": dataset.labels_text[index],
                "split": "validation" if index in validation_set else "fit",
                "label_protected_fit_example": index in protected_set,
            }
        )
    pd.DataFrame(records).to_csv(split_path, index=False)
    return fitting, validation


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

    def forward(self, sequence, structure, seq_mask, str_mask):
        seq_hidden = self.seq_proj(self.seq_pool(sequence, seq_mask))
        str_hidden = self.str_proj(self.str_pool(structure, str_mask))
        gate = self.gate(torch.cat([seq_hidden, str_hidden], dim=1))
        fused = gate * str_hidden + (1.0 - gate) * seq_hidden
        return self.classifier(fused)


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


def build_model(name, num_classes, args):
    if name == "gaca":
        return GaCA(num_classes, args.hidden_dim, args.dropout)
    if name == "seq_only":
        return SingleModality(
            num_classes, args.hidden_dim, args.dropout, modality="sequence"
        )
    if name == "str_only":
        return SingleModality(
            num_classes, args.hidden_dim, args.dropout, modality="structure"
        )
    if name == "mean_vector_gate":
        return MeanVectorGate(num_classes, args.hidden_dim, args.dropout)
    if name == "attention_concat":
        return AttentionConcat(num_classes, args.hidden_dim, args.dropout)
    if name == "cross_attention":
        return CrossAttention(
            num_classes,
            args.cross_attention_dim,
            args.dropout,
            args.cross_attention_heads,
        )
    raise KeyError(name)


def move_batch(batch, device):
    sequence, structure, seq_mask, str_mask, labels, protein_ids = batch
    return (
        sequence.to(device, non_blocking=True),
        structure.to(device, non_blocking=True),
        seq_mask.to(device, non_blocking=True),
        str_mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        protein_ids,
    )


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    sample_count = 0
    correct = 0
    for batch in loader:
        sequence, structure, seq_mask, str_mask, labels, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(sequence, structure, seq_mask, str_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach()) * labels.shape[0]
        sample_count += labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum())
    return running_loss / sample_count, correct / sample_count


@torch.no_grad()
def validation_metrics(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    sample_count = 0
    true_labels = []
    predictions = []
    for batch in loader:
        sequence, structure, seq_mask, str_mask, labels, _ = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(sequence, structure, seq_mask, str_mask)
            loss = criterion(logits, labels)
        running_loss += float(loss) * labels.shape[0]
        sample_count += labels.shape[0]
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
    return running_loss / sample_count, accuracy_score(true_labels, predictions)


def hierarchical_correct(reference, prediction):
    reference_fields = reference.split(".")
    prediction_fields = prediction.split(".")
    return [
        reference_fields[:level] == prediction_fields[:level]
        for level in range(1, 5)
    ]


@torch.no_grad()
def evaluate_test(model, dataset, loader, device, use_amp):
    model.eval()
    rows = []
    true_indices = []
    predicted_indices = []
    for batch in loader:
        sequence, structure, seq_mask, str_mask, labels, protein_ids = move_batch(
            batch, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(sequence, structure, seq_mask, str_mask)
        probabilities = F.softmax(logits.float(), dim=1)
        top_scores, predictions = probabilities.max(dim=1)
        true_indices.extend(labels.cpu().numpy())
        predicted_indices.extend(predictions.cpu().numpy())
        reference_text = dataset.label_encoder.inverse_transform(labels.cpu().numpy())
        prediction_text = dataset.label_encoder.inverse_transform(
            predictions.cpu().numpy()
        )
        for protein_id, reference, prediction, score in zip(
            protein_ids,
            reference_text,
            prediction_text,
            top_scores.cpu().numpy(),
        ):
            correct = hierarchical_correct(reference, prediction)
            rows.append(
                {
                    "protein_id": protein_id,
                    "true_ec": reference,
                    "predicted_ec": prediction,
                    "top_class_softmax_score": float(score),
                    "level_1_correct": correct[0],
                    "level_2_correct": correct[1],
                    "level_3_correct": correct[2],
                    "level_4_correct": correct[3],
                }
            )

    result = pd.DataFrame(rows)
    labels_for_macro = sorted(set(true_indices))
    metrics = {
        "samples": len(result),
        "level_1_accuracy": float(result["level_1_correct"].mean()),
        "level_2_accuracy": float(result["level_2_correct"].mean()),
        "level_3_accuracy": float(result["level_3_correct"].mean()),
        "level_4_accuracy": float(result["level_4_correct"].mean()),
        "macro_f1_reference_classes": float(
            f1_score(
                true_indices,
                predicted_indices,
                labels=labels_for_macro,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(true_indices, predicted_indices)),
        "reference_class_count": len(labels_for_macro),
    }
    return metrics, result


def make_loader(dataset, indices, batch_size, shuffle, args):
    target = Subset(dataset, indices) if indices is not None else dataset
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        target,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        generator=generator,
    )


def run_model(
    name,
    train_dataset,
    fitting_indices,
    validation_indices,
    test_datasets,
    output_dir,
    args,
    device,
):
    model_dir = output_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    batch_size = (
        args.cross_attention_batch_size
        if name == "cross_attention"
        else args.batch_size
    )
    use_amp = bool(args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    fitting_loader = make_loader(
        train_dataset, fitting_indices, batch_size, True, args
    )
    validation_loader = make_loader(
        train_dataset, validation_indices, batch_size, False, args
    )

    seed_everything(args.seed)
    model = build_model(name, len(train_dataset.label_encoder.classes_), args).to(
        device
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_state = None
    patience_count = 0
    print(
        f"[model] {name}: parameters={parameter_count:,}, batch={batch_size}",
        flush=True,
    )
    if args.refit_only_epochs is not None:
        tuning_path = model_dir / "tuning_history.csv"
        if not tuning_path.exists():
            raise FileNotFoundError(
                "--refit-only-epochs requires an existing tuning_history.csv"
            )
        existing_history = pd.read_csv(tuning_path)
        best_epoch = int(args.refit_only_epochs)
        if best_epoch < 1:
            raise ValueError("--refit-only-epochs must be positive")
        selected = existing_history.loc[
            existing_history["epoch"].astype(int) == best_epoch
        ]
        if selected.empty:
            raise ValueError(f"Epoch {best_epoch} is absent from {tuning_path}")
        best_validation_accuracy = float(
            selected.iloc[0]["validation_accuracy"]
        )
        print(
            f"[resume] {name}: skip tuning and refit from scratch for "
            f"{best_epoch} epochs (saved val_acc={best_validation_accuracy:.5f})",
            flush=True,
        )
    else:
        for epoch in range(1, args.max_epochs + 1):
            started = time.time()
            train_loss, train_accuracy = train_epoch(
                model,
                fitting_loader,
                optimizer,
                criterion,
                device,
                scaler,
                use_amp,
            )
            validation_loss, validation_accuracy = validation_metrics(
                model, validation_loader, criterion, device, use_amp
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "elapsed_seconds": time.time() - started,
                }
            )
            print(
                f"[tune] {name} epoch={epoch:03d} "
                f"train_loss={train_loss:.5f} train_acc={train_accuracy:.5f} "
                f"val_loss={validation_loss:.5f} val_acc={validation_accuracy:.5f}",
                flush=True,
            )
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                )
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    break

        pd.DataFrame(history).to_csv(
            model_dir / "tuning_history.csv", index=False
        )
        torch.save(best_state, model_dir / "best_validation_checkpoint.pth")
    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_everything(args.seed)
    final_model = build_model(
        name, len(train_dataset.label_encoder.classes_), args
    ).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    final_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    full_loader = make_loader(
        train_dataset,
        np.arange(len(train_dataset)),
        batch_size,
        True,
        args,
    )
    final_history = []
    for epoch in range(1, best_epoch + 1):
        started = time.time()
        train_loss, train_accuracy = train_epoch(
            final_model,
            full_loader,
            final_optimizer,
            criterion,
            device,
            final_scaler,
            use_amp,
        )
        final_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "elapsed_seconds": time.time() - started,
            }
        )
        print(
            f"[refit] {name} epoch={epoch:03d}/{best_epoch:03d} "
            f"loss={train_loss:.5f} acc={train_accuracy:.5f}",
            flush=True,
        )
    pd.DataFrame(final_history).to_csv(
        model_dir / "full_refit_history.csv", index=False
    )
    torch.save(final_model.state_dict(), model_dir / "final_full_training_model.pth")

    model_metrics = {
        "model": name,
        "trainable_parameters": parameter_count,
        "fitting_samples": len(fitting_indices),
        "validation_samples": len(validation_indices),
        "full_refit_samples": len(train_dataset),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "batch_size": batch_size,
        "amp": use_amp,
        "test_results": {},
    }
    for dataset_name, dataset in test_datasets.items():
        loader = make_loader(dataset, None, batch_size, False, args)
        metrics, predictions = evaluate_test(
            final_model, dataset, loader, device, use_amp
        )
        metrics["dataset"] = dataset_name
        model_metrics["test_results"][dataset_name] = metrics
        predictions.to_csv(
            model_dir / f"predictions_{dataset_name}.csv", index=False
        )
        print(
            f"[test] {name} {dataset_name}: "
            f"L1={metrics['level_1_accuracy']:.5f} "
            f"L2={metrics['level_2_accuracy']:.5f} "
            f"L3={metrics['level_3_accuracy']:.5f} "
            f"L4={metrics['level_4_accuracy']:.5f}",
            flush=True,
        )

    (model_dir / "metrics.json").write_text(
        json.dumps(model_metrics, indent=2), encoding="utf-8"
    )
    return model_metrics


def load_datasets(feature_root):
    stores = {}
    for split, (seq_name, str_name) in STORE_NAMES.items():
        stores[split] = (
            MemmapFeatureStore(feature_root / seq_name),
            MemmapFeatureStore(feature_root / str_name),
        )
    train_dataset = ProteinDataset(
        CSV_PATHS["train"],
        *stores["train"],
        fit_label_encoder=True,
    )
    tests = {
        split: ProteinDataset(
            CSV_PATHS[split],
            *stores[split],
            label_encoder=train_dataset.label_encoder,
        )
        for split in ("lt30", "30_50")
    }
    return train_dataset, tests


def write_environment(output_dir, args, device):
    information = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arguments": vars(args),
    }
    (output_dir / "environment_and_config.json").write_text(
        json.dumps(information, indent=2, default=str), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["gaca"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cross-attention-batch-size", type=int, default=2)
    parser.add_argument("--cross-attention-dim", type=int, default=512)
    parser.add_argument("--cross-attention-heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--refit-only-epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    write_environment(args.output_dir, args, device)

    train_dataset, test_datasets = load_datasets(args.feature_root)
    fitting_indices, validation_indices = save_or_load_split(
        train_dataset,
        args.output_dir,
        args.validation_ratio,
        args.seed,
    )
    fitting_labels = set(train_dataset.labels[fitting_indices])
    all_labels = set(train_dataset.labels)
    if fitting_labels != all_labels:
        raise RuntimeError("The fitting split does not cover every training label")

    pd.DataFrame(
        {
            "class_index": np.arange(len(train_dataset.label_encoder.classes_)),
            "ec_label": train_dataset.label_encoder.classes_,
        }
    ).to_csv(args.output_dir / "label_mapping.csv", index=False)
    dataset_summary = {
        "effective_training_samples": len(train_dataset),
        "fitting_samples": len(fitting_indices),
        "validation_samples": len(validation_indices),
        "training_classes": len(train_dataset.label_encoder.classes_),
        "fitting_classes": len(fitting_labels),
        "test_samples": {
            name: len(dataset) for name, dataset in test_datasets.items()
        },
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )
    print(f"[data] {json.dumps(dataset_summary)}", flush=True)

    aggregate_path = args.output_dir / "all_model_metrics.json"
    if aggregate_path.exists():
        all_metrics = json.loads(aggregate_path.read_text(encoding="utf-8"))
    else:
        all_metrics = {}
    for name in args.models:
        all_metrics[name] = run_model(
            name,
            train_dataset,
            fitting_indices,
            validation_indices,
            test_datasets,
            args.output_dir,
            args,
            device,
        )
        aggregate_path.write_text(
            json.dumps(all_metrics, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
