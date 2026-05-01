#!/usr/bin/env python
import argparse
import copy
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from aux_classifier import AuxCNNClassifier, AuxLSTMClassifier, AuxTransformerClassifier
from baseline_classifier import SimpleBaselineClassifier
from data import create_multimodal_dataset_from_config, load_config
from train import collate_fn_multimodal, set_seed, unify_config


def build_downstream_classifier(
    name: str,
    in_channels: int,
    time_len: int,
    num_classes: int = 2,
    hidden: int = 256,
    depth: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return SimpleBaselineClassifier(
            in_channels, time_len, num_classes=num_classes, hidden=hidden, depth=depth, dropout=dropout
        )
    if name == "cnn1d":
        return AuxCNNClassifier(in_channels, num_classes=num_classes, hidden=hidden, dropout=dropout)
    if name == "lstm":
        return AuxLSTMClassifier(in_channels, num_classes=num_classes, hidden=hidden, layers=max(1, depth), dropout=dropout)
    if name == "transformer":
        return AuxTransformerClassifier(
            in_channels,
            num_classes=num_classes,
            d_model=hidden,
            nhead=4,
            layers=max(1, depth),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported downstream classifier: {name}")


def compute_classification_metrics(probs_or_logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    if probs_or_logits.ndim != 2:
        raise ValueError("Expected [N, C] tensor for probs_or_logits")
    probs = probs_or_logits.detach().cpu().float().numpy()
    y_true = labels.detach().cpu().long().numpy()

    if probs.shape[1] == 1:
        pos_prob = probs[:, 0]
        preds = (pos_prob >= 0.5).astype(int)
    else:
        row_sums = probs.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            exp = np.exp(probs - probs.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
        pos_prob = probs[:, 1]
        preds = probs.argmax(axis=1)

    acc = float((preds == y_true).mean())
    f1_bin = float(f1_score(y_true, preds, average="binary", zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, pos_prob))
    except Exception:
        roc_auc = 0.0
    try:
        pr_auc = float(average_precision_score(y_true, pos_prob))
    except Exception:
        pr_auc = 0.0
    return {"acc": acc, "f1_bin": f1_bin, "roc_auc": roc_auc, "pr_auc": pr_auc}


def _prepare_completed_config(base_config: Dict, completed_dir: str, sources: List[str] | None = None) -> Dict:
    cfg = copy.deepcopy(base_config)
    names = ["FM", "OD", "MEFAR"] if not sources else sources
    cfg["data_files"] = [os.path.join(completed_dir, f"{name}_completed.csv") for name in names]
    cfg["cache_enabled"] = False
    cfg["resume_training"] = False
    return cfg


def _build_loaders(cfg: Dict, seed: int, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    dataset = create_multimodal_dataset_from_config(cfg, phase="encode")
    n_total = len(dataset)
    n_train = int(cfg["train_ratio"] * n_total)
    n_val = int(cfg["val_ratio"] * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    sample = next(iter(train_loader))
    in_channels = sample[0].size(1)
    time_len = sample[0].size(2)
    return train_loader, val_loader, test_loader, in_channels, time_len


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> Tuple[float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            x, labels = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            loss = ce(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probs = torch.softmax(logits.detach(), dim=1)
            total_loss += float(loss.detach().item()) * labels.size(0)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    n = sum(t.size(0) for t in all_labels)
    return total_loss / max(n, 1), torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def run_single_backbone(base_config: Dict, completed_dir: str, backbone: str, seed: int, device: torch.device) -> Dict[str, float]:
    set_seed(seed)
    cfg = _prepare_completed_config(base_config, completed_dir)
    train_loader, val_loader, test_loader, in_channels, time_len = _build_loaders(cfg, seed, device)
    model = build_downstream_classifier(backbone, in_channels, time_len, num_classes=int(cfg.get("num_classes", 2))).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.get("baseline_lr", 1e-3)))
    max_epochs = int(cfg.get("baseline_max_epochs", 20))
    patience = int(cfg.get("baseline_patience", 10))

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    wait = 0
    for _ in range(max_epochs):
        _run_epoch(model, train_loader, device, optimizer=optimizer)
        _, val_probs, val_labels = _run_epoch(model, val_loader, device, optimizer=None)
        val_metrics = compute_classification_metrics(val_probs, val_labels)
        if val_metrics["roc_auc"] > best_val:
            best_val = val_metrics["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    _, test_probs, test_labels = _run_epoch(model, test_loader, device, optimizer=None)
    return compute_classification_metrics(test_probs, test_labels)


def summarize_backbone_runs(base_config: Dict, completed_dir: str, backbone: str, seeds: List[int], device: torch.device) -> Dict[str, Tuple[float, float]]:
    metrics_by_name: Dict[str, List[float]] = {}
    for seed in seeds:
        metrics = run_single_backbone(base_config, completed_dir, backbone, seed, device)
        for k, v in metrics.items():
            metrics_by_name.setdefault(k, []).append(v)
    summary: Dict[str, Tuple[float, float]] = {}
    for k, values in metrics_by_name.items():
        arr = np.asarray(values, dtype=float)
        summary[k] = (float(arr.mean()), float(arr.std(ddof=0)))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate completed datasets with downstream classifiers")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--completed_dir", type=str, required=True)
    parser.add_argument("--backbones", nargs="+", default=["mlp", "lstm", "cnn1d", "transformer"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = unify_config(load_config(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.seeds if args.seeds else list(range(42, 42 + args.runs))
    results = {
        backbone: summarize_backbone_runs(cfg, args.completed_dir, backbone, seeds, device)
        for backbone in args.backbones
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
