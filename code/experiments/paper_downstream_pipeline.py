#!/usr/bin/env python
import argparse
import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from ablation_experiments import EXP1_VARIANTS, EXP2_VARIANTS
from aux_classifier import build_aux_classifier
from center_loss import CenterLoss
from data import create_multimodal_dataset_from_config, load_config
from downstream_eval import run_single_backbone
from export_completed import export_completed_datasets
from model import TGATUNet
from train import (
    AMP_AVAILABLE,
    BackfillManager,
    collate_fn_multimodal,
    eval_loop,
    set_seed,
    train_aux_classifier,
    train_one_epoch,
    unify_config,
)

try:
    from torch.cuda.amp.grad_scaler import GradScaler
except Exception:  # pragma: no cover
    GradScaler = None


ALL_VARIANTS = {**EXP1_VARIANTS, **EXP2_VARIANTS}


def resolve_variants(names: List[str]) -> Dict[str, Dict]:
    resolved = {}
    for name in names:
        if name not in ALL_VARIANTS:
            raise ValueError(f"Unknown variant: {name}")
        resolved[name] = copy.deepcopy(ALL_VARIANTS[name])
    return resolved


def build_protocol_snapshot(config_path: str, base_config: Dict, variants: List[str], seeds: List[int], backbones: List[str]) -> Dict:
    aux_cfg = dict(base_config.get("aux_classifier", {}) or {})
    gate_cfg = dict(base_config.get("backfill_gate", {}) or {})
    return {
        "config_path": config_path,
        "variants": list(variants),
        "seeds": list(seeds),
        "backbones": list(backbones),
        "stage1": {
            "epochs": int(base_config.get("epochs", 100)),
            "early_stopping_patience": int(base_config.get("early_stopping_patience", 20)),
            "batch_size": int(base_config.get("batch_size", 32)),
            "num_workers": int(base_config.get("num_workers", 0)),
            "learning_rate": float(base_config.get("learning_rate", 1e-3)),
            "train_ratio": float(base_config.get("train_ratio", 0.6)),
            "val_ratio": float(base_config.get("val_ratio", 0.2)),
            "aux_type": str(aux_cfg.get("type", "disabled")),
            "aux_epochs": int(aux_cfg.get("epochs", 0)),
            "aux_patience": int(aux_cfg.get("patience", 0)),
            "gate_warmup_epochs": int(gate_cfg.get("warmup_epochs", 0)),
            "gate_interval": int(gate_cfg.get("interval", 1)),
            "gate_enable_stats": bool(gate_cfg.get("enable_stats", False)),
        },
    }


def _prepare_variant_config(base_config: Dict, variant: Dict, variant_name: str, run_tag: str) -> Tuple[Dict, bool, bool]:
    config = copy.deepcopy(base_config)
    aux_enabled = variant["aux_enabled"]
    backfill_enabled = variant["backfill_enabled"]

    gate_override = variant.get("backfill_gate_override")
    if gate_override is not None:
        if "backfill_gate" not in config or config["backfill_gate"] is None:
            config["backfill_gate"] = {}
        config["backfill_gate"] = {**config["backfill_gate"], **gate_override}

    config.setdefault("aux_classifier", {})["enabled"] = bool(aux_enabled)
    loss_cfg = dict(config.get("loss_config", {}) or {})
    if "cls_w_override" in variant:
        loss_cfg["cls_diff_weight"] = float(variant["cls_w_override"])
    config["loss_config"] = loss_cfg

    root = Path(config.get("log_dir", "Logs")).parent.resolve()
    out_root = root / "paper_runs" / variant_name / run_tag
    config["log_dir"] = str(out_root / "logs")
    config["tensorboard_dir"] = str(out_root / "runs")
    config["checkpoint_dir"] = str(out_root / "checkpoints")
    config["resume_training"] = False
    return config, aux_enabled, backfill_enabled


def train_variant_and_export(base_config: Dict, variant_name: str, variant: Dict, seed: int, device: torch.device, logger: logging.Logger) -> Path:
    set_seed(seed)
    run_tag = f"seed_{seed}"
    config, aux_enabled, backfill_enabled = _prepare_variant_config(base_config, variant, variant_name, run_tag)

    dataset = create_multimodal_dataset_from_config(config, phase="encode")
    n_total = len(dataset)
    n_train = int(config["train_ratio"] * n_total)
    n_val = int(config["val_ratio"] * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    batch_size = int(config.get("batch_size", 32))
    pin_memory = device.type == "cuda"
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch[0].size(1)
    time_len = sample_batch[0].size(2)

    model = TGATUNet(
        in_channels=in_channels,
        hidden_channels=config.get("hidden_channels", 64),
        out_channels=in_channels,
        num_classes=config.get("num_classes", 2),
        encoder_layers=config.get("encoder_layers", 3),
        decoder_layers=config.get("decoder_layers", 3),
        heads=config.get("gat_heads", 4),
        time_k=config.get("time_k", 1),
        trans_nhead=config.get("trans_nhead", 4),
        trans_layers=config.get("trans_layers", 2),
        trans_dim_feedforward=config.get("trans_dim_feedforward", 512),
        sanitize_nans=True,
        clamp_value=1e4,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 1e-5)))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=float(config.get("lr_factor", 0.5)), patience=int(config.get("lr_patience", 10)))
    use_amp = AMP_AVAILABLE and bool(config.get("use_amp", False))
    scaler = GradScaler() if (use_amp and GradScaler is not None) else None

    epochs = int(config.get("epochs", 100))
    loss_cfg = config.get("loss_config", {})
    missing_w = float(config.get("missing_recon_weight", 1.0))
    present_w = float(config.get("present_recon_weight", 0.0))
    recon_w = float(loss_cfg.get("recon_weight", 1.0))
    cls_w = float(loss_cfg.get("cls_diff_weight", 0.5))
    backfill_gate = config.get("backfill_gate", {"enabled": False})
    aux_cfg = config.get("aux_classifier", {})

    center_loss_weight = float(config.get("center_loss_weight", 0.0))
    center_loss_fn = None
    if center_loss_weight > 0:
        feat_dim = int(config.get("hidden_channels", 64))
        center_loss_fn = CenterLoss(num_classes=config.get("num_classes", 2), feat_dim=feat_dim).to(device)

    aux_classifier = None
    backfill_manager = None
    aux_ce = nn.CrossEntropyLoss(reduction="none")

    def _freeze_aux(aux_model):
        if aux_model is None:
            return
        for p in aux_model.parameters():
            p.requires_grad = False

    if aux_enabled:
        aux_classifier = build_aux_classifier(aux_cfg, in_channels, time_len, num_classes=config.get("num_classes", 2)).to(device)
        train_aux_classifier(aux_classifier, train_loader, device, int(aux_cfg.get("epochs", 20)), float(aux_cfg.get("lr", 1e-3)), val_loader=val_loader, patience=aux_cfg.get("patience", None))
        aux_classifier.eval()
        _freeze_aux(aux_classifier)

    if backfill_enabled and aux_enabled:
        backfill_manager = BackfillManager(mark_present=bool(backfill_gate.get("mark_present", True)))

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = int(config.get("early_stopping_patience", 20))
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_gate = backfill_gate
        if isinstance(backfill_gate, dict):
            epoch_gate = dict(backfill_gate)
            warmup = int(backfill_gate.get("warmup_epochs", 0))
            interval = int(backfill_gate.get("interval", 1))
            if epoch_gate.get("enabled", True):
                if epoch <= warmup:
                    epoch_gate["enabled"] = False
                elif interval > 1:
                    first = warmup + 1
                    epoch_gate["enabled"] = ((epoch - first) % interval == 0)

        train_one_epoch(
            model, train_loader, optimizer, device,
            accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
            use_mixed_precision=use_amp, scaler=scaler,
            recon_weight=recon_w, cls_diff_weight=cls_w,
            cls_diff_clip=loss_cfg.get("cls_diff_clip", 10.0),
            center_loss_fn=center_loss_fn, center_loss_weight=center_loss_weight,
            missing_weight=missing_w, present_weight=present_w,
            aux_classifier=aux_classifier, aux_criterion=aux_ce,
            backfill_manager=backfill_manager, backfill_gate=epoch_gate,
        )
        val_results = eval_loop(
            model, val_loader, device, return_auc=True,
            backfill_manager=backfill_manager, use_completed_input=True,
            cls_diff_weight=cls_w, cls_diff_clip=loss_cfg.get("cls_diff_clip", 10.0),
            aux_classifier=aux_classifier, missing_weight=missing_w, present_weight=present_w,
        )
        val_loss = val_results[0]
        scheduler.step(val_loss)

        if aux_enabled and backfill_manager is not None and backfill_manager.new_data:
            retrain_min = int(aux_cfg.get("retrain_min_new", 0))
            if retrain_min <= 0 or backfill_manager.pending_new >= retrain_min:
                train_aux_classifier(aux_classifier, train_loader, device, int(aux_cfg.get("epochs", 20)), float(aux_cfg.get("lr", 1e-3)), backfill_manager=backfill_manager, val_loader=val_loader, patience=aux_cfg.get("patience", None))
                aux_classifier.eval()
                _freeze_aux(aux_classifier)
                backfill_manager.clear_new_flag()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"
    torch.save({"model_state_dict": best_state, "config": config, "seed": seed}, ckpt_path)
    model.load_state_dict(best_state)

    completed_dir = ckpt_dir.parent / "completed"
    export_completed_datasets(model, dataset, config, device, str(completed_dir))
    logger.info("[%s][seed=%s] exported completed datasets to %s", variant_name, seed, completed_dir)
    return completed_dir


def run_pipeline_for_variant(base_config: Dict, variant_name: str, seeds: List[int], backbones: List[str], device: torch.device, logger: logging.Logger) -> Dict[str, Dict[str, Tuple[float, float]]]:
    metrics_by_backbone: Dict[str, Dict[str, List[float]]] = {b: {} for b in backbones}
    variant = ALL_VARIANTS[variant_name]

    for seed in seeds:
        completed_dir = train_variant_and_export(base_config, variant_name, variant, seed, device, logger)
        for backbone in backbones:
            metrics = run_single_backbone(base_config, str(completed_dir), backbone, seed, device)
            for k, v in metrics.items():
                metrics_by_backbone[backbone].setdefault(k, []).append(v)

    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for backbone, metric_map in metrics_by_backbone.items():
        summary[backbone] = {}
        for metric_name, values in metric_map.items():
            arr = np.asarray(values, dtype=float)
            summary[backbone][metric_name] = (float(arr.mean()), float(arr.std(ddof=0)))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned downstream evaluation pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--backbones", nargs="+", default=["mlp", "lstm", "cnn1d", "transformer"])
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    base_config = unify_config(raw_cfg)
    resolve_variants(args.variants)

    log_dir = Path(base_config.get("log_dir", "Logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"paper_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("paper_pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.seeds if args.seeds else list(range(42, 42 + args.runs))
    protocol_snapshot = build_protocol_snapshot(args.config, base_config, args.variants, seeds, args.backbones)
    logger.info("protocol snapshot: %s", json.dumps(protocol_snapshot, ensure_ascii=False, sort_keys=True))

    all_results = {}
    for variant_name in args.variants:
        logger.info("=== variant %s ===", variant_name)
        all_results[variant_name] = run_pipeline_for_variant(base_config, variant_name, seeds, args.backbones, device, logger)

    payload = json.dumps(all_results, indent=2, ensure_ascii=False)
    print(payload)
    out_path = args.out or str(log_dir / f"paper_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload)
    logger.info("results saved to %s", out_path)


if __name__ == "__main__":
    main()
