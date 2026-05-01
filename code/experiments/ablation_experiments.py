"""
消融实验运行器 (Ablation Experiments Runner)

实验1: 门控回填机制消融
  - no_backfill       : 仅标签引导（aux分类器用于损失），无回填
  - backfill_no_gate  : 回填全部缺失样本，无门控筛选
  - full_system       : 完整系统（标签引导 + 门控回填）

实验2: 门的组件消融（在回填开启前提下）
  - acc_gate_only     : 仅准确性门（不检查矩统计）
  - moment_gate_only  : 仅矩门（均值/方差一致性检查，不检查准确性提升）
  - both_gates        : 准确性门 + 矩门（完整系统）

支持多次运行（不同随机种子），输出 mean ± std。

用法:
    python ablation_experiments.py --config config.yaml --runs 3 --exp 1 2
"""

import argparse
import copy
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

try:
    from torch.cuda.amp.grad_scaler import GradScaler
    AMP_AVAILABLE = True
except Exception:
    AMP_AVAILABLE = False

from data import create_multimodal_dataset_from_config, load_config, check_label_distribution
from center_loss import CenterLoss
from aux_classifier import build_aux_classifier
from model import TGATUNet
from train import (
    unify_config, set_seed, collate_fn_multimodal,
    BackfillManager, train_one_epoch, eval_loop,
    train_aux_classifier
)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Experiment 1: backfill mechanism ablation
EXP1_VARIANTS = {
    "no_backfill": {
        "desc": "仅标签引导（无回填）",
        "aux_enabled": True,
        "backfill_enabled": False,   # no backfill manager
        "backfill_gate_override": {"enabled": False},
    },
    "backfill_no_guide": {
        "desc": "仅回填（无标签引导：λ_acc=0，buffer接受所有）",
        "aux_enabled": True,
        "backfill_enabled": True,
        # λ_acc=0：关闭CE差分损失，label guidance完全移除
        "cls_w_override": 0.0,
        "backfill_gate_override": {
            "enabled": True,
            "disable_acc_gate": True,   # 不做准确性筛选
            "enable_stats": False,       # 不做矩一致性筛选
            "accept_correct": False,
            "warmup_epochs": 0,
            "interval": 1,
            "dynamic_interval": False,
            "max_accept_per_epoch": 99999,
            "keep_max_per_batch": 0,
        },
    },
    "full_system": {
        "desc": "完整系统（标签引导 + 门控回填）",
        "aux_enabled": True,
        "backfill_enabled": True,
        "backfill_gate_override": None,  # use config as-is
    },
}

# Experiment 2: gate component ablation
EXP2_VARIANTS = {
    "acc_gate_only": {
        "desc": "仅准确性门",
        "aux_enabled": True,
        "backfill_enabled": True,
        "backfill_gate_override": {
            "enable_stats": False,       # disable moment check
            "disable_acc_gate": False,   # keep accuracy check
        },
    },
    "moment_gate_only": {
        "desc": "仅矩门（均值/方差一致性）",
        "aux_enabled": True,
        "backfill_enabled": True,
        "backfill_gate_override": {
            "enable_stats": True,        # keep moment check
            "disable_acc_gate": True,    # skip accuracy check
            "accept_correct": False,
            "keep_max_per_batch": 0,     # disable CE-ranked top-k so only moment gate remains
        },
    },
    "both_gates": {
        "desc": "准确性门 + 矩门（完整门控）",
        "aux_enabled": True,
        "backfill_enabled": True,
        "backfill_gate_override": {
            "enable_stats": True,
            "disable_acc_gate": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(base_config: Dict, variant: Dict, seed: int, device: torch.device,
               logger: logging.Logger) -> Dict[str, float]:
    """Run one experiment variant with the given seed. Returns test metrics."""
    set_seed(seed)
    config = copy.deepcopy(base_config)

    # Apply variant settings
    aux_enabled = variant["aux_enabled"]
    backfill_enabled = variant["backfill_enabled"]

    # Override backfill_gate if specified
    gate_override = variant.get("backfill_gate_override")
    if gate_override is not None:
        if "backfill_gate" not in config or config["backfill_gate"] is None:
            config["backfill_gate"] = {}
        config["backfill_gate"] = {**config["backfill_gate"], **gate_override}

    if not aux_enabled:
        config.setdefault("aux_classifier", {})["enabled"] = False
    else:
        config.setdefault("aux_classifier", {})["enabled"] = True

    # Build dataset
    dataset = create_multimodal_dataset_from_config(config, phase='encode')
    n_total = len(dataset)
    n_train = int(config['train_ratio'] * n_total)
    n_val = int(config['val_ratio'] * n_total)
    n_test = n_total - n_train - n_val

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    batch_size = int(config.get('batch_size', 32))
    pin_memory = (device.type == 'cuda')
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn_multimodal)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            collate_fn=collate_fn_multimodal)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             collate_fn=collate_fn_multimodal)

    # Infer input channels
    try:
        sb = next(iter(train_loader))
        in_channels = sb[0].size(1)
        if in_channels > 512 and sb[0].size(2) < 32:
            in_channels = sb[0].size(2)
        time_len = sb[0].size(2)
    except Exception:
        in_channels = config.get('input_channels', 32)
        time_len = int(config.get('window_size', 320) // max(1, int(config.get('sampling_rate', 1))))

    # Build model
    model = TGATUNet(
        in_channels=in_channels,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=in_channels,
        num_classes=config.get('num_classes', 2),
        encoder_layers=config.get('encoder_layers', 3),
        decoder_layers=config.get('decoder_layers', 3),
        heads=config.get('gat_heads', 4),
        time_k=config.get('time_k', 1),
        trans_nhead=config.get('trans_nhead', 4),
        trans_layers=config.get('trans_layers', 2),
        trans_dim_feedforward=config.get('trans_dim_feedforward', 512),
        sanitize_nans=True,
        clamp_value=1e4,
    ).to(device)

    optimizer = Adam(model.parameters(),
                     lr=float(config['learning_rate']),
                     weight_decay=float(config.get('weight_decay', 1e-5)))
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=float(config.get('lr_factor', 0.5)),
                                  patience=int(config.get('lr_patience', 10)))
    use_amp = AMP_AVAILABLE and bool(config.get('use_amp', False))
    scaler = GradScaler() if use_amp else None

    epochs = int(config.get('epochs', 100))
    loss_cfg = config.get('loss_config', {})
    missing_w = float(config.get('missing_recon_weight', 1.0))
    present_w = float(config.get('present_recon_weight', 0.0))
    recon_w = float(loss_cfg.get('recon_weight', 1.0))
    cls_w = float(loss_cfg.get('cls_diff_weight', 0.5))
    # Variant-level override: e.g. backfill_no_guide sets cls_w=0 to remove label guidance
    if 'cls_w_override' in variant:
        cls_w = float(variant['cls_w_override'])
    backfill_gate = config.get('backfill_gate', {'enabled': False})
    aux_cfg = config.get('aux_classifier', {})

    center_loss_weight = float(config.get('center_loss_weight', 0.0))
    center_loss_fn = None
    if center_loss_weight > 0:
        feat_dim = int(config.get('hidden_channels', 64))
        center_loss_fn = CenterLoss(
            num_classes=config.get('num_classes', 2), feat_dim=feat_dim).to(device)

    # Auxiliary classifier
    aux_classifier = None
    backfill_manager = None

    def _freeze_aux(aux_model):
        if aux_model is None:
            return
        for p in aux_model.parameters():
            p.requires_grad = False

    if aux_enabled:
        aux_classifier = build_aux_classifier(
            aux_cfg, in_channels, time_len,
            num_classes=config.get('num_classes', 2)).to(device)
        aux_epochs = int(aux_cfg.get('epochs', 20))
        aux_lr = float(aux_cfg.get('lr', 1e-3))
        aux_patience = aux_cfg.get('patience', None)
        train_aux_classifier(aux_classifier, train_loader, device,
                             aux_epochs, aux_lr, val_loader=val_loader,
                             patience=aux_patience)
        aux_classifier.eval()
        _freeze_aux(aux_classifier)

    if backfill_enabled and aux_enabled:
        backfill_manager = BackfillManager(
            mark_present=bool(backfill_gate.get('mark_present', True)))

    aux_ce = nn.CrossEntropyLoss(reduction='none')

    best_val_loss = float('inf')
    patience = int(config.get('early_stopping_patience', 20))
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Determine active gate for this epoch
        epoch_gate = backfill_gate
        if isinstance(backfill_gate, dict):
            epoch_gate = dict(backfill_gate)
            warmup = int(backfill_gate.get('warmup_epochs', 0))
            interval = int(backfill_gate.get('interval', 1))
            if epoch_gate.get('enabled', True):
                if epoch <= warmup:
                    epoch_gate['enabled'] = False
                elif interval > 1:
                    first = warmup + 1
                    epoch_gate['enabled'] = ((epoch - first) % interval == 0)

        tr_results = train_one_epoch(
            model, train_loader, optimizer, device,
            accumulate_grad_batches=int(config.get('accumulate_grad_batches', 1)),
            use_mixed_precision=use_amp, scaler=scaler,
            recon_weight=recon_w,
            cls_diff_weight=cls_w,
            cls_diff_clip=loss_cfg.get('cls_diff_clip', 10.0),
            center_loss_fn=center_loss_fn,
            center_loss_weight=center_loss_weight,
            missing_weight=missing_w, present_weight=present_w,
            aux_classifier=aux_classifier,
            aux_criterion=aux_ce,
            backfill_manager=backfill_manager,
            backfill_gate=epoch_gate,
        )

        val_results = eval_loop(
            model, val_loader, device, return_auc=True,
            backfill_manager=backfill_manager,
            use_completed_input=True,
            cls_diff_weight=cls_w,
            cls_diff_clip=loss_cfg.get('cls_diff_clip', 10.0),
            aux_classifier=aux_classifier,
            missing_weight=missing_w, present_weight=present_w,
        )
        val_loss = val_results[0]
        scheduler.step(val_loss)

        # Retrain aux if backfill added data
        if (aux_enabled and backfill_manager is not None
                and backfill_manager.new_data):
            retrain_min = int(aux_cfg.get('retrain_min_new', 0))
            if retrain_min <= 0 or backfill_manager.pending_new >= retrain_min:
                train_aux_classifier(aux_classifier, train_loader, device,
                                     int(aux_cfg.get('epochs', 20)),
                                     float(aux_cfg.get('lr', 1e-3)),
                                     backfill_manager=backfill_manager,
                                     val_loader=val_loader,
                                     patience=aux_cfg.get('patience', None))
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

    # Test evaluation
    model.load_state_dict(best_state)
    test_results = eval_loop(
        model, test_loader, device, return_auc=True,
        backfill_manager=backfill_manager,
        use_completed_input=True,
        cls_diff_weight=cls_w,
        cls_diff_clip=loss_cfg.get('cls_diff_clip', 10.0),
        aux_classifier=aux_classifier,
        missing_weight=missing_w, present_weight=present_w,
    )

    if len(test_results) == 8:
        test_loss, test_recon, test_acc, test_wd, test_miss, test_pres, test_cls, test_auc = test_results
    elif len(test_results) == 7:
        test_loss, test_recon, test_acc, test_wd, test_miss, test_pres, test_cls = test_results
        test_auc = 0.0
    else:
        test_loss, test_recon, test_acc, test_wd, test_miss, test_pres = test_results
        test_cls = 0.0
        test_auc = 0.0

    return {
        "test_loss": test_loss,
        "test_recon": test_recon,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_miss_recon": test_miss,
    }


# ---------------------------------------------------------------------------
# Multi-run with statistics
# ---------------------------------------------------------------------------

def run_variant_multiple(base_config: Dict, name: str, variant: Dict,
                         seeds: List[int], device: torch.device,
                         logger: logging.Logger) -> Dict[str, Tuple[float, float]]:
    """Run a variant multiple times; return {metric: (mean, std)}."""
    all_metrics: Dict[str, List[float]] = {}
    for run_idx, seed in enumerate(seeds):
        logger.info(f"  [{name}] run {run_idx + 1}/{len(seeds)}, seed={seed}")
        try:
            metrics = run_single(base_config, variant, seed, device, logger)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)
        except Exception as e:
            logger.warning(f"  [{name}] run failed (seed={seed}): {e}")
    summary: Dict[str, Tuple[float, float]] = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals, dtype=float)
        summary[k] = (float(arr.mean()), float(arr.std(ddof=0)))
    return summary


def print_table(results: Dict[str, Dict], variants: Dict, exp_name: str):
    """Print a formatted results table."""
    metrics_order = ["test_auc", "test_acc", "test_recon", "test_miss_recon", "test_loss"]
    metric_labels = {
        "test_auc": "AUC",
        "test_acc": "Acc",
        "test_recon": "Recon-MSE",
        "test_miss_recon": "Miss-MSE",
        "test_loss": "Loss",
    }

    header_cols = ["Variant", "Description"] + [metric_labels.get(m, m) for m in metrics_order]
    col_widths = [20, 30] + [18] * len(metrics_order)

    def row_str(cols):
        return " | ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))

    sep = "-" * sum(col_widths + [3 * len(col_widths)])

    print(f"\n{'='*80}")
    print(f"  {exp_name} 消融实验结果（mean ± std）")
    print(f"{'='*80}")
    print(row_str(header_cols))
    print(sep)

    for vname, vdef in variants.items():
        if vname not in results:
            continue
        summary = results[vname]
        row = [vname, vdef["desc"]]
        for m in metrics_order:
            if m in summary:
                mean, std = summary[m]
                row.append(f"{mean:.4f} ± {std:.4f}")
            else:
                row.append("N/A")
        print(row_str(row))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="消融实验运行器")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--runs", type=int, default=3,
                        help="每个变体运行次数（不同种子），默认3次")
    parser.add_argument("--exp", type=int, nargs="+", default=[1, 2],
                        choices=[1, 2],
                        help="运行哪些实验 (1=回填机制消融, 2=门组件消融)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="指定随机种子列表，默认 42 43 44 ...")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="只运行指定变体名（e.g. no_backfill full_system）")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    base_config = unify_config(raw_cfg)

    # Setup logging
    os.makedirs(base_config.get('log_dir', 'Logs'), exist_ok=True)
    log_path = os.path.join(
        base_config.get('log_dir', 'Logs'),
        f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger("ablation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    seeds = args.seeds if args.seeds else list(range(42, 42 + args.runs))
    logger.info(f"随机种子: {seeds}")

    exp_map = {
        1: ("实验1：门控回填机制消融", EXP1_VARIANTS),
        2: ("实验2：门组件消融", EXP2_VARIANTS),
    }

    all_tables = {}

    for exp_id in args.exp:
        exp_name, variants = exp_map[exp_id]
        logger.info(f"\n{'='*60}")
        logger.info(f"开始 {exp_name}")
        logger.info(f"{'='*60}")

        exp_results = {}
        for vname, vdef in variants.items():
            if args.variants and vname not in args.variants:
                continue
            logger.info(f"\n--- 变体: {vname} ({vdef['desc']}) ---")
            summary = run_variant_multiple(
                base_config, vname, vdef, seeds, device, logger)
            exp_results[vname] = summary
            for metric, (mean, std) in summary.items():
                logger.info(f"  {metric}: {mean:.4f} ± {std:.4f}")

        all_tables[exp_id] = (exp_name, variants, exp_results)

    # Print summary tables
    for exp_id, (exp_name, variants, exp_results) in all_tables.items():
        print_table(exp_results, variants, exp_name)

    # Save results to file
    results_path = os.path.join(
        base_config.get('log_dir', 'Logs'),
        f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(results_path, 'w', encoding='utf-8') as f:
        for exp_id, (exp_name, variants, exp_results) in all_tables.items():
            f.write(f"\n{exp_name}\n{'='*60}\n")
            for vname, summary in exp_results.items():
                vdesc = variants[vname]["desc"]
                f.write(f"\n{vname} ({vdesc}):\n")
                for metric, (mean, std) in summary.items():
                    f.write(f"  {metric}: {mean:.4f} ± {std:.4f}\n")
    logger.info(f"\n结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
