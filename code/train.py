# train.py - v99

"""
Copied and aligned from V13=78. v99 normalizes config keys for reproducibility.
 Accepts legacy keys (lr, ckpt_dir, use_mixed_precision, train_split, sample_rate, patience)
 Exposes unified keys internally (learning_rate, checkpoint_dir, use_amp, train_ratio/val_ratio, sampling_rate, early_stopping_patience)
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import collections
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

try:
    from torch.cuda.amp.grad_scaler import GradScaler
    from torch.cuda.amp.autocast_mode import autocast
    AMP_AVAILABLE = True
except Exception:
    AMP_AVAILABLE = False

from data import create_multimodal_dataset_from_config, load_config, check_label_distribution
from center_loss import CenterLoss
from aux_classifier import build_aux_classifier
from project_paths import (
    default_cache_dir,
    default_checkpoint_dir,
    default_log_dir,
    default_tensorboard_dir,
    normalize_config_paths,
)

try:
    from simple_multimodal_integration import create_simple_multimodal_criterion
    MULTIMODAL_CRITERION_AVAILABLE = True
except Exception:
    MULTIMODAL_CRITERION_AVAILABLE = False

try:
    from domain_adaptation import DomainAdaptationModule, get_lambda_schedule
    DOMAIN_ADAPTATION_AVAILABLE = True
except Exception:
    DomainAdaptationModule = None
    get_lambda_schedule = None
    DOMAIN_ADAPTATION_AVAILABLE = False


def collate_fn_multimodal(batch):
    tensors, labels, masks, sources, presents, missings, indices = [], [], [], [], [], [], []
    for item in batch:
        if not isinstance(item, (list, tuple)):
            continue
        if len(item) >= 7:
            tensor, label, modal_mask, source_id, present_mask, missing_mask, sample_idx = item[:7]
        elif len(item) >= 6:
            tensor, label, modal_mask, source_id, present_mask, missing_mask = item[:6]
            sample_idx = None
        elif len(item) >= 4:
            tensor, label, modal_mask, source_id = item[:4]
            present_mask = modal_mask.clone()
            missing_mask = (1.0 - present_mask)
            sample_idx = None
        else:
            continue
        tensors.append(tensor)
        labels.append(label)
        masks.append(modal_mask)
        sources.append(source_id)
        presents.append(present_mask)
        missings.append(missing_mask)
        if sample_idx is not None:
            indices.append(sample_idx)
    if indices:
        return (torch.stack(tensors), torch.stack(labels), torch.stack(masks),
                torch.stack(sources), torch.stack(presents), torch.stack(missings),
                torch.stack(indices))
    return (torch.stack(tensors), torch.stack(labels), torch.stack(masks),
            torch.stack(sources), torch.stack(presents), torch.stack(missings))


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger("v99")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def unify_config(cfg: Dict) -> Dict:
    c = dict(cfg) if cfg else {}
    # learning rate
    if 'learning_rate' not in c and 'lr' in c:
        c['learning_rate'] = c['lr']
    c.setdefault('learning_rate', 1e-4)
    # checkpoint/tensorboard dirs
    if 'checkpoint_dir' not in c and 'ckpt_dir' in c:
        c['checkpoint_dir'] = c['ckpt_dir']
    c.setdefault('checkpoint_dir', str(default_checkpoint_dir()))
    c.setdefault('tensorboard_dir', c.get('tb_dir', str(default_tensorboard_dir())))
    # splits
    train_ratio = c.get('train_ratio', c.get('train_split', None))
    val_ratio = c.get('val_ratio', c.get('val_split', None))
    test_ratio = c.get('test_ratio', None)
    if train_ratio is None and val_ratio is None:
        train_ratio, val_ratio = 0.7, 0.15
    if train_ratio is None:
        if val_ratio is not None and test_ratio is not None:
            train_ratio = max(0.0, 1.0 - float(val_ratio) - float(test_ratio))
        else:
            train_ratio = 0.7
    if val_ratio is None:
        if test_ratio is not None:
            val_ratio = max(0.0, 1.0 - float(train_ratio) - float(test_ratio))
        else:
            val_ratio = 0.15
    c['train_ratio'] = float(train_ratio)
    c['val_ratio'] = float(val_ratio)
    # AMP flag
    if 'use_amp' not in c:
        c['use_amp'] = bool(c.get('use_mixed_precision', False))
    # patience
    if 'early_stopping_patience' not in c and 'patience' in c:
        c['early_stopping_patience'] = c['patience']
    c.setdefault('early_stopping_patience', 20)
    # lr scheduler keys
    c.setdefault('lr_factor', 0.5)
    c.setdefault('lr_patience', 10)
    # logging dirs
    c.setdefault('log_dir', str(default_log_dir()))
    # batch size, epochs
    c.setdefault('batch_size', 32)
    c.setdefault('epochs', 100)
    # loss_config defaults
    lc = dict(c.get('loss_config', {}) or {})
    lc.setdefault('recon_weight', 1.0)
    lc.setdefault('cls_improvement_weight', 2.0)
    lc.setdefault('accuracy_reward_scale', 2.0)
    lc.setdefault('accuracy_threshold', 0.05)
    lc.setdefault('min_improvement_margin', 0.05)
    lc.setdefault('dynamic_weighting', True)
    lc.setdefault('loss_smoothing', False)
    lc.setdefault('smoothing_factor', 0.9)
    lc.setdefault('cls_diff_weight', 1.0)
    lc.setdefault('cls_diff_clip', 10.0)
    c['loss_config'] = lc
    # completion extensions
    c.setdefault('missing_recon_weight', 1.0)
    c.setdefault('present_recon_weight', 0.0)
    c.setdefault('stage1_epochs', 0)
    # save interval
    c.setdefault('save_interval', 10)
    # resume
    c.setdefault('resume_training', False)
    c.setdefault('cache_dir', str(default_cache_dir()))
    return normalize_config_paths(c)

def select_stage(epoch: int, config: Dict) -> str:
    staged = config.get('staged_training', {}) if isinstance(config, dict) else {}
    stage1_epochs = int(staged.get('stage1_epochs', config.get('stage1_epochs', 0)) or 0)
    if stage1_epochs > 0 and int(epoch) <= stage1_epochs:
        return str(staged.get('stage1_name', 'warmup'))
    return str(staged.get('stage2_name', 'supervised'))


def get_training_stage(epoch: int, config: Dict) -> str:
    return select_stage(epoch, config)


def _domain_component_weight(domain_cfg: Dict, name: str) -> float:
    component = domain_cfg.get(name, {}) if isinstance(domain_cfg, dict) else {}
    if not isinstance(component, dict) or not bool(component.get('enabled', False)):
        return 0.0
    return float(component.get('weight', 0.0))


def stage_loss_weights(stage: str, config: Dict) -> Dict[str, float]:
    cfg = config or {}
    loss_cfg = cfg.get('loss_config', {}) if isinstance(cfg, dict) else {}
    recon_weight = float(loss_cfg.get('recon_weight', 1.0))
    cls_weight = float(loss_cfg.get('cls_diff_weight', loss_cfg.get('cls_improvement_weight', 0.0)))
    if str(stage).lower() in {'warmup', 'a', 'stage1'}:
        cls_weight = 0.0

    domain_cfg = cfg.get('domain_adaptation', {}) if isinstance(cfg, dict) else {}
    domain_weight = 0.0
    if isinstance(domain_cfg, dict) and bool(domain_cfg.get('enabled', False)):
        domain_weight = (
            _domain_component_weight(domain_cfg, 'adversarial')
            + _domain_component_weight(domain_cfg, 'mmd')
            + _domain_component_weight(domain_cfg, 'coral')
        )
    return {
        'recon_weight': recon_weight,
        'cls_weight': cls_weight,
        'cls_diff_weight': cls_weight,
        'domain_weight': domain_weight,
    }


def batch_loss_flags(stage: str = 'default', batch_kind: str = 'P') -> Dict[str, bool]:
    normalized_stage = str(stage or 'default').lower()
    normalized_kind = str(batch_kind or 'P').upper()
    classification = normalized_kind == 'P' and normalized_stage not in {'warmup', 'a', 'stage1'}
    return {'recon': True, 'classification': classification, 'domain': True}


def route_classification_for_batch(batch_meta: Optional[Any] = None, *, batch_kind: Optional[str] = None,
                                   source_ids: Optional[torch.Tensor] = None, n_p_sources: int = 0,
                                   stage: str = 'default'):
    if isinstance(batch_meta, dict):
        batch_kind = batch_meta.get('batch_kind', batch_kind)
        source_ids = batch_meta.get('source_ids', source_ids)
        n_p_sources = int(batch_meta.get('n_p_sources', n_p_sources) or 0)
        stage = batch_meta.get('stage', stage)
    if batch_kind is not None:
        return batch_loss_flags(stage=stage, batch_kind=batch_kind)['classification']
    if source_ids is not None and n_p_sources > 0:
        return source_ids < int(n_p_sources)
    return batch_loss_flags(stage=stage, batch_kind='P')['classification']


class BackfillManager:
    def __init__(self, mark_present: bool = True, ttl: int = 0):
        """
        ttl: Time-To-Live（epoch 数）。0 = 禁用（永不自动过期）。
             写入 epoch e 的样本，在 current_epoch >= e + ttl 时被 expire() 删除。
        """
        self.data = {}
        self.timestamps = {}   # sample_idx -> 写入时的 epoch
        self.mark_present = mark_present
        self.ttl = int(ttl)
        self.new_data = False
        self.pending_new = 0
        self.last_epoch_stats = {'total': 0, 'accepted': 0}

    def apply(self, batch, present_mask, missing_mask, sample_indices):
        if sample_indices is None:
            return batch, present_mask, missing_mask
        if isinstance(sample_indices, torch.Tensor):
            idx_list = sample_indices.detach().cpu().tolist()
        else:
            idx_list = list(sample_indices)
        for i, idx in enumerate(idx_list):
            filled = self.data.get(int(idx))
            if filled is None:
                continue
            filled = filled.to(batch.device, non_blocking=True)
            batch[i] = filled
            if self.mark_present:
                present_mask[i] = 1.0
                missing_mask[i] = 0.0
        return batch, present_mask, missing_mask

    def update(self, idx, completed_tensor, epoch: int = 0):
        key = int(idx)
        self.data[key] = completed_tensor.detach().cpu()
        self.timestamps[key] = int(epoch)
        self.new_data = True
        self.pending_new += 1

    def expire(self, current_epoch: int):
        """删除超过 TTL 的条目。ttl=0 时不删除任何条目。"""
        if self.ttl <= 0:
            return
        expired = [k for k, ts in self.timestamps.items() if current_epoch >= ts + self.ttl]
        for k in expired:
            del self.data[k]
            del self.timestamps[k]
        if expired:
            print(f"[Backfill] TTL expire: removed {len(expired)} entries at epoch {current_epoch}")

    def revalidate(self, model, device, mse_threshold: float):
        """用当前模型重建所有存储的补全，删除 MSE 过高的条目。"""
        if not self.data:
            return
        model.eval()
        to_remove = []
        with torch.no_grad():
            for k, tensor in self.data.items():
                # tensor: [C, T]，转为 [T, C] 送入模型
                window = tensor.t().to(device)
                try:
                    result = model(window, return_latent=True)
                    recon = result[0]  # recon: [C, T]
                    mse = (recon.cpu() - tensor).pow(2).mean().item()
                    if mse > mse_threshold:
                        to_remove.append(k)
                except Exception:
                    pass
        for k in to_remove:
            del self.data[k]
            del self.timestamps[k]
        if to_remove:
            print(f"[Backfill] Revalidation: removed {len(to_remove)} stale entries (MSE > {mse_threshold:.4f})")

    def clear_new_flag(self):
        self.new_data = False
        self.pending_new = 0

    def set_epoch_stats(self, total, accepted):
        self.last_epoch_stats = {'total': int(total), 'accepted': int(accepted)}


def safe_ce_loss(logits: torch.Tensor, labels: torch.Tensor, clamp_value: float = 1e4):
    if torch.isnan(logits).any() or (~torch.isfinite(logits)).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
        logits = torch.clamp(logits, -clamp_value, clamp_value)
    return nn.CrossEntropyLoss()(logits, labels)


def forward_batch_parallel(model, input_batch, device, present_masks=None):
    windows = input_batch.transpose(1, 2)  # [B, T, C]
    if hasattr(model, 'forward_batch') and callable(getattr(model, 'forward_batch')):
        try:
            res = model.forward_batch(windows, present_masks=present_masks)
        except TypeError:
            res = model.forward_batch(windows)
        if len(res) == 3:
            return res
        recon, logits = res[0], res[-1]
        latent = recon.mean(dim=-1) if recon.ndim >= 3 else recon
        return recon, logits, latent

    outs, logits_list, latents = [], [], []
    for i in range(windows.size(0)):
        pm = present_masks[i] if present_masks is not None else None
        res = model(windows[i], return_latent=True, present_mask=pm)
        if len(res) == 3:
            o, l, latent = res
        else:
            o, l = res[0], res[-1]
            latent = o.mean(dim=-1) if o.ndim >= 2 else o
        outs.append(o)
        logits_list.append(l)
        latents.append(latent)
    return torch.stack(outs, 0), torch.stack(logits_list, 0), torch.stack(latents, 0)


def _wasserstein_distance(x, y):
    try:
        B = x.size(0)
        x_flat = x.view(B, -1).float()
        y_flat = y.view(B, -1).float()
        x_sorted, _ = torch.sort(x_flat, dim=1)
        y_sorted, _ = torch.sort(y_flat, dim=1)
        wd = torch.mean(torch.abs(x_sorted - y_sorted))
        return wd
    except Exception:
        return torch.tensor(0.0, device=x.device)


def train_aux_classifier(aux_model, dataloader, device, epochs: int, lr: float,
                         backfill_manager: Optional[BackfillManager] = None,
                         val_loader=None, patience: Optional[int] = None):
    aux_model.to(device)
    for param in aux_model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
    opt = Adam(aux_model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    wait = 0
    max_epochs = int(epochs)
    for ep in range(1, max_epochs + 1):
        aux_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_data in dataloader:
            if len(batch_data) >= 7:
                batch, labels, modal_mask, source_ids, present_mask, missing_mask, sample_indices = batch_data[:7]
            else:
                batch, labels, modal_mask, source_ids, present_mask, missing_mask = batch_data[:6]
                sample_indices = None
            batch = batch.to(device)
            labels = labels.to(device)
            if backfill_manager is not None:
                batch, _, _ = backfill_manager.apply(batch, present_mask.to(device), missing_mask.to(device), sample_indices)
            logits = aux_model(batch)
            loss = ce(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                preds = logits.argmax(1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += float(loss.detach().item()) * labels.size(0)
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            print(f"[AuxCls] Epoch {ep}/{max_epochs}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        if val_loader is not None:
            aux_model.eval()
            v_loss = 0.0
            v_correct = 0
            v_samples = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    if len(vbatch) >= 7:
                        vb, vlabels, vmodal_mask, vsource_ids, vpresent_mask, vmissing_mask, vsample_indices = vbatch[:7]
                    else:
                        vb, vlabels, vmodal_mask, vsource_ids, vpresent_mask, vmissing_mask = vbatch[:6]
                        vsample_indices = None
                    vb = vb.to(device)
                    vlabels = vlabels.to(device)
                    if backfill_manager is not None:
                        vb, _, _ = backfill_manager.apply(vb, vpresent_mask.to(device), vmissing_mask.to(device), vsample_indices)
                    vlogits = aux_model(vb)
                    vbatch_loss = ce(vlogits, vlabels)
                    preds = vlogits.argmax(1)
                    v_correct += (preds == vlabels).sum().item()
                    v_samples += vlabels.size(0)
                    v_loss += float(vbatch_loss.detach().item()) * vlabels.size(0)
            if v_samples > 0:
                val_loss = v_loss / v_samples
                val_acc = v_correct / v_samples
                print(f"[AuxCls] Val {ep}/{max_epochs}: loss={val_loss:.4f}, acc={val_acc:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if patience is not None and wait >= int(patience):
                        print(f"[AuxCls] Early stopping at epoch {ep} (best_val_loss={best_val_loss:.4f})")
                        break


def train_one_epoch(model, dataloader, optimizer, device, accumulate_grad_batches,
                    use_mixed_precision, scaler, recon_weight, cls_diff_weight, cls_diff_clip,
                    center_loss_fn=None, center_loss_weight=0.0,
                    struct_drop=False, struct_drop_weight=0.0,
                    missing_weight=1.0, present_weight=0.0,
                    aux_classifier=None, aux_criterion=None,
                    backfill_manager=None, backfill_gate=None,
                    current_epoch: int = 0,
                    domain_weight: float = 0.0, domain_module=None, n_p_sources: int = 0):
    model.train()
    total_loss = total_recon = total_cls = total_wd = 0.0
    total_recon_missing = 0.0
    total_recon_present = 0.0
    total_center = total_consistency = 0.0
    total_domain = total_domain_adv = total_domain_mmd = total_domain_coral = 0.0
    total_domain_acc = 0.0
    total_p_recon = total_pp_recon = 0.0
    total_p_samples = total_pp_samples = 0
    total_correct = 0
    total_samples = 0
    step = 0
    gate_total = 0
    gate_accepted = 0
    epoch_accept_limit = int(backfill_gate.get('max_accept_per_epoch', 0)) if isinstance(backfill_gate, dict) else 0

    for batch_data in tqdm(dataloader, desc="Training"):
        if len(batch_data) >= 7:
            batch, labels, modal_mask, source_ids, present_mask, missing_mask, sample_indices = batch_data[:7]
        else:
            batch, labels, modal_mask, source_ids, present_mask, missing_mask = batch_data[:6]
            sample_indices = None
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        source_ids = source_ids.to(device, non_blocking=True)
        present_mask = present_mask.to(device, non_blocking=True)
        missing_mask = missing_mask.to(device, non_blocking=True)
        if sample_indices is not None:
            sample_indices = sample_indices.to(device, non_blocking=True)

        if backfill_manager is not None:
            batch, present_mask, missing_mask = backfill_manager.apply(
                batch, present_mask, missing_mask, sample_indices
            )

        if step % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        with autocast(enabled=(use_mixed_precision and scaler is not None and AMP_AVAILABLE)):
            recon, logits, latents = forward_batch_parallel(model, batch, device, present_masks=present_mask)
            diff = (recon - batch)
            mse_per = diff.pow(2)
            present_mask_exp = present_mask.unsqueeze(-1)
            missing_mask_exp = missing_mask.unsqueeze(-1)
            pres_mse = (mse_per * present_mask_exp).sum() / (present_mask_exp.sum() * mse_per.size(-1) + 1e-6)
            miss_mse = (mse_per * missing_mask_exp).sum() / (missing_mask_exp.sum() * mse_per.size(-1) + 1e-6)
            recon_loss = (missing_weight * miss_mse) + (present_weight * pres_mse)
            completed = batch * present_mask_exp + recon * missing_mask_exp

            cls_loss = torch.tensor(0.0, device=device)
            ce_diff = torch.tensor(0.0, device=device)
            preds = torch.zeros(labels.size(0), device=device, dtype=torch.long)
            cls_mask = route_classification_for_batch(
                source_ids=source_ids, n_p_sources=int(n_p_sources or 0), stage='supervised'
            )
            if isinstance(cls_mask, torch.Tensor):
                cls_mask = cls_mask.to(device=device, dtype=torch.bool)
            if aux_classifier is not None and (not isinstance(cls_mask, torch.Tensor) or bool(cls_mask.any().item())):
                aux_classifier.eval()
                aux_logits = aux_classifier(batch)
                aux_logits_completed = aux_classifier(completed)
                if isinstance(cls_mask, torch.Tensor):
                    ce_original = safe_ce_loss(aux_logits[cls_mask], labels[cls_mask])
                    ce_completed = safe_ce_loss(aux_logits_completed[cls_mask], labels[cls_mask])
                else:
                    ce_original = safe_ce_loss(aux_logits, labels)
                    ce_completed = safe_ce_loss(aux_logits_completed, labels)
                if cls_diff_clip is not None:
                    ce_diff = torch.clamp(ce_original - ce_completed, min=-float(cls_diff_clip), max=float(cls_diff_clip))
                else:
                    ce_diff = ce_original - ce_completed
                cls_loss = ce_completed
                preds = aux_logits_completed.argmax(1)

            center_loss = torch.tensor(0.0, device=device)
            if center_loss_fn is not None and center_loss_weight > 0:
                center_loss = center_loss_fn(latents, labels) * center_loss_weight

            domain_loss = torch.tensor(0.0, device=device)
            domain_losses = {}
            if domain_module is not None and float(domain_weight) > 0.0:
                domain_features = latents.float()
                if domain_features.ndim > 2:
                    domain_features = domain_features.reshape(domain_features.size(0), -1)
                valid_domain = source_ids >= 0
                if int(valid_domain.sum().item()) > 1:
                    domain_losses = domain_module(domain_features[valid_domain], source_ids[valid_domain].long())
                    domain_loss = domain_losses.get('total', domain_loss)

            loss = (recon_weight * recon_loss) - (cls_diff_weight * ce_diff) + center_loss + (float(domain_weight) * domain_loss)

        if scaler is not None and use_mixed_precision and AMP_AVAILABLE:
            scaler.scale(loss).backward()
            if step % accumulate_grad_batches == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if step % accumulate_grad_batches == 0:
                optimizer.step()

        wd = _wasserstein_distance(batch, recon)

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        batch_size_n = labels.size(0)
        total_recon += float(recon_loss.item()) * batch_size_n
        if int(n_p_sources or 0) > 0:
            p_mask = source_ids < int(n_p_sources)
        else:
            p_mask = torch.ones_like(source_ids, dtype=torch.bool)
        pp_mask = ~p_mask
        if p_mask.any():
            total_p_recon += float(recon_loss.item()) * int(p_mask.sum().item())
            total_p_samples += int(p_mask.sum().item())
        if pp_mask.any():
            total_pp_recon += float(recon_loss.item()) * int(pp_mask.sum().item())
            total_pp_samples += int(pp_mask.sum().item())
        total_recon_missing += float(miss_mse.item()) * labels.size(0)
        total_recon_present += float(pres_mse.item()) * labels.size(0)
        total_cls += float(cls_loss.item()) * labels.size(0)
        total_loss += float(loss.item()) * labels.size(0)
        total_wd += float(wd.item()) * labels.size(0)
        total_center += float(center_loss.item()) * labels.size(0)
        total_domain += float(domain_loss.detach().item()) * labels.size(0)
        if domain_losses:
            total_domain_adv += float(domain_losses.get('adversarial', torch.tensor(0.0, device=device)).detach().item()) * labels.size(0)
            total_domain_mmd += float(domain_losses.get('mmd', torch.tensor(0.0, device=device)).detach().item()) * labels.size(0)
            total_domain_coral += float(domain_losses.get('coral', torch.tensor(0.0, device=device)).detach().item()) * labels.size(0)
            total_domain_acc += float(domain_losses.get('domain_acc', 0.0)) * labels.size(0)
        step += 1

        if backfill_manager is not None and isinstance(backfill_gate, dict) and backfill_gate.get('enabled', True):
            has_missing = (missing_mask.sum(dim=1) > 0)
            if has_missing.any() and aux_classifier is not None:
                gate_total += int(has_missing.sum().item())
                aux_logits = aux_classifier(batch)
                aux_logits_completed = aux_classifier(completed)
                preds_o = aux_logits.argmax(1)
                preds_c = aux_logits_completed.argmax(1)
                correct_o = preds_o == labels
                correct_c = preds_c == labels
                # Ablation: disable_acc_gate=True → skip accuracy improvement check (moment gate only)
                disable_acc_gate = bool(backfill_gate.get('disable_acc_gate', False))
                if disable_acc_gate:
                    # Accept all missing samples initially; moment gate (enable_stats) will filter
                    accept = has_missing.clone()
                else:
                    acc_improve = (correct_c.int() - correct_o.int())
                    min_acc_improve = float(backfill_gate.get('min_acc_improve', 1.0))
                    accept = (acc_improve >= min_acc_improve)
                    if bool(backfill_gate.get('accept_correct', False)):
                        pres_mse_per = (mse_per * present_mask_exp).sum(dim=(1, 2)) / (present_mask.sum(dim=1) * mse_per.size(-1) + 1e-6)
                        keep_factor = float(backfill_gate.get('keep_present_mse_factor', 0.7))
                        thresh = pres_mse_per.mean() * keep_factor
                        accept = accept | (correct_o & correct_c & (pres_mse_per <= thresh))
                    accept = accept & has_missing
                if bool(backfill_gate.get('enable_stats', False)):
                    mean_tol = float(backfill_gate.get('mean_tol', 0.1))
                    std_tol = float(backfill_gate.get('std_tol', 0.1))
                    for i in range(batch.size(0)):
                        if not accept[i]:
                            continue
                        obs = batch[i][present_mask[i].bool()]
                        imp = completed[i][missing_mask[i].bool()]
                        if obs.numel() == 0 or imp.numel() == 0:
                            continue
                        if abs(obs.mean().item() - imp.mean().item()) > mean_tol:
                            accept[i] = False
                            continue
                        if abs(obs.std().item() - imp.std().item()) > std_tol:
                            accept[i] = False
                keep_max = int(backfill_gate.get('keep_max_per_batch', 0))
                if keep_max > 0 and accept.sum().item() > keep_max:
                    if aux_criterion is not None:
                        ce_o = aux_criterion(aux_logits, labels)
                        ce_c = aux_criterion(aux_logits_completed, labels)
                        scores = (ce_o - ce_c).detach()
                    else:
                        ce_o = torch.nn.functional.cross_entropy(aux_logits, labels, reduction='none')
                        ce_c = torch.nn.functional.cross_entropy(aux_logits_completed, labels, reduction='none')
                        scores = (ce_o - ce_c).detach()
                    top_idx = torch.topk(scores, k=keep_max).indices
                    mask = torch.zeros_like(accept, dtype=torch.bool)
                    mask[top_idx] = True
                    accept = accept & mask
                for i in range(batch.size(0)):
                    if epoch_accept_limit > 0 and gate_accepted >= epoch_accept_limit:
                        break
                    if accept[i] and sample_indices is not None:
                        backfill_manager.update(sample_indices[i].item(), completed[i], epoch=current_epoch)
                        gate_accepted += 1

    if backfill_manager is not None:
        backfill_manager.set_epoch_stats(gate_total, gate_accepted)

    n = max(total_samples, 1)
    return (total_loss / n, total_recon / n, total_cls / n, (total_correct / n),
            total_wd / n, total_center / n, total_consistency / n,
            total_recon_missing / n, total_recon_present / n,
            total_domain_adv / n, total_domain_mmd / n, total_domain_coral / n, total_domain_acc / n,
            total_domain / n,
            total_p_recon / max(total_p_samples, 1),
            total_pp_recon / max(total_pp_samples, 1))


def eval_loop(model, dataloader, device, return_auc=False, backfill_manager=None,
              use_completed_input=True, cls_diff_weight=0.0, cls_diff_clip=None,
              aux_classifier=None, missing_weight=0.0, present_weight=1.0):
    model.eval()
    total_loss = total_recon = total_cls = total_wd = 0.0
    total_recon_missing = 0.0
    total_recon_present = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Eval"):
            if len(batch_data) >= 7:
                batch, labels, modal_mask, source_ids, present_mask, missing_mask, sample_indices = batch_data[:7]
            else:
                batch, labels, modal_mask, source_ids, present_mask, missing_mask = batch_data[:6]
                sample_indices = None
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            present_mask = present_mask.to(device, non_blocking=True)
            missing_mask = missing_mask.to(device, non_blocking=True)
            if sample_indices is not None:
                sample_indices = sample_indices.to(device, non_blocking=True)
            if backfill_manager is not None:
                batch, present_mask, missing_mask = backfill_manager.apply(
                    batch, present_mask, missing_mask, sample_indices
                )

            recon, logits, latents = forward_batch_parallel(model, batch, device, present_masks=present_mask)
            diff = (recon - batch)
            mse_per = diff.pow(2)
            present_mask_exp = present_mask.unsqueeze(-1)
            missing_mask_exp = missing_mask.unsqueeze(-1)
            pres_mse = (mse_per * present_mask_exp).sum() / (present_mask_exp.sum() * mse_per.size(-1) + 1e-6)
            miss_mse = (mse_per * missing_mask_exp).sum() / (missing_mask_exp.sum() * mse_per.size(-1) + 1e-6)
            recon_loss = (missing_weight * miss_mse) + (present_weight * pres_mse)
            completed = batch * present_mask_exp + recon * missing_mask_exp

            cls_loss = torch.tensor(0.0, device=device)
            ce_diff = torch.tensor(0.0, device=device)
            if aux_classifier is not None:
                aux_classifier.eval()
                logits_eval = aux_classifier(completed) if use_completed_input else aux_classifier(batch)
                logits_orig = aux_classifier(batch)
                ce_original = safe_ce_loss(logits_orig, labels)
                ce_completed = safe_ce_loss(logits_eval, labels)
                if cls_diff_clip is not None:
                    ce_diff = torch.clamp(ce_original - ce_completed, min=-float(cls_diff_clip), max=float(cls_diff_clip))
                else:
                    ce_diff = ce_original - ce_completed
                cls_loss = ce_completed
                preds = logits_eval.argmax(1)
                probs = torch.softmax(logits_eval, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            loss = recon_loss - (cls_diff_weight * ce_diff)
            wd = _wasserstein_distance(batch, recon)
            total_recon += float(recon_loss.item()) * labels.size(0)
            total_recon_missing += float(miss_mse.item()) * labels.size(0)
            total_recon_present += float(pres_mse.item()) * labels.size(0)
            total_cls += float(cls_loss.item()) * labels.size(0)
            total_loss += float(loss.item()) * labels.size(0)
            total_wd += float(wd.item()) * labels.size(0)

    n = max(total_samples, 1)
    base_results = (total_loss / n, total_recon / n, (total_correct / n), total_wd / n,
                    (total_recon_missing / n), (total_recon_present / n), (total_cls / n))
    if return_auc and all_probs and all_labels:
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        try:
            if len(np.unique(all_labels)) == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except Exception:
            auc = 0.0
        return base_results + (auc,)
    return base_results


def main():
    parser = argparse.ArgumentParser(description='v99 Multimodal Time Series Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    config = unify_config(raw_cfg)

    set_seed(config.get('seed', 42))
    logger = setup_logging(config.get('log_dir', 'Logs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = create_multimodal_dataset_from_config(config, phase='encode')
    logger.info(f"数据集创建完成，滑动窗口数量: {len(dataset)}")

    getattr(dataset, 'analyze_dataset_label_distribution', lambda: None)()
    label_counter, all_labels = check_label_distribution(dataset)
    logger.info(f"标签分布: {dict(label_counter)} | 所有标签: {sorted(list(all_labels))}")

    n_total = len(dataset)
    n_train = int(config['train_ratio'] * n_total)
    n_val = int(config['val_ratio'] * n_total)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    batch_size = int(config.get('batch_size', 32))
    pin_memory = (device.type == 'cuda')

    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn_multimodal)

    try:
        sample_batch = next(iter(train_loader))
        # sample_batch[0] shape: [B, C, T]; we want C
        if sample_batch and isinstance(sample_batch, (list, tuple)):
            in_channels = sample_batch[0].size(1)
            # sanity: if T is suspiciously small and C huge, maybe transposed; correct
            if in_channels > 512 and sample_batch[0].size(2) < 32:
                in_channels = sample_batch[0].size(2)
            time_len = sample_batch[0].size(2)
        else:
            in_channels = config.get('input_channels', 32)
            time_len = int(config.get('window_size', 320) // max(1, int(config.get('sampling_rate', 1))))
    except Exception:
        in_channels = config.get('input_channels', 32)
        time_len = int(config.get('window_size', 320) // max(1, int(config.get('sampling_rate', 1))))

    # 消融实验配置
    ablation_config = config.get('ablation_study', {})
    ablation_enabled = ablation_config.get('enabled', False)
    ablation_variant = ablation_config.get('variant', 'full')
    
    # 根据消融实验变体设置模型开关
    disable_transformer = False
    
    if ablation_enabled:
        if ablation_variant == 'no_transformer':
            disable_transformer = True
            logger.info("[Ablation] disable Transformer bottleneck")
        elif ablation_variant == 'minimal':
            disable_transformer = True
            logger.info("[Ablation] minimal model (no Transformer)")
        elif ablation_variant == 'full':
            logger.info("[Ablation] full model")
        else:
            logger.warning(f"Unknown ablation variant {ablation_variant}, using full model")

    from model import TGATUNet
    # Sensitivity analysis: read noise_sigma from config
    noise_sigma = float(config.get('noise_sigma', 0.0))
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
        simple_encoder_fallback=False,
        disable_model_layernorm=config.get('disable_model_layernorm', False),
        disable_transformer=disable_transformer,
        noise_sigma=noise_sigma,
        channel_attn=bool(config.get('channel_attn', True)),
        channel_attn_heads=int(config.get('channel_attn_heads', 4)),
        channel_attn_dropout=float(config.get('channel_attn_dropout', 0.1)),
        use_unet_skip=bool(config.get('use_unet_skip', True)),
        trans_use_pos_enc=bool(config.get('trans_use_pos_enc', True)),
        use_mask_input=bool(config.get('use_mask_input', True)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建完成，参数量: {n_params:,}")
    
    if ablation_enabled:
        logger.info(f"消融实验变体: {ablation_variant}")
        logger.info(f"  - Transformer: {'禁用' if disable_transformer else '启用'}")

    domain_cfg = config.get('domain_adaptation', {}) if isinstance(config.get('domain_adaptation', {}), dict) else {}
    domain_module = None
    if bool(domain_cfg.get('enabled', False)) and DOMAIN_ADAPTATION_AVAILABLE:
        adv_raw_w = float(domain_cfg.get('adversarial', {}).get('weight', 0.0)) if bool(domain_cfg.get('adversarial', {}).get('enabled', False)) else 0.0
        mmd_raw_w = float(domain_cfg.get('mmd', {}).get('weight', 0.0)) if bool(domain_cfg.get('mmd', {}).get('enabled', False)) else 0.0
        coral_raw_w = float(domain_cfg.get('coral', {}).get('weight', 0.0)) if bool(domain_cfg.get('coral', {}).get('enabled', False)) else 0.0
        domain_raw_total = max(adv_raw_w + mmd_raw_w + coral_raw_w, 1e-12)
        domain_module = DomainAdaptationModule(
            feature_dim=int(config.get('hidden_channels', 64)),
            num_domains=int(domain_cfg.get('num_domains', len(getattr(dataset, 'source_to_id', {})) or 1)),
            use_adversarial=bool(domain_cfg.get('adversarial', {}).get('enabled', False)),
            use_mmd=bool(domain_cfg.get('mmd', {}).get('enabled', False)),
            use_coral=bool(domain_cfg.get('coral', {}).get('enabled', False)),
            adversarial_weight=adv_raw_w / domain_raw_total,
            mmd_weight=mmd_raw_w / domain_raw_total,
            coral_weight=coral_raw_w / domain_raw_total,
            discriminator_hidden=int(domain_cfg.get('adversarial', {}).get('hidden_dim', 256)),
            discriminator_layers=int(domain_cfg.get('adversarial', {}).get('num_layers', 2)),
            discriminator_dropout=float(domain_cfg.get('adversarial', {}).get('dropout', 0.1)),
        ).to(device)
        logger.info(f"跨域模块启用: domains={domain_cfg.get('num_domains', len(getattr(dataset, 'source_to_id', {})) or 1)}")
    elif bool(domain_cfg.get('enabled', False)):
        logger.warning("跨域配置已启用，但 domain_adaptation 模块不可用，跳过跨域损失")

    optim_params = list(model.parameters())
    if domain_module is not None:
        optim_params += list(domain_module.parameters())
    optimizer = Adam(optim_params, lr=float(config['learning_rate']), weight_decay=float(config.get('weight_decay', 1e-5)))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=float(config.get('lr_factor', 0.5)), patience=int(config.get('lr_patience', 10)), verbose=True)
    use_amp = AMP_AVAILABLE and bool(config.get('use_amp', False))
    scaler = GradScaler() if use_amp else None
    writer = SummaryWriter(config.get('tensorboard_dir', 'runs'))

    epochs = int(config.get('epochs', 100))
    best_val_loss = float('inf')
    patience = int(config.get('early_stopping_patience', 20))
    patience_counter = 0
    
    # Disable early stopping for ablation study
    if ablation_enabled:
        patience = epochs + 1  # Set patience > epochs to effectively disable early stopping
        logger.info(f"Ablation study mode: Early stopping disabled (will train for full {epochs} epochs)")

    loss_cfg = config.get('loss_config', {})
    logger.info("=== 训练配置 ===")
    logger.info(f"batch_size={batch_size}, lr={config['learning_rate']}, AMP={use_amp}")

    ckpt_dir = config.get('checkpoint_dir', 'Checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    resume = bool(config.get('resume_training', False))
    resume_path = config.get('resume_path', None)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    last_ckpt_path = os.path.join(ckpt_dir, 'last_checkpoint.pth')
    start_epoch = 1
    if resume:
        load_path = None
        if resume_path and os.path.exists(resume_path):
            load_path = resume_path
        elif os.path.exists(last_ckpt_path):
            load_path = last_ckpt_path
        elif os.path.exists(ckpt_path):
            load_path = ckpt_path
        if load_path:
            try:
                checkpoint = torch.load(load_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if domain_module is not None and checkpoint.get('domain_module_state_dict') is not None:
                        domain_module.load_state_dict(checkpoint['domain_module_state_dict'])
                    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
                    scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
                    best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
                    start_epoch = int(checkpoint.get('epoch', 0)) + 1
                    if use_amp and 'scaler_state_dict' in checkpoint and scaler is not None:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logger.info(f"恢复训练，checkpoint={load_path}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")
            except Exception as e:
                logger.warning(f"恢复失败: {e}")

    # center loss setup
    center_loss_weight = float(config.get('center_loss_weight', 0.0))
    struct_drop = bool(config.get('enable_structured_drop', False))
    struct_drop_weight = float(config.get('structured_drop_weight', 0.0))
    if center_loss_weight > 0:
        feat_dim = int(config.get('hidden_channels', 64))
        center_loss_fn = CenterLoss(num_classes=config.get('num_classes', 2), feat_dim=feat_dim).to(device)
    else:
        center_loss_fn = None
    debug_cfg = {
        'debug_nan': config.get('debug_nan', False),
        'debug_detect_anomaly': config.get('debug_detect_anomaly', False),
        'debug_dump_first_bad_batch': config.get('debug_dump_first_bad_batch', False),
        'debug_param_scan': config.get('debug_param_scan', False)
    }
    last_param_scan = {}

    # Auxiliary classifier and backfill gate
    aux_cfg = config.get('aux_classifier', {})
    aux_enabled = bool(aux_cfg.get('enabled', False))
    backfill_gate = config.get('backfill_gate', {'enabled': True, 'min_ce_improve': 0.0, 'mark_present': True})
    gate_warmup_epochs = int(backfill_gate.get('warmup_epochs', 0)) if isinstance(backfill_gate, dict) else 0
    gate_interval = int(backfill_gate.get('interval', 1)) if isinstance(backfill_gate, dict) else 1
    dynamic_interval = bool(backfill_gate.get('dynamic_interval', False)) if isinstance(backfill_gate, dict) else False
    interval_min = int(backfill_gate.get('interval_min', 1)) if isinstance(backfill_gate, dict) else 1
    interval_max = int(backfill_gate.get('interval_max', gate_interval)) if isinstance(backfill_gate, dict) else gate_interval
    interval_step = int(backfill_gate.get('interval_step', 1)) if isinstance(backfill_gate, dict) else 1
    acc_rate_low = float(backfill_gate.get('acc_rate_low', 0.001)) if isinstance(backfill_gate, dict) else 0.001
    acc_rate_high = float(backfill_gate.get('acc_rate_high', 0.01)) if isinstance(backfill_gate, dict) else 0.01
    gate_interval_current = gate_interval
    force_gate_next = False
    backfill_ttl = int(backfill_gate.get('ttl', 0)) if isinstance(backfill_gate, dict) else 0
    backfill_manager = BackfillManager(
        mark_present=bool(backfill_gate.get('mark_present', True)),
        ttl=backfill_ttl
    ) if aux_enabled else None
    aux_classifier = None
    aux_ce = nn.CrossEntropyLoss(reduction='none')
    def _freeze_aux(aux_model):
        if aux_model is None:
            return
        for param in aux_model.parameters():
            param.requires_grad = False
    def _init_aux():
        return build_aux_classifier(aux_cfg, in_channels, time_len, num_classes=config.get('num_classes', 2)).to(device)
    use_aux_for_cls = bool(aux_enabled)

    if aux_enabled:
        aux_classifier = _init_aux()
        aux_epochs = int(aux_cfg.get('epochs', 20))
        aux_lr = float(aux_cfg.get('lr', 1e-3))
        aux_patience = aux_cfg.get('patience', None)
        logger.info(f"[AuxCls] init type={aux_cfg.get('type', 'mlp')} epochs={aux_epochs} lr={aux_lr} patience={aux_patience}")
        train_aux_classifier(aux_classifier, train_loader, device, aux_epochs, aux_lr,
                             backfill_manager=backfill_manager, val_loader=val_loader, patience=aux_patience)
        aux_classifier.eval()
        _freeze_aux(aux_classifier)

    # ===== 原有训练模式 =====
    missing_w = float(config.get('missing_recon_weight', 1.0))
    present_w = float(config.get('present_recon_weight', 0.0))
    recon_w_global = float(loss_cfg.get('recon_weight', 1.0))
    cls_w_global = float(loss_cfg.get('cls_diff_weight', loss_cfg.get('cls_improvement_weight', 2.0)))
    
    # 消融实验: 如果禁用分类器，设置分类损失权重为0

    for epoch in range(start_epoch, epochs + 1):
        # Optionally skip backfill gating on some epochs (warmup/interval); training still runs every epoch.
        epoch_backfill_gate = backfill_gate
        if isinstance(backfill_gate, dict):
            epoch_backfill_gate = dict(backfill_gate)
            if bool(epoch_backfill_gate.get('enabled', True)):
                if epoch <= gate_warmup_epochs:
                    epoch_backfill_gate['enabled'] = False
                else:
                    if gate_interval_current > 1:
                        first_gate_epoch = gate_warmup_epochs + 1
                        epoch_backfill_gate['enabled'] = ((epoch - first_gate_epoch) % gate_interval_current == 0)
            if force_gate_next and isinstance(epoch_backfill_gate, dict):
                epoch_backfill_gate['enabled'] = True
                force_gate_next = False
        gate_active = bool(epoch_backfill_gate.get('enabled', True)) if isinstance(epoch_backfill_gate, dict) else True

        retrain_min_new = int(aux_cfg.get('retrain_min_new', 0))
        should_retrain_aux = (aux_enabled and backfill_manager is not None and backfill_manager.new_data and
                              (retrain_min_new <= 0 or backfill_manager.pending_new >= retrain_min_new))
        if should_retrain_aux:
            reset_ratio = float(aux_cfg.get('reset_on_backfill_ratio', 0.0))
            stats = backfill_manager.last_epoch_stats
            total = max(1, stats.get('total', 0))
            ratio = stats.get('accepted', 0) / total
            if reset_ratio > 0 and ratio >= reset_ratio:
                logger.info(f"[AuxCls] reset due to backfill ratio {ratio:.4f} >= {reset_ratio:.4f}")
                aux_classifier = _init_aux()
            aux_epochs = int(aux_cfg.get('epochs', 20))
            aux_lr = float(aux_cfg.get('lr', 1e-3))
            aux_patience = aux_cfg.get('patience', None)
            if retrain_min_new > 0:
                logger.info(f"[AuxCls] retrain on backfilled data (new={backfill_manager.pending_new} >= {retrain_min_new}; epochs={aux_epochs}, lr={aux_lr}, patience={aux_patience})")
            else:
                logger.info(f"[AuxCls] retrain on backfilled data (epochs={aux_epochs}, lr={aux_lr}, patience={aux_patience})")
            train_aux_classifier(aux_classifier, train_loader, device, aux_epochs, aux_lr,
                                 backfill_manager=backfill_manager, val_loader=val_loader, patience=aux_patience)
            aux_classifier.eval()
            _freeze_aux(aux_classifier)
            backfill_manager.clear_new_flag()
        stage = select_stage(epoch, config)
        weights = stage_loss_weights(stage, config)
        recon_w = float(weights.get('recon_weight', recon_w_global))
        cls_w = float(weights.get('cls_diff_weight', cls_w_global))
        domain_w = float(weights.get('domain_weight', 0.0))
        if domain_module is not None and get_lambda_schedule is not None:
            adv_cfg = domain_cfg.get('adversarial', {}) if isinstance(domain_cfg, dict) else {}
            lam = get_lambda_schedule(
                epoch=epoch,
                max_epochs=max(epochs, 1),
                schedule_type=str(adv_cfg.get('lambda_schedule', 'linear')),
                gamma=float(adv_cfg.get('lambda_gamma', 10.0)),
            )
            domain_module.set_lambda(float(lam))
        # train_one_epoch returns base metrics plus domain and P/PP reconstruction metrics.
        (tr_loss, tr_recon, tr_cls, tr_in_acc, tr_wd, tr_center, tr_cons, tr_miss, tr_pres,
         tr_domain_adv, tr_domain_mmd, tr_domain_coral, tr_domain_acc, tr_domain,
         tr_p_recon, tr_pp_recon) = train_one_epoch(
            model, train_loader, optimizer, device,
            accumulate_grad_batches=int(config.get('accumulate_grad_batches', 1)),
            use_mixed_precision=use_amp, scaler=scaler,
            recon_weight=recon_w,
            cls_diff_weight=cls_w,
            cls_diff_clip=loss_cfg.get('cls_diff_clip', 10.0),
            center_loss_fn=center_loss_fn, center_loss_weight=center_loss_weight,
            struct_drop=struct_drop, struct_drop_weight=struct_drop_weight,
            missing_weight=missing_w, present_weight=present_w,
            aux_classifier=aux_classifier,
            aux_criterion=aux_ce,
            backfill_manager=backfill_manager,
            backfill_gate=epoch_backfill_gate,
            current_epoch=epoch,
            domain_weight=domain_w,
            domain_module=domain_module,
            n_p_sources=len(config.get('p_data_files', []) or []),
        )

        # TTL 过期清理
        if backfill_manager is not None:
            backfill_manager.expire(current_epoch=epoch)

        # 定期再验证
        revalidate_interval = int(config.get('backfill_revalidate_interval', 0))
        revalidate_threshold = float(config.get('backfill_revalidate_mse_threshold', 1.0))
        if (backfill_manager is not None and revalidate_interval > 0
                and epoch % revalidate_interval == 0
                and len(backfill_manager.data) > 0):
            logger.info(f"[Backfill] Running revalidation at epoch {epoch}...")
            backfill_manager.revalidate(model, device, mse_threshold=revalidate_threshold)

        val_results = eval_loop(
            model,
            val_loader,
            device,
            return_auc=True,
            backfill_manager=backfill_manager,
            use_completed_input=True,
            cls_diff_weight=cls_w,
            cls_diff_clip=loss_cfg.get('cls_diff_clip', 10.0),
            aux_classifier=aux_classifier if use_aux_for_cls else None,
            missing_weight=missing_w,
            present_weight=present_w
        )
        if len(val_results) == 8:
            val_loss, val_recon, val_in_acc, val_wd, val_miss, val_pres, val_cls, val_auc = val_results
        elif len(val_results) == 7:
            val_loss, val_recon, val_in_acc, val_wd, val_miss, val_pres, val_cls = val_results
            val_auc = 0.0
        else:
            val_loss, val_recon, val_in_acc, val_wd, val_miss, val_pres = val_results
            val_cls = 0.0
            val_auc = 0.0
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}: stage={stage} recon_w={recon_w:.3f} cls_w={cls_w:.3f} domain_w={domain_w:.3f} miss_w={missing_w:.3f} pres_w={present_w:.3f}; "
            f"train_loss={tr_loss:.6f}, train_recon={tr_recon:.6f}, train_miss={tr_miss:.6f}, train_pres={tr_pres:.6f}, train_cls={tr_cls:.6f}, "
            f"domain={tr_domain:.6f}, domain_acc={tr_domain_acc:.4f}, p_recon={tr_p_recon:.6f}, pp_recon={tr_pp_recon:.6f}, "
            f"in_acc={tr_in_acc:.4f}, train_wd={tr_wd:.6f}, center={tr_center:.6f}, cons={tr_cons:.6f}; "
            f"val_loss={val_loss:.6f}, val_recon={val_recon:.6f}, val_miss={val_miss:.6f}, val_pres={val_pres:.6f}, val_cls={val_cls:.6f}, val_completed_acc={val_in_acc:.4f}, val_auc={val_auc:.4f}, val_wd={val_wd:.6f}; lr={cur_lr:.6e}")
        if backfill_manager is not None:
            stats = backfill_manager.last_epoch_stats
            if gate_active:
                logger.info(f"[Backfill] accepted={stats.get('accepted', 0)}/{stats.get('total', 0)}")
            else:
                logger.info("[Backfill] skipped (gate disabled this epoch)")
            if dynamic_interval and gate_active:
                total = stats.get('total', 0)
                if total > 0:
                    rate = stats.get('accepted', 0) / total
                    new_interval = gate_interval_current
                    if rate < acc_rate_low:
                        new_interval = min(interval_max, gate_interval_current + interval_step)
                    elif rate > acc_rate_high:
                        new_interval = max(interval_min, gate_interval_current - interval_step)
                    if new_interval != gate_interval_current:
                        logger.info(f"[Backfill] interval update: {gate_interval_current} -> {new_interval} (rate={rate:.6f})")
                        gate_interval_current = new_interval

        # TensorBoard logging
        writer.add_scalar('Train/Loss', tr_loss, epoch)
        writer.add_scalar('Train/Recon_Loss', tr_recon, epoch)
        writer.add_scalar('Train/Missing_Recon', tr_miss, epoch)
        writer.add_scalar('Train/Present_Recon', tr_pres, epoch)
        writer.add_scalar('Train/Input_Acc', tr_in_acc, epoch)
        writer.add_scalar('Train/Wasserstein', tr_wd, epoch)
        writer.add_scalar('Train/Total_Loss', tr_loss, epoch)
        writer.add_scalar('Train/Center_Loss', tr_center, epoch)
        writer.add_scalar('Train/Consistency_Loss', tr_cons, epoch)
        writer.add_scalar('Train/Domain_Loss', tr_domain, epoch)
        writer.add_scalar('Train/Domain_Adversarial', tr_domain_adv, epoch)
        writer.add_scalar('Train/Domain_MMD', tr_domain_mmd, epoch)
        writer.add_scalar('Train/Domain_CORAL', tr_domain_coral, epoch)
        writer.add_scalar('Train/Domain_Acc', tr_domain_acc, epoch)
        writer.add_scalar('Train/P_Recon', tr_p_recon, epoch)
        writer.add_scalar('Train/PP_Recon', tr_pp_recon, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Recon_Loss', val_recon, epoch)
        writer.add_scalar('Val/Missing_Recon', val_miss, epoch)
        writer.add_scalar('Val/Present_Recon', val_pres, epoch)
        writer.add_scalar('Val/Completed_Acc', val_in_acc, epoch)
        writer.add_scalar('Val/Cls_Loss', val_cls, epoch)
        writer.add_scalar('Val/AUC', val_auc, epoch)
        writer.add_scalar('Val/Wasserstein', val_wd, epoch)
        writer.add_scalar('Val/Generated_MSE', val_recon, epoch)
        # Cross-stats removed
        writer.add_scalar('LR', cur_lr, epoch)

        # Checkpoint & early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'domain_module_state_dict': domain_module.state_dict() if domain_module is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'in_channels': in_channels
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, ckpt_path)
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                       os.path.join(ckpt_dir, 'best_model_weights_only.pth'))
            logger.info(f"保存最佳模型到 {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发于第 {epoch} 轮")
                break

        if epoch % int(config.get('save_interval', 10)) == 0:
            periodic = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'domain_module_state_dict': domain_module.state_dict() if domain_module is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            p_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(periodic, p_path)
        last_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'domain_module_state_dict': domain_module.state_dict() if domain_module is not None else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        if scaler is not None:
            last_ckpt['scaler_state_dict'] = scaler.state_dict()
        torch.save(last_ckpt, last_ckpt_path)

    logger.info("开始测试评估...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        w_path = os.path.join(ckpt_dir, 'best_model_weights_only.pth')
        if os.path.exists(w_path):
            model.load_state_dict(torch.load(w_path, map_location=device))

    test_results = eval_loop(
        model,
        test_loader,
        device,
        return_auc=True,
        backfill_manager=backfill_manager,
        use_completed_input=True,
        cls_diff_weight=cls_w_global,
        cls_diff_clip=loss_cfg.get('cls_diff_clip', 10.0),
        aux_classifier=aux_classifier if use_aux_for_cls else None,
        missing_weight=missing_w,
        present_weight=present_w
    )
    if len(test_results) == 8:
        test_loss, test_recon, test_in_acc, test_wd, test_miss, test_pres, test_cls, test_auc = test_results
    elif len(test_results) == 7:
        test_loss, test_recon, test_in_acc, test_wd, test_miss, test_pres, test_cls = test_results
        test_auc = 0.0
    else:
        test_loss, test_recon, test_in_acc, test_wd, test_miss, test_pres = test_results
        test_cls = 0.0
        test_auc = 0.0
    logger.info(
        f"测试: loss={test_loss:.6f}, recon={test_recon:.6f}, cls={test_cls:.6f}, completed_acc={test_in_acc:.4f}, auc={test_auc:.4f}, wd={test_wd:.6f}")
    writer.add_scalar('Test/Loss', test_loss)
    writer.add_scalar('Test/Recon_Loss', test_recon)
    writer.add_scalar('Test/Missing_Recon', test_miss)
    writer.add_scalar('Test/Present_Recon', test_pres)
    writer.add_scalar('Test/Completed_Acc', test_in_acc)
    writer.add_scalar('Test/Cls_Loss', test_cls)
    writer.add_scalar('Test/AUC', test_auc)
    writer.add_scalar('Test/Wasserstein', test_wd)
    writer.add_scalar('Test/Generated_MSE', test_recon)

    # Cross-stats removed
    writer.close()


if __name__ == '__main__':
    main()
