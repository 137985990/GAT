#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
import torch

from data import create_multimodal_dataset_from_config, load_config
from model import TGATUNet


def _normalize_columns(df):
    new_cols = []
    for col in df.columns:
        if col == 'F':
            new_cols.append(col)
        elif str(col).lower() in ['block', 'id', 'session']:
            new_cols.append(str(col).lower())
        else:
            new_cols.append(str(col).strip().lower())
    df.columns = new_cols
    return df


def export_completed_datasets(model, dataset, config, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = list(dataset.feature_cols)
    block_col = dataset.block_col
    window_size = int(dataset.window_size)
    step_size = int(dataset.step_size)
    sampling_rate = int(dataset.sampling_rate)
    normalize = dataset.normalize

    data_files = config.get('data_files', [])
    data_dir = config.get('data_dir', '')
    file_map = {}
    for f in data_files:
        path = os.path.join(data_dir, f) if not os.path.isabs(f) and not os.path.exists(f) else f
        fname = os.path.basename(path).lower()
        if 'fm' in fname:
            file_map['FM'] = path
        elif 'od' in fname:
            file_map['OD'] = path
        elif 'mefar' in fname:
            file_map['MEFAR'] = path

    model.eval()
    for source in sorted(file_map.keys()):
        print(f"[Export] start source={source}")
        df_src = pd.read_csv(file_map[source])
        df_src = _normalize_columns(df_src)
        orig_cols = list(df_src.columns)

        for col in feature_cols:
            if col not in df_src.columns:
                df_src[col] = 0.0
            df_src[col] = pd.to_numeric(df_src[col], errors='coerce').fillna(0.0)

        n_rows = len(df_src)
        if n_rows == 0:
            continue

        present_mask = dataset.source_feature_presence.get(
            source, torch.ones(len(feature_cols), dtype=torch.float32)
        )
        missing_cols = (1.0 - present_mask).numpy().astype(bool)

        recon_sum = np.zeros((n_rows, len(feature_cols)), dtype=np.float32)
        recon_count = np.zeros((n_rows, len(feature_cols)), dtype=np.int32)

        if block_col in df_src.columns:
            block_groups = df_src.groupby(block_col, sort=False)
        else:
            block_groups = [(0, df_src)]

        for _, block_df in block_groups:
            values = block_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            n_block = values.shape[0]
            if n_block < window_size:
                continue
            for start in range(0, n_block - window_size + 1, step_size):
                window = values[start:start + window_size]
                window = window[::sampling_rate]
                if normalize == 'zscore':
                    mean = window.mean(axis=0, keepdims=True)
                    std = window.std(axis=0, keepdims=True)
                    window_in = (window - mean) / (std + 1e-6)
                elif normalize == 'minmax':
                    min_v = window.min(axis=0, keepdims=True)
                    max_v = window.max(axis=0, keepdims=True)
                    window_in = (window - min_v) / (max_v - min_v + 1e-6)
                else:
                    window_in = window

                with torch.no_grad():
                    inp = torch.from_numpy(window_in).float().to(device)
                    recon, _, _ = model(inp, return_latent=True)
                recon = recon.detach().cpu().numpy().T

                if normalize == 'zscore':
                    recon = recon * (std + 1e-6) + mean
                elif normalize == 'minmax':
                    recon = recon * (max_v - min_v + 1e-6) + min_v

                for t in range(recon.shape[0]):
                    row_pos = start + (t * sampling_rate)
                    if row_pos >= n_block:
                        break
                    end_pos = min(row_pos + sampling_rate, n_block)
                    for rp in range(row_pos, end_pos):
                        row_idx = block_df.index[rp]
                        recon_sum[row_idx, missing_cols] += recon[t, missing_cols]
                        recon_count[row_idx, missing_cols] += 1

        for j, col in enumerate(feature_cols):
            if not missing_cols[j]:
                continue
            counts = recon_count[:, j]
            filled = recon_sum[:, j]
            filled = np.where(counts > 0, filled / np.maximum(counts, 1), df_src[col].to_numpy())
            if col in df_src.columns:
                series = df_src[col].to_numpy()
                if np.isnan(series).any():
                    series = np.where(np.isnan(series), filled, series)
                else:
                    series = filled
                df_src[col] = series
            else:
                df_src[col] = filled

        out_cols = list(orig_cols)
        for col in feature_cols:
            if col not in out_cols:
                out_cols.append(col)
        df_src = df_src.loc[:, out_cols]

        out_path = os.path.join(out_dir, f"{source}_completed.csv")
        df_src.to_csv(out_path, index=False)
        print(f"[Export] saved {out_path} rows={len(df_src)}")


def main():
    parser = argparse.ArgumentParser(description='Export completed datasets')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint (default: best_model.pth)')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory (default: data_dir)')
    parser.add_argument('--source', type=str, default='', help='Only export one source (FM/OD/MEFAR)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = create_multimodal_dataset_from_config(cfg, phase='encode')
    in_channels = len(dataset.feature_cols)

    from model import TGATUNet
    model = TGATUNet(
        in_channels=in_channels,
        hidden_channels=cfg.get('hidden_channels', 64),
        out_channels=in_channels,
        num_classes=cfg.get('num_classes', 2),
        encoder_layers=cfg.get('encoder_layers', 3),
        decoder_layers=cfg.get('decoder_layers', 3),
        heads=cfg.get('gat_heads', 4),
        time_k=cfg.get('time_k', 1),
        trans_nhead=cfg.get('trans_nhead', 4),
        trans_layers=cfg.get('trans_layers', 2),
        trans_dim_feedforward=cfg.get('trans_dim_feedforward', 512),
        sanitize_nans=True,
        clamp_value=1e4,
        simple_encoder_fallback=False,
        disable_model_layernorm=cfg.get('disable_model_layernorm', False),
        disable_transformer=False,
        noise_sigma=float(cfg.get('noise_sigma', 0.0))
    ).to(device)

    ckpt_dir = cfg.get('checkpoint_dir', 'Checkpoints')
    ckpt_path = args.checkpoint.strip() or os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")

    out_dir = args.out_dir.strip() or cfg.get('data_dir', 'Data')
    if args.source:
        cfg = dict(cfg)
        cfg['data_files'] = [p for p in cfg.get('data_files', []) if args.source.lower() in os.path.basename(p).lower()]
    export_completed_datasets(model, dataset, cfg, device, out_dir)


if __name__ == '__main__':
    main()
