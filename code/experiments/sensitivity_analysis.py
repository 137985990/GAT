#!/usr/bin/env python
import argparse
import copy
import os
import sys
import subprocess
import yaml

from data import load_config
from train import unify_config


def _tag_float(val: float) -> str:
    return str(val).replace('.', 'p')

def _parse_list(raw: str):
    return [float(x) for x in raw.split(',') if x.strip() != '']


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for noise and beta")
    parser.add_argument('--config', type=str, default='config.yaml', help='Base config.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per run')
    parser.add_argument('--only_noise', action='store_true', help='Sweep noise only')
    parser.add_argument('--only_beta', action='store_true', help='Sweep beta only')
    parser.add_argument('--beta', type=float, default=0.5, help='Fixed beta when sweeping noise')
    parser.add_argument('--noise', type=float, default=0.0, help='Fixed noise when sweeping beta')
    parser.add_argument('--beta_list', type=str, default=None, help='Comma-separated beta list override')
    parser.add_argument('--noise_list', type=str, default=None, help='Comma-separated noise list override')
    parser.add_argument('--alpha_list', type=str, default=None, help='Comma-separated recon_weight list')
    parser.add_argument('--no_early_stop', action='store_true', help='Disable early stopping for main training')
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    base_cfg = unify_config(raw_cfg)

    noise_list = [0.0, 0.01, 0.05, 0.1]
    beta_list = [0.1, 0.2, 0.5, 1.0]
    if args.noise_list:
        noise_list = _parse_list(args.noise_list)
    if args.beta_list:
        beta_list = _parse_list(args.beta_list)
    alpha_list = None
    if args.alpha_list:
        alpha_list = _parse_list(args.alpha_list)
    if args.only_noise and args.only_beta:
        raise ValueError("Use only one of --only_noise or --only_beta")
    if args.only_noise:
        beta_list = [float(args.beta)]
    if args.only_beta:
        noise_list = [float(args.noise)]

    out_cfg_dir = os.path.join(base_cfg.get('log_dir', 'Logs'), 'sensitivity_configs')
    os.makedirs(out_cfg_dir, exist_ok=True)

    for noise_sigma in noise_list:
        for beta in beta_list:
            run_alpha_list = alpha_list if alpha_list is not None else [None]
            for alpha in run_alpha_list:
                cfg = copy.deepcopy(raw_cfg)
                cfg['noise_sigma'] = float(noise_sigma)
                cfg.setdefault('loss_config', {})
                cfg['loss_config']['cls_diff_weight'] = float(beta)
                if alpha is not None:
                    cfg['loss_config']['recon_weight'] = float(alpha)
                cfg['epochs'] = int(args.epochs)
                cfg['resume_training'] = False
                if args.no_early_stop:
                    cfg['early_stopping_patience'] = int(args.epochs) + 1

                tag = f"noise{_tag_float(noise_sigma)}_beta{_tag_float(beta)}"
                if alpha is not None:
                    tag += f"_alpha{_tag_float(float(alpha))}"
                cfg['log_dir'] = os.path.join('Logs', f'sens_{tag}')
                cfg['tensorboard_dir'] = os.path.join('runs', f'sens_{tag}')
                cfg['checkpoint_dir'] = os.path.join('Checkpoints', f'sens_{tag}')

                cfg_path = os.path.join(out_cfg_dir, f'config_{tag}.yaml')
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

                cmd = [sys.executable, 'train.py', '--config', cfg_path]
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
