# unify_eeg.py
"""
EEG 频带统一脚本

将不同数据集的 EEG 数据统一成标准的 5 频带格式：
- eeg_delta (0.5-4 Hz)
- eeg_theta (4-8 Hz)
- eeg_alpha (8-13 Hz)
- eeg_beta  (13-30 Hz)
- eeg_gamma (30+ Hz)

支持的数据集：
1. FM:    4电极 × 5频带 → 取平均
2. MEFAR: 单通道 × 8频带(细分) → 合并子频带
3. DROZY: 原始EEG × 5电极 → FFT计算 → 取平均
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch
from typing import Optional


# ============== FM 处理 ==============
def unify_fm_eeg(df: pd.DataFrame) -> pd.DataFrame:
    """
    FM 数据集: 4电极(tp9, af7, af8, tp10) × 5频带
    统一为 5 个平均频带功率
    """
    df = df.copy()
    
    electrodes = ['tp9', 'af7', 'af8', 'tp10']
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    
    for band in bands:
        cols = [f'{band}_{elec}' for elec in electrodes]
        # 检查列是否存在
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            df[f'eeg_{band}'] = df[existing_cols].mean(axis=1)
        else:
            df[f'eeg_{band}'] = 0.0
    
    print(f"[FM] Unified EEG: {len(df)} samples, bands: {[f'eeg_{b}' for b in bands]}")
    return df


# ============== MEFAR 处理 ==============
def unify_mefar_eeg(df: pd.DataFrame) -> pd.DataFrame:
    """
    MEFAR 数据集: 单通道 × 8频带(细分)
    - delta, theta
    - alpha1, alpha2 → 合并为 alpha
    - beta1, beta2 → 合并为 beta
    - gamma1, gamma2 → 合并为 gamma
    """
    df = df.copy()
    
    # Delta 和 Theta 直接使用
    df['eeg_delta'] = df['delta'] if 'delta' in df.columns else 0.0
    df['eeg_theta'] = df['theta'] if 'theta' in df.columns else 0.0
    
    # Alpha: 合并 alpha1 和 alpha2
    alpha_cols = [c for c in ['alpha1', 'alpha2'] if c in df.columns]
    df['eeg_alpha'] = df[alpha_cols].mean(axis=1) if alpha_cols else 0.0
    
    # Beta: 合并 beta1 和 beta2
    beta_cols = [c for c in ['beta1', 'beta2'] if c in df.columns]
    df['eeg_beta'] = df[beta_cols].mean(axis=1) if beta_cols else 0.0
    
    # Gamma: 合并 gamma1 和 gamma2
    gamma_cols = [c for c in ['gamma1', 'gamma2'] if c in df.columns]
    df['eeg_gamma'] = df[gamma_cols].mean(axis=1) if gamma_cols else 0.0
    
    print(f"[MEFAR] Unified EEG: {len(df)} samples")
    return df


# ============== DROZY 处理 ==============
def compute_band_power(signal: np.ndarray, fs: int = 32, 
                       band: tuple = (8, 13),
                       scale_factor: float = 1e6) -> float:
    """
    计算信号在指定频带的平均功率
    
    Args:
        signal: 1D 信号数组
        fs: 采样率
        band: (low_freq, high_freq) 频带范围
        scale_factor: 信号缩放因子（EDF微伏数据需要放大）
    
    Returns:
        对数功率（更稳定）
    """
    if len(signal) < fs:  # 太短的信号
        return 0.0
    
    try:
        # 缩放信号（微伏级别 → 标准单位）
        signal = signal * scale_factor
        
        # Welch 方法计算 PSD
        nperseg = min(len(signal), fs * 2)  # 2秒窗口
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        
        # 提取频带功率
        idx = (freqs >= band[0]) & (freqs < band[1])
        if idx.sum() == 0:
            return 0.0
        
        power = np.mean(psd[idx])
        
        # 对数变换（EEG 分析常用）
        if power > 0:
            return float(np.log10(power + 1e-10))
        return 0.0
    except Exception:
        return 0.0


def unify_drozy_eeg(df: pd.DataFrame, fs: int = 32, 
                    window_size: int = 320) -> pd.DataFrame:
    """
    DROZY 数据集: 原始 EEG 波形 → FFT → 5 频带功率
    
    电极: Fz, Cz, C3, C4, Pz
    
    注意: 这个函数对整个 DataFrame 逐窗口计算频带功率
    为了效率，我们对每个 block 的数据进行处理
    """
    df = df.copy()
    
    electrodes = ['eeg_fz', 'eeg_cz', 'eeg_c3', 'eeg_c4', 'eeg_pz']
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, min(50, fs/2 - 1))  # 受采样率限制
    }
    
    # 检查是否有原始 EEG 列
    existing_elec = [e for e in electrodes if e in df.columns]
    if not existing_elec:
        print(f"[DROZY] Warning: No raw EEG columns found, setting to 0")
        for band in bands:
            df[f'eeg_{band}'] = 0.0
        return df
    
    # 初始化输出列
    for band in bands:
        df[f'eeg_{band}'] = 0.0
    
    # 按 block 处理（提高效率）
    if 'block' in df.columns:
        blocks = df['block'].unique()
        print(f"[DROZY] Processing {len(blocks)} blocks...")
        
        for block_id in blocks:
            mask = df['block'] == block_id
            block_df = df.loc[mask]
            
            # 对每个频带计算功率
            for band_name, (low, high) in bands.items():
                powers = []
                for elec in existing_elec:
                    signal = block_df[elec].values.astype(float)
                    # 滑动窗口计算（简化：整段计算一次，然后广播）
                    power = compute_band_power(signal, fs, (low, high))
                    powers.append(power)
                
                # 取所有电极的平均
                mean_power = np.mean(powers) if powers else 0.0
                df.loc[mask, f'eeg_{band_name}'] = mean_power
    else:
        # 没有 block 信息，整体计算
        for band_name, (low, high) in bands.items():
            powers = []
            for elec in existing_elec:
                signal = df[elec].values.astype(float)
                power = compute_band_power(signal, fs, (low, high))
                powers.append(power)
            df[f'eeg_{band_name}'] = np.mean(powers) if powers else 0.0
    
    print(f"[DROZY] Unified EEG: {len(df)} samples")
    return df


# ============== 主处理函数 ==============
def unify_dataset_eeg(input_path: str, output_path: str, 
                      dataset_type: str, fs: int = 32) -> pd.DataFrame:
    """
    统一单个数据集的 EEG 格式
    
    Args:
        input_path: 输入 CSV 路径
        output_path: 输出 CSV 路径
        dataset_type: 'FM', 'MEFAR', 'DROZY', 'OD'
        fs: 采样率
    
    Returns:
        处理后的 DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_type}: {input_path}")
    print('='*60)
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if dataset_type.upper() == 'FM':
        df = unify_fm_eeg(df)
    elif dataset_type.upper() == 'MEFAR':
        df = unify_mefar_eeg(df)
    elif dataset_type.upper() == 'DROZY':
        df = unify_drozy_eeg(df, fs=fs)
    elif dataset_type.upper() == 'OD':
        # OD 没有 EEG，添加空列
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            df[f'eeg_{band}'] = 0.0
        print(f"[OD] No EEG data, added zero columns")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # 保存
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # 统计
    eeg_cols = [f'eeg_{b}' for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
    print(f"\nEEG statistics:")
    for col in eeg_cols:
        if col in df.columns:
            vals = df[col]
            print(f"  {col}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
                  f"min={vals.min():.6f}, max={vals.max():.6f}")
    
    return df


def process_all_datasets(data_dir: str, fs: int = 32):
    """
    处理所有数据集
    """
    data_dir = Path(data_dir)
    
    datasets = [
        ('FM_original.csv', 'FM_unified.csv', 'FM'),
        ('MEFAR_original.csv', 'MEFAR_unified.csv', 'MEFAR'),
        ('OD_original.csv', 'OD_unified.csv', 'OD'),
        ('DROZY_original.csv', 'DROZY_unified.csv', 'DROZY'),
    ]
    
    results = {}
    for input_name, output_name, dtype in datasets:
        input_path = data_dir / input_name
        output_path = data_dir / output_name
        
        if not input_path.exists():
            print(f"Skipping {input_name}: file not found")
            continue
        
        try:
            df = unify_dataset_eeg(str(input_path), str(output_path), dtype, fs)
            results[dtype] = df
        except Exception as e:
            print(f"Error processing {dtype}: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*60)
    print("SUMMARY: Unified EEG Features")
    print("="*60)
    print(f"{'Dataset':<10} {'Samples':<12} {'EEG Present':<15}")
    print("-"*40)
    for dtype, df in results.items():
        eeg_cols = [c for c in df.columns if c.startswith('eeg_')]
        has_eeg = any(df[c].abs().sum() > 0 for c in eeg_cols)
        print(f"{dtype:<10} {len(df):<12} {'Yes' if has_eeg else 'No (zeros)':<15}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Unify EEG features across datasets')
    parser.add_argument('--data_dir', type=str, default='../Data',
                        help='Data directory')
    parser.add_argument('--fs', type=int, default=32,
                        help='Sampling rate (Hz)')
    parser.add_argument('--single', type=str, default=None,
                        help='Process single dataset: FM, MEFAR, OD, or DROZY')
    
    args = parser.parse_args()
    
    if args.single:
        data_dir = Path(args.data_dir)
        mapping = {
            'FM': ('FM_original.csv', 'FM_unified.csv'),
            'MEFAR': ('MEFAR_original.csv', 'MEFAR_unified.csv'),
            'OD': ('OD_original.csv', 'OD_unified.csv'),
            'DROZY': ('DROZY_original.csv', 'DROZY_unified.csv'),
        }
        if args.single.upper() in mapping:
            inp, out = mapping[args.single.upper()]
            unify_dataset_eeg(str(data_dir / inp), str(data_dir / out), 
                             args.single.upper(), args.fs)
    else:
        process_all_datasets(args.data_dir, args.fs)
