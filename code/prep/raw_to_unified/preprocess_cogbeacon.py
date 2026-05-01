# preprocess_cogbeacon.py
"""
CogBeacon 数据集预处理脚本
将 CogBeacon 的 EEG 和疲劳自报数据转换为与 FM/OD/MEFAR/DROZY/WESAD 兼容的统一 CSV 格式

CogBeacon 数据集包含:
- EEG 数据: Muse 2 头带 (4 电极: TP9, AF7, AF8, TP10, 采样率 ~220 Hz)
  - 原始 EEG 值
  - 频带功率: alpha, beta, delta, gamma, theta
  - 归一化频带功率 (As, Bs, Ds, Gs, Ts)
  - 绝对频带功率 (Aa, Ab, Ad, Ag, At)
  - 连接质量 (h, c)
- 疲劳自报: 每个 round 累计按钮次数 (0 或递增的数字)
- 用户表现: Score, Response, Time 等

疲劳标签策略:
- 方案1: 使用疲劳自报的变化率 (按钮次数增加 = 疲劳增加)
- 方案2: 使用时间进展 (会话后期 = 更疲劳)
- 方案3: 结合表现下降 (Score 下降 = 疲劳)

数据结构:
- eeg/user_X_v_m/  每个文件夹是一个会话
  - 0_1, 0_2, ...  每个文件是一个 round
- fatigue_self_report/user_X_v_m.csv  疲劳自报
- user_performance/user_X_v_m_XX.XX.csv  用户表现
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional, List
import warnings
import re

# 目标采样率 (与其他数据集一致)
TARGET_SR = 32

# Muse 2 采样率 (约 220 Hz，但实际可能有变化)
MUSE_SR = 220


def resample_signal(data: np.ndarray, original_sr: float, target_sr: float) -> np.ndarray:
    """
    重采样信号到目标采样率
    
    Args:
        data: 原始信号 (N, C) 或 (N,)
        original_sr: 原始采样率
        target_sr: 目标采样率
    
    Returns:
        重采样后的信号
    """
    if original_sr == target_sr:
        return data
    
    if len(data) == 0:
        return np.array([])
    
    # 计算新长度
    original_len = data.shape[0]
    new_len = int(original_len * target_sr / original_sr)
    
    if new_len <= 0:
        return np.array([])
    
    # 处理多维数据
    if data.ndim == 1:
        return scipy_signal.resample(data, new_len)
    else:
        # 对每个通道分别重采样
        resampled = np.zeros((new_len, data.shape[1]))
        for i in range(data.shape[1]):
            resampled[:, i] = scipy_signal.resample(data[:, i], new_len)
        return resampled


def parse_eeg_file(file_path: str) -> Dict[str, np.ndarray]:
    """
    解析单个 EEG round 文件
    
    文件格式:
    eeg 819.20 815.91 838.94 832.36   # 原始 EEG (4 电极)
    a 0.265 0.087 0.063 0.293         # Alpha 频带功率
    b 0.055 0.124 0.080 0.079         # Beta 频带功率
    d 0.493 0.658 0.631 0.382         # Delta 频带功率
    g 0.037 0.072 0.073 0.068         # Gamma 频带功率
    t 0.147 0.058 0.150 0.176         # Theta 频带功率
    h 1.0 1.0 1.0 1.0                 # 头带连接质量 (每电极)
    c 1.0                              # 整体连接质量
    Aa/Ab/Ad/Ag/At                     # 绝对频带功率
    as/bs/ds/gs/ts                     # 归一化频带功率 (分数)
    
    Returns:
        dict with keys: eeg, alpha, beta, delta, gamma, theta, etc.
    """
    result = {
        'eeg': [],      # 原始 EEG, shape (N, 4)
        'alpha': [],    # 相对 alpha 功率, shape (N, 4)
        'beta': [],     # 相对 beta 功率, shape (N, 4)
        'delta': [],    # 相对 delta 功率, shape (N, 4)
        'gamma': [],    # 相对 gamma 功率, shape (N, 4)
        'theta': [],    # 相对 theta 功率, shape (N, 4)
        'horseshoe': [],  # 连接质量, shape (N, 4)
        'connection': [],  # 整体连接, shape (N,)
    }
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                key = parts[0]
                try:
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                
                # 映射到我们的键
                if key == 'eeg' and len(values) == 4:
                    result['eeg'].append(values)
                elif key == 'a' and len(values) == 4:
                    result['alpha'].append(values)
                elif key == 'b' and len(values) == 4:
                    result['beta'].append(values)
                elif key == 'd' and len(values) == 4:
                    result['delta'].append(values)
                elif key == 'g' and len(values) == 4:
                    result['gamma'].append(values)
                elif key == 't' and len(values) == 4:
                    result['theta'].append(values)
                elif key == 'h' and len(values) == 4:
                    result['horseshoe'].append(values)
                elif key == 'c' and len(values) >= 1:
                    result['connection'].append(values[0])
                # 忽略 Aa, Ab, Ad, Ag, At, as, bs, ds, gs, ts
    
    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}")
        return result
    
    # 转换为 numpy 数组
    for key in result:
        if result[key]:
            result[key] = np.array(result[key])
        else:
            result[key] = np.array([])
    
    return result


def parse_fatigue_csv(csv_path: str) -> List[int]:
    """
    解析疲劳自报 CSV 文件
    
    文件格式: 每行一个数字，表示累计按钮次数
    
    Returns:
        list of cumulative button presses per round
    """
    fatigue_values = []
    
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        fatigue_values.append(int(float(line)))
                    except ValueError:
                        fatigue_values.append(0)
    except Exception as e:
        print(f"Warning: Failed to parse {csv_path}: {e}")
    
    return fatigue_values


def compute_fatigue_labels(fatigue_cumulative: List[int], 
                           n_rounds: int,
                           method: str = 'cumulative') -> np.ndarray:
    """
    从累计疲劳按钮次数计算二元疲劳标签
    
    Args:
        fatigue_cumulative: 累计按钮次数列表 (每个 round)
        n_rounds: 总 round 数
        method: 
            'cumulative' - 基于累计次数的中位数分割
            'time' - 基于时间进展 (前半 = 清醒, 后半 = 疲劳)
            'change' - 基于变化率 (按钮次数增加 = 疲劳开始)
    
    Returns:
        每个 round 的疲劳标签 (0 = 清醒, 1 = 疲劳)
    """
    if not fatigue_cumulative:
        # 如果没有疲劳数据，使用时间方法
        method = 'time'
    
    labels = np.zeros(n_rounds, dtype=int)
    
    if method == 'cumulative':
        # 使用累计次数的中位数作为阈值
        if len(fatigue_cumulative) >= n_rounds:
            values = fatigue_cumulative[:n_rounds]
        else:
            # 填充到 n_rounds
            values = fatigue_cumulative + [fatigue_cumulative[-1] if fatigue_cumulative else 0] * (n_rounds - len(fatigue_cumulative))
        
        max_val = max(values) if values else 0
        if max_val > 0:
            # 如果有按钮按压，使用中位数分割
            median = np.median(values)
            labels = np.array([1 if v >= median and v > 0 else 0 for v in values])
        else:
            # 如果全部为 0，使用时间方法
            half = n_rounds // 2
            labels[half:] = 1
    
    elif method == 'time':
        # 简单的时间分割: 后半部分 = 疲劳
        half = n_rounds // 2
        labels[half:] = 1
    
    elif method == 'change':
        # 基于变化率: 第一次按钮增加后 = 疲劳开始
        if len(fatigue_cumulative) >= 2:
            for i in range(1, min(len(fatigue_cumulative), n_rounds)):
                if fatigue_cumulative[i] > fatigue_cumulative[i-1]:
                    labels[i:] = 1
                    break
            else:
                # 没有变化，使用时间方法
                half = n_rounds // 2
                labels[half:] = 1
        else:
            half = n_rounds // 2
            labels[half:] = 1
    
    return labels


def process_single_session(eeg_dir: str, fatigue_csv: str, 
                          session_name: str, user_id: int) -> Optional[pd.DataFrame]:
    """
    处理单个会话的数据
    
    Args:
        eeg_dir: EEG 数据目录 (包含 0_1, 0_2, ... 文件)
        fatigue_csv: 疲劳自报 CSV 路径
        session_name: 会话名称 (如 'user_0_v_m')
        user_id: 用户 ID (数字)
    
    Returns:
        处理后的 DataFrame
    """
    print(f"  Processing session: {session_name}")
    
    eeg_dir = Path(eeg_dir)
    
    # 获取所有 round 文件
    round_files = sorted([f for f in eeg_dir.iterdir() if f.is_file()], 
                        key=lambda x: (int(x.name.split('_')[0]) if '_' in x.name else 0,
                                      int(x.name.split('_')[1]) if '_' in x.name else 0))
    
    if not round_files:
        print(f"    No round files found in {eeg_dir}")
        return None
    
    print(f"    Found {len(round_files)} rounds")
    
    # 解析疲劳数据
    fatigue_cumulative = parse_fatigue_csv(fatigue_csv) if os.path.exists(fatigue_csv) else []
    print(f"    Fatigue data: {len(fatigue_cumulative)} values, max={max(fatigue_cumulative) if fatigue_cumulative else 0}")
    
    # 计算每个 round 的疲劳标签
    fatigue_labels = compute_fatigue_labels(fatigue_cumulative, len(round_files), method='cumulative')
    
    all_round_dfs = []
    
    for round_idx, round_file in enumerate(round_files):
        # 解析 EEG 数据
        eeg_data = parse_eeg_file(str(round_file))
        
        if len(eeg_data['alpha']) == 0:
            # 如果没有频带数据，跳过这个 round
            continue
        
        # 使用频带功率数据 (比原始 EEG 更可靠)
        n_samples = len(eeg_data['alpha'])
        
        # 获取这个 round 的疲劳标签
        round_fatigue = fatigue_labels[round_idx] if round_idx < len(fatigue_labels) else 0
        
        # 创建 DataFrame
        round_df = pd.DataFrame({
            # 加速度 (CogBeacon 没有，设为 0)
            'acc_x': np.zeros(n_samples),
            'acc_y': np.zeros(n_samples),
            'acc_z': np.zeros(n_samples),
            
            # PPG/BVP (CogBeacon 没有)
            'ppg': np.zeros(n_samples),
            
            # GSR/EDA (CogBeacon 没有)
            'gsr': np.zeros(n_samples),
            
            # HR (CogBeacon 没有)
            'hr': np.zeros(n_samples),
            
            # 皮肤温度 (CogBeacon 没有)
            'skt': np.zeros(n_samples),
            
            # ECG (CogBeacon 没有)
            'ecg': np.zeros(n_samples),
            
            # 呼吸 (CogBeacon 没有)
            'breathing': np.zeros(n_samples),
            
            # EMG (CogBeacon 没有)
            'emg': np.zeros(n_samples),
            
            # EEG 频带功率 (平均 4 电极)
            'eeg_alpha': np.mean(eeg_data['alpha'], axis=1) if len(eeg_data['alpha']) > 0 else np.zeros(n_samples),
            'eeg_beta': np.mean(eeg_data['beta'], axis=1) if len(eeg_data['beta']) > 0 else np.zeros(n_samples),
            'eeg_delta': np.mean(eeg_data['delta'], axis=1) if len(eeg_data['delta']) > 0 else np.zeros(n_samples),
            'eeg_gamma': np.mean(eeg_data['gamma'], axis=1) if len(eeg_data['gamma']) > 0 else np.zeros(n_samples),
            'eeg_theta': np.mean(eeg_data['theta'], axis=1) if len(eeg_data['theta']) > 0 else np.zeros(n_samples),
            
            # 详细 EEG 通道 (与 FM 格式兼容)
            # Muse 2 电极: TP9, AF7, AF8, TP10 (对应列 0, 1, 2, 3)
            'alpha_tp9': eeg_data['alpha'][:, 0] if len(eeg_data['alpha']) > 0 else np.zeros(n_samples),
            'alpha_af7': eeg_data['alpha'][:, 1] if len(eeg_data['alpha']) > 0 else np.zeros(n_samples),
            'alpha_af8': eeg_data['alpha'][:, 2] if len(eeg_data['alpha']) > 0 else np.zeros(n_samples),
            'alpha_tp10': eeg_data['alpha'][:, 3] if len(eeg_data['alpha']) > 0 else np.zeros(n_samples),
            
            'beta_tp9': eeg_data['beta'][:, 0] if len(eeg_data['beta']) > 0 else np.zeros(n_samples),
            'beta_af7': eeg_data['beta'][:, 1] if len(eeg_data['beta']) > 0 else np.zeros(n_samples),
            'beta_af8': eeg_data['beta'][:, 2] if len(eeg_data['beta']) > 0 else np.zeros(n_samples),
            'beta_tp10': eeg_data['beta'][:, 3] if len(eeg_data['beta']) > 0 else np.zeros(n_samples),
            
            'delta_tp9': eeg_data['delta'][:, 0] if len(eeg_data['delta']) > 0 else np.zeros(n_samples),
            'delta_af7': eeg_data['delta'][:, 1] if len(eeg_data['delta']) > 0 else np.zeros(n_samples),
            'delta_af8': eeg_data['delta'][:, 2] if len(eeg_data['delta']) > 0 else np.zeros(n_samples),
            'delta_tp10': eeg_data['delta'][:, 3] if len(eeg_data['delta']) > 0 else np.zeros(n_samples),
            
            'gamma_tp9': eeg_data['gamma'][:, 0] if len(eeg_data['gamma']) > 0 else np.zeros(n_samples),
            'gamma_af7': eeg_data['gamma'][:, 1] if len(eeg_data['gamma']) > 0 else np.zeros(n_samples),
            'gamma_af8': eeg_data['gamma'][:, 2] if len(eeg_data['gamma']) > 0 else np.zeros(n_samples),
            'gamma_tp10': eeg_data['gamma'][:, 3] if len(eeg_data['gamma']) > 0 else np.zeros(n_samples),
            
            'theta_tp9': eeg_data['theta'][:, 0] if len(eeg_data['theta']) > 0 else np.zeros(n_samples),
            'theta_af7': eeg_data['theta'][:, 1] if len(eeg_data['theta']) > 0 else np.zeros(n_samples),
            'theta_af8': eeg_data['theta'][:, 2] if len(eeg_data['theta']) > 0 else np.zeros(n_samples),
            'theta_tp10': eeg_data['theta'][:, 3] if len(eeg_data['theta']) > 0 else np.zeros(n_samples),
            
            # 眼部特征 (CogBeacon 没有)
            'space_distance': np.zeros(n_samples),
            'distance_to_eye_center': np.zeros(n_samples),
            'pose_pca': np.zeros(n_samples),
            
            # EOG (CogBeacon 没有)
            'eog_v': np.zeros(n_samples),
            'eog_h': np.zeros(n_samples),
            
            # 标签和元数据
            'p': np.zeros(n_samples),
            'm': np.zeros(n_samples),
            'f': np.full(n_samples, round_fatigue),
            'id': user_id,
            'session': hash(session_name) % 1000,  # 会话 ID (哈希到小数字)
            'round': round_idx,
        })
        
        all_round_dfs.append(round_df)
    
    if not all_round_dfs:
        return None
    
    # 合并所有 round
    session_df = pd.concat(all_round_dfs, ignore_index=True)
    
    # 重采样到目标采样率 (Muse ~220 Hz -> 32 Hz)
    # 注意: 频带功率数据采样率不明确，这里假设与 EEG 相同
    # 实际上频带功率可能是每秒更新一次，需要根据实际情况调整
    original_len = len(session_df)
    # 估算原始采样率 (通常 Muse 频带功率每秒约 10 个样本)
    estimated_sr = 10  # 假设频带功率采样率约 10 Hz
    
    if estimated_sr != TARGET_SR:
        # 简单下采样
        downsample_factor = max(1, int(len(session_df) * TARGET_SR / (estimated_sr * len(session_df))))
        # 由于频带功率已经是低采样率，可能不需要重采样
        # 这里保持原样
    
    print(f"    Samples: {len(session_df)}, Fatigue distribution: {session_df['f'].value_counts().to_dict()}")
    
    return session_df


def extract_user_id(session_name: str) -> int:
    """
    从会话名称提取用户 ID
    
    Args:
        session_name: 如 'user_0_v_m', 'user_10b_v_o'
    
    Returns:
        用户 ID (数字)
    """
    # 提取 user_ 后面的数字部分
    match = re.match(r'user_(\d+)b?_', session_name)
    if match:
        return int(match.group(1))
    return 0


def process_cogbeacon_dataset(cogbeacon_root: str, output_path: str):
    """
    处理整个 CogBeacon 数据集
    
    Args:
        cogbeacon_root: CogBeacon 数据集根目录
        output_path: 输出 CSV 路径
    """
    cogbeacon_root = Path(cogbeacon_root)
    eeg_root = cogbeacon_root / 'eeg'
    fatigue_root = cogbeacon_root / 'fatigue_self_report'
    
    # 获取所有会话目录
    session_dirs = sorted([d for d in eeg_root.iterdir() if d.is_dir()])
    print(f"Found {len(session_dirs)} sessions")
    
    all_dfs = []
    
    for session_dir in session_dirs:
        session_name = session_dir.name
        user_id = extract_user_id(session_name)
        fatigue_csv = fatigue_root / f"{session_name}.csv"
        
        try:
            df = process_single_session(
                str(session_dir), 
                str(fatigue_csv), 
                session_name, 
                user_id
            )
            if df is not None and len(df) > 0:
                df['source'] = 'CogBeacon'
                all_dfs.append(df)
        except Exception as e:
            print(f"  Error processing {session_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_dfs:
        raise RuntimeError("No data processed successfully")
    
    # 合并所有数据
    print("\nMerging all sessions...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 添加 block 列 (用于 data.py 兼容性)
    combined_df['block'] = combined_df['id'] * 1000 + combined_df['session']
    
    print(f"Total samples: {len(combined_df)}")
    print(f"Total users: {combined_df['id'].nunique()}")
    print(f"Total sessions: {combined_df['session'].nunique()}")
    print(f"Label distribution: {combined_df['f'].value_counts().to_dict()}")
    
    # 保存
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return combined_df


def print_dataset_summary(df: pd.DataFrame):
    """打印数据集摘要"""
    print("\n" + "="*60)
    print("CogBeacon Dataset Summary")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\nLabel distribution (f):")
    for label, count in df['f'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        label_name = "awake" if label == 0 else "fatigue"
        print(f"  {label} ({label_name}): {count:,} ({pct:.1f}%)")
    
    print(f"\nUsers: {df['id'].nunique()}")
    print(f"User IDs: {sorted(df['id'].unique())}")
    print(f"Sessions: {df['session'].nunique()}")
    
    print(f"\nSignal channels with data (non-zero):")
    signal_cols = ['ecg', 'ppg', 'gsr', 'hr', 'skt', 'breathing', 'emg', 
                   'acc_x', 'acc_y', 'acc_z', 
                   'eeg_alpha', 'eeg_beta', 'eeg_delta', 'eeg_gamma', 'eeg_theta']
    for col in signal_cols:
        if col in df.columns:
            non_zero = (df[col] != 0).sum()
            pct = non_zero / len(df) * 100
            status = "OK" if pct > 50 else "MISSING" if pct < 1 else "PARTIAL"
            print(f"  {col}: {pct:.1f}% non-zero [{status}]")
    
    print(f"\nEEG channel statistics:")
    eeg_cols = ['alpha_tp9', 'alpha_af7', 'alpha_af8', 'alpha_tp10',
                'beta_tp9', 'beta_af7', 'beta_af8', 'beta_tp10']
    for col in eeg_cols[:4]:  # 只显示 alpha
        if col in df.columns:
            mean_val = df[col][df[col] != 0].mean()
            std_val = df[col][df[col] != 0].std()
            print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CogBeacon dataset')
    parser.add_argument('--cogbeacon_root', type=str, 
                        default='../Data/CogBeacon',
                        help='Path to CogBeacon dataset root')
    parser.add_argument('--output', type=str, 
                        default='../Data/CogBeacon_unified.csv',
                        help='Output CSV path')
    
    args = parser.parse_args()
    
    print("CogBeacon Preprocessing Script")
    print("="*60)
    print(f"Input: {args.cogbeacon_root}")
    print(f"Output: {args.output}")
    print(f"Target sampling rate: {TARGET_SR} Hz")
    print("="*60)
    
    df = process_cogbeacon_dataset(args.cogbeacon_root, args.output)
    print_dataset_summary(df)
