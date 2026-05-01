# preprocess_wesad.py
"""
WESAD 数据集预处理脚本
将 WESAD 的 PKL 数据转换为与 FM/OD/MEFAR/DROZY 兼容的统一 CSV 格式

WESAD 数据集包含:
- 胸带 (RespiBAN, 700 Hz): ECG, EDA, EMG, ACC(3轴), Temp, Resp
- 手环 (Empatica E4): BVP(64Hz), EDA(4Hz), TEMP(4Hz), ACC(32Hz)
- 标签: 0=过渡, 1=基线, 2=压力, 3=愉悦, 4=冥想
- 问卷: PANAS, STAI, DIM, SSSQ

疲劳标签策略:
- 方案1: 使用原始标签 (1=基线/清醒 -> 0, 2=压力后 -> 1)
- 方案2: 使用 SSSQ Engagement (低Engagement = 高疲劳)
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional, List
import warnings

# 目标采样率
TARGET_SR = 32

# WESAD 采样率
CHEST_SR = 700  # 胸带
WRIST_BVP_SR = 64
WRIST_ACC_SR = 32
WRIST_EDA_SR = 4
WRIST_TEMP_SR = 4

# 标签映射 (WESAD原始 -> 疲劳检测)
# 1=基线(清醒), 2=压力(诱导疲劳), 3=愉悦, 4=冥想
# 策略: 只保留 1(清醒->0) 和 2(压力后->1)
LABEL_MAPPING = {
    0: -1,  # 过渡期，排除
    1: 0,   # 基线 -> 清醒
    2: 1,   # 压力 -> 疲劳 (压力任务后容易疲劳)
    3: 0,   # 愉悦 -> 清醒 (情绪积极，不疲劳)
    4: 0,   # 冥想 -> 清醒 (放松状态)
    5: -1,  # 未定义
    6: -1,  # 未定义
    7: -1,  # 未定义
}


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


def parse_questionnaire(quest_path: str) -> Dict:
    """
    解析 WESAD 问卷文件
    
    Returns:
        dict with keys: PANAS, STAI, DIM, SSSQ
    """
    result = {
        'PANAS': [],
        'STAI': [],
        'DIM': [],
        'SSSQ': [],
        'ORDER': [],
        'START': [],
        'END': []
    }
    
    try:
        with open(quest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';;'):
                    continue
                
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                
                key = parts[0].replace('#', '').strip()
                if key in result:
                    # 提取数值
                    values = []
                    for p in parts[1:]:
                        p = p.strip()
                        if p:
                            try:
                                values.append(float(p))
                            except ValueError:
                                values.append(p)
                    if values:
                        result[key].append(values)
    except Exception as e:
        print(f"Warning: Failed to parse questionnaire {quest_path}: {e}")
    
    return result


def compute_sssq_fatigue(sssq_scores: List) -> Optional[float]:
    """
    从 SSSQ 分数计算疲劳水平
    SSSQ 有多个维度，Engagement 维度与疲劳相关
    低 Engagement = 高疲劳
    
    Returns:
        疲劳分数 (0-1 范围)，None 表示无法计算
    """
    if not sssq_scores or len(sssq_scores) == 0:
        return None
    
    # SSSQ 最后一行通常是汇总分数
    # 取平均作为 engagement 代理
    all_scores = []
    for row in sssq_scores:
        for val in row:
            if isinstance(val, (int, float)):
                all_scores.append(val)
    
    if not all_scores:
        return None
    
    avg_score = np.mean(all_scores)
    # SSSQ 评分范围通常 1-5，转换为疲劳分数
    # 高分 = 高 engagement = 低疲劳
    # 归一化到 0-1，然后反转
    fatigue = 1.0 - (avg_score - 1) / 4.0
    return np.clip(fatigue, 0, 1)


def process_single_subject(pkl_path: str, quest_path: str, 
                           subject_id: str) -> Optional[pd.DataFrame]:
    """
    处理单个被试的数据
    
    Args:
        pkl_path: PKL 文件路径
        quest_path: 问卷 CSV 路径
        subject_id: 被试 ID (如 'S2')
    
    Returns:
        处理后的 DataFrame
    """
    print(f"Processing {subject_id}...")
    
    # 读取 PKL 文件
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"  Error loading {pkl_path}: {e}")
        return None
    
    # 提取信号
    chest = data['signal']['chest']
    wrist = data['signal']['wrist']
    labels = data['label']
    
    # 检查数据长度
    chest_len = chest['ECG'].shape[0]
    print(f"  Chest samples: {chest_len} (@ {CHEST_SR} Hz = {chest_len/CHEST_SR/60:.1f} min)")
    
    # 1. 重采样胸带信号 (700 Hz -> 32 Hz)
    print("  Resampling chest signals...")
    ecg = resample_signal(chest['ECG'].flatten(), CHEST_SR, TARGET_SR)
    eda_chest = resample_signal(chest['EDA'].flatten(), CHEST_SR, TARGET_SR)
    emg = resample_signal(chest['EMG'].flatten(), CHEST_SR, TARGET_SR)
    temp_chest = resample_signal(chest['Temp'].flatten(), CHEST_SR, TARGET_SR)
    resp = resample_signal(chest['Resp'].flatten(), CHEST_SR, TARGET_SR)
    acc_chest = resample_signal(chest['ACC'], CHEST_SR, TARGET_SR)
    
    # 2. 重采样手环信号
    print("  Resampling wrist signals...")
    bvp = resample_signal(wrist['BVP'].flatten(), WRIST_BVP_SR, TARGET_SR)
    eda_wrist = resample_signal(wrist['EDA'].flatten(), WRIST_EDA_SR, TARGET_SR)
    temp_wrist = resample_signal(wrist['TEMP'].flatten(), WRIST_TEMP_SR, TARGET_SR)
    acc_wrist = resample_signal(wrist['ACC'], WRIST_ACC_SR, TARGET_SR)  # 已经是 32 Hz
    
    # 3. 重采样标签 (700 Hz -> 32 Hz)
    print("  Resampling labels...")
    labels_resampled = resample_signal(labels.astype(float), CHEST_SR, TARGET_SR)
    labels_resampled = np.round(labels_resampled).astype(int)
    
    # 4. 对齐所有信号到相同长度 (取最短)
    target_len = min(len(ecg), len(bvp), len(eda_wrist), len(labels_resampled))
    print(f"  Target length: {target_len} samples (@ {TARGET_SR} Hz = {target_len/TARGET_SR/60:.1f} min)")
    
    # 截断到相同长度
    ecg = ecg[:target_len]
    eda_chest = eda_chest[:target_len]
    emg = emg[:target_len]
    temp_chest = temp_chest[:target_len]
    resp = resp[:target_len]
    acc_chest = acc_chest[:target_len]
    bvp = bvp[:target_len]
    eda_wrist = eda_wrist[:target_len]
    temp_wrist = temp_wrist[:target_len]
    acc_wrist = acc_wrist[:target_len]
    labels_resampled = labels_resampled[:target_len]
    
    # 5. 映射标签到疲劳标签
    fatigue_labels = np.array([LABEL_MAPPING.get(l, -1) for l in labels_resampled])
    
    # 6. 创建 DataFrame
    df = pd.DataFrame({
        # 时间戳
        'timestamp': pd.date_range(start='2018-01-01', periods=target_len, freq=f'{1000//TARGET_SR}ms'),
        
        # 加速度 (使用手环的，因为更接近可穿戴设备场景)
        'acc_x': acc_wrist[:, 0] if acc_wrist.shape[0] > 0 else np.zeros(target_len),
        'acc_y': acc_wrist[:, 1] if acc_wrist.shape[0] > 0 else np.zeros(target_len),
        'acc_z': acc_wrist[:, 2] if acc_wrist.shape[0] > 0 else np.zeros(target_len),
        
        # PPG/BVP
        'ppg': bvp,
        
        # GSR/EDA (使用手环的)
        'gsr': eda_wrist,
        
        # HR (从 BVP 计算，这里简化为 0，后续可以用 HeartPy 计算)
        'hr': np.zeros(target_len),
        
        # 皮肤温度 (使用手环的)
        'skt': temp_wrist,
        
        # ECG (胸带)
        'ecg': ecg,
        
        # 呼吸 (胸带)
        'breathing': resp,
        
        # EMG (胸带)
        'emg': emg,
        
        # EEG 频带功率 (WESAD 没有 EEG，设为 0)
        'eeg_alpha': np.zeros(target_len),
        'eeg_beta': np.zeros(target_len),
        'eeg_delta': np.zeros(target_len),
        'eeg_gamma': np.zeros(target_len),
        'eeg_theta': np.zeros(target_len),
        
        # 详细 EEG 通道 (与 FM 格式兼容)
        'alpha_tp9': np.zeros(target_len),
        'alpha_af7': np.zeros(target_len),
        'alpha_af8': np.zeros(target_len),
        'alpha_tp10': np.zeros(target_len),
        'beta_tp9': np.zeros(target_len),
        'beta_af7': np.zeros(target_len),
        'beta_af8': np.zeros(target_len),
        'beta_tp10': np.zeros(target_len),
        'delta_tp9': np.zeros(target_len),
        'delta_af7': np.zeros(target_len),
        'delta_af8': np.zeros(target_len),
        'delta_tp10': np.zeros(target_len),
        'gamma_tp9': np.zeros(target_len),
        'gamma_af7': np.zeros(target_len),
        'gamma_af8': np.zeros(target_len),
        'gamma_tp10': np.zeros(target_len),
        'theta_tp9': np.zeros(target_len),
        'theta_af7': np.zeros(target_len),
        'theta_af8': np.zeros(target_len),
        'theta_tp10': np.zeros(target_len),
        
        # 眼部特征 (WESAD 没有)
        'space_distance': np.zeros(target_len),
        'distance_to_eye_center': np.zeros(target_len),
        'pose_pca': np.zeros(target_len),
        
        # EOG (WESAD 没有，但 DROZY 有)
        'eog_v': np.zeros(target_len),
        'eog_h': np.zeros(target_len),
        
        # 标签和元数据
        'p': np.zeros(target_len),
        'm': np.zeros(target_len),
        'f': fatigue_labels,
        'id': int(subject_id[1:]),  # S2 -> 2
        'session': 1,
        'block': int(subject_id[1:]) * 100 + 1,  # 唯一 block ID
    })
    
    # 7. 过滤掉无效标签 (f == -1)
    valid_mask = df['f'] >= 0
    df = df[valid_mask].reset_index(drop=True)
    
    print(f"  Valid samples: {len(df)} ({valid_mask.sum()/len(valid_mask)*100:.1f}%)")
    print(f"  Label distribution: {df['f'].value_counts().to_dict()}")
    
    return df


def process_wesad_dataset(wesad_root: str, output_path: str):
    """
    处理整个 WESAD 数据集
    
    Args:
        wesad_root: WESAD 数据集根目录 (包含 S2, S3, ... 子目录)
        output_path: 输出 CSV 路径
    """
    wesad_root = Path(wesad_root)
    
    # 查找所有被试目录
    subject_dirs = sorted([d for d in wesad_root.iterdir() if d.is_dir() and d.name.startswith('S')])
    print(f"Found {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")
    
    all_dfs = []
    
    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        pkl_path = subj_dir / f"{subject_id}.pkl"
        quest_path = subj_dir / f"{subject_id}_quest.csv"
        
        if not pkl_path.exists():
            print(f"Skipping {subject_id}: PKL file not found")
            continue
        
        try:
            df = process_single_subject(str(pkl_path), str(quest_path), subject_id)
            if df is not None and len(df) > 0:
                df['source'] = 'WESAD'
                all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_dfs:
        raise RuntimeError("No data processed successfully")
    
    # 合并所有数据
    print("\nMerging all subjects...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"Total samples: {len(combined_df)}")
    print(f"Total subjects: {combined_df['id'].nunique()}")
    print(f"Label distribution: {combined_df['f'].value_counts().to_dict()}")
    
    # 保存
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return combined_df


def print_dataset_summary(df: pd.DataFrame):
    """打印数据集摘要"""
    print("\n" + "="*60)
    print("WESAD Dataset Summary")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Duration: {len(df)/TARGET_SR/60:.1f} minutes")
    print(f"\nLabel distribution (f):")
    for label, count in df['f'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        label_name = "awake" if label == 0 else "fatigue"
        print(f"  {label} ({label_name}): {count:,} ({pct:.1f}%)")
    
    print(f"\nSubjects: {df['id'].nunique()}")
    print(f"Subject IDs: {sorted(df['id'].unique())}")
    
    print(f"\nSignal channels with data (non-zero):")
    signal_cols = ['ecg', 'ppg', 'gsr', 'hr', 'skt', 'breathing', 'emg', 
                   'acc_x', 'acc_y', 'acc_z', 'eeg_alpha', 'eeg_beta']
    for col in signal_cols:
        if col in df.columns:
            non_zero = (df[col] != 0).sum()
            pct = non_zero / len(df) * 100
            status = "OK" if pct > 50 else "MISSING" if pct < 1 else "PARTIAL"
            print(f"  {col}: {pct:.1f}% non-zero [{status}]")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WESAD dataset')
    parser.add_argument('--wesad_root', type=str, 
                        default='../Data/WESAD/WESAD',
                        help='Path to WESAD dataset root')
    parser.add_argument('--output', type=str, 
                        default='../Data/WESAD_unified.csv',
                        help='Output CSV path')
    
    args = parser.parse_args()
    
    print("WESAD Preprocessing Script")
    print("="*60)
    print(f"Input: {args.wesad_root}")
    print(f"Output: {args.output}")
    print(f"Target sampling rate: {TARGET_SR} Hz")
    print("="*60)
    
    df = process_wesad_dataset(args.wesad_root, args.output)
    print_dataset_summary(df)
