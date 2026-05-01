# preprocess_drozy.py
"""
DROZY 数据集预处理脚本
将 DROZY 的 EDF (PSG) 数据转换为与 FM/OD/MEFAR 兼容的 CSV 格式

DROZY 数据集包含:
- EEG: Fz, Cz, C3, C4, Pz (5通道)
- EOG: EOG-V, EOG-H (2通道)
- EMG: EMG (1通道)
- ECG: ECG (1通道)
- KSS 标签: 1-9 (Karolinska Sleepiness Scale)

标签映射: KSS >= 7 -> 疲劳(1), KSS < 7 -> 清醒(0)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: mne not installed. Run: pip install mne")

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False
    print("Warning: pyedflib not installed. Run: pip install pyedflib")


def load_kss_labels(kss_file: str) -> dict:
    """
    加载 KSS 标签文件
    格式: 14行 x 3列 (14个被试, 每人3次测试)
    返回: {(subject_id, test_id): kss_score}
    """
    kss_dict = {}
    with open(kss_file, 'r') as f:
        for subj_idx, line in enumerate(f, start=1):
            scores = line.strip().split()
            for test_idx, score in enumerate(scores, start=1):
                kss_dict[(subj_idx, test_idx)] = int(score)
    return kss_dict


def kss_to_fatigue(kss: int, threshold: int = 7) -> int:
    """
    KSS 到二值疲劳标签的映射
    KSS >= threshold -> 疲劳(1)
    KSS < threshold -> 清醒(0)
    KSS = 0 表示缺失数据，标记为 -1
    """
    if kss == 0:
        return -1  # 缺失
    return 1 if kss >= threshold else 0


def read_edf_file(edf_path: str) -> tuple:
    """
    读取 EDF 文件，返回信号数据和通道信息
    返回: (data_dict, sampling_rate)
    """
    if MNE_AVAILABLE:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            data = raw.get_data()
            ch_names = raw.ch_names
            sfreq = raw.info['sfreq']
            data_dict = {ch: data[i] for i, ch in enumerate(ch_names)}
            return data_dict, sfreq
        except Exception as e:
            print(f"MNE failed for {edf_path}: {e}")
    
    if PYEDFLIB_AVAILABLE:
        try:
            f = pyedflib.EdfReader(edf_path)
            n_channels = f.signals_in_file
            ch_names = f.getSignalLabels()
            sfreq = f.getSampleFrequency(0)
            data_dict = {}
            for i in range(n_channels):
                data_dict[ch_names[i]] = f.readSignal(i)
            f.close()
            return data_dict, sfreq
        except Exception as e:
            print(f"pyedflib failed for {edf_path}: {e}")
    
    raise RuntimeError(f"Cannot read EDF file: {edf_path}. Install mne or pyedflib.")


def process_single_recording(edf_path: str, kss_score: int, 
                             subject_id: int, test_id: int,
                             target_sr: int = 32) -> pd.DataFrame:
    """
    处理单个 EDF 录制文件
    
    Args:
        edf_path: EDF 文件路径
        kss_score: KSS 评分 (1-9)
        subject_id: 被试 ID
        test_id: 测试 ID
        target_sr: 目标采样率 (与现有数据对齐)
    
    Returns:
        DataFrame with aligned features
    """
    data_dict, original_sr = read_edf_file(edf_path)
    
    # DROZY PSG 通道映射到标准名称
    channel_mapping = {
        'Fz': 'eeg_fz',
        'Cz': 'eeg_cz', 
        'C3': 'eeg_c3',
        'C4': 'eeg_c4',
        'Pz': 'eeg_pz',
        'EOG-V': 'eog_v',
        'EOG-H': 'eog_h',
        'EMG': 'emg',
        'ECG': 'ecg',
    }
    
    # 排除非生理信号通道
    exclude_channels = ['Cam-Sync', 'PVT', 'Oz']
    
    # 重采样因子
    resample_factor = int(original_sr / target_sr)
    
    # 处理每个通道
    processed_data = {}
    n_samples = None
    
    for orig_name, new_name in channel_mapping.items():
        if orig_name in data_dict:
            signal = data_dict[orig_name]
            # 简单下采样 (取平均)
            if resample_factor > 1:
                n_new = len(signal) // resample_factor
                signal = signal[:n_new * resample_factor].reshape(-1, resample_factor).mean(axis=1)
            processed_data[new_name] = signal
            if n_samples is None:
                n_samples = len(signal)
    
    if n_samples is None:
        print(f"Warning: No valid channels in {edf_path}")
        return None
    
    # 创建 DataFrame
    df = pd.DataFrame(processed_data)
    
    # 添加与现有数据兼容的列 (设为 0，表示缺失)
    # 共享模态 (DROZY 没有这些，设为 0)
    for col in ['acc_x', 'acc_y', 'acc_z', 'ppg', 'gsr', 'hr', 'skt']:
        df[col] = 0.0
    
    # EEG 频带功率 (DROZY 是原始信号，需要后续计算或设为0)
    # 这里简化处理，设为 0，表示需要从原始 EEG 计算
    for band in ['alpha', 'beta', 'delta', 'gamma', 'theta']:
        for ch in ['tp9', 'af7', 'af8', 'tp10']:
            df[f'{band}_{ch}'] = 0.0
    
    # 呼吸信号 (DROZY 没有)
    df['breathing'] = 0.0
    
    # 眼部特征 (DROZY 没有，但有 EOG)
    for col in ['space_distance', 'distance_to_eye_center', 'pose_pca']:
        df[col] = 0.0
    
    # 标签和元数据
    fatigue_label = kss_to_fatigue(kss_score)
    df['F'] = fatigue_label
    df['kss'] = kss_score  # 保留原始 KSS 评分
    df['p'] = 0
    df['m'] = 0
    df['f'] = fatigue_label
    df['id'] = subject_id
    df['session'] = test_id
    df['block'] = subject_id * 100 + test_id  # 唯一 block ID
    df['source'] = 'DROZY'
    
    # 时间戳
    df['timestamp'] = pd.date_range(start='2016-01-01', periods=len(df), freq=f'{1000//target_sr}ms')
    
    return df


def process_drozy_dataset(drozy_root: str, output_path: str, target_sr: int = 32):
    """
    处理整个 DROZY 数据集
    
    Args:
        drozy_root: DROZY 数据集根目录
        output_path: 输出 CSV 路径
        target_sr: 目标采样率
    """
    drozy_root = Path(drozy_root)
    psg_dir = drozy_root / 'psg'
    kss_file = drozy_root / 'KSS.txt'
    
    if not psg_dir.exists():
        raise FileNotFoundError(f"PSG directory not found: {psg_dir}")
    if not kss_file.exists():
        raise FileNotFoundError(f"KSS file not found: {kss_file}")
    
    # 加载 KSS 标签
    kss_labels = load_kss_labels(str(kss_file))
    print(f"Loaded KSS labels for {len(kss_labels)} recordings")
    
    # 处理所有 EDF 文件
    all_dfs = []
    edf_files = sorted(psg_dir.glob('*.edf'))
    
    print(f"Found {len(edf_files)} EDF files")
    
    for edf_path in edf_files:
        # 解析文件名: SUBJECT-TEST.edf
        name = edf_path.stem
        parts = name.split('-')
        if len(parts) != 2:
            print(f"Skipping invalid filename: {name}")
            continue
        
        try:
            subject_id = int(parts[0])
            test_id = int(parts[1])
        except ValueError:
            print(f"Skipping invalid filename: {name}")
            continue
        
        # 获取 KSS 标签
        kss_score = kss_labels.get((subject_id, test_id), 0)
        if kss_score == 0:
            print(f"Skipping {name}: missing KSS label")
            continue
        
        print(f"Processing {name}: subject={subject_id}, test={test_id}, KSS={kss_score}")
        
        try:
            df = process_single_recording(
                str(edf_path), kss_score, subject_id, test_id, target_sr
            )
            if df is not None:
                all_dfs.append(df)
                print(f"  -> {len(df)} samples, fatigue={kss_to_fatigue(kss_score)}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    if not all_dfs:
        raise RuntimeError("No data processed successfully")
    
    # 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples: {len(combined_df)}")
    print(f"Label distribution: {combined_df['F'].value_counts().to_dict()}")
    
    # 保存
    combined_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return combined_df


def print_dataset_summary(df: pd.DataFrame):
    """打印数据集摘要"""
    print("\n" + "="*60)
    print("DROZY Dataset Summary")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nLabel distribution (F):")
    print(df['F'].value_counts())
    print(f"\nKSS distribution:")
    print(df['kss'].value_counts().sort_index())
    print(f"\nSubjects: {df['id'].nunique()}")
    print(f"Sessions per subject: {df.groupby('id')['session'].nunique().mean():.1f}")
    print(f"\nAvailable channels:")
    signal_cols = [c for c in df.columns if c not in ['timestamp', 'F', 'kss', 'p', 'm', 'f', 'id', 'session', 'block', 'source']]
    non_zero_cols = [c for c in signal_cols if df[c].abs().sum() > 0]
    print(f"  Non-zero: {non_zero_cols}")
    zero_cols = [c for c in signal_cols if df[c].abs().sum() == 0]
    print(f"  Zero (missing): {len(zero_cols)} columns")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess DROZY dataset')
    parser.add_argument('--drozy_root', type=str, default='../Data/DROZY/DROZY',
                        help='Path to DROZY dataset root')
    parser.add_argument('--output', type=str, default='../Data/DROZY_original.csv',
                        help='Output CSV path')
    parser.add_argument('--target_sr', type=int, default=32,
                        help='Target sampling rate (Hz)')
    
    args = parser.parse_args()
    
    if not MNE_AVAILABLE and not PYEDFLIB_AVAILABLE:
        print("ERROR: Please install mne or pyedflib to read EDF files")
        print("  pip install mne")
        print("  or")
        print("  pip install pyedflib")
        exit(1)
    
    df = process_drozy_dataset(args.drozy_root, args.output, args.target_sr)
    print_dataset_summary(df)
