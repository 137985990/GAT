
# impute.py - v99
# 使用训练好的模型补全缺失模态

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# 从训练脚本导入必要的函数和类
from train import forward_batch_parallel, unify_config
from data import load_config
from model import TGATUNet


def detect_missing_modalities(df: pd.DataFrame, feature_cols: List[str], 
                               source_name: str) -> torch.Tensor:
    """
    检测哪些模态在数据中缺失（和 data.py 的 source_feature_presence 逻辑一致）
    
    Args:
        df: 数据DataFrame
        feature_cols: 特征列名列表
        source_name: 数据源名称（用于日志）
    
    Returns:
        presence_mask: [C] 张量，1表示存在，0表示缺失
    """
    presence = []
    for col in feature_cols:
        if col not in df.columns:
            presence.append(0)
            continue
        
        series = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        col_vals = series.to_numpy(dtype=float, copy=True)
        
        if col_vals.size == 0:
            presence.append(0)
            continue
        
        max_abs = float(np.max(np.abs(col_vals)))
        var_val = float(np.var(col_vals))
        
        # 和 data.py 第137-141行逻辑一致
        if (max_abs < 1e-8) or (var_val < 1e-8):
            presence.append(0)
        else:
            presence.append(1)
    
    presence_tensor = torch.tensor(presence, dtype=torch.float32)
    
    # 打印检测结果
    present_cols = [feature_cols[i] for i, p in enumerate(presence) if p == 1]
    missing_cols = [feature_cols[i] for i, p in enumerate(presence) if p == 0]
    print(f"\n[{source_name}] 模态检测结果:")
    print(f"  存在模态 ({len(present_cols)}): {', '.join(present_cols[:5])}{'...' if len(present_cols) > 5 else ''}")
    print(f"  缺失模态 ({len(missing_cols)}): {', '.join(missing_cols)}")
    
    return presence_tensor


def sliding_window_split(data_array: np.ndarray, window_size: int, step_size: int) -> List[np.ndarray]:
    """
    滑动窗口切割时序数据
    
    Args:
        data_array: [T, C] 数组
        window_size: 窗口大小
        step_size: 滑动步长
    
    Returns:
        windows: list of [window_size, C] 数组
    """
    T, C = data_array.shape
    windows = []
    
    for start in range(0, T - window_size + 1, step_size):
        window = data_array[start:start + window_size]
        windows.append(window)
    
    return windows


def reconstruct_from_windows(windows: List[np.ndarray], step_size: int, 
                             total_length: int) -> np.ndarray:
    """
    从重叠窗口重建完整时序（加权平均处理重叠区域）
    
    Args:
        windows: list of [window_size, C] 数组
        step_size: 滑动步长
        total_length: 原始时序总长度
    
    Returns:
        reconstructed: [total_length, C] 数组
    """
    if not windows:
        return np.array([])
    
    window_size, C = windows[0].shape
    
    # 累积数组和计数数组
    accumulator = np.zeros((total_length, C), dtype=np.float64)
    counter = np.zeros((total_length, C), dtype=np.float64)
    
    # 对每个窗口使用三角权重（中心权重高，边缘权重低）
    triangle_weight = np.concatenate([
        np.linspace(0.5, 1.0, window_size // 2),
        np.linspace(1.0, 0.5, window_size - window_size // 2)
    ])
    triangle_weight = triangle_weight.reshape(-1, 1)  # [window_size, 1]
    
    for i, window in enumerate(windows):
        start = i * step_size
        end = start + window_size
        
        if end > total_length:
            end = total_length
            window = window[:end - start]
            triangle_weight_trimmed = triangle_weight[:end - start]
        else:
            triangle_weight_trimmed = triangle_weight
        
        accumulator[start:end] += window * triangle_weight_trimmed
        counter[start:end] += triangle_weight_trimmed
    
    # 避免除零
    counter = np.maximum(counter, 1e-8)
    reconstructed = accumulator / counter
    
    return reconstructed


def normalize_window(window: np.ndarray, method: str = 'zscore', 
                     present_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    窗口归一化（和 data.py 第186-195行逻辑一致）
    
    Args:
        window: [T, C] 数组
        method: 归一化方法
        present_mask: [C] 数组，1表示存在的通道（用于统计计算）
    
    Returns:
        normalized: 归一化后的数组
        stats: 归一化统计信息（用于逆归一化）
    """
    if method == 'zscore':
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        
        # 对于缺失通道（std≈0），使用全局统计或邻近通道统计
        if present_mask is not None:
            present_mask = present_mask.reshape(1, -1)  # [1, C]
            # 计算存在通道的全局统计
            present_channels = window[:, present_mask[0] == 1]
            if present_channels.size > 0:
                global_mean = present_channels.mean()
                global_std = present_channels.std()
                global_std = max(global_std, 1e-6)
            else:
                global_mean = 0.0
                global_std = 1.0
            
            # 对于std太小的通道（缺失通道），用全局统计替换
            small_std_mask = std < 1e-6
            std = np.where(small_std_mask, global_std, std)
            mean = np.where(small_std_mask, global_mean, mean)
        
        normalized = (window - mean) / (std + 1e-6)
        stats = {'mean': mean, 'std': std, 'method': 'zscore'}
    elif method == 'minmax':
        min_v = window.min(axis=0, keepdims=True)
        max_v = window.max(axis=0, keepdims=True)
        
        # 对于缺失通道，使用全局范围
        if present_mask is not None:
            present_mask = present_mask.reshape(1, -1)
            present_channels = window[:, present_mask[0] == 1]
            if present_channels.size > 0:
                global_min = present_channels.min()
                global_max = present_channels.max()
            else:
                global_min = 0.0
                global_max = 1.0
            
            small_range_mask = (max_v - min_v) < 1e-6
            min_v = np.where(small_range_mask, global_min, min_v)
            max_v = np.where(small_range_mask, global_max, max_v)
        
        normalized = (window - min_v) / (max_v - min_v + 1e-6)
        stats = {'min': min_v, 'max': max_v, 'method': 'minmax'}
    else:
        normalized = window
        stats = {'method': 'none'}
    
    return normalized, stats


def denormalize_window(window: np.ndarray, stats: Dict) -> np.ndarray:
    """
    逆归一化
    """
    method = stats.get('method', 'none')
    
    if method == 'zscore':
        return window * stats['std'] + stats['mean']
    elif method == 'minmax':
        return window * (stats['max'] - stats['min']) + stats['min']
    else:
        return window


def impute_single_csv(input_csv: str, output_csv: str, config: Dict, 
                     model: nn.Module, device: torch.device,
                     dataset_name: Optional[str] = None):
    """
    补全单个CSV文件
    
    Args:
        input_csv: 输入CSV路径
        output_csv: 输出CSV路径
        config: 配置字典
        model: 训练好的模型
        device: 设备
        dataset_name: 数据集名称（FM/OD/MEFAR）
    """
    print(f"\n{'='*60}")
    print(f"开始处理: {input_csv}")
    print(f"{'='*60}")
    
    # 1. 构建特征列（和 data.py 第221-230行一致）
    common_modalities = config.get('common_modalities', [])
    dataset_modalities = config.get('dataset_modalities', {})
    
    feature_cols = list(common_modalities)
    for _, mods in dataset_modalities.items():
        for mod in (mods.get('have', []) + mods.get('need', [])):
            if mod not in feature_cols:
                feature_cols.append(mod)
    
    print(f"\n总特征列数: {len(feature_cols)}")
    
    # 2. 读取CSV
    print(f"\n读取CSV文件...")
    df_original = pd.read_csv(input_csv)
    
    # 标准化列名（和 data.py 第59-68行一致）
    new_cols = []
    for col in df_original.columns:
        if col == 'F':
            new_cols.append(col)
        elif str(col).lower() in ['block', 'id', 'session']:
            new_cols.append(str(col).lower())
        else:
            new_cols.append(str(col).strip().lower())
    df_original.columns = new_cols
    
    print(f"原始数据形状: {df_original.shape}")
    print(f"原始列: {list(df_original.columns[:10])}{'...' if len(df_original.columns) > 10 else ''}")
    
    # 3. 补齐缺失列为0（和 data.py 第69-71行一致）
    df_full = df_original.copy()
    for col in feature_cols:
        if col not in df_full.columns:
            df_full[col] = 0.0
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0.0)
    
    # 4. 检测缺失模态
    if dataset_name is None:
        fname = os.path.basename(input_csv).lower()
        if 'fm' in fname:
            dataset_name = 'FM'
        elif 'od' in fname:
            dataset_name = 'OD'
        elif 'mefar' in fname:
            dataset_name = 'MEFAR'
        else:
            dataset_name = 'UNKNOWN'
    
    present_mask = detect_missing_modalities(df_full, feature_cols, dataset_name)
    missing_mask = 1.0 - present_mask
    
    # 5. 按 block 分组处理
    block_col = config.get('block_col', 'block')
    if block_col not in df_full.columns:
        print(f"警告: 未找到 '{block_col}' 列，将整个文件视为单个block")
        df_full[block_col] = 0
    
    window_size = int(config.get('window_size', 320))
    step_size = int(config.get('step_size', 32))
    norm_method = config.get('norm_method', 'zscore')
    
    print(f"\n滑动窗口参数: window_size={window_size}, step_size={step_size}, norm={norm_method}")
    
    # 6. 准备输出DataFrame
    df_output = df_original.copy()
    
    # 为缺失的列添加到输出（初始化为0）
    for col in feature_cols:
        if col not in df_output.columns:
            df_output[col] = 0.0
    
    model.eval()
    total_windows = 0
    
    # 7. 逐block处理
    blocks = df_full[block_col].unique()
    print(f"\n共 {len(blocks)} 个 blocks")
    
    with torch.no_grad():
        for block_id in tqdm(blocks, desc="处理blocks"):
            block_df = df_full[df_full[block_col] == block_id]
            block_idx = df_full[block_col] == block_id
            
            # 提取特征数组 [T, C]
            data_array = block_df[feature_cols].values.astype(np.float32)
            T, C = data_array.shape
            
            if T < window_size:
                print(f"  警告: block {block_id} 长度 {T} < window_size {window_size}，跳过")
                continue
            
            # 滑动窗口切割
            windows = sliding_window_split(data_array, window_size, step_size)
            total_windows += len(windows)
            
            # 对每个窗口进行归一化、推理、逆归一化
            imputed_windows = []
            all_stats = []
            
            # 调试：打印第一个窗口的信息
            debug_first_window = (block_id == blocks[0])
            
            for win_idx, window in enumerate(windows):
                # 归一化（传入 present_mask 以正确处理缺失通道）
                window_norm, stats = normalize_window(window, method=norm_method, 
                                                     present_mask=present_mask.numpy())
                all_stats.append(stats)
                
                # 转为张量 [C, T]
                window_tensor = torch.from_numpy(window_norm.T).float().to(device)
                
                # 推理（和 train.py 第251行一致）
                recon, _, _ = forward_batch_parallel(model, window_tensor.unsqueeze(0), device)
                recon = recon.squeeze(0)  # [C, T]
                
                # 调试第一个窗口
                if debug_first_window and win_idx == 0:
                    print(f"\n[调试] Block {block_id}, 窗口 0:")
                    print(f"  原始窗口形状: {window.shape}")
                    print(f"  归一化后范围: [{window_norm.min():.4f}, {window_norm.max():.4f}]")
                    print(f"  重建输出形状: {recon.shape}")
                    print(f"  重建输出范围: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
                    print(f"  missing_mask: {missing_mask}")
                    print(f"  missing_mask 总和: {missing_mask.sum().item()}")
                    
                    # 检查缺失列的重建值
                    missing_indices = torch.where(missing_mask == 1)[0]
                    if len(missing_indices) > 0:
                        print(f"  缺失通道索引: {missing_indices.tolist()}")
                        for idx in missing_indices[:3]:  # 只看前3个
                            print(f"    通道 {idx} ({feature_cols[idx]}): 重建前5值={recon[idx, :5].cpu().numpy()}")
                
                # 混合：缺失模态用重建值，存在模态保留原值（和 train.py 第258-262行一致）
                window_tensor_mixed = torch.where(
                    missing_mask.unsqueeze(-1).to(device) == 1,
                    recon,
                    window_tensor
                )
                
                # 转回 numpy [T, C]
                window_imputed = window_tensor_mixed.T.cpu().numpy()
                
                # 逆归一化
                window_denorm = denormalize_window(window_imputed, stats)
                
                # 调试第一个窗口的逆归一化结果
                if debug_first_window and win_idx == 0:
                    print(f"  逆归一化后范围: [{window_denorm.min():.4f}, {window_denorm.max():.4f}]")
                    if len(missing_indices) > 0:
                        for idx in missing_indices[:3]:
                            print(f"    通道 {idx} ({feature_cols[idx]}): 逆归一化后前5值={window_denorm[:5, idx]}")
                
                imputed_windows.append(window_denorm)
            
            # 从窗口重建完整时序
            reconstructed = reconstruct_from_windows(imputed_windows, step_size, T)
            
            # 更新输出DataFrame（只更新缺失模态的列）
            missing_cols = [feature_cols[i] for i in range(len(feature_cols)) if missing_mask[i] == 1]
            for col_idx, col_name in enumerate(feature_cols):
                if col_name in missing_cols:
                    df_output.loc[block_idx, col_name] = reconstructed[:, col_idx]
    
    print(f"\n总共处理窗口数: {total_windows}")
    
    # 8. 保存输出CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    
    # 保持列顺序：原始列 + 新增列
    original_cols = [c for c in df_original.columns if c in df_output.columns]
    new_cols = [c for c in feature_cols if c not in original_cols and c in df_output.columns]
    output_cols = original_cols + new_cols
    
    df_output[output_cols].to_csv(output_csv, index=False)
    print(f"\n补全完成！保存到: {output_csv}")
    print(f"输出形状: {df_output[output_cols].shape}")
    print(f"新增列: {new_cols}")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型补全缺失模态')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='../Checkpoints/best_model.pth', 
                       help='模型checkpoint路径')
    parser.add_argument('--input_csv', type=str, help='输入CSV文件路径（单文件模式）')
    parser.add_argument('--output_csv', type=str, help='输出CSV文件路径（单文件模式）')
    parser.add_argument('--dataset_name', type=str, choices=['FM', 'OD', 'MEFAR'], 
                       help='数据集名称（可选，会自动从文件名推断）')
    parser.add_argument('--batch_mode', action='store_true', 
                       help='批量模式：处理config中的所有data_files')
    
    args = parser.parse_args()
    
    # 加载配置
    print("加载配置文件...")
    config = load_config(args.config)
    config = unify_config(config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型（和 train.py 第645-652行一致）
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        in_channels = checkpoint.get('in_channels', 32)
        model_state = checkpoint['model_state_dict']
    else:
        # 只有权重的情况
        in_channels = config.get('input_channels', 32)
        model_state = checkpoint
    
    print(f"in_channels: {in_channels}")
    
    # 检查ablation_study配置（从checkpoint或config读取）
    ablation_config = {}
    if isinstance(checkpoint, dict):
        ablation_config = checkpoint.get('ablation_config', {})
    if not ablation_config:
        ablation_config = config.get('ablation_study', {})
    
    disable_transformer = ablation_config.get('disable_transformer', False)
    disable_classifier = ablation_config.get('disable_classifier', False)
    
    # 也检查 variant 字段
    variant = ablation_config.get('variant', '')
    if variant == 'no_classifier':
        disable_classifier = True
    elif variant == 'no_transformer':
        disable_transformer = True
    
    print(f"Ablation: disable_transformer={disable_transformer}, disable_classifier={disable_classifier}")
    
    # 创建模型（和 train.py 第458-472行一致）
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
        disable_classifier=disable_classifier
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    
    # 执行补全
    if args.batch_mode:
        # 批量模式：处理config中的所有文件
        print("\n批量模式：处理所有数据集")
        data_files = config.get('data_files', [])
        data_dir = config.get('data_dir', '')
        
        for data_file in data_files:
            # 构建完整路径
            if not os.path.isabs(data_file):
                input_path = os.path.join(data_dir, data_file) if data_dir else data_file
            else:
                input_path = data_file
            
            if not os.path.exists(input_path):
                # 尝试相对于配置文件的路径
                config_dir = os.path.dirname(args.config)
                input_path = os.path.join(config_dir, data_file)
            
            if not os.path.exists(input_path):
                print(f"警告: 文件不存在，跳过: {data_file}")
                continue
            
            # 构建输出路径：将 _original 替换为 _completed
            basename = os.path.basename(input_path)
            if '_original' in basename:
                output_basename = basename.replace('_original', '_completed')
            else:
                name, ext = os.path.splitext(basename)
                output_basename = f"{name}_completed{ext}"
            
            output_path = os.path.join(data_dir if data_dir else os.path.dirname(input_path), 
                                      output_basename)
            
            # 推断数据集名称
            dataset_name = None
            if 'fm' in basename.lower():
                dataset_name = 'FM'
            elif 'od' in basename.lower():
                dataset_name = 'OD'
            elif 'mefar' in basename.lower():
                dataset_name = 'MEFAR'
            
            impute_single_csv(input_path, output_path, config, model, device, dataset_name)
    
    else:
        # 单文件模式
        if not args.input_csv or not args.output_csv:
            parser.error("单文件模式需要 --input_csv 和 --output_csv 参数")
        
        impute_single_csv(args.input_csv, args.output_csv, config, model, device, args.dataset_name)
    
    print("\n所有补全任务完成！")


if __name__ == '__main__':
    main()
