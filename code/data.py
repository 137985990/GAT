# data.py - v99
import os
import json
import hashlib
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Sequence
from project_paths import normalize_config_paths, resolve_existing_path

def balance_fm_labels(df, source, label_col='F', strategy='undersample', target_ratio=None):
    if source != 'FM' or label_col not in df.columns:
        return df
    counts = df[label_col].value_counts().sort_index()
    if len(counts) < 2:
        return df
    groups = {lbl: df[df[label_col] == lbl].copy() for lbl in counts.index}
    if strategy == 'undersample':
        min_count = counts.min()
        parts = []
        for lbl, g in groups.items():
            if len(g) > min_count:
                parts.append(g.sample(n=min_count, random_state=42))
            else:
                parts.append(g)
        return pd.concat(parts, ignore_index=True)
    elif strategy == 'oversample':
        max_count = counts.max()
        parts = []
        for lbl, g in groups.items():
            if len(g) < max_count:
                reps = max_count - len(g)
                parts.append(pd.concat([g, g.sample(n=reps, replace=True, random_state=42)], ignore_index=True))
            else:
                parts.append(g)
        return pd.concat(parts, ignore_index=True)
    else:
        return df

def load_config(config_path: str) -> dict:
    resolved_config_path = resolve_existing_path(config_path)
    with open(resolved_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict):
        return normalize_config_paths(config, config_path=resolved_config_path)
    return config

def canonical_source_name(source: str) -> str:
    normalized = str(source or '').strip()
    key = normalized.lower().replace('-', '').replace('_', '').replace(' ', '')
    aliases = {
        'fatigueset': 'FM',
        'fm': 'FM',
        'vpfd': 'OD',
        'od': 'OD',
        'mefar': 'MEFAR',
        'drozy': 'DROZY',
        'cogbeacon': 'CogBeacon',
        'wesad': 'WESAD',
        'wearableexamstress': 'WearableExamStress',
        'sleepedf': 'SleepEDF',
    }
    return aliases.get(key, normalized)


def infer_source_name(file_path: str, df: Optional[pd.DataFrame] = None) -> str:
    if df is not None and 'source' in df.columns and not df['source'].dropna().empty:
        return canonical_source_name(str(df['source'].dropna().iloc[0]).strip())
    fname = os.path.basename(file_path).lower()
    if 'fatigueset' in fname or fname.startswith('fm') or '_fm' in fname:
        return canonical_source_name('FM')
    if 'vpfd' in fname or fname.startswith('od') or '_od' in fname:
        return canonical_source_name('OD')
    if 'mefar' in fname:
        return canonical_source_name('MEFAR')
    if 'drozy' in fname:
        return canonical_source_name('DROZY')
    if 'cogbeacon' in fname:
        return canonical_source_name('CogBeacon')
    if 'wearableexamstress' in fname or 'wearable_exam_stress' in fname:
        return canonical_source_name('WearableExamStress')
    if 'sleepedf' in fname or 'sleep_edf' in fname:
        return canonical_source_name('SleepEDF')
    if 'wesad' in fname:
        return canonical_source_name('WESAD')
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return canonical_source_name(stem.replace('_P', '').replace('_PP', ''))


def build_source_to_id(sources: Sequence[str], preferred_order: Optional[Sequence[str]] = None) -> Dict[str, int]:
    ordered = []
    for src in preferred_order or []:
        if src in sources and src not in ordered:
            ordered.append(src)
    for src in sorted(set(sources)):
        if src not in ordered:
            ordered.append(src)
    return {src: i for i, src in enumerate(ordered)}


def _source_block_offset(source: str, source_to_id: Optional[Dict[str, int]] = None) -> int:
    if source_to_id and source in source_to_id:
        return int(source_to_id[source]) * 1000
    return build_source_to_id([source]).get(source, 9) * 1000


def load_and_merge_multimodal_datasets(data_files: List[str], feature_cols: List[str],
                                       dataset_modalities_config: Optional[Dict] = None,
                                       balancing_config: Optional[Dict] = None,
                                       source_order: Optional[List[str]] = None) -> pd.DataFrame:
    all_dfs = []
    expected_sources = [infer_source_name(p) for p in data_files]
    source_to_id = build_source_to_id(expected_sources, source_order)
    print(f"开始加载{len(data_files)}个多模态数据集...")
    for i, file_path in enumerate(data_files):
        print(f"\n处理文件 {i+1}/{len(data_files)}: {file_path}")
        df = pd.read_csv(file_path)
        source = infer_source_name(file_path, df)
        if source not in source_to_id:
            source_to_id[source] = len(source_to_id)
        block_offset = _source_block_offset(source, source_to_id)
        # normalize columns (keep F uppercase)
        new_cols = []
        for col in df.columns:
            if col == 'F':
                new_cols.append(col)
            elif str(col).lower() in ['block', 'id', 'session']:
                new_cols.append(str(col).lower())
            else:
                new_cols.append(str(col).strip().lower())
        df.columns = new_cols
        if 'block' in df.columns:
            df['block'] = df['block'] + block_offset
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # label column handling
        if 'F' in df.columns:
            df['F'] = pd.to_numeric(df['F'], errors='coerce').fillna(0)
            label_col = 'F'
        else:
            label_col = 'f' if 'f' in df.columns else None
            if label_col:
                df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0)

        df['source'] = source
        if (balancing_config and balancing_config.get('enabled', False) and
            source in balancing_config.get('target_datasets', []) and (label_col or 'F') in df.columns):
            eff_label_col = label_col or 'F'
            df = balance_fm_labels(df, source, eff_label_col, balancing_config.get('strategy', 'undersample'), balancing_config.get('target_ratio'))
        all_dfs.append(df)
    print(f"\n合并{len(all_dfs)}个数据集...")
    return pd.concat(all_dfs, ignore_index=True)


def _build_cache_key(data_files: List[str], feature_cols: List[str],
                     dataset_modalities: Optional[Dict], balancing: Optional[Dict]) -> str:
    meta = {
        "files": [],
        "feature_cols": feature_cols,
        "dataset_modalities": dataset_modalities or {},
        "balancing": balancing or {},
    }
    for p in data_files:
        try:
            stat = os.stat(p)
            meta["files"].append({"path": os.path.abspath(p), "mtime": stat.st_mtime, "size": stat.st_size})
        except FileNotFoundError:
            meta["files"].append({"path": os.path.abspath(p), "mtime": None, "size": None})
    raw = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

class SlidingWindowDataset(Dataset):
    def __init__(self, data: pd.DataFrame, block_col: str, feature_cols: List[str],
                 window_size: int, step_size: int, sampling_rate: int = 1,
                 normalize: Optional[str] = None, label_col: str = 'F', phase: str = 'encode',
                 dataset_modalities: Optional[Dict] = None, common_modalities: Optional[List[str]] = None):
        super().__init__()
        self.data = data.copy()
        self.block_col = block_col
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.label_col = label_col
        self.phase = phase
        self.dataset_modalities = dataset_modalities or {}
        self.common_modalities = common_modalities or []
        self.global_source_order = [
            'FM', 'OD', 'MEFAR', 'DROZY', 'CogBeacon',
            'WESAD', 'WearableExamStress', 'SleepEDF'
        ]
        self.blocks = []
        self.block_sources = []
        for _, block_df in self.data.groupby(self.block_col):
            self.blocks.append(block_df)
            src = block_df['source'].iloc[0] if 'source' in block_df.columns else 'UNKNOWN'
            self.block_sources.append(src)
        self._generate_window_indices()
        # Build stable global source ids so P and PP domains do not restart at 0.
        uniq_sources = sorted(list({s for s in self.block_sources}))
        self.source_to_id = build_source_to_id(uniq_sources, self.global_source_order)
        # modality presence per source - use explicit config if provided, otherwise fallback to variance detection
        self.source_feature_presence = {}
        for src in uniq_sources:
            if src in self.dataset_modalities:
                # Use explicit config: common_modalities + have list are present
                have_list = self.dataset_modalities[src].get('have', [])
                present_features = set(self.common_modalities) | set(have_list)
                presence = []
                for col in self.feature_cols:
                    if col in present_features:
                        presence.append(1)
                    else:
                        presence.append(0)
                self.source_feature_presence[src] = torch.tensor(presence, dtype=torch.float32)
                print(f"[Data] {src}: have={list(present_features)[:5]}..., need={self.dataset_modalities[src].get('need', [])[:3]}...")
            else:
                # Fallback to variance-based detection for unknown sources
                src_blocks = [b for b,s in zip(self.blocks, self.block_sources) if s == src]
                if not src_blocks:
                    continue
                concat = pd.concat(src_blocks, axis=0)
                presence = []
                for col in self.feature_cols:
                    series = pd.to_numeric(concat[col], errors='coerce').fillna(0.0)
                    col_vals = series.to_numpy(dtype=float, copy=True)
                    if col_vals.size == 0:
                        presence.append(0)
                        continue
                    max_abs = float(np.max(np.abs(col_vals)))
                    var_val = float(np.var(col_vals))
                    if (max_abs < 1e-8) or (var_val < 1e-8):
                        presence.append(0)
                    else:
                        presence.append(1)
                self.source_feature_presence[src] = torch.tensor(presence, dtype=torch.float32)
                print(f"[Data] {src}: using variance-based detection (fallback)")

    def _generate_window_indices(self):
        self.indices = []
        for b_idx, block in enumerate(self.blocks):
            # robust label column fallback if specified not present
            if self.label_col not in block.columns:
                for cand in [self.label_col.lower(), 'f', 'label', 'y', 'target']:
                    if cand in block.columns:
                        self.label_col = cand
                        break
            if self.label_col not in block.columns:
                # create a dummy zero label column to avoid crash
                block[self.label_col] = 0
            labels = block[self.label_col].values
            change_points = np.where(np.diff(labels) != 0)[0] + 1
            seg_starts = np.concatenate(([0], change_points))
            seg_ends = np.concatenate((change_points, [len(labels)]))
            for seg_start, seg_end in zip(seg_starts, seg_ends):
                if seg_end - seg_start < self.window_size:
                    continue
                for start in range(seg_start, seg_end - self.window_size + 1, self.step_size):
                    self.indices.append((b_idx, start, labels[seg_start], self.block_sources[b_idx]))
        np.random.shuffle(self.indices)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b_idx, start, seg_label, source_dataset = self.indices[idx]
        block = self.blocks[b_idx]
        window_df = block.iloc[start:start + self.window_size]
        data_array = window_df[self.feature_cols].values[::self.sampling_rate]
        df_window = pd.DataFrame(data_array, columns=self.feature_cols)
        for col in df_window.columns:
            df_window[col] = pd.to_numeric(df_window[col], errors='coerce').fillna(0.0)
        data_array = df_window.values.astype(np.float32)
        expected = self.window_size // self.sampling_rate
        if data_array.shape[0] != expected:
            if data_array.shape[0] < expected:
                pad = np.zeros((expected - data_array.shape[0], data_array.shape[1]), dtype=np.float32)
                data_array = np.vstack([data_array, pad])
            else:
                data_array = data_array[:expected]
        if self.normalize == 'zscore':
            mean = data_array.mean(axis=0, keepdims=True)
            std = data_array.std(axis=0, keepdims=True)
            data_array = (data_array - mean) / (std + 1e-6)
        elif self.normalize == 'minmax':
            min_v = data_array.min(axis=0, keepdims=True)
            max_v = data_array.max(axis=0, keepdims=True)
            data_array = (data_array - min_v) / (max_v - min_v + 1e-6)
        tensor = torch.from_numpy(data_array.T).float()
        label = torch.tensor(int(seg_label), dtype=torch.long)
        src_id = self.source_to_id.get(source_dataset, -1)
        modal_mask = self.source_feature_presence.get(
            source_dataset,
            torch.ones(len(self.feature_cols), dtype=torch.float32)
        )
        # modal_mask 当前表示该 source 统计推断出的“存在”特征 (1=存在,0=缺失)
        present_mask = modal_mask.clone()
        missing_mask = (1.0 - present_mask)
        return tensor, label, modal_mask, torch.tensor(src_id, dtype=torch.long), present_mask, missing_mask, torch.tensor(idx, dtype=torch.long)


def create_multimodal_dataset_from_config(config: Dict, data_files: Optional[List[str]] = None, phase: str = 'encode') -> SlidingWindowDataset:
    if data_files is None:
        files = list(config.get('data_files', []) or [])
        pp_files = list(config.get('pp_data_files', []) or [])
        if pp_files:
            seen = set(files)
            files.extend([f for f in pp_files if f not in seen])
        data_dir = config.get('data_dir', '')
        data_files = [os.path.join(data_dir, f) if not os.path.isabs(f) and not os.path.exists(f) else f for f in files]
    common_modalities = config.get('common_modalities', [])
    dataset_modalities = config.get('dataset_modalities', {})
    feature_cols = list(common_modalities)
    # Keep feature union for compatibility
    for _, mods in dataset_modalities.items():
        for mod in (mods.get('have', []) + mods.get('need', [])):
            if mod not in feature_cols:
                feature_cols.append(mod)
    balancing = config.get('label_balancing', {})
    source_order = config.get('domain_source_order') or [
        infer_source_name(p) for p in (config.get('p_data_files', []) + config.get('pp_data_files', []))
    ]
    cache_enabled = bool(config.get('cache_enabled', False))
    cache_dir = config.get('cache_dir', os.path.join(config.get('data_dir', ''), 'cache'))
    cache_key = config.get('cache_key', None)
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        if not cache_key:
            cache_key = _build_cache_key(data_files, feature_cols, dataset_modalities, balancing)
        cache_path = os.path.join(cache_dir, f"combined_{cache_key}.pkl")
    else:
        cache_path = None

    if cache_enabled and cache_path and os.path.exists(cache_path):
        print(f"\n加载缓存数据: {cache_path}")
        combined = pd.read_pickle(cache_path)
    else:
        combined = load_and_merge_multimodal_datasets(
            data_files, feature_cols, dataset_modalities, balancing, source_order=source_order
        )
        if cache_enabled and cache_path:
            print(f"\n保存缓存数据: {cache_path}")
            combined.to_pickle(cache_path)
    # If global normalization already applied externally, allow disabling per-window normalization
    norm_method = None if config.get('disable_internal_window_norm', False) else config.get('norm_method')
    dataset = SlidingWindowDataset(
        data=combined,
        block_col=config.get('block_col', 'block'),
        feature_cols=feature_cols,
        window_size=int(config.get('window_size', 320)),
        step_size=int(config.get('step_size', 96)),
        sampling_rate=int(config.get('sampling_rate', config.get('sample_rate', 1))),
        normalize=norm_method,
        label_col=config.get('label_col', 'F'),
        phase=phase,
        dataset_modalities=dataset_modalities,
        common_modalities=common_modalities
    )
    return dataset


def check_label_distribution(dataset):
    import collections
    counter = collections.Counter()
    labels = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        # support (tensor,label) shape
        label = sample[1] if isinstance(sample, (list, tuple)) and len(sample) >= 2 else sample
        # normalize label to python int
        try:
            if isinstance(label, torch.Tensor):
                v = int(label.item())
            elif isinstance(label, np.ndarray):
                if label.size == 0:
                    continue
                v = int(label.reshape(-1)[0])
            elif isinstance(label, (list, tuple)):
                if len(label) == 0:
                    continue
                v = int(label[0])
            else:
                v = int(label)
        except Exception:
            # skip uncastable label
            continue
        counter[v] += 1
        labels.add(v)
    print("标签分布:", dict(counter))
    print("所有标签:", sorted(list(labels)))
    return counter, labels


# (moved to top)
