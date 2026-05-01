"""Authoritative clean replacement for model.py (v99).

Usage:
    from V99.model import TGATUNet
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from graph import build_graph

_nan_once = {"hit": False}

def _nan_guard(tag: str, t: torch.Tensor | None):
    if _nan_once["hit"] or t is None:
        return
    if torch.isnan(t).any() or (~torch.isfinite(t)).any():
        print(f"[NaNGuard][{tag}] shape={tuple(t.shape)} min={t.min():.2e} max={t.max():.2e} mean={t.mean():.2e}")
        _nan_once["hit"] = True

class AttentionPooling(nn.Module):
    """
    对 [T, hid] 的节点表示做注意力加权求和，输出 [hid]。
    比 mean pooling 保留更多时序结构信息。
    """
    def __init__(self, hid: int):
        super().__init__()
        self.score = nn.Linear(hid, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [T, hid]
        weights = torch.softmax(self.score(h), dim=0)  # [T, 1]
        return (weights * h).sum(dim=0)                # [hid]


class ChannelCrossAttention(nn.Module):
    """
    跨通道注意力：在时间步维度做 self-attention，每个时间步的 embed=C 携带所有通道信息，
    使各时间步能聚合彼此的跨通道相关性，等效于学习跨模态依赖。
    输入 x: [T, C]  →  输出 [T, C]
    """
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % heads == 0, f"channels({channels}) 须被 heads({heads}) 整除"
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, C] → [1, T, C]，batch=1, seq=T, embed=C
        residual = x
        x_3d = x.unsqueeze(0)             # [1, T, C]
        attn_out, _ = self.attn(x_3d, x_3d, x_3d)  # [1, T, C]
        attn_out = attn_out.squeeze(0)    # [T, C]
        return self.norm(residual + self.drop(attn_out))


class GraphEncoder(nn.Module):
    def __init__(self, in_ch: int, hid: int, layers: int = 3, heads: int = 4, sanitize: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([GATConv(in_ch, hid // heads, heads=heads)])
        for _ in range(layers - 1):
            self.layers.append(GATConv(hid, hid // heads, heads=heads))
        self.act = nn.ReLU()
        self.sanitize = sanitize
    def forward(self, x, edge_index, return_intermediates: bool = False):
        intermediates = []
        for i, gat in enumerate(self.layers):
            x = self.act(gat(x, edge_index))
            if self.sanitize:
                if torch.isnan(x).any() or (~torch.isfinite(x)).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                x = torch.clamp(x, -1e4, 1e4)
            _nan_guard(f"enc{i}", x)
            intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x

class SinusoidalPE(nn.Module):
    """固定正弦余弦位置编码，支持任意序列长度。"""
    def __init__(self, hid: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, hid)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hid, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / hid))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:hid // 2])
        self.register_buffer('pe', pe)  # [max_len, hid]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, hid] 或 [1, T, hid]
        if x.dim() == 3:
            T = x.size(1)
            return x + self.pe[:T].unsqueeze(0)
        T = x.size(0)
        return x + self.pe[:T]


class TransformerBottleneck(nn.Module):
    def __init__(self, hid: int, heads: int = 4, layers: int = 2, ff: int = 512,
                 drop: float = 0.1, use_pos_enc: bool = True):
        super().__init__()
        block = nn.TransformerEncoderLayer(hid, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(block, num_layers=layers)
        self.pre_ln = nn.LayerNorm(hid)
        self.post_ln = nn.LayerNorm(hid)
        self.pos_enc = SinusoidalPE(hid) if use_pos_enc else None

    def forward(self, x):
        # x: [1, T, hid]（从 TGATUNet.forward 传入时已 unsqueeze(0)）
        x = self.pre_ln(x)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        y = self.enc(x)
        y = self.post_ln(y)
        _nan_guard("trans", y)
        return y

class GraphDecoder(nn.Module):
    def __init__(self, hid: int, out_ch: int, layers: int = 3, heads: int = 4, sanitize: bool = False):
        super().__init__()
        blocks: list[nn.Module] = [GATConv(hid, hid // heads, heads=heads)]
        for _ in range(layers - 2):
            blocks.append(GATConv(hid, hid // heads, heads=heads))
        blocks.append(GATConv(hid, out_ch, heads=1))
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.ReLU()
        self.sanitize = sanitize
    def forward(self, x, edge_index, skip_list=None):
        """
        skip_list: List[[T, hid]]，长度等于 decoder 层数（含最后一层）
                   skip_list[0] 对应最深的编码器特征，skip_list[-1] 对应最浅
        """
        for i in range(len(self.blocks) - 1):
            if skip_list is not None and i < len(skip_list):
                x = x + skip_list[i]
            gat = self.blocks[i]
            x = self.act(gat(x, edge_index))
            if self.sanitize:
                if torch.isnan(x).any() or (~torch.isfinite(x)).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                x = torch.clamp(x, -1e4, 1e4)
            _nan_guard(f"dec{i}", x)
        # 最后一层（输出层）
        last_skip_idx = len(self.blocks) - 1
        if skip_list is not None and last_skip_idx < len(skip_list):
            x = x + skip_list[last_skip_idx]
        x = self.blocks[-1](x, edge_index)
        _nan_guard("dec_last", x)
        return x

class TGATUNet(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, num_classes:int=2,
                 encoder_layers:int=3, decoder_layers:int=3, heads:int=4, time_k:int=1,
                 trans_nhead:int=4, trans_layers:int=2, trans_dim_feedforward:int=512,
                 simple_encoder_fallback: bool=False, sanitize_nans: bool = True, clamp_value: float = 1e4,
                 disable_model_layernorm: bool = False,
                 disable_transformer: bool = False,
                 noise_sigma: float = 0.0,
                 channel_attn: bool = True,
                 channel_attn_heads: int = 4,
                 channel_attn_dropout: float = 0.1,
                 use_unet_skip: bool = True,
                 trans_use_pos_enc: bool = True,
                 use_mask_input: bool = True,
                 **_):
        super().__init__()
        self.time_k = time_k
        self.disable_transformer = disable_transformer
        self.noise_sigma = noise_sigma  # Sensitivity analysis: noise injection level
        self.num_classes = num_classes
        self.use_mask_input = use_mask_input
        self.in_channels = int(in_channels)
        # 若使用 mask 输入，encoder 的输入维从 in_channels 变为 in_channels * 2
        enc_in = in_channels * 2 if use_mask_input else in_channels
        self.pre_norm = nn.Identity() if disable_model_layernorm else nn.LayerNorm(enc_in)
        # 跨通道注意力模块（可选），适配 enc_in（mask 拼接后的维度）
        if channel_attn:
            effective_heads = max(1, min(int(channel_attn_heads), enc_in))
            while effective_heads > 1 and (enc_in % effective_heads != 0):
                effective_heads -= 1
            self.channel_attn = ChannelCrossAttention(enc_in, heads=effective_heads,
                                                       dropout=channel_attn_dropout)
        else:
            self.channel_attn = None
        self.encoder = GraphEncoder(enc_in, hidden_channels, layers=encoder_layers, heads=heads, sanitize=sanitize_nans)
        
        # Ablation: 可选的Transformer bottleneck
        if disable_transformer:
            # 用Identity替代Transformer，直接传递encoder输出
            self.bottleneck = nn.Identity()
            self.skip = nn.Linear(hidden_channels, hidden_channels)  # skip只需要encoder输出
        else:
            self.bottleneck = TransformerBottleneck(hidden_channels, heads=trans_nhead, layers=trans_layers, ff=trans_dim_feedforward, use_pos_enc=trans_use_pos_enc)
            self.skip = nn.Linear(hidden_channels*2, hidden_channels)  # 原来的concat方式
        
        if disable_model_layernorm and not disable_transformer:
            # Monkey patch forward to skip internal norms without altering attribute types
            original_forward = self.bottleneck.forward
            def patched_forward(x):
                # call encoder on raw x (skip pre/post layer norms)
                y = self.bottleneck.enc(x)
                _nan_guard("trans", y)
                return y
            self.bottleneck.forward = patched_forward  # type: ignore
        
        self.decoder = GraphDecoder(hidden_channels, out_channels, layers=decoder_layers, heads=heads, sanitize=sanitize_nans)
        self.pool = AttentionPooling(hidden_channels)

        self.simple_encoder_fallback = simple_encoder_fallback
        self.sanitize_nans = sanitize_nans
        self.clamp_value = clamp_value
        self.use_unet_skip = use_unet_skip
        if simple_encoder_fallback:
            self.fb = nn.Linear(in_channels, hidden_channels)
    def forward(self, window: torch.Tensor, phase: str='encode', return_latent: bool=True,
                present_mask=None):
        device = next(self.parameters()).device
        if window.device != device:
            window = window.to(device)
        if window.dim() == 2 and window.size(0) == self.in_channels and window.size(1) != self.in_channels:
            window = window.t().contiguous()
        data = build_graph(window, time_k=self.time_k)
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        # 调试：前 3 次打印图结构信息，帮助确认边数量是否异常导致表示恒定
        if not hasattr(self, '_edge_dbg'): self._edge_dbg = 0
        if self._edge_dbg < 3:
            try:
                print(f"[GraphDebug] edge_index shape={edge_index.shape}, nodes={x.shape[0]}, edges={edge_index.size(1)}")
            except Exception:
                pass
            self._edge_dbg += 1
        _nan_guard('input', x)
        # 将 present_mask 拼接到输入特征（若启用）
        if self.use_mask_input:
            if present_mask is None:
                mask_feat = torch.ones(x.size(0), x.size(1), device=x.device)
            else:
                pm = present_mask.to(x.device)
                mask_feat = pm.unsqueeze(0).expand(x.size(0), -1)  # [T, C]
            x = torch.cat([x, mask_feat], dim=-1)  # [T, 2C]
        x = self.pre_norm(x)
        # 跨通道注意力（可选）
        if self.channel_attn is not None:
            x = self.channel_attn(x)
            if self.sanitize_nans:
                x = torch.nan_to_num(x, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
                x = torch.clamp(x, -self.clamp_value, self.clamp_value)
        if self.use_unet_skip:
            h_enc, enc_intermediates = self.encoder(x, edge_index, return_intermediates=True)
        else:
            h_enc = self.encoder(x, edge_index)
            enc_intermediates = None
        if self.sanitize_nans:
            if torch.isnan(h_enc).any() or (~torch.isfinite(h_enc)).any():
                h_enc = torch.nan_to_num(h_enc, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
            h_enc = torch.clamp(h_enc, -self.clamp_value, self.clamp_value)
        if (torch.isnan(h_enc).any() or (~torch.isfinite(h_enc)).any()) and self.simple_encoder_fallback:
            h_enc = self.fb(x)
            _nan_guard('fallback', h_enc)
        
        # Sensitivity analysis: inject Gaussian noise to encoder output
        if self.noise_sigma > 0.0 and self.training:
            noise = torch.randn_like(h_enc) * self.noise_sigma
            h_enc = h_enc + noise
            if self.sanitize_nans:
                h_enc = torch.clamp(h_enc, -self.clamp_value, self.clamp_value)
        
        # Ablation: 根据是否禁用Transformer选择不同的路径
        if self.disable_transformer:
            # 直接使用encoder输出，跳过Transformer
            h = self.skip(h_enc)
        else:
            # 原来的Transformer路径
            h_trans = self.bottleneck(h_enc.unsqueeze(0)).squeeze(0)
            _nan_guard('trans_out', h_trans)
            if self.sanitize_nans:
                if torch.isnan(h_trans).any() or (~torch.isfinite(h_trans)).any():
                    h_trans = torch.nan_to_num(h_trans, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
                h_trans = torch.clamp(h_trans, -self.clamp_value, self.clamp_value)
            h = self.skip(torch.cat([h_enc, h_trans], dim=-1))
        if self.sanitize_nans:
            if torch.isnan(h).any() or (~torch.isfinite(h)).any():
                h = torch.nan_to_num(h, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
            h = torch.clamp(h, -self.clamp_value, self.clamp_value)
        pooled = self.pool(h)
        logits = torch.zeros(self.num_classes, device=pooled.device)
        latent = pooled
        skip_list = list(reversed(enc_intermediates)) if enc_intermediates is not None else None
        recon = self.decoder(h, edge_index, skip_list=skip_list)
        if self.sanitize_nans:
            if torch.isnan(recon).any() or (~torch.isfinite(recon)).any():
                recon = torch.nan_to_num(recon, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
            recon = torch.clamp(recon, -self.clamp_value, self.clamp_value)
        recon = recon.t()
        if phase == 'encode':
            return (recon, logits, latent) if return_latent else (recon, logits)
        elif phase == 'decode':
            return (recon, None, latent) if return_latent else (recon, None)
        return (recon, logits, latent) if return_latent else (recon, logits)
    def forward_batch(self, windows_batch: torch.Tensor):
        device = next(self.parameters()).device
        if windows_batch.device != device:
            windows_batch = windows_batch.to(device)
        outs, logits, latents = [], [], []
        for w in windows_batch:
            out_tuple = self.forward(w, return_latent=True)
            if len(out_tuple) == 3:
                recon, logit, latent = out_tuple
            elif len(out_tuple) == 2:
                recon, logit = out_tuple
                latent = recon.mean(dim=-1) if recon.ndim >= 2 else recon
            else:
                recon = out_tuple[0]
                logit = torch.zeros(1, device=recon.device)
                latent = recon.mean(dim=-1) if recon.ndim >= 2 else recon
            outs.append(recon); logits.append(logit); latents.append(latent)
        return torch.stack(outs), torch.stack(logits), torch.stack(latents)

# End of clean TGATUNet definition
