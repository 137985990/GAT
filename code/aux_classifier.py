import torch
import torch.nn as nn

from baseline_classifier import SimpleBaselineClassifier


class AuxCNNClassifier(nn.Module):
    """Lightweight 1D CNN classifier for [B, C, T] input."""
    def __init__(self, in_channels: int, num_classes: int = 2, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class AuxLSTMClassifier(nn.Module):
    """Simple LSTM classifier for [B, C, T] input."""
    def __init__(self, in_channels: int, num_classes: int = 2, hidden: int = 128, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class AuxTransformerClassifier(nn.Module):
    """Minimal Transformer encoder classifier for [B, C, T] input."""
    def __init__(self, in_channels: int, num_classes: int = 2, d_model: int = 128,
                 nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.proj(x)
        h = self.encoder(x)
        pooled = h.mean(dim=1)
        return self.fc(pooled)


def build_aux_classifier(aux_cfg: dict, in_channels: int, time_len: int, num_classes: int = 2) -> nn.Module:
    ctype = str(aux_cfg.get('type', 'mlp')).lower()
    hidden = int(aux_cfg.get('hidden', 256))
    depth = int(aux_cfg.get('depth', 2))
    dropout = float(aux_cfg.get('dropout', 0.1))
    if ctype == 'mlp':
        return SimpleBaselineClassifier(in_channels, time_len, num_classes=num_classes, hidden=hidden, depth=depth, dropout=dropout)
    if ctype == 'cnn':
        return AuxCNNClassifier(in_channels, num_classes=num_classes, hidden=hidden, dropout=dropout)
    if ctype == 'lstm':
        layers = int(aux_cfg.get('layers', 1))
        return AuxLSTMClassifier(in_channels, num_classes=num_classes, hidden=hidden, layers=layers, dropout=dropout)
    if ctype == 'transformer':
        d_model = int(aux_cfg.get('d_model', hidden))
        nhead = int(aux_cfg.get('nhead', 4))
        layers = int(aux_cfg.get('layers', 2))
        return AuxTransformerClassifier(in_channels, num_classes=num_classes, d_model=d_model, nhead=nhead, layers=layers, dropout=dropout)
    raise ValueError(f"Unknown aux classifier type: {ctype}")
