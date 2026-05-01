import torch
import torch.nn as nn


class SimpleBaselineClassifier(nn.Module):
    """A simple MLP baseline that flattens per-window channel*time input.
    Expects input shape [B, C, T].
    """
    def __init__(self, in_channels: int, time_len: int, num_classes: int = 2, hidden: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        input_dim = in_channels * time_len
        d = input_dim
        for i in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: [B, C, T]
        b = x.size(0)
        flat = x.view(b, -1)
        return self.net(flat)


def train_baseline(model, train_loader, device, epochs=5, lr=1e-3, use_present_mask=True,
                   val_loader=None, max_epochs=None, patience=20):
    """Train baseline classifier with optional early stopping.
    Args:
        epochs: (deprecated) kept for backward compatibility when max_epochs not provided.
        max_epochs: maximum training epochs (overrides epochs if given)
        patience: early stopping patience based on validation accuracy (or training acc if no val_loader)
    Returns: best_loss, best_acc
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_loss = float('inf')
    wait = 0
    total_epochs = int(max_epochs) if max_epochs is not None else int(epochs)
    for ep in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            if not isinstance(batch, (list, tuple)):
                continue
            if len(batch) >= 6:
                x, labels, modal_mask, source_ids, present_mask, missing_mask = batch[:6]
            elif len(batch) >= 2:
                x, labels = batch[:2]
                present_mask = None
            else:
                continue
            x = x.to(device)
            labels = labels.to(device)
            if use_present_mask and present_mask is not None:
                pm = present_mask.to(device).unsqueeze(-1)
                x = x * pm
            logits = model(x)
            loss = ce(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                preds = logits.argmax(1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += float(loss.detach().item()) * labels.size(0)
        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validation
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_samples = 0
            with torch.no_grad():
                for vb in val_loader:
                    if not isinstance(vb, (list, tuple)):
                        continue
                    if len(vb) >= 6:
                        x, labels, modal_mask, source_ids, present_mask, missing_mask = vb[:6]
                    elif len(vb) >= 2:
                        x, labels = vb[:2]
                        present_mask = None
                    else:
                        continue
                    x = x.to(device)
                    labels = labels.to(device)
                    if use_present_mask and present_mask is not None:
                        pm = present_mask.to(device).unsqueeze(-1)
                        x = x * pm
                    logits = model(x)
                    loss = ce(logits, labels)
                    preds = logits.argmax(1)
                    v_correct += (preds == labels).sum().item()
                    v_samples += labels.size(0)
                    v_loss += float(loss.detach().item()) * labels.size(0)
            val_loss = v_loss / max(v_samples, 1)
            val_acc = v_correct / max(v_samples, 1)
        else:
            val_loss = train_loss
            val_acc = train_acc

        improved = False
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            wait = 0
            improved = True
        else:
            wait += 1

        print(f"[Baseline] Epoch {ep} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}{' *' if improved else ''}")
        if wait >= patience:
            print(f"[Baseline] Early stopping at epoch {ep} (best_acc={best_acc:.4f})")
            break

    return best_loss, best_acc
