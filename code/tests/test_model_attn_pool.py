import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import AttentionPooling

def test_attention_pooling_shape():
    T, hid = 20, 32
    pool = AttentionPooling(hid)
    h = torch.randn(T, hid)
    out = pool(h)
    assert out.shape == (hid,), f"期望 ({hid},)，实际 {out.shape}"

def test_attention_pooling_not_uniform():
    """注意力权重不应全部相等"""
    torch.manual_seed(42)
    T, hid = 20, 32
    pool = AttentionPooling(hid)
    h = torch.randn(T, hid)
    with torch.no_grad():
        scores = pool.score(h).squeeze(-1)  # [T]
        weights = torch.softmax(scores, dim=0)
    assert weights.std().item() > 1e-4, "注意力权重过于均匀"

def test_attention_pooling_differentiable():
    T, hid = 8, 16
    pool = AttentionPooling(hid)
    h = torch.randn(T, hid, requires_grad=True)
    out = pool(h)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None
