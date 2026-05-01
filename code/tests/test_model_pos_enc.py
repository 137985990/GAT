import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import TransformerBottleneck

def test_positional_encoding_breaks_permutation_symmetry():
    """有 PE 时，打乱输入顺序后输出应与原始顺序不同"""
    torch.manual_seed(0)
    T, hid = 16, 32
    bottleneck = TransformerBottleneck(hid, heads=4, layers=2, ff=64, use_pos_enc=True)
    bottleneck.eval()

    x = torch.randn(T, hid)
    perm = torch.randperm(T)
    x_perm = x[perm]

    with torch.no_grad():
        out = bottleneck(x)
        out_perm = bottleneck(x_perm)

    assert not torch.allclose(out[perm], out_perm, atol=1e-4), \
        "PE 无效：打乱输入后输出与原输出置换结果相同"

def test_transformer_bottleneck_shape_unchanged():
    """输出形状不变"""
    T, hid = 20, 32
    bn = TransformerBottleneck(hid, heads=4, layers=2, ff=64, use_pos_enc=True)
    x = torch.randn(T, hid)
    out = bn(x)
    assert out.shape == (T, hid)
