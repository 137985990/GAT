import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import GraphEncoder, GraphDecoder, TGATUNet

def _make_chain_graph(T, C):
    """构造链式图 edge_index，T 个节点"""
    edges = []
    for i in range(T):
        for j in range(max(0, i-1), min(T, i+2)):
            edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.randn(T, C)
    return x, edge_index

def test_graph_encoder_returns_intermediates():
    T, C, hid = 16, 8, 32
    x, ei = _make_chain_graph(T, C)
    enc = GraphEncoder(C, hid, layers=3, heads=4)
    out, intermediates = enc(x, ei, return_intermediates=True)
    assert out.shape == (T, hid)
    assert len(intermediates) == 3, f"期望 3 个中间层，实际 {len(intermediates)}"
    for h in intermediates:
        assert h.shape == (T, hid)

def test_graph_decoder_accepts_skips():
    T, hid, C = 16, 32, 8
    _, ei = _make_chain_graph(T, C)
    dec = GraphDecoder(hid, C, layers=3, heads=4)
    h = torch.randn(T, hid)
    skips = [torch.randn(T, hid) for _ in range(3)]
    out = dec(h, ei, skip_list=skips)
    assert out.shape == (T, C)

def test_tgatunet_unet_skip_end_to_end():
    T, C = 32, 8
    model = TGATUNet(in_channels=C, hidden_channels=32, out_channels=C,
                     encoder_layers=3, decoder_layers=3, heads=4,
                     use_unet_skip=True)
    window = torch.randn(T, C)
    recon, logits, latent = model(window, return_latent=True)
    assert recon.shape == (C, T)
