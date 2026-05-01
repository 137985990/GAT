"""
集成冒烟测试：所有新特性同时开启时的端到端前向 + 反向测试。
不检查准确率，只验证流程可通、无 NaN、梯度可流。
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import TGATUNet

def test_full_model_forward_all_features():
    """所有新特性同时开启时的前向传播"""
    T, C, hid = 40, 8, 32
    model = TGATUNet(
        in_channels=C, hidden_channels=hid, out_channels=C,
        encoder_layers=3, decoder_layers=3, heads=4,
        channel_attn=True, channel_attn_heads=4,
        use_unet_skip=True,
        trans_use_pos_enc=True,
        use_mask_input=True,
    )
    model.eval()
    window = torch.randn(T, C)
    mask = torch.ones(C)
    mask[C//2:] = 0.0

    with torch.no_grad():
        recon, logits, latent = model(window, present_mask=mask, return_latent=True)

    assert recon.shape == (C, T), f"recon shape: {recon.shape}"
    assert latent.shape == (hid,), f"latent shape: {latent.shape}"
    assert not torch.isnan(recon).any(), "recon 含 NaN"
    assert not torch.isnan(latent).any(), "latent 含 NaN"

def test_model_backward_pass():
    """梯度能正常反传"""
    T, C, hid = 20, 6, 16
    model = TGATUNet(
        in_channels=C, hidden_channels=hid, out_channels=C,
        encoder_layers=2, decoder_layers=2, heads=2,
        channel_attn=True, channel_attn_heads=2,
        use_unet_skip=True,
        trans_use_pos_enc=True,
        use_mask_input=True,
    )
    window = torch.randn(T, C)
    mask = torch.ones(C)
    recon, _, _ = model(window, present_mask=mask, return_latent=True)
    target = torch.randn(C, T)
    loss = (recon - target).pow(2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "没有参数收到梯度，反传失败"

def test_all_features_disabled():
    """所有新特性关闭时退化为旧行为也能正常运行"""
    T, C, hid = 32, 8, 32
    model = TGATUNet(
        in_channels=C, hidden_channels=hid, out_channels=C,
        encoder_layers=2, decoder_layers=2, heads=4,
        channel_attn=False,
        use_unet_skip=False,
        trans_use_pos_enc=False,
        use_mask_input=False,
    )
    model.eval()
    window = torch.randn(T, C)
    with torch.no_grad():
        recon, logits, latent = model(window, return_latent=True)
    assert recon.shape == (C, T)
    assert not torch.isnan(recon).any()
