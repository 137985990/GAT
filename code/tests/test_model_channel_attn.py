import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import ChannelCrossAttention, TGATUNet

def test_channel_cross_attention_shape():
    """输入 [T, C]，输出形状相同，且不等于输入（有信息流动）"""
    T, C = 32, 8
    x = torch.randn(T, C)
    attn = ChannelCrossAttention(C, heads=4)
    out = attn(x)
    assert out.shape == (T, C), f"期望 ({T},{C})，实际 {out.shape}"
    assert not torch.allclose(out, x), "输出与输入完全相同，通道注意力无效"

def test_tgatunet_uses_channel_attn():
    """TGATUNet 启用 channel_attn 时前向不崩溃，输出形状正确"""
    T, C = 32, 8
    model = TGATUNet(in_channels=C, hidden_channels=32, out_channels=C,
                     encoder_layers=2, decoder_layers=2, heads=4,
                     channel_attn=True, channel_attn_heads=4)
    window = torch.randn(T, C)
    recon, logits, latent = model(window, return_latent=True)
    assert recon.shape == (C, T), f"recon shape 错误: {recon.shape}"

def test_tgatunet_no_channel_attn_backward_compat():
    """channel_attn=False 时行为与旧版一致"""
    T, C = 32, 8
    model = TGATUNet(in_channels=C, hidden_channels=32, out_channels=C,
                     encoder_layers=2, decoder_layers=2, heads=4,
                     channel_attn=False)
    window = torch.randn(T, C)
    recon, logits, latent = model(window, return_latent=True)
    assert recon.shape == (C, T)


def test_tgatunet_channel_attn_auto_adjusts_incompatible_head_count():
    """当 enc_in 不能被请求的 head 数整除时，自动退到最近可用 head 数。"""
    T, C = 32, 17
    model = TGATUNet(
        in_channels=C,
        hidden_channels=32,
        out_channels=C,
        encoder_layers=2,
        decoder_layers=2,
        heads=4,
        channel_attn=True,
        channel_attn_heads=4,
        use_mask_input=True,
    )
    assert model.channel_attn is not None
    assert model.channel_attn.attn.num_heads == 2

    window = torch.randn(T, C)
    mask = torch.ones(C)
    recon, logits, latent = model(window, present_mask=mask, return_latent=True)
    assert recon.shape == (C, T)
