import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import TGATUNet

def test_mask_input_changes_output():
    """相同窗口数据，不同 present_mask，输出应不同"""
    T, C = 32, 8
    model = TGATUNet(in_channels=C, hidden_channels=32, out_channels=C,
                     encoder_layers=2, decoder_layers=2, heads=4,
                     use_mask_input=True)
    model.eval()
    window = torch.randn(T, C)
    mask_all = torch.ones(C)
    mask_half = torch.zeros(C)
    mask_half[:C//2] = 1.0

    with torch.no_grad():
        recon_all, _, _ = model(window, present_mask=mask_all, return_latent=True)
        recon_half, _, _ = model(window, present_mask=mask_half, return_latent=True)

    assert not torch.allclose(recon_all, recon_half, atol=1e-4), \
        "mask 输入无效：不同 mask 产生了相同输出"

def test_mask_input_shape():
    T, C = 32, 8
    model = TGATUNet(in_channels=C, hidden_channels=32, out_channels=C,
                     encoder_layers=2, decoder_layers=2, heads=4,
                     use_mask_input=True)
    window = torch.randn(T, C)
    mask = torch.ones(C)
    recon, logits, latent = model(window, present_mask=mask, return_latent=True)
    assert recon.shape == (C, T)

def test_no_mask_input_backward_compat():
    """use_mask_input=False 时，不传 mask 也能正常运行"""
    T, C = 32, 8
    model = TGATUNet(in_channels=C, hidden_channels=32, out_channels=C,
                     encoder_layers=2, decoder_layers=2, heads=4,
                     use_mask_input=False)
    window = torch.randn(T, C)
    recon, logits, latent = model(window, return_latent=True)
    assert recon.shape == (C, T)
