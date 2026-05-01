import unittest
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model import TGATUNet


class TestModelInputOrientation(unittest.TestCase):
    def test_channel_first_window_is_transposed_before_graph_encoding(self):
        model = TGATUNet(in_channels=20, hidden_channels=16, out_channels=20, num_classes=2)
        window = torch.randn(20, 128)
        recon, logits, latent = model(window, return_latent=True)
        self.assertEqual(tuple(recon.shape), (20, 128))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(latent.ndim, 1)


if __name__ == '__main__':
    unittest.main()
