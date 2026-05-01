import unittest
import sys
from pathlib import Path

import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from train import train_aux_classifier


class TinyAuxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 2)

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


def make_loader():
    batch = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [0.5, 0.0, 0.5]],
            [[0.0, 1.0, 0.0], [0.2, 0.8, 0.2]],
            [[1.0, 1.0, 0.0], [0.4, 0.3, 0.2]],
            [[0.0, 0.0, 1.0], [0.7, 0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    modal_mask = torch.ones(4, 2, dtype=torch.float32)
    source_ids = torch.zeros(4, dtype=torch.long)
    present_mask = torch.ones(4, 2, dtype=torch.float32)
    missing_mask = torch.zeros(4, 2, dtype=torch.float32)
    sample_indices = torch.arange(4, dtype=torch.long)
    return [(batch, labels, modal_mask, source_ids, present_mask, missing_mask, sample_indices)]


class TestAuxClassifierRetrain(unittest.TestCase):
    def test_train_aux_classifier_reenables_frozen_parameters(self):
        model = TinyAuxClassifier()
        for param in model.parameters():
            param.requires_grad = False

        before = model.fc.weight.detach().clone()
        train_aux_classifier(
            model,
            make_loader(),
            torch.device("cpu"),
            epochs=1,
            lr=1e-2,
        )

        self.assertTrue(any(param.requires_grad for param in model.parameters()))
        self.assertFalse(torch.equal(before, model.fc.weight.detach()))


if __name__ == "__main__":
    unittest.main()
