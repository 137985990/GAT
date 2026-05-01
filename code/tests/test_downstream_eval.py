import math
import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


class TestDownstreamEval(unittest.TestCase):
    def test_resolve_variants_accepts_requested_names(self):
        from paper_downstream_pipeline import resolve_variants

        variants = resolve_variants([
            "no_backfill",
            "backfill_no_guide",
            "full_system",
            "acc_gate_only",
            "moment_gate_only",
            "both_gates",
        ])
        self.assertEqual(
            set(variants.keys()),
            {
                "no_backfill",
                "backfill_no_guide",
                "full_system",
                "acc_gate_only",
                "moment_gate_only",
                "both_gates",
            },
        )

    def test_build_protocol_snapshot_reports_key_runtime_settings(self):
        from paper_downstream_pipeline import build_protocol_snapshot

        cfg = {
            "epochs": 100,
            "early_stopping_patience": 30,
            "batch_size": 128,
            "num_workers": 0,
            "learning_rate": 5e-4,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "aux_classifier": {"epochs": 20, "patience": 3, "type": "cnn"},
            "backfill_gate": {"warmup_epochs": 4, "interval": 5, "enable_stats": True},
        }

        snapshot = build_protocol_snapshot(
            "config.yaml",
            cfg,
            ["no_backfill", "full_system"],
            [42, 43, 44],
            ["mlp", "cnn1d"],
        )

        self.assertEqual(snapshot["config_path"], "config.yaml")
        self.assertEqual(snapshot["variants"], ["no_backfill", "full_system"])
        self.assertEqual(snapshot["seeds"], [42, 43, 44])
        self.assertEqual(snapshot["backbones"], ["mlp", "cnn1d"])
        self.assertEqual(snapshot["stage1"]["epochs"], 100)
        self.assertEqual(snapshot["stage1"]["aux_epochs"], 20)
        self.assertEqual(snapshot["stage1"]["gate_warmup_epochs"], 4)

    def test_moment_gate_only_disables_accuracy_ranked_topk_filter(self):
        from paper_downstream_pipeline import resolve_variants

        variant = resolve_variants(["moment_gate_only"])["moment_gate_only"]
        gate = variant["backfill_gate_override"]

        self.assertTrue(gate["enable_stats"])
        self.assertTrue(gate["disable_acc_gate"])
        self.assertEqual(gate.get("keep_max_per_batch"), 0)

    def test_build_downstream_classifier_supports_all_backbones(self):
        from downstream_eval import build_downstream_classifier

        in_channels = 17
        time_len = 10
        names = ["mlp", "cnn1d", "lstm", "transformer"]
        models = [
            build_downstream_classifier(name, in_channels, time_len, num_classes=2)
            for name in names
        ]

        batch = torch.randn(4, in_channels, time_len)
        for model in models:
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (4, 2))

    def test_compute_classification_metrics_returns_expected_keys(self):
        from downstream_eval import compute_classification_metrics

        probs = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        metrics = compute_classification_metrics(probs, labels)

        self.assertEqual(set(metrics.keys()), {"acc", "f1_bin", "roc_auc", "pr_auc"})
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
