import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDomainAdaptationIntegration(unittest.TestCase):

    def test_domain_adaptation_importable(self):
        import torch
        from domain_adaptation import DomainAdaptationModule

        da = DomainAdaptationModule(
            feature_dim=32,
            num_domains=3,
            use_adversarial=True,
            use_mmd=True,
            use_coral=False,
        )
        feats = torch.randn(8, 32)
        labels = torch.randint(0, 3, (8,))
        out = da(feats, labels)
        self.assertIn("total", out)
        self.assertIn("adversarial", out)
        self.assertIn("mmd", out)
        self.assertTrue(out["total"].item() >= 0.0)

    def test_stage_loss_weights_includes_domain_weight(self):
        import train

        cfg = {
            "loss_config": {"recon_weight": 1.0, "cls_diff_weight": 0.5},
            "domain_adaptation": {
                "enabled": True,
                "adversarial": {"enabled": True, "weight": 0.1},
                "mmd": {"enabled": True, "weight": 0.05},
            },
        }
        weights = train.stage_loss_weights("B", cfg)
        self.assertIn("domain_weight", weights)
        self.assertAlmostEqual(weights["domain_weight"], 0.15, places=5)

    def test_stage_loss_weights_domain_weight_zero_when_disabled(self):
        import train

        cfg = {
            "loss_config": {"recon_weight": 1.0, "cls_diff_weight": 0.5},
            "domain_adaptation": {"enabled": False},
        }
        weights = train.stage_loss_weights("B", cfg)
        self.assertIn("domain_weight", weights)
        self.assertEqual(weights["domain_weight"], 0.0)

    def test_train_one_epoch_has_domain_weight_param(self):
        """train_one_epoch signature must include domain_weight parameter."""
        import inspect
        import train

        sig = inspect.signature(train.train_one_epoch)
        self.assertIn("domain_weight", sig.parameters)
        self.assertIn("domain_module", sig.parameters)
        self.assertIn("n_p_sources", sig.parameters)

    def test_train_one_epoch_returns_16_tuple(self):
        """train_one_epoch must return 16 values including domain, p_recon, pp_recon."""
        import torch
        import train

        # Build a minimal dummy dataloader with a single P batch
        B, C, T = 4, 3, 10
        dummy_batch = (
            torch.randn(B, C, T),           # batch tensors
            torch.zeros(B, dtype=torch.long),  # labels
            torch.ones(B, C),               # modal_mask
            torch.zeros(B, dtype=torch.long),  # source_ids
            torch.ones(B, C),               # present_mask
            torch.zeros(B, C),              # missing_mask
            torch.arange(B),               # sample_indices
        )

        class TinyLoader:
            def __iter__(self):
                yield dummy_batch

        # Build a trivial model stub
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(C * T, C * T)
                self.in_channels = C

            def forward_batch(self, windows):
                B2, T2, C2 = windows.shape
                flat = windows.reshape(B2, -1)
                out = self.fc(flat).reshape(B2, T2, C2).permute(0, 2, 1)
                logits = torch.zeros(B2, 2)
                latents = torch.zeros(B2, C2)
                return out, logits, latents

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        result = train.train_one_epoch(
            model=model,
            dataloader=TinyLoader(),
            optimizer=optimizer,
            device=device,
            accumulate_grad_batches=1,
            use_mixed_precision=False,
            scaler=None,
            recon_weight=1.0,
            cls_diff_weight=0.0,
            cls_diff_clip=10.0,
        )
        self.assertEqual(len(result), 16, f"Expected 16-tuple, got {len(result)}-tuple")
        # indices 13, 14, 15 are domain, p_recon, pp_recon
        tr_domain, tr_p_recon, tr_pp_recon = result[13], result[14], result[15]
        self.assertIsInstance(float(tr_domain), float)
        self.assertIsInstance(float(tr_p_recon), float)
        self.assertIsInstance(float(tr_pp_recon), float)


if __name__ == "__main__":
    unittest.main()
