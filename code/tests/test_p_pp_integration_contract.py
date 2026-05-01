import unittest

import train


class TestPPIntegrationContract(unittest.TestCase):
    def test_warmup_pp_disables_classification(self):
        stage = train.select_stage(epoch=1, config={"stage1_epochs": 2, "stage2_epochs": 3})
        self.assertEqual(stage, "warmup")
        weights = train.stage_loss_weights(stage, {"loss_config": {"recon_weight": 1.0, "cls_diff_weight": 0.5}})
        flags = train.batch_loss_flags(stage, "PP")
        self.assertTrue(weights["recon_weight"] >= 0.0)
        self.assertEqual(weights["cls_weight"], 0.0)
        self.assertFalse(flags["classification"])

    def test_supervised_p_keeps_classification(self):
        stage = train.select_stage(epoch=3, config={"stage1_epochs": 2, "stage2_epochs": 3})
        self.assertEqual(stage, "supervised")
        weights = train.stage_loss_weights(stage, {"loss_config": {"recon_weight": 1.0, "cls_diff_weight": 0.5}})
        flags = train.batch_loss_flags(stage, "P")
        self.assertTrue(weights["recon_weight"] >= 0.0)
        self.assertTrue(weights["cls_weight"] >= 0.0)
        self.assertTrue(flags["classification"])


if __name__ == "__main__":
    unittest.main()
