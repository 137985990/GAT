import os
import sys
import types
import importlib.util
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
from typing import cast
import unittest


class TestPpLossRoutingAPI(unittest.TestCase):
    def test_route_classification_api_exists(self):
        """
        Test-first: the training module must expose a routing API that distinguishes
        P vs PP batches so that PP batches skip classification loss while P batches
        retain classification-related routing. This API does not exist yet and
        this test should fail until the feature is implemented.
        """
        # Inject minimal dummy modules so importing train.py does not raise
        # ImportError for missing project modules. We do NOT execute training.
        dummy_modules = {}

        m = types.ModuleType("data")
        m.__dict__["create_multimodal_dataset_from_config"] = (
            lambda cfg, phase="encode": None
        )
        m.__dict__["load_config"] = lambda path: {}
        m.__dict__["check_label_distribution"] = lambda ds: ({}, [])
        dummy_modules["data"] = m

        m = types.ModuleType("center_loss")

        class CenterLoss:
            def __init__(self, *a, **k):
                pass

        m.__dict__["CenterLoss"] = CenterLoss
        dummy_modules["center_loss"] = m

        m = types.ModuleType("aux_classifier")
        m.__dict__["build_aux_classifier"] = lambda cfg, in_ch, tlen, num_classes=2: (
            None
        )
        dummy_modules["aux_classifier"] = m

        m = types.ModuleType("model")

        class TGATUNet:
            def __init__(self, *a, **k):
                pass

            def to(self, device):
                return self

            def parameters(self):
                return []

            def state_dict(self):
                return {}

        m.__dict__["TGATUNet"] = TGATUNet
        dummy_modules["model"] = m

        # optional integration module
        m = types.ModuleType("simple_multimodal_integration")
        m.__dict__["create_simple_multimodal_criterion"] = lambda: None
        dummy_modules["simple_multimodal_integration"] = m

        # Insert into sys.modules temporarily
        saved = {}
        for name, mod in dummy_modules.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mod

        try:
            # Locate train.py relative to this test file or fall back to absolute path
            candidates = [
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "train.py")
                ),
                r"E:\GPTGAT2\code\train.py",
                "train.py",
            ]
            train_path = None
            for c in candidates:
                if c and os.path.exists(c):
                    train_path = c
                    break
            self.assertIsNotNone(
                train_path,
                "train.py not found for import (expected at E:/GPTGAT2/code/train.py)",
            )

            spec = importlib.util.spec_from_file_location("train", train_path)
            if spec is None:
                self.fail(
                    "importlib.spec_from_file_location returned None for train.py"
                )
            spec = cast(ModuleSpec, spec)
            module = importlib.util.module_from_spec(spec)
            sys.modules["train"] = module
            # load module (executes top-level but we provided dummy deps)
            loader = getattr(spec, "loader", None)
            if loader is None:
                self.fail("spec.loader is None for train.py spec")
            loader = cast(Loader, loader)
            loader.exec_module(module)

            # Desired API to be implemented in the future:
            # - train.route_classification_for_batch(batch_meta) -> bool or mask
            # The feature: PP batches skip classification loss, P batches retain it.
            self.assertTrue(
                hasattr(module, "route_classification_for_batch"),
                "Expected train.route_classification_for_batch to exist: it should route PP vs P batches (PP skip classification loss, P retain).",
            )
        finally:
            # restore sys.modules
            for name, mod in saved.items():
                if mod is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = mod


if __name__ == "__main__":
    unittest.main()
