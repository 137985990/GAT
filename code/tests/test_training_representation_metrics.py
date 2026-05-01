import inspect
import unittest


class TestTrainingRepresentationMetricsAPI(unittest.TestCase):
    """Test-first: declare the helper API surface we will implement next.

    The test intentionally does not execute training code. It inspects
    train.py to assert the presence of minimal helper names and
    signatures required for representation metrics, classification-gap
    metrics, and anti-forgetting losses. The test should fail cleanly
    if any helper is missing (AttributeError) rather than raising
    syntax/import errors.
    """

    _train_module = None

    @classmethod
    def setUpClass(cls):
        # import lazily to allow import-time heavy deps to exist; we
        # expect train.py to import fine in the environment. Use
        # importlib to load module by path via runpy-like technique.
        import importlib.util
        from pathlib import Path

        train_path = Path(__file__).resolve().parents[1] / "train.py"
        spec = importlib.util.spec_from_file_location("train", str(train_path))
        assert spec is not None, "unable to locate train.py"
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None, "unable to load train.py"
        # execute module in its own namespace
        loader.exec_module(module)  # type: ignore[attr-defined]
        cls._train_module = module

    def _assert_callable(self, name):
        m = self._train_module
        self.assertTrue(hasattr(m, name), f"train.py must define '{name}'")
        obj = getattr(m, name)
        self.assertTrue(callable(obj), f"'{name}' must be callable")
        return obj

    def test_representation_metrics_helpers_exist(self):
        """Representation / collapse metrics helpers expected:

        - representation_collapse_score(latents: Tensor) -> float
        - representation_pairwise_distance(a: Tensor, b: Tensor) -> Tensor
        - compute_representation_stats(latents: Tensor) -> dict
        """
        names = [
            "representation_collapse_score",
            "representation_pairwise_distance",
            "compute_representation_stats",
        ]
        for n in names:
            self._assert_callable(n)

    def test_classification_gap_helpers_exist(self):
        """Classification gap helpers between original/completed logits.

        - classification_gap(orig_logits, completed_logits, labels) -> float
        - classification_loss_gap(orig_loss, completed_loss) -> float
        """
        names = ["classification_gap", "classification_loss_gap"]
        for n in names:
            self._assert_callable(n)

    def test_antiforgetting_helpers_exist(self):
        """Anti-forgetting loss helpers:

        - anti_forgetting_loss(current_reps, reference_reps, margin=0.0) -> Tensor
        - compute_representation_drift(current_reps, reference_reps) -> float
        """
        names = ["anti_forgetting_loss", "compute_representation_drift"]
        for n in names:
            self._assert_callable(n)

    def test_helpers_have_expected_argcounts(self):
        # Check a couple of signatures to ensure minimal args are present.
        rep_score = self._assert_callable("representation_collapse_score")
        sig = inspect.signature(rep_score)
        self.assertGreaterEqual(len(sig.parameters), 1)

        clf_gap = self._assert_callable("classification_gap")
        sig2 = inspect.signature(clf_gap)
        self.assertGreaterEqual(len(sig2.parameters), 3)


if __name__ == "__main__":
    # Running the test module directly should execute the suite and
    # fail if helpers are missing. This file is intentionally
    # minimal and test-first: implement helpers in train.py next.
    unittest.main()
