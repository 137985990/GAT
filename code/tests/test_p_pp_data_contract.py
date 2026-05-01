import unittest
import importlib
import os
import sys

# Ensure project root is on sys.path so `data` module can be imported when tests run
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# The project data module under test
data = importlib.import_module("data")


class TestPppDataContract(unittest.TestCase):
    """Tests for explicit P / PP metadata contract and alternating loader helper.

    This test encodes the desired API which does not yet exist in data.py:
    - datasets should expose explicit metadata flags for 'P' (labeled) and 'PP' (unlabeled)
    - there should be a helper function `alternating_loader(dataset_p, dataset_pp)` that
      yields alternating batches (or samples) from P and PP sources.

    The test is intentionally minimal and will fail because the current codebase
    does not implement this contract.
    """

    def test_p_pp_metadata_presence_on_dataset(self):
        # Create a tiny fake SlidingWindowDataset-like object (no real data needed)
        class FakeDS:
            def __init__(self, tag):
                self.tag = tag

        p = FakeDS("P")
        pp = FakeDS("PP")

        # Expectation: SlidingWindowDataset (or dataset objects) should carry explicit attributes
        # `is_P` and `is_PP` or a metadata dict `pp_metadata` describing the contract.
        # We assert the attribute/structure exists on real dataset types from data module.

        # Check SlidingWindowDataset class exists
        self.assertTrue(
            hasattr(data, "SlidingWindowDataset"),
            "SlidingWindowDataset class missing in data module",
        )

        DS = data.SlidingWindowDataset

        # We do not instantiate the full dataset (would require csv files). Instead assert the
        # class defines or documents the P/PP metadata contract via attributes or annotations.
        has_flag_attrs = any(
            name in DS.__dict__ for name in ("is_P", "is_PP", "pp_metadata")
        )
        # Also accept a classmethod/property named `get_pp_metadata`
        has_getter = hasattr(DS, "get_pp_metadata") or hasattr(DS, "pp_metadata")

        self.assertTrue(
            has_flag_attrs or has_getter,
            "SlidingWindowDataset must expose P/PP metadata (is_P/is_PP/pp_metadata/get_pp_metadata)\n"
            "This test defines the expected contract for explicit P/PP dataset metadata.",
        )

    def test_alternating_loader_helper_api(self):
        # The project should define a helper function alternating_loader(dataset_p, dataset_pp)
        # that yields items alternating between P and PP. We only assert presence and basic
        # callability/signature; we do not run heavy I/O.
        self.assertTrue(
            hasattr(data, "alternating_loader"),
            "Module `data` must define `alternating_loader(dataset_p, dataset_pp)` helper",
        )

        alt = getattr(data, "alternating_loader", None)
        self.assertTrue(callable(alt), "alternating_loader must be callable")

        # Verify it accepts exactly two positional arguments (dataset_p, dataset_pp)
        try:
            import inspect

            sig = inspect.signature(alt)
            params = list(sig.parameters.values())
            # allow for optional kwargs but require at least two positional params
            self.assertGreaterEqual(
                len(params), 2, "alternating_loader must accept at least two parameters"
            )
        except Exception:
            # If inspecting signature fails, still fail the test to indicate missing contract
            self.fail(
                "Could not introspect alternating_loader signature; expected function taking dataset_p and dataset_pp"
            )


if __name__ == "__main__":
    unittest.main()
