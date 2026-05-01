import unittest

import train


class TestPpLoaderSetup(unittest.TestCase):
    def setUp(self):
        # minimal example config describing desired future API usage
        self.minimal_cfg = {
            "datasets": ["WESAD_P", "WESAD_PP"],
            "batch_size": 8,
            "train_ratio": 0.7,
            "val_ratio": 0.2,
        }

    def test_build_p_pp_loaders_api_exists(self):
        """Desired API: train.build_p_pp_loaders_from_config(cfg) should exist.

        This test asserts the function is present and callable. It is written
        to fail now (test-first) because the helper is intentionally missing
        in the current codebase. The failure will clearly indicate the
        missing feature, not an import/syntax problem.
        """

        self.assertTrue(
            hasattr(train, "build_p_pp_loaders_from_config"),
            "train.build_p_pp_loaders_from_config(cfg) must be implemented: builds separate P and PP loaders from a config",
        )

        fn = getattr(train, "build_p_pp_loaders_from_config")
        self.assertTrue(callable(fn), "build_p_pp_loaders_from_config must be callable")

    def test_detect_forbidden_overlap(self):
        """Desired behaviour: providing both WESAD_P and WESAD_PP in a single run
        should be detected as forbidden and raise ValueError.

        The test expresses the desired contract; it will fail now if the
        builder is not implemented.
        """

        self.assertTrue(
            hasattr(train, "build_p_pp_loaders_from_config"),
            "build_p_pp_loaders_from_config must exist to enforce dataset overlap checks",
        )

        fn = getattr(train, "build_p_pp_loaders_from_config")

        with self.assertRaises(
            ValueError, msg="Overlapping P and PP datasets must raise ValueError"
        ):
            fn({"datasets": ["WESAD_P", "WESAD_PP"]})


if __name__ == "__main__":
    unittest.main()
