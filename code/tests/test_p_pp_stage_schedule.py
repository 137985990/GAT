import unittest


class TestPppStageScheduleAPI(unittest.TestCase):
    """Test the desired future API for staged hybrid training.

    This file is intentionally a test-first specification. It should fail
    against the current codebase because the staged-training API does not
    yet exist. The test verifies the shape of the API we want to implement:

    - a helper named `select_stage` or `get_training_stage` that, given an
      epoch number and a config mapping, returns a stage id string like
      'warmup' or 'supervised'.
    - a stage routing helper `stage_loss_weights` that returns loss weight
      mapping for the active stage (e.g., recon vs cls weights).

    The test only imports train and asserts these symbols are present and
    that their call signatures behave as the API docstring describes.
    """

    def test_stage_helpers_exist_and_raise_on_missing(self):
        import train

        # The API is not implemented yet; when added the test should assert
        # concrete behavior. For now, ensure importing train doesn't error
        # (sanity) and that the expected names are not present which makes
        # the test fail for the right reason (missing-feature) per instructions.

        # Expected helper names for staged hybrid training
        expected_names = [
            "select_stage",
            "get_training_stage",
            "stage_loss_weights",
        ]

        present = [name for name in expected_names if hasattr(train, name)]

        # We want this test to fail because the API is missing. If any of the
        # helpers exist, that's fine; require ALL to be present to pass.
        missing = [n for n in expected_names if n not in present]

        # If the API has been implemented already (unexpected), perform a
        # minimal runtime check: call the function with a sample epoch and
        # config and assert it returns expected stage keys/weights shapes.
        if not missing:
            # call select stage with epoch and config
            cfg = {"stage1_epochs": 2, "stage2_epochs": 5}
            # prefer available name
            selector = getattr(
                train, "select_stage", getattr(train, "get_training_stage")
            )
            stage = selector(epoch=1, config=cfg)
            assert isinstance(stage, str)

            weights = train.stage_loss_weights(stage, config=cfg)
            assert isinstance(weights, dict)
            assert "recon_weight" in weights and "cls_weight" in weights

        # Force failure when API missing so test encodes missing-feature
        self.assertEqual(len(missing), 0, f"Staged-training API missing: {missing}")


if __name__ == "__main__":
    unittest.main()
