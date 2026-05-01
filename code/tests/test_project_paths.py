import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from project_paths import normalize_config_paths, project_root


class TestProjectPaths(unittest.TestCase):
    def test_relative_artifact_paths_resolve_to_root_artifacts(self):
        normalized = normalize_config_paths(
            {
                "log_dir": "Logs",
                "checkpoint_dir": "Checkpoints",
                "tensorboard_dir": "runs",
                "cache_dir": "Data/cache",
            }
        )
        root = project_root()
        self.assertEqual(normalized["log_dir"], str(root / "artifacts" / "logs"))
        self.assertEqual(normalized["checkpoint_dir"], str(root / "artifacts" / "checkpoints"))
        self.assertEqual(normalized["tensorboard_dir"], str(root / "artifacts" / "runs"))
        self.assertEqual(normalized["cache_dir"], str(root / "artifacts" / "cache" / "datasets"))

    def test_legacy_data_dir_resolves_to_repo_data(self):
        normalized = normalize_config_paths({"data_dir": "Data"})
        self.assertEqual(normalized["data_dir"], str(project_root() / "Data"))


if __name__ == "__main__":
    unittest.main()
