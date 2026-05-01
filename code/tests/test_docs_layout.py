import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDocsLayout(unittest.TestCase):
    def test_docs_layout_separates_active_archive_and_runbooks(self):
        self.assertTrue((REPO_ROOT / "docs" / "plans" / "active").is_dir())
        self.assertTrue((REPO_ROOT / "docs" / "plans" / "archive").is_dir())
        self.assertTrue((REPO_ROOT / "docs" / "runbooks").is_dir())
        self.assertTrue((REPO_ROOT / "docs" / "archive" / "legacy-paper").is_dir())
        self.assertFalse((REPO_ROOT / "docs" / "paper").exists())

    def test_active_archive_and_runbook_files_are_reclassified(self):
        self.assertTrue(
            (REPO_ROOT / "docs" / "plans" / "archive" / "2026-03-15-model-p-pp-training-implementation.md").exists()
        )
        self.assertTrue(
            (REPO_ROOT / "docs" / "plans" / "active" / "2026-04-21-phase2-domain-experiments.md").exists()
        )
        self.assertTrue(
            (REPO_ROOT / "docs" / "runbooks" / "2026-03-15-model-p-pp-training-runbook.md").exists()
        )


if __name__ == "__main__":
    unittest.main()
