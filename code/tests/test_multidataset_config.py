import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data import load_config


class TestMultidatasetConfig(unittest.TestCase):
    def test_default_config_uses_current_p_pp_dataset_set(self):
        cfg = load_config(CODE_ROOT / "config.yaml")

        self.assertEqual(len(cfg["p_data_files"]), 5)
        self.assertEqual(len(cfg["pp_data_files"]), 3)
        self.assertEqual(cfg["domain_adaptation"]["num_domains"], 8)

        names = "\n".join(cfg["p_data_files"] + cfg["pp_data_files"])
        for expected in [
            "Fatigueset_P.csv",
            "VPFD_P.csv",
            "MEFAR_P.csv",
            "DROZY_P.csv",
            "CogBeacon_P.csv",
            "WESAD_PP.csv",
            "WearableExamStress_PP.csv",
            "SleepEDF_PP.csv",
        ]:
            self.assertIn(expected, names)


    def test_domain_source_order_uses_five_p_then_three_pp(self):
        from data import build_source_to_id, infer_source_name

        cfg = load_config(CODE_ROOT / "config.yaml")
        source_order = [infer_source_name(path) for path in cfg["p_data_files"] + cfg["pp_data_files"]]
        mapping = build_source_to_id(source_order, source_order)

        self.assertEqual(mapping, {
            "FM": 0,
            "OD": 1,
            "MEFAR": 2,
            "DROZY": 3,
            "CogBeacon": 4,
            "WESAD": 5,
            "WearableExamStress": 6,
            "SleepEDF": 7,
        })

    def test_p_only_domain_count_matches_labeled_sources(self):
        cfg = load_config(CODE_ROOT / "config.p_only.yaml")

        self.assertEqual(len(cfg["p_data_files"]), 5)
        self.assertEqual(cfg["pp_data_files"], [])
        self.assertEqual(cfg["domain_adaptation"]["num_domains"], 5)


if __name__ == "__main__":
    unittest.main()
