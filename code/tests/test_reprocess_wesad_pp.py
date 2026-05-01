import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessWESADPP(unittest.TestCase):
    def test_reprocess_removes_labels_and_keeps_expected_columns(self):
        df = pd.DataFrame({
            "acc_x": [1.0],
            "acc_y": [2.0],
            "acc_z": [3.0],
            "ppg": [4.0],
            "gsr": [5.0],
            "hr": [6.0],
            "skt": [7.0],
            "ecg": [8.0],
            "breathing": [9.0],
            "emg": [10.0],
            "label": [1],
            "f": [1],
            "fatigue": [1],
            "kss": [7],
            "id": [2],
            "session": [1],
            "block": [201],
            "source": ["WESAD"],
        })
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'WESAD_P.csv'
            df.to_csv(p, index=False)
            import reprocess_wesad_pp as mod
            out = mod.reprocess_wesad_pp(p)

        expected = [
            'acc_x', 'acc_y', 'acc_z', 'ppg', 'gsr', 'hr', 'skt',
            'ecg', 'breathing', 'emg', 'id', 'session', 'block', 'source'
        ]
        self.assertEqual(out.columns.tolist(), expected)
        for col in ['label', 'f', 'F', 'fatigue', 'kss']:
            self.assertNotIn(col, out.columns)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, 'source'], 'WESAD')


if __name__ == '__main__':
    unittest.main()
