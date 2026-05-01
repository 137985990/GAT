import unittest
import tempfile
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessWESAD(unittest.TestCase):
    def test_reprocess_drops_eeg_and_timestamp_and_renames_label(self):
        df = pd.DataFrame({
            'timestamp': ['t0', 't1'],
            'acc_x': [1.0, 2.0],
            'acc_y': [0.0, 0.0],
            'acc_z': [0.0, 0.0],
            'ppg': [0.1, 0.2],
            'gsr': [0.0, 0.0],
            'hr': [0.0, 0.0],
            'skt': [33.0, 33.1],
            'ecg': [0.5, 0.6],
            'breathing': [1.0, 1.1],
            'emg': [2.0, 2.1],
            'eeg_alpha': [0.0, 0.0],
            'alpha_tp9': [0.0, 0.0],
            'space_distance': [0.0, 0.0],
            'eog_v': [0.0, 0.0],
            'p': [0.0, 0.0],
            'm': [0.0, 0.0],
            'f': [0, 1],
            'id': [2, 2],
            'session': [1, 1],
            'block': [201, 201],
            'source': ['WESAD', 'WESAD'],
        })

        with tempfile.TemporaryDirectory() as td:
            in_csv = Path(td) / 'wesad.csv'
            df.to_csv(in_csv, index=False)

            import reprocess_wesad as mod
            out = mod.reprocess_wesad(in_csv)

        self.assertIn('label', out.columns)
        self.assertNotIn('f', out.columns)
        self.assertNotIn('timestamp', out.columns)
        self.assertNotIn('eeg_alpha', out.columns)
        self.assertNotIn('alpha_tp9', out.columns)
        self.assertNotIn('space_distance', out.columns)
        self.assertNotIn('eog_v', out.columns)
        self.assertNotIn('p', out.columns)
        self.assertNotIn('m', out.columns)

        self.assertEqual(out.loc[0, 'label'], 0)
        self.assertEqual(out.loc[1, 'label'], 1)


if __name__ == '__main__':
    unittest.main()
