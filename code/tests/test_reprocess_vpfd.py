import unittest
import tempfile
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessVPFD(unittest.TestCase):
    def test_reprocess_drops_dup_acc_and_time_and_renames_label(self):
        df = pd.DataFrame({
            'space_distance': [0.1, 0.2],
            'distance_to_eye_center': [1.0, 1.1],
            'pose_pca': [0.0, 0.0],
            'hdistance_to_eye_center': [0.3, 0.4],
            'intersection_pca': [0.5, 0.6],
            'gaze_duration': [0.0, 1.0],
            'large_changes_per_minute': [2.0, 3.0],
            'acc_x': [1.0, 2.0],
            'acc_y': [3.0, 4.0],
            'acc_z': [5.0, 6.0],
            'acc_x.1': [10.0, 20.0],
            'acc_y.1': [30.0, 40.0],
            'acc_z.1': [50.0, 60.0],
            'ppg': [0.0, 0.0],
            'gsr': [0.0, 0.0],
            'hr': [60, 61],
            'skt': [33.0, 33.1],
            'fatigue': [1, 3],
            'f': [0, 1],
            'block': [1, 1],
            'time': [0.0, 0.03125],
            'eeg_delta': [0.0, 0.0],
        })

        with tempfile.TemporaryDirectory() as td:
            in_csv = Path(td) / 'vpfd.csv'
            df.to_csv(in_csv, index=False)

            import reprocess_vpfd as mod
            out = mod.reprocess_vpfd(in_csv)

        self.assertIn('label', out.columns)
        self.assertNotIn('f', out.columns)
        self.assertNotIn('acc_x.1', out.columns)
        self.assertNotIn('acc_y.1', out.columns)
        self.assertNotIn('acc_z.1', out.columns)
        self.assertNotIn('time', out.columns)
        self.assertNotIn('fatigue', out.columns)
        self.assertNotIn('eeg_delta', out.columns)

        self.assertEqual(out.loc[0, 'label'], 0)
        self.assertEqual(out.loc[1, 'label'], 1)


if __name__ == '__main__':
    unittest.main()
