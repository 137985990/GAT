import unittest
import tempfile
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessFatigueset(unittest.TestCase):
    def test_reprocess_computes_5_band_means_and_renames_label(self):
        # Minimal 2-row sample with 4-electrode band columns
        df = pd.DataFrame({
            'timestamp': ['t0', 't1'],
            'acc_x': [1.0, 2.0],
            'acc_y': [0.0, 0.0],
            'acc_z': [0.0, 0.0],
            'ppg': [0.1, 0.2],
            'gsr': [0.0, 0.0],
            'hr': [60, 61],
            'skt': [33.0, 33.1],
            'ecg': [0.0, 0.0],
            'breathing': [0.0, 0.0],
            'alpha_tp9': [1.0, 2.0],
            'alpha_af7': [3.0, 4.0],
            'alpha_af8': [5.0, 6.0],
            'alpha_tp10': [7.0, 8.0],
            'beta_tp9': [1.0, 1.0],
            'beta_af7': [1.0, 1.0],
            'beta_af8': [1.0, 1.0],
            'beta_tp10': [1.0, 1.0],
            'delta_tp9': [0.0, 1.0],
            'delta_af7': [0.0, 1.0],
            'delta_af8': [0.0, 1.0],
            'delta_tp10': [0.0, 1.0],
            'gamma_tp9': [2.0, 2.0],
            'gamma_af7': [2.0, 2.0],
            'gamma_af8': [2.0, 2.0],
            'gamma_tp10': [2.0, 2.0],
            'theta_tp9': [4.0, 4.0],
            'theta_af7': [4.0, 4.0],
            'theta_af8': [4.0, 4.0],
            'theta_tp10': [4.0, 4.0],
            'f': [0, 1],
            'id': [1, 1],
            'session': ['s1', 's1'],
            'block': [1, 1],
        })

        with tempfile.TemporaryDirectory() as td:
            in_csv = Path(td) / 'fatigueset.csv'
            df.to_csv(in_csv, index=False)

            import reprocess_fatigueset_5eeg as mod
            out = mod.reprocess_fatigueset(in_csv)

        # Must have label (not f), and 5 eeg bands
        self.assertIn('label', out.columns)
        self.assertNotIn('timestamp', out.columns)
        self.assertNotIn('alpha_tp9', out.columns)
        for c in ['eeg_delta','eeg_theta','eeg_alpha','eeg_beta','eeg_gamma']:
            self.assertIn(c, out.columns)

        # alpha mean of 4 electrodes
        self.assertAlmostEqual(out.loc[0, 'eeg_alpha'], (1+3+5+7)/4)
        self.assertAlmostEqual(out.loc[1, 'eeg_alpha'], (2+4+6+8)/4)

        # f copied to label
        self.assertEqual(out.loc[0, 'label'], 0)
        self.assertEqual(out.loc[1, 'label'], 1)


if __name__ == '__main__':
    unittest.main()
