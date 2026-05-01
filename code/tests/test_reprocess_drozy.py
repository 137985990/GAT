import unittest
import tempfile
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessDROZY(unittest.TestCase):
    def test_reprocess_drops_raw_and_aux_and_renames_label(self):
        df = pd.DataFrame({
            'timestamp': ['t0', 't1'],
            'eeg_fz': [0.1, 0.2],
            'eeg_cz': [0.1, 0.2],
            'alpha_tp9': [0.0, 0.0],
            'eog_v': [1.0, 2.0],
            'eog_h': [3.0, 4.0],
            'emg': [5.0, 6.0],
            'ecg': [7.0, 8.0],
            'acc_x': [0.0, 1.0],
            'acc_y': [0.0, 1.0],
            'acc_z': [0.0, 1.0],
            'ppg': [0.0, 0.0],
            'gsr': [0.0, 0.0],
            'hr': [0.0, 0.0],
            'skt': [0.0, 0.0],
            'breathing': [0.0, 0.0],
            'eeg_delta': [0.9, 0.8],
            'eeg_theta': [0.7, 0.6],
            'eeg_alpha': [0.5, 0.4],
            'eeg_beta': [0.3, 0.2],
            'eeg_gamma': [0.0, 0.0],
            'kss': [7, 7],
            'F': [1, 1],
            'p': [0.0, 0.0],
            'm': [0.0, 0.0],
            'f': [0, 1],
            'id': [1, 1],
            'session': [1, 1],
            'block': [101, 101],
            'source': ['DROZY', 'DROZY'],
        })

        with tempfile.TemporaryDirectory() as td:
            in_csv = Path(td) / 'drozy.csv'
            df.to_csv(in_csv, index=False)

            import reprocess_drozy as mod
            out = mod.reprocess_drozy(in_csv)

        self.assertIn('label', out.columns)
        self.assertNotIn('f', out.columns)
        for c in ['timestamp','eeg_fz','eeg_cz','alpha_tp9','kss','F','p','m','breathing']:
            self.assertNotIn(c, out.columns)

        # keep eog/ecg/emg and eeg bands
        for c in ['eog_v','eog_h','emg','ecg','eeg_delta','eeg_theta','eeg_alpha','eeg_beta']:
            self.assertIn(c, out.columns)

        self.assertNotIn('eeg_gamma', out.columns)



        self.assertEqual(out.loc[0, 'label'], 0)
        self.assertEqual(out.loc[1, 'label'], 1)


if __name__ == '__main__':
    unittest.main()
