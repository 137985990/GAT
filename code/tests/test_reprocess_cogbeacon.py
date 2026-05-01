import unittest
import tempfile
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessCogBeacon(unittest.TestCase):
    def test_reprocess_keeps_only_eeg_bands_and_meta(self):
        df = pd.DataFrame({
            'acc_x': [0.0, 0.0],
            'ppg': [0.0, 0.0],
            'eeg_delta': [0.1, 0.2],
            'eeg_theta': [0.3, 0.4],
            'eeg_alpha': [0.5, 0.6],
            'eeg_beta': [0.7, 0.8],
            'eeg_gamma': [0.9, 1.0],
            'alpha_tp9': [1.0, 2.0],
            'f': [0, 1],
            'id': [10, 10],
            'session': [123, 123],
            'round': [0, 0],
            'block': [5010, 5010],
            'source': ['CogBeacon', 'CogBeacon'],
        })

        with tempfile.TemporaryDirectory() as td:
            in_csv = Path(td) / 'cog.csv'
            df.to_csv(in_csv, index=False)

            import reprocess_cogbeacon as mod
            out = mod.reprocess_cogbeacon(in_csv)

        # Only EEG bands + meta
        for c in ['eeg_delta','eeg_theta','eeg_alpha','eeg_beta','eeg_gamma','label','id','session','block','source']:
            self.assertIn(c, out.columns)

        for c in ['acc_x','ppg','alpha_tp9','round','f']:
            self.assertNotIn(c, out.columns)

        self.assertEqual(out.loc[0, 'label'], 0)
        self.assertEqual(out.loc[1, 'label'], 1)


if __name__ == '__main__':
    unittest.main()
