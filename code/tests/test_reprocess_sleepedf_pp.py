import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessSleepEDFPP(unittest.TestCase):
    def test_extract_window_features_without_labels(self):
        import reprocess_sleepedf_pp as mod
        fs = 100
        t = np.arange(0, 30, 1/fs)
        eeg = np.sin(2 * np.pi * 10 * t)
        eog = np.sin(2 * np.pi * 2 * t)
        emg = np.ones_like(t) * 0.5
        resp = np.ones_like(t) * 0.2
        temp = np.ones_like(t) * 36.0
        df = mod.build_window_rows(
            record_name='SC4001E0-PSG.edf',
            subject_id=1,
            session_id=1,
            eeg1=eeg,
            eeg2=eeg,
            eog=eog,
            emg=emg,
            breathing=resp,
            skt=temp,
            fs=fs,
            window_seconds=30,
        )
        self.assertEqual(len(df), 1)
        for c in ['eeg_delta','eeg_theta','eeg_alpha','eeg_beta','eog_h','emg','breathing','skt','id','session','block','source']:
            self.assertIn(c, df.columns)
        for c in ['label','sleep_stage','hypnogram']:
            self.assertNotIn(c, df.columns)
        self.assertEqual(df.loc[0, 'source'], 'SleepEDF')


if __name__ == '__main__':
    unittest.main()
