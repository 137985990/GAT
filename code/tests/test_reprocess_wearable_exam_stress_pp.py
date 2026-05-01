import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


class TestReprocessWearableExamStressPP(unittest.TestCase):
    def test_builds_pp_from_subject_exam_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'Data'
            exam = root / 'S1' / 'Midterm 1'
            exam.mkdir(parents=True)
            (exam / 'info.txt').write_text('info', encoding='utf-8')
            (exam / 'tags.csv').write_text('', encoding='utf-8')
            (exam / 'IBI.csv').write_text('0,IBI\n0.5,0.5\n', encoding='utf-8')
            (exam / 'ACC.csv').write_text('0,0,0\n32,32,32\n1,2,3\n4,5,6\n', encoding='utf-8')
            (exam / 'BVP.csv').write_text('0\n64\n10\n20\n30\n40\n', encoding='utf-8')
            (exam / 'EDA.csv').write_text('0\n4\n0.1\n', encoding='utf-8')
            (exam / 'HR.csv').write_text('0\n1\n60\n', encoding='utf-8')
            (exam / 'TEMP.csv').write_text('0\n4\n33\n', encoding='utf-8')

            import reprocess_wearable_exam_stress_pp as mod
            out = mod.reprocess_directory(root)

        expected = ['acc_x','acc_y','acc_z','ppg','gsr','hr','skt','id','session','block','source']
        self.assertEqual(out.columns.tolist(), expected)
        self.assertEqual(len(out), 2)
        self.assertEqual(out['id'].iloc[0], 1)
        self.assertEqual(out['session'].iloc[0], 1)
        self.assertEqual(out['block'].iloc[0], 11)
        self.assertEqual(out['source'].iloc[0], 'WearableExamStress')


if __name__ == '__main__':
    unittest.main()
