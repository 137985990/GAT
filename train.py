# Compatibility wrapper for tests and legacy scripts that import `train` from the project root.
import os
import sys

_CODE_DIR = os.path.join(os.path.dirname(__file__), 'code')
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from code.train import *  # noqa: F401,F403
