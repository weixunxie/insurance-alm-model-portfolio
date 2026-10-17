"""Interactive dashboard for the NAIC-constrained insurance ALM model portfolio.

The model package lives in src/alm; put src/ on the import path so `import alm` works
both locally and under gunicorn without installing the package.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
