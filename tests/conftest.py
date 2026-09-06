"""
Makes the project root importable from tests/ without needing to install
the project as a package. pytest adds this file's directory's parent to
sys.path automatically only for rootdir-relative imports in some configs,
so we do it explicitly here to be robust regardless of how pytest is
invoked (from repo root, from tests/, or via `python -m pytest`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
