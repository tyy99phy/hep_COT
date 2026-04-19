"""Conftest / shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running pytest from repo root without install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
