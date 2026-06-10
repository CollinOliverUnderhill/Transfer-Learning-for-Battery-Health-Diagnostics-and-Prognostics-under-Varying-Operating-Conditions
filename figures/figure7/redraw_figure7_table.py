#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


FIGURECODES = Path(__file__).resolve().parents[1] / "figurecodes"
sys.path.insert(0, str(FIGURECODES))

from figure07_selected_soh_feature_combination import main  # noqa: E402


if __name__ == "__main__":
    main()
