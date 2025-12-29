#!/usr/bin/env python3
"""Wrapper to run the sanity suite from tools/ as requested."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runpy.run_path(str(ROOT / "scripts" / "run_sanity_suite.py"), run_name="__main__")


if __name__ == "__main__":
    main()
