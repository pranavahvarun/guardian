"""Shared printing helpers for the review outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "testing" / "outputs"
W = 86


def title(name, question):
    print()
    print("=" * W)
    print(f"  {name}")
    print(f"  {question}")
    print("=" * W)


def section(text):
    print()
    print(f"  {text}")
    print("  " + "-" * (W - 2))


def rows(lines):
    for line in lines:
        print("  " + line)


def need(path, command):
    """Stop with a clear message if a saved result is missing."""
    if not Path(path).exists():
        print(f"\n  Missing saved result: {Path(path).relative_to(ROOT)}")
        print(f"  Create it once with:  {command}\n")
        sys.exit(1)
    return Path(path)
