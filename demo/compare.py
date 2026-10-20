"""
demo/compare.py -- review output 5: the Emergency Risk Estimation Module
against an impact-only detector, SAVED result of
py -m testing.compare_impact_only (simulated cases, locked before the run).

    py -m demo.compare
"""

from __future__ import annotations

from demo.common import OUT, W, need, title


def main():
    d = OUT / "compare"
    f = d / "compare_first_run.txt"
    if not f.exists() and (d / "compare_partial.txt").exists():
        f = d / "compare_partial.txt"
    txt = need(f, "py -m testing.compare_impact_only").read_text()
    title("5. GUARDIAN vs AN IMPACT-ONLY DETECTOR",
          "Same simulated cases, same module outputs: who tells the caregiver when it matters?")
    print()
    print(txt.rstrip())
    print("=" * W)


if __name__ == "__main__":
    main()
