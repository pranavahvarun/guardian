"""
demo/stats.py -- the saved results of all three modules, one after another
(the same as demo.motion, demo.activity and demo.physio).

    py -m demo.stats
"""

from __future__ import annotations

from demo import activity, motion, physio


def main():
    motion.main()
    activity.main()
    physio.main()


if __name__ == "__main__":
    main()
