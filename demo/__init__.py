"""
demo/ -- review outputs (Stage 7.7). Plain text.

Saved results (instant):
    py -m demo.stats        all three modules below, one after another
    py -m demo.motion       Impact and Immobility Detection Module
    py -m demo.activity     Activity Recognition Module
    py -m demo.physio       Heart Rate Monitoring Module

Run live on the data (shows progress, takes a few minutes):
    py -m demo.motion_eval
    py -m demo.activity_eval            (add --from-raw to recompute features)
    py -m demo.physio_eval --root "D:\\MVP\\Sem VII\\PDP\\data"

Emergency Risk Estimation Module (replays saved module outputs through fusion):
    py -m demo.fusion       or: 1-4, results, found

Fusion vs an impact-only detector (saved result of testing/compare_impact_only.py):
    py -m demo.compare

Alerts as a caregiver sees them (needs Docker, after a live run):
    py -m demo.show_alerts --user 5

Optional live demo (two windows, needs Docker):
    py -m core.watch --user 5
    py -m simulator.live hard_fall_long_lie --speed 2
"""
