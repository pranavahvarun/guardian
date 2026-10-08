"""
streams/activity -- activity stream (owner: Sudharshan), activity_v1.

    engine.py    his model code (RandomForest, 78 features, 3-window smoothing)
    features.py  his 78 features, unchanged
    stream.py    contract wrapper: our window -> his input, his output -> last_*

The pipeline uses ActivityStream from here (core/pipeline.py STREAMS).
Model: models/activity_rf.joblib + models/activity_config.json.
"""

from .stream import ActivityStream, CLASSES  # noqa: F401
