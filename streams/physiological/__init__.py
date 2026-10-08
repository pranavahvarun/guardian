"""
streams/physiological -- heart-rate stream (owner: Mithuna).

    ppg.py, quality.py, hr.py, features.py, inference.py   her code
    stream.py   contract wrapper: PPG buffer, 64 Hz resampling, renamed
                heart-rate bands, quality from agreement of two HR estimates

The pipeline uses PhysiologicalStream from here (core/pipeline.py STREAMS).
Model: models/physiological_xgb.joblib (XGBoost, trained on PPG-DaLiA).
"""

from .stream import PhysiologicalStream  # noqa: F401
