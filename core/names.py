"""
core/names.py -- display names of the modules (renamed by function after the
review feedback, 20 Oct 2026).

Only what is PRINTED changes. The internal keys stay as they were, because
they are stored in the database, in saved outputs and in the frozen fusion
settings (stream "motion", "activity", "physiological", "fusion").

    key            module name (slides, report, screens)
    motion         Impact and Immobility Detection Module
    activity       Activity Recognition Module
    physiological  Heart Rate Monitoring Module
    context        Situational Context Module (inside core/fusion.py for now)
    fusion         Emergency Risk Estimation Module
    alerts         Alert Escalation Module
    explanation    Alert Explanation Module (planned, not built for this review)
"""

MODULES = {
    "motion": "Impact and Immobility Detection Module",
    "activity": "Activity Recognition Module",
    "physiological": "Heart Rate Monitoring Module",
    "context": "Situational Context Module",
    "fusion": "Emergency Risk Estimation Module",
    "alerts": "Alert Escalation Module",
    "explanation": "Alert Explanation Module",
}

# short column names on the live screens, and which module each comes from
LEGEND = [
    "impact, still (s) = Impact and Immobility Detection Module   |   "
    "activity = Activity Recognition Module",
    "heart = Heart Rate Monitoring Module   |   emergency = Emergency Risk Estimation Module"
    "   |   alerts = Alert Escalation Module",
]


def name(key: str) -> str:
    return MODULES.get(key, key)
