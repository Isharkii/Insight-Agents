"""Shared failure code constants for pipeline error handling."""

CRITICAL_FAILURES = [
    "empty_kpi",
    "empty_forecast",
    "empty_risk",
]

OPTIONAL_FAILURES = [
    "missing_segmentation",
    "missing_scraping",
]

