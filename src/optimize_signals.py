"""
Adaptive Signal Timing Optimization
--------------------------------------
Recommends traffic signal green-light duration based on predicted
congestion state, replacing fixed-cycle timing with adaptive logic.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "simulated_traffic_data.csv")


# Base green-light duration (seconds) per traffic state
SIGNAL_TIMING_RULES = {
    "Free Flow": 30,
    "Moderate": 45,
    "Congested": 65,
    "Critical": 90,
}

STATE_LABELS = {0: "Free Flow", 1: "Moderate", 2: "Congested", 3: "Critical"}


def recommend_signal_timing(traffic_state):
    """Return recommended green-light duration in seconds for a given state."""
    if isinstance(traffic_state, int):
        traffic_state = STATE_LABELS.get(traffic_state, "Moderate")
    return SIGNAL_TIMING_RULES.get(traffic_state, 45)


def optimize_intersection_schedule(df, intersection_id):
    """
    Generate a signal timing schedule for a specific intersection
    based on its historical/predicted traffic states.
    """
    subset = df[df["intersection_id"] == intersection_id].copy()
    subset["recommended_green_seconds"] = subset["traffic_state"].apply(recommend_signal_timing)

    # Estimate congestion reduction vs a fixed 45-second baseline
    baseline = 45
    subset["timing_adjustment"] = subset["recommended_green_seconds"] - baseline

    return subset[["timestamp", "traffic_state", "vehicle_count",
                    "recommended_green_seconds", "timing_adjustment"]]


def adaptive_change_rate(df, baseline=45):
    """
    Share of intervals (%) where the adaptive plan recommends a different
    green time than a fixed-cycle baseline.

    This measures how often adaptive timing would act, not a measured
    reduction in congestion: that would need a traffic simulator or a
    before/after field test.
    """
    recommended = df["traffic_state"].apply(recommend_signal_timing)
    return round((recommended != baseline).mean() * 100, 1)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    schedule = optimize_intersection_schedule(df, intersection_id=1)
    print("Sample signal timing schedule (Intersection 1):")
    print(schedule.head(10))

    rate = adaptive_change_rate(df)
    print(f"\nIntervals where adaptive timing differs from fixed 45s: {rate}%")
