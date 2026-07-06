"""
Preprocessing and Feature Engineering
---------------------------------------
Cleans raw traffic sensor data and engineers features used for
congestion classification.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "simulated_traffic_data.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def engineer_features(df):
    """Add time-based and rolling features useful for classification."""
    df = df.copy()

    # Cyclical encoding for hour (captures 23->0 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Rush hour flag
    df["is_rush_hour"] = df["hour"].isin([7, 8, 17, 18]).astype(int)

    # Weekend flag
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # Rolling average vehicle count per intersection (last 4 readings = last hour)
    df = df.sort_values(["intersection_id", "timestamp"])
    df["rolling_avg_vehicles"] = (
        df.groupby("intersection_id")["vehicle_count"]
        .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    )

    return df


def encode_labels(df, target_col="traffic_state"):
    """Encode traffic state labels for model training."""
    state_map = {"Free Flow": 0, "Moderate": 1, "Congested": 2, "Critical": 3}
    df = df.copy()
    df["traffic_state_encoded"] = df[target_col].map(state_map)
    return df, state_map


def get_feature_columns():
    return [
        "vehicle_count", "avg_speed_kmh", "density",
        "hour_sin", "hour_cos", "is_rush_hour", "is_weekend",
        "rolling_avg_vehicles"
    ]


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    df, state_map = encode_labels(df)
    print("Feature engineering complete.")
    print(df[get_feature_columns() + ["traffic_state"]].head())
    out_path = os.path.join(BASE_DIR, "..", "data", "processed_traffic_data.csv")
    df.to_csv(out_path, index=False)
