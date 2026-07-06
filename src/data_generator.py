"""
IoT Traffic Sensor Data Simulator
----------------------------------
Simulates traffic sensor readings (vehicle count, speed, density) across
multiple intersections, as would be collected from real IoT sensors.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "simulated_traffic_data.csv")


def generate_traffic_data(num_intersections=5, days=7, readings_per_hour=4, seed=42):
    """
    Generate simulated IoT traffic sensor data.

    Parameters
    ----------
    num_intersections : int
        Number of traffic intersections to simulate.
    days : int
        Number of days of data to generate.
    readings_per_hour : int
        Sensor readings taken per hour (e.g. 4 = every 15 minutes).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated traffic sensor dataset.
    """
    np.random.seed(seed)

    start_time = datetime(2026, 1, 1, 0, 0)
    total_readings = days * 24 * readings_per_hour
    records = []

    for intersection_id in range(1, num_intersections + 1):
        base_traffic = np.random.randint(20, 60)  # baseline vehicles/interval

        for i in range(total_readings):
            timestamp = start_time + timedelta(minutes=(60 // readings_per_hour) * i)
            hour = timestamp.hour

            # Rush hour multiplier (7-9am, 5-7pm)
            if hour in [7, 8, 17, 18]:
                multiplier = np.random.uniform(2.2, 3.0)
            elif hour in [9, 10, 16, 19]:
                multiplier = np.random.uniform(1.4, 1.8)
            elif 0 <= hour <= 5:
                multiplier = np.random.uniform(0.1, 0.3)
            else:
                multiplier = np.random.uniform(0.8, 1.3)

            vehicle_count = max(0, int(base_traffic * multiplier + np.random.normal(0, 5)))
            avg_speed = max(5, 60 - (vehicle_count * 0.4) + np.random.normal(0, 4))
            density = round(vehicle_count / 60, 2)  # vehicles per meter (simplified)

            # Congestion label based on vehicle count and speed
            if vehicle_count > 100 or avg_speed < 15:
                state = "Critical"
            elif vehicle_count > 70 or avg_speed < 25:
                state = "Congested"
            elif vehicle_count > 40 or avg_speed < 40:
                state = "Moderate"
            else:
                state = "Free Flow"

            records.append({
                "intersection_id": intersection_id,
                "timestamp": timestamp,
                "hour": hour,
                "day_of_week": timestamp.strftime("%A"),
                "vehicle_count": vehicle_count,
                "avg_speed_kmh": round(avg_speed, 1),
                "density": density,
                "traffic_state": state
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_traffic_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} traffic sensor records across {df['intersection_id'].nunique()} intersections")
    print(df.head(10))
    print("\nTraffic state distribution:")
    print(df["traffic_state"].value_counts())
