"""
Smart Traffic Management Dashboard
------------------------------------
Streamlit app for monitoring simulated intersection traffic states
and viewing adaptive signal timing recommendations.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from optimize_signals import recommend_signal_timing, estimate_congestion_reduction

st.set_page_config(page_title="Smart Traffic Management", layout="wide")

st.title("🚦 Smart Traffic Management System")
st.caption("IoT-based traffic monitoring and adaptive signal optimization")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "simulated_traffic_data.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

try:
    df = load_data()

    intersections = sorted(df["intersection_id"].unique())
    selected = st.sidebar.selectbox("Select Intersection", intersections)

    filtered = df[df["intersection_id"] == selected]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Vehicle Count", f"{filtered['vehicle_count'].mean():.0f}")
    col2.metric("Avg Speed (km/h)", f"{filtered['avg_speed_kmh'].mean():.1f}")
    col3.metric("Current State", filtered.iloc[-1]["traffic_state"])
    col4.metric("Est. Congestion Reduction", f"{estimate_congestion_reduction(df)}%")

    st.subheader("Vehicle Count Over Time")
    st.line_chart(filtered.set_index("timestamp")["vehicle_count"])

    st.subheader("Traffic State Distribution")
    st.bar_chart(filtered["traffic_state"].value_counts())

    st.subheader("Recommended Signal Timing")
    filtered = filtered.copy()
    filtered["recommended_green_seconds"] = filtered["traffic_state"].apply(recommend_signal_timing)
    st.dataframe(
        filtered[["timestamp", "traffic_state", "vehicle_count", "avg_speed_kmh", "recommended_green_seconds"]].tail(20),
        use_container_width=True
    )

except FileNotFoundError:
    st.error("Data not found. Run `python src/data_generator.py` first to generate simulated traffic data.")
