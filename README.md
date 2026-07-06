# Smart Traffic Management System

An IoT-integrated data analytics solution for real-time traffic monitoring and urban congestion management. The system simulates traffic sensor data across multiple intersections, classifies congestion states using machine learning, and recommends adaptive signal timing to reduce congestion.

---

## Problem Statement

Urban congestion costs cities billions annually in lost productivity and emissions. Traditional traffic signals use fixed timing cycles that cannot adapt to real-time conditions. This project builds a data-driven traffic management system that classifies congestion in real time and recommends adaptive signal timing.

---

## Solution Overview

| Component | Details |
|---|---|
| Data Simulation | IoT sensor data generator — vehicle count, speed, density across 5 intersections, 7 days |
| Feature Engineering | Cyclical time encoding, rush-hour flags, rolling averages |
| Classification | Random Forest classifier predicting traffic state (Free Flow / Moderate / Congested / Critical) |
| Optimization | Rule-based adaptive signal timing engine |
| Dashboard | Streamlit app for real-time monitoring and timing recommendations |

---

## Tech Stack

- **Language:** Python 3.12
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **Tools:** Jupyter Notebook, Git

---

## Project Structure

```
smart-traffic-management/
│
├── data/
│   └── simulated_traffic_data.csv       # Generated IoT sensor data (3,360 records)
│
├── notebooks/
│   ├── 01_EDA_Traffic_Patterns.ipynb
│   ├── 02_Congestion_Classification.ipynb
│   └── 03_Signal_Optimization.ipynb
│
├── src/
│   ├── data_generator.py                # IoT sensor data simulation
│   ├── preprocess.py                    # Data cleaning and feature engineering
│   ├── classify_congestion.py           # Traffic state classifier
│   └── optimize_signals.py              # Adaptive signal timing logic
│
├── dashboard/
│   └── app.py                           # Streamlit monitoring dashboard
│
├── models/
│   └── congestion_classifier.pkl        # Trained model (generated on run)
│
├── requirements.txt
└── README.md
```

---

## Key Features

- Simulates realistic traffic patterns including rush-hour spikes and overnight lulls
- Classifies traffic into 4 states: **Free Flow · Moderate · Congested · Critical**
- Adaptive signal timing recommendations per traffic state
- Interactive Streamlit dashboard for live monitoring per intersection

---

## Results

Trained and evaluated on 3,360 simulated sensor readings across 5 intersections (80/20 train-test split):

| Metric | Value |
|---|---|
| Classification accuracy | **99.7%** |
| Precision (weighted avg) | 1.00 |
| Recall (weighted avg) | 1.00 |
| Estimated congestion reduction (adaptive vs. fixed 45s timing) | **~22%** |

**Top predictive features:** vehicle count, traffic density, average speed, and rolling average vehicle count.

*Note: high accuracy reflects the rule-based simulation used to generate labels — the pipeline (feature engineering, classification, adaptive optimization) is designed to transfer directly to real IoT sensor feeds.*

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/NadavaluruSashank/Smart-Traffic-Management.git
cd Smart-Traffic-Management

# Install dependencies
pip install -r requirements.txt

# 1. Generate simulated IoT sensor data
python src/data_generator.py

# 2. Train the congestion classifier
cd src && python classify_congestion.py && cd ..

# 3. Test the signal optimizer
cd src && python optimize_signals.py && cd ..

# 4. Launch the live dashboard
streamlit run dashboard/app.py
```

---

## Use Cases

- Smart city traffic planning and simulation
- Municipal congestion management systems
- Transportation analytics and reporting for urban planning teams

---

## Author

**Nadavaluru Sashank**
Data Scientist · MSc Data Science & AI @ DSTI School of Engineering, Paris
[LinkedIn](https://www.linkedin.com/in/sashank-nadavaluru) · [GitHub](https://github.com/NadavaluruSashank)
