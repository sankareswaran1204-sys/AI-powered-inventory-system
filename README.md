# AI-Powered Intelligent Product and Inventory Management System

A college mini-project: a Streamlit inventory management dashboard built on a
synthetic retail dataset, with three applied machine learning components.

## Features

### 1. Demand Forecasting (Machine Learning)
- Global **Random Forest Regression** model trained on lag / rolling-mean /
  calendar features across all products' daily demand history.
- Recursive multi-step forecasting (predict day N, feed it back in as a lag
  feature to predict day N+1, etc.)
- Benchmarked against a simple **Linear Regression** trend baseline.
- Feature importance chart included.

### 2. Reorder Point Optimization
- Classic inventory-control statistics computed from each product's **actual**
  demand mean & standard deviation and its (synthetic) supplier lead time:
  - `Safety Stock = Z * σ_demand * √(Lead Time)`
  - `Reorder Point = (avg_daily_demand * Lead Time) + Safety Stock`
  - **EOQ** (Economic Order Quantity) as a bonus classical optimization:
    `EOQ = √(2DS / H)`
- Configurable service level (90% / 95% / 97.5% / 99%) via Z-factor.

### 3. Anomaly Detection
- **Isolation Forest** (unsupervised ML) on engineered transaction features
  (quantity, per-product z-score, day-of-week, days since last transaction).
- Compared against a **Z-score threshold baseline**.
- The synthetic data generator injects ~1.5% ground-truth anomalies (demand
  spikes, bulk restocks) purely for **evaluation** — precision / recall / F1
  are computed and shown side by side for both methods.

## Dataset

Synthetically generated (see `data_generator.py`):
- **60 products** across 6 categories, 5 suppliers, with cost, price, lead
  time, ordering cost, and holding cost.
- **~12,000+ transactions** (sales, restocks, returns) over a 200-day window,
  with per-product demand patterns: rising / falling / stable / seasonal-spike,
  plus injected anomalies.

Regenerate the dataset anytime with the "🔄 Regenerate synthetic data" button
in the sidebar, or by running `python data_generator.py`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# generate the synthetic dataset (already included, but you can regenerate)
python data_generator.py

# launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project Structure

```
inventory_streamlit/
├── app.py               # Streamlit UI (6 pages: Overview, Forecasting,
│                         #   Reorder Optimization, Anomaly Detection,
│                         #   Product Catalog, About/Data)
├── data_generator.py     # Synthetic data generator
├── ml_models.py           # ML models: forecasting, reorder optimization,
│                         #   anomaly detection
├── requirements.txt
└── data/
    ├── products.csv
    └── transactions.csv
```

## Tech Stack

Streamlit · pandas · NumPy · scikit-learn (RandomForestRegressor,
IsolationForest, LinearRegression) · Plotly

## Suggested Report / Viva Talking Points

- Why Random Forest over plain linear regression for demand forecasting
  (captures non-linear weekly/seasonal patterns via lag + calendar features).
- Why the reorder point formula uses **σ (standard deviation)**, not just the
  mean — it accounts for demand variability, which is the actual reason
  safety stock exists.
- Why Isolation Forest suits anomaly detection here: it isolates anomalies via
  random recursive splits, needing fewer splits for outliers — no need to
  assume a specific data distribution, unlike a pure z-score method.
- Precision/recall trade-off: Isolation Forest vs Z-score baseline, and how
  `contamination` / threshold parameters shift that trade-off.
