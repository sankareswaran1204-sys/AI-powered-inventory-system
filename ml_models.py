"""
ML layer for the Inventory Management System.

1. DEMAND FORECASTING
   A global regression model (Random Forest) trained on lag/rolling/calendar
   features across ALL products' daily demand series, plus a simple
   per-product Linear Regression trend baseline for comparison. The RF model
   is used for recursive multi-step-ahead forecasting.

2. REORDER POINT OPTIMIZATION
   Classic inventory-control statistics computed from each product's actual
   demand distribution and lead time:
       Safety Stock   = Z * sigma_demand * sqrt(lead_time)
       Reorder Point  = (avg_daily_demand * lead_time) + Safety Stock
       EOQ (bonus)    = sqrt(2 * D * S / H)   [Economic Order Quantity]
   Z is the service-level factor (e.g. Z=1.65 for 95% service level).

3. ANOMALY DETECTION
   Isolation Forest (unsupervised) on transaction-level engineered features
   (quantity, per-product z-score, day-of-week deviation), compared against
   a simple statistical z-score threshold baseline. Since the synthetic data
   generator kept ground-truth anomaly labels, precision/recall/F1 can be
   computed for evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score

Z_TABLE = {"90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}


# --------------------------------------------------------------------------
# Data prep: build a daily demand panel (product_id, date, qty) from raw
# transaction records, filling missing days with 0 demand.
# --------------------------------------------------------------------------
def build_daily_demand_panel(transactions_df):
    sales = transactions_df[transactions_df["transaction_type"] == "SALE"].copy()
    sales["date"] = pd.to_datetime(sales["date"])

    daily = (
        sales.groupby(["product_id", "date"])["quantity"].sum().reset_index()
        .rename(columns={"quantity": "demand"})
    )

    full_frames = []
    date_min, date_max = sales["date"].min(), sales["date"].max()
    all_dates = pd.date_range(date_min, date_max, freq="D")

    for pid, grp in daily.groupby("product_id"):
        grp = grp.set_index("date").reindex(all_dates, fill_value=0)
        grp["product_id"] = pid
        grp = grp.rename_axis("date").reset_index()
        full_frames.append(grp[["product_id", "date", "demand"]])

    return pd.concat(full_frames, ignore_index=True)


def _add_features(panel):
    panel = panel.sort_values(["product_id", "date"]).copy()
    g = panel.groupby("product_id")["demand"]
    panel["lag_1"] = g.shift(1)
    panel["lag_7"] = g.shift(7)
    panel["roll_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    panel["roll_mean_14"] = g.shift(1).rolling(14).mean().reset_index(level=0, drop=True)
    panel["dow"] = panel["date"].dt.dayofweek
    panel["day_of_month"] = panel["date"].dt.day
    panel["month"] = panel["date"].dt.month
    return panel


FEATURE_COLS = ["lag_1", "lag_7", "roll_mean_7", "roll_mean_14", "dow", "day_of_month", "month", "product_id"]


def train_demand_model(transactions_df):
    """Train a global Random Forest demand-forecasting model."""
    panel = build_daily_demand_panel(transactions_df)
    feat = _add_features(panel).dropna()

    X = feat[FEATURE_COLS]
    y = feat["demand"]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    # quick in-sample R^2 for reporting in the UI
    r2 = model.score(X, y)
    return model, panel, r2


def forecast_product(model, panel, product_id, horizon_days=14):
    """Recursive multi-step forecast for a single product using the trained
    global RF model."""
    hist = panel[panel["product_id"] == product_id].sort_values("date").copy()
    if hist.empty:
        return pd.DataFrame(columns=["date", "forecast_demand"])

    series = list(hist["demand"].values)
    last_date = hist["date"].max()

    preds = []
    for step in range(1, horizon_days + 1):
        d = last_date + pd.Timedelta(days=step)
        lag_1 = series[-1]
        lag_7 = series[-7] if len(series) >= 7 else np.mean(series)
        roll_mean_7 = np.mean(series[-7:]) if len(series) >= 7 else np.mean(series)
        roll_mean_14 = np.mean(series[-14:]) if len(series) >= 14 else np.mean(series)

        row = pd.DataFrame([{
            "lag_1": lag_1, "lag_7": lag_7, "roll_mean_7": roll_mean_7,
            "roll_mean_14": roll_mean_14, "dow": d.dayofweek,
            "day_of_month": d.day, "month": d.month, "product_id": product_id,
        }])[FEATURE_COLS]

        pred = max(0.0, float(model.predict(row)[0]))
        preds.append({"date": d, "forecast_demand": round(pred, 2)})
        series.append(pred)

    return pd.DataFrame(preds)


def linear_trend_baseline(panel, product_id, horizon_days=14):
    """Simple per-product linear regression trend, used as a comparison
    baseline against the RF model."""
    hist = panel[panel["product_id"] == product_id].sort_values("date").reset_index(drop=True)
    if len(hist) < 5:
        return pd.DataFrame(columns=["date", "forecast_demand"])

    x = np.arange(len(hist)).reshape(-1, 1)
    y = hist["demand"].values
    lr = LinearRegression().fit(x, y)

    future_x = np.arange(len(hist), len(hist) + horizon_days).reshape(-1, 1)
    preds = np.clip(lr.predict(future_x), 0, None)
    last_date = hist["date"].max()
    dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]
    return pd.DataFrame({"date": dates, "forecast_demand": np.round(preds, 2)})


# --------------------------------------------------------------------------
# Reorder Point Optimization
# --------------------------------------------------------------------------
def reorder_optimization(products_df, panel, service_level="95%", horizon_days=14, forecast_fn=None,
                          model=None):
    z = Z_TABLE[service_level]
    results = []

    for _, prod in products_df.iterrows():
        pid = prod["product_id"]
        hist = panel[panel["product_id"] == pid]["demand"]
        if hist.empty:
            continue

        avg_daily_demand = float(hist.mean())
        std_daily_demand = float(hist.std(ddof=0))
        lead_time = int(prod["lead_time_days"])

        safety_stock = z * std_daily_demand * np.sqrt(lead_time)
        reorder_point = avg_daily_demand * lead_time + safety_stock

        # EOQ (Economic Order Quantity) - bonus classical optimization
        annual_demand = avg_daily_demand * 365
        holding_cost = prod["cost_price"] * prod["holding_cost_pct"]
        eoq = np.sqrt((2 * annual_demand * prod["ordering_cost"]) / holding_cost) if holding_cost > 0 else np.nan

        results.append({
            "product_id": pid,
            "sku": prod["sku"],
            "name": prod["name"],
            "category": prod["category"],
            "lead_time_days": lead_time,
            "avg_daily_demand": round(avg_daily_demand, 2),
            "std_daily_demand": round(std_daily_demand, 2),
            "safety_stock": round(safety_stock, 1),
            "reorder_point": round(reorder_point, 1),
            "eoq": round(eoq, 0) if not np.isnan(eoq) else None,
            "service_level": service_level,
        })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Anomaly Detection
# --------------------------------------------------------------------------
def _txn_features(transactions_df):
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek

    # per-product, per-transaction-type mean/std for z-score features
    stats = (
        df.groupby(["product_id", "transaction_type"])["quantity"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "grp_mean", "std": "grp_std"})
    )
    df = df.merge(stats, on=["product_id", "transaction_type"], how="left")
    df["grp_std"] = df["grp_std"].replace(0, np.nan).fillna(1.0)
    df["zscore"] = (df["quantity"] - df["grp_mean"]) / df["grp_std"]

    # days since previous transaction for this product (gap anomalies)
    df = df.sort_values(["product_id", "date"])
    df["days_since_prev"] = df.groupby("product_id")["date"].diff().dt.days.fillna(0)

    return df


def detect_anomalies_isolation_forest(transactions_df, contamination=0.015):
    df = _txn_features(transactions_df)

    feature_cols = ["quantity", "zscore", "days_since_prev", "dow"]
    X = df[feature_cols].fillna(0)

    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    df["if_flag"] = model.fit_predict(X) == -1  # -1 = anomaly
    df["if_score"] = -model.decision_function(X)  # higher = more anomalous

    return df


def detect_anomalies_zscore(transactions_df, threshold=3.0):
    df = _txn_features(transactions_df)
    df["z_flag"] = df["zscore"].abs() > threshold
    return df


def evaluate_against_ground_truth(df, pred_col):
    """Compute precision/recall/F1 of a predicted flag column against the
    synthetic ground-truth `is_anomaly` label (evaluation-only; the label is
    never used as a model feature)."""
    y_true = df["is_anomaly"].astype(int)
    y_pred = df[pred_col].astype(int)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "flagged_count": int(y_pred.sum()),
        "true_anomaly_count": int(y_true.sum()),
    }
