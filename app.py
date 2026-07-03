import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import ml_models as ml
from data_generator import main as regenerate_data

st.set_page_config(
    page_title="AI Inventory Management System",
    page_icon="📦",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    h1, h2, h3 { font-family: 'Trebuchet MS', sans-serif; color: #1C2541; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E5EA;
        border-left: 4px solid #0B7A75;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .anomaly-tag {
        background-color: #FFE3D3; color: #B4491F; padding: 2px 8px;
        border-radius: 4px; font-size: 0.8em; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    products = pd.read_csv(DATA_DIR / "products.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    return products, transactions


@st.cache_data
def get_panel(transactions):
    return ml.build_daily_demand_panel(transactions)


@st.cache_resource
def get_trained_model(transactions_hash):
    products, transactions = load_data()
    model, panel, r2 = ml.train_demand_model(transactions)
    return model, panel, r2


products_df, transactions_df = load_data()
panel = get_panel(transactions_df)

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("📦 AI Inventory System")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Demand Forecasting", "Reorder Optimization", "Anomaly Detection", "Product Catalog", "About / Data"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Products: **{len(products_df)}**")
st.sidebar.caption(f"Transactions: **{len(transactions_df):,}**")
st.sidebar.caption(f"Date range: {transactions_df['date'].min().date()} → {transactions_df['date'].max().date()}")

if st.sidebar.button("🔄 Regenerate synthetic data"):
    with st.spinner("Regenerating synthetic dataset..."):
        regenerate_data()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# ==========================================================================
# PAGE: OVERVIEW
# ==========================================================================
if page == "Overview":
    st.title("Inventory Overview")

    latest_stock = (
        transactions_df.sort_values("date")
        .groupby("product_id")["stock_after"].last()
        .reset_index().rename(columns={"stock_after": "current_stock"})
    )
    merged = products_df.merge(latest_stock, on="product_id", how="left")
    merged["current_stock"] = merged["current_stock"].fillna(0)
    low_stock = merged[merged["current_stock"] <= merged["reorder_level_baseline"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", len(merged))
    c2.metric("Total Transactions", f"{len(transactions_df):,}")
    c3.metric("Low Stock Items", len(low_stock))
    c4.metric("Inventory Value", f"₹{(merged['current_stock'] * merged['cost_price']).sum():,.0f}")

    st.markdown("### Sales Trend (last 30 days, all products)")
    sales = transactions_df[transactions_df["transaction_type"] == "SALE"]
    daily_rev = (
        sales.assign(revenue=sales["quantity"] * sales["product_id"].map(
            products_df.set_index("product_id")["selling_price"]))
        .groupby("date")["revenue"].sum().reset_index()
    )
    daily_rev = daily_rev.tail(30)
    fig = px.area(daily_rev, x="date", y="revenue", color_discrete_sequence=["#0B7A75"])
    fig.update_layout(height=320, margin=dict(t=10, l=10, r=10, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sales by Category")
        cat_sales = sales.merge(products_df[["product_id", "category"]], on="product_id")
        cat_totals = cat_sales.groupby("category")["quantity"].sum().reset_index()
        fig2 = px.bar(cat_totals, x="category", y="quantity", color="category",
                      color_discrete_sequence=px.colors.qualitative.Prism)
        fig2.update_layout(height=320, showlegend=False, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### ⚠️ Products Needing Attention (low stock)")
        st.dataframe(
            low_stock[["sku", "name", "category", "current_stock", "reorder_level_baseline"]]
            .sort_values("current_stock")
            .rename(columns={"reorder_level_baseline": "reorder_level"}),
            hide_index=True, use_container_width=True, height=320,
        )


# ==========================================================================
# PAGE: DEMAND FORECASTING
# ==========================================================================
elif page == "Demand Forecasting":
    st.title("📈 Demand Forecasting (Machine Learning)")
    st.caption("Global Random Forest regression model with lag / rolling-mean / calendar features, "
               "used for recursive multi-step forecasting. Compared against a per-product linear trend baseline.")

    with st.spinner("Training demand forecasting model..."):
        model, trained_panel, r2 = get_trained_model(len(transactions_df))

    st.info(f"Model trained on {len(products_df)} products' daily demand history · In-sample R² = **{r2:.3f}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        product_choice = st.selectbox(
            "Select product",
            products_df["name"] + " (" + products_df["sku"] + ")",
        )
        pid = products_df.iloc[(products_df["name"] + " (" + products_df["sku"] + ")").tolist().index(product_choice)]["product_id"]
    with col2:
        horizon = st.slider("Forecast horizon (days)", 7, 30, 14)

    rf_forecast = ml.forecast_product(model, trained_panel, pid, horizon_days=horizon)
    lr_forecast = ml.linear_trend_baseline(trained_panel, pid, horizon_days=horizon)

    hist = trained_panel[trained_panel["product_id"] == pid].tail(60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["demand"], name="Historical demand",
                              line=dict(color="#1C2541")))
    fig.add_trace(go.Scatter(x=rf_forecast["date"], y=rf_forecast["forecast_demand"],
                              name="Random Forest forecast", line=dict(color="#0B7A75", dash="dash")))
    if not lr_forecast.empty:
        fig.add_trace(go.Scatter(x=lr_forecast["date"], y=lr_forecast["forecast_demand"],
                                  name="Linear trend baseline", line=dict(color="#E8A33D", dash="dot")))
    fig.update_layout(height=420, margin=dict(t=20, l=10, r=10, b=10), plot_bgcolor="white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    total_forecast = rf_forecast["forecast_demand"].sum()
    st.metric(f"Total forecasted demand — next {horizon} days (RF model)", f"{total_forecast:.1f} units")

    with st.expander("View forecast table"):
        st.dataframe(rf_forecast.rename(columns={"forecast_demand": "RF forecast"}), hide_index=True)

    with st.expander("Feature importance (what drives the RF model's predictions)"):
        importances = pd.DataFrame({
            "feature": ml.FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fig_imp = px.bar(importances, x="importance", y="feature", orientation="h",
                          color_discrete_sequence=["#0B7A75"])
        fig_imp.update_layout(height=300, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================================================
# PAGE: REORDER OPTIMIZATION
# ==========================================================================
elif page == "Reorder Optimization":
    st.title("🎯 Reorder Point Optimization")
    st.caption("Safety stock + reorder point computed from each product's actual demand mean/variance and "
               "synthetic lead time, at a configurable service level. EOQ shown as a bonus classical optimization.")

    service_level = st.select_slider("Target service level (Z-factor)", options=list(ml.Z_TABLE.keys()), value="95%")

    with st.spinner("Computing reorder points..."):
        reorder_df = ml.reorder_optimization(products_df, panel, service_level=service_level)

    latest_stock = (
        transactions_df.sort_values("date").groupby("product_id")["stock_after"].last()
        .reset_index().rename(columns={"stock_after": "current_stock"})
    )
    reorder_df = reorder_df.merge(latest_stock, on="product_id", how="left")
    reorder_df["current_stock"] = reorder_df["current_stock"].fillna(0)
    reorder_df["needs_reorder"] = reorder_df["current_stock"] <= reorder_df["reorder_point"]
    reorder_df["suggested_order_qty"] = np.where(
        reorder_df["needs_reorder"],
        (reorder_df["reorder_point"] - reorder_df["current_stock"]).clip(lower=0).round(0),
        0,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Products needing reorder", int(reorder_df["needs_reorder"].sum()))
    c2.metric("Avg. safety stock (units)", f"{reorder_df['safety_stock'].mean():.1f}")
    c3.metric("Avg. reorder point (units)", f"{reorder_df['reorder_point'].mean():.1f}")

    st.markdown("### Reorder Recommendations")
    display_df = reorder_df.sort_values("needs_reorder", ascending=False)[
        ["sku", "name", "category", "current_stock", "avg_daily_demand", "std_daily_demand",
         "lead_time_days", "safety_stock", "reorder_point", "eoq", "suggested_order_qty", "needs_reorder"]
    ]
    st.dataframe(display_df, hide_index=True, use_container_width=True, height=420)

    st.markdown("### Reorder Point vs Current Stock")
    top20 = reorder_df.sort_values("needs_reorder", ascending=False).head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top20["sku"], y=top20["current_stock"], name="Current stock", marker_color="#0B7A75"))
    fig.add_trace(go.Scatter(x=top20["sku"], y=top20["reorder_point"], name="Reorder point",
                              mode="markers+lines", marker=dict(color="#E8A33D", size=8)))
    fig.update_layout(height=380, margin=dict(t=20, l=10, r=10, b=10), plot_bgcolor="white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Formulas used"):
        st.latex(r"\text{Safety Stock} = Z \times \sigma_{demand} \times \sqrt{\text{Lead Time}}")
        st.latex(r"\text{Reorder Point} = (\bar{d} \times \text{Lead Time}) + \text{Safety Stock}")
        st.latex(r"\text{EOQ} = \sqrt{\dfrac{2 D S}{H}}\quad\text{(D = annual demand, S = ordering cost, H = holding cost)}")


# ==========================================================================
# PAGE: ANOMALY DETECTION
# ==========================================================================
elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection in Stock Movements")
    st.caption("Isolation Forest (unsupervised ML) vs. a statistical Z-score baseline, evaluated against "
               "injected ground-truth anomalies in the synthetic dataset.")

    contamination = st.slider("Expected anomaly rate (Isolation Forest contamination)", 0.005, 0.05, 0.015, 0.005)
    z_threshold = st.slider("Z-score threshold (baseline method)", 2.0, 4.0, 3.0, 0.1)

    with st.spinner("Running anomaly detection models..."):
        if_df = ml.detect_anomalies_isolation_forest(transactions_df, contamination=contamination)
        z_df = ml.detect_anomalies_zscore(transactions_df, threshold=z_threshold)

    if_eval = ml.evaluate_against_ground_truth(if_df, "if_flag")
    z_eval = ml.evaluate_against_ground_truth(z_df, "z_flag")

    st.markdown("### Model Comparison (against injected ground-truth anomalies)")
    comp_df = pd.DataFrame([
        {"Model": "Isolation Forest", **if_eval},
        {"Model": "Z-score baseline", **z_eval},
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp_df, x="Model", y=["precision", "recall", "f1"], barmode="group",
                     color_discrete_sequence=["#0B7A75", "#E8A33D", "#1C2541"])
        fig.update_layout(height=320, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Why Isolation Forest?**")
        st.write(
            "It isolates points using random feature splits — anomalies need fewer splits to isolate, "
            "so it scores quantity, per-product z-score, day-of-week, and transaction gap jointly, "
            "rather than thresholding a single variable in isolation."
        )

    st.markdown("### Flagged Transactions (Isolation Forest)")
    flagged = if_df[if_df["if_flag"]].merge(
        products_df[["product_id", "name", "sku"]], on="product_id"
    ).sort_values("if_score", ascending=False)
    st.dataframe(
        flagged[["date", "sku", "name", "transaction_type", "quantity", "zscore", "if_score", "is_anomaly"]]
        .rename(columns={"is_anomaly": "actually_anomaly (ground truth)"})
        .head(50),
        hide_index=True, use_container_width=True, height=400,
    )

    st.markdown("### Anomalies Over Time")
    daily_anom = if_df.groupby(if_df["date"].dt.date)["if_flag"].sum().reset_index()
    fig2 = px.bar(daily_anom, x="date", y="if_flag", color_discrete_sequence=["#B4491F"])
    fig2.update_layout(height=280, margin=dict(t=10, l=10, r=10, b=10), yaxis_title="Anomalies flagged")
    st.plotly_chart(fig2, use_container_width=True)


# ==========================================================================
# PAGE: PRODUCT CATALOG
# ==========================================================================
elif page == "Product Catalog":
    st.title("🗂️ Product Catalog")

    latest_stock = (
        transactions_df.sort_values("date").groupby("product_id")["stock_after"].last()
        .reset_index().rename(columns={"stock_after": "current_stock"})
    )
    merged = products_df.merge(latest_stock, on="product_id", how="left")

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect("Category", options=sorted(merged["category"].unique()))
    with col2:
        sup_filter = st.multiselect("Supplier", options=sorted(merged["supplier"].unique()))
    with col3:
        search = st.text_input("Search by name / SKU")

    filtered = merged.copy()
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if sup_filter:
        filtered = filtered[filtered["supplier"].isin(sup_filter)]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False) | filtered["sku"].str.contains(search, case=False)
        ]

    st.dataframe(
        filtered[["sku", "name", "category", "supplier", "cost_price", "selling_price",
                  "current_stock", "reorder_level_baseline", "lead_time_days"]],
        hide_index=True, use_container_width=True, height=500,
    )
    st.caption(f"Showing {len(filtered)} of {len(merged)} products")


# ==========================================================================
# PAGE: ABOUT / DATA
# ==========================================================================
elif page == "About / Data":
    st.title("ℹ️ About This Project")
    st.markdown("""
### AI-Powered Intelligent Product and Inventory Management System
A college mini-project demonstrating three applied ML components on top of a
synthetic retail inventory dataset:

1. **Demand Forecasting** — Random Forest regression (lag + rolling-mean + calendar
   features) with recursive multi-step forecasting, benchmarked against a
   linear-trend baseline.
2. **Reorder Point Optimization** — statistical safety-stock / reorder-point
   formulas driven by each product's empirical demand mean & standard deviation,
   plus Economic Order Quantity (EOQ) as a bonus classical optimization.
3. **Anomaly Detection** — Isolation Forest (unsupervised) vs. Z-score baseline
   on engineered transaction features, evaluated with precision/recall/F1
   against injected ground-truth anomalies.

**Dataset**: synthetically generated — 60 products across 6 categories / 5
suppliers, with 200 days of simulated daily transactions (sales, restocks,
returns) including per-product trend patterns (rising / falling / stable /
seasonal-spike) and injected anomalies for evaluation.
    """)

    st.markdown("### Dataset samples")
    tab1, tab2 = st.tabs(["Products", "Transactions"])
    with tab1:
        st.dataframe(products_df.head(20), hide_index=True, use_container_width=True)
    with tab2:
        st.dataframe(transactions_df.head(20), hide_index=True, use_container_width=True)

    st.markdown("### Tech Stack")
    st.write("Streamlit · pandas · NumPy · scikit-learn (RandomForest, IsolationForest, LinearRegression) · Plotly")
