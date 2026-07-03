"""
Synthetic data generator for the AI-Powered Intelligent Product & Inventory
Management System.

Generates:
  - data/products.csv           : ~60 products across 6 categories / 5 suppliers
                                   with cost, price, lead time, ordering/holding
                                   cost (for EOQ), and a baseline reorder level.
  - data/transactions.csv       : 3000+ daily inventory movement records
                                   (SALE / RESTOCK / RETURN / ADJUSTMENT) over
                                   a 200-day window, generated with per-product
                                   demand patterns (trend + weekly seasonality +
                                   noise) and a small number of *injected*
                                   anomalies (sudden spikes / drops) whose ground
                                   truth label is kept in `is_anomaly` purely for
                                   evaluating the anomaly-detection model
                                   afterwards (it is NOT used as a model input).

Run directly:  python data_generator.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CATEGORIES = ["Electronics", "Groceries", "Stationery", "Home & Kitchen", "Apparel", "Personal Care"]
SUPPLIERS = ["Global Traders Ltd", "Metro Wholesale", "Prime Distributors", "Sunrise Imports", "Bluepeak Supply Co"]

PRODUCT_NAME_POOL = {
    "Electronics": ["Wireless Mouse", "USB-C Cable", "Bluetooth Earbuds", "Laptop Stand", "Power Bank 10000mAh",
                    "HDMI Cable 2m", "Wireless Keyboard", "Webcam HD", "Portable SSD 500GB", "Smart Plug"],
    "Groceries": ["Basmati Rice 5kg", "Cooking Oil 1L", "Wheat Flour 5kg", "Sugar 1kg", "Green Tea Pack",
                  "Instant Noodles Box", "Coffee Powder 200g", "Salt 1kg", "Honey 500g", "Peanut Butter 400g"],
    "Stationery": ["A4 Notebook", "Gel Pen Pack", "Sticky Notes", "Highlighter Set", "Stapler",
                   "Whiteboard Marker", "File Folder Pack", "Pencil Box", "Correction Pen", "Sketch Pens Set"],
    "Home & Kitchen": ["Desk Organizer", "Stainless Steel Bottle", "Non-stick Pan", "LED Desk Lamp", "Storage Box",
                       "Kitchen Scale", "Cutting Board", "Ceramic Mug Set", "Laundry Basket", "Wall Clock"],
    "Apparel": ["Cotton T-Shirt", "Formal Shirt", "Denim Jeans", "Hooded Sweatshirt", "Running Shorts",
               "Woolen Socks Pack", "Casual Cap", "Rain Jacket", "Track Pants", "Polo T-Shirt"],
    "Personal Care": ["Hand Sanitizer 200ml", "Face Wash 100ml", "Shampoo 400ml", "Toothpaste 150g", "Body Lotion 250ml",
                      "Sunscreen SPF50", "Shaving Cream", "Hair Oil 200ml", "Lip Balm", "Deodorant Spray"],
}

TREND_TYPES = ["rising", "falling", "stable", "seasonal_spike"]


def generate_products(n_per_category=10):
    rows = []
    pid = 1
    for cat in CATEGORIES:
        names = PRODUCT_NAME_POOL[cat]
        for name in names[:n_per_category]:
            cost = round(rng.uniform(20, 900), 2)
            margin = rng.uniform(0.3, 1.2)
            price = round(cost * (1 + margin), 2)
            lead_time = int(rng.integers(2, 15))
            ordering_cost = round(rng.uniform(50, 300), 2)      # cost to place one order (for EOQ)
            holding_cost_pct = round(rng.uniform(0.1, 0.25), 3)  # % of unit cost per year (for EOQ)
            reorder_level_baseline = int(rng.integers(10, 40))
            trend = rng.choice(TREND_TYPES, p=[0.3, 0.2, 0.3, 0.2])

            rows.append({
                "product_id": pid,
                "sku": f"SKU-{1000 + pid}",
                "name": name,
                "category": cat,
                "supplier": rng.choice(SUPPLIERS),
                "cost_price": cost,
                "selling_price": price,
                "lead_time_days": lead_time,
                "ordering_cost": ordering_cost,
                "holding_cost_pct": holding_cost_pct,
                "reorder_level_baseline": reorder_level_baseline,
                "demand_trend": trend,
            })
            pid += 1
    return pd.DataFrame(rows)


def generate_transactions(products_df, n_days=200, anomaly_rate=0.015):
    start_date = date.today() - timedelta(days=n_days)
    records = []
    txn_id = 1

    for _, prod in products_df.iterrows():
        base_demand = rng.uniform(1.5, 8.0)
        trend = prod["demand_trend"]
        stock = int(rng.integers(60, 200))  # starting stock

        # initial stock load transaction
        records.append({
            "transaction_id": txn_id, "product_id": prod["product_id"], "date": start_date,
            "transaction_type": "RESTOCK", "quantity": stock, "stock_after": stock,
            "is_anomaly": False,
        })
        txn_id += 1

        for i in range(1, n_days + 1):
            day = start_date + timedelta(days=i)
            weekday = day.weekday()

            # --- expected demand (lambda) based on trend pattern ---
            if trend == "rising":
                lam = base_demand * (1 + i / n_days * 1.6)
            elif trend == "falling":
                lam = max(0.15, base_demand * (1 - i / n_days * 0.75))
            elif trend == "seasonal_spike":
                lam = base_demand * (2.8 if weekday in (5, 6) else 0.9)
            else:  # stable
                lam = base_demand

            qty_sold = int(round(max(0, rng.normal(lam, lam * 0.4 + 0.3))))

            # --- inject anomaly: rare, sudden demand spike or drop ---
            is_anomaly = False
            if rng.random() < anomaly_rate and qty_sold >= 0:
                spike_type = rng.choice(["spike", "bulk_return"])
                if spike_type == "spike":
                    qty_sold = int(qty_sold + rng.uniform(15, 40))
                is_anomaly = True

            if qty_sold > 0:
                qty_sold = min(qty_sold, stock)  # cannot sell more than available
                if qty_sold > 0:
                    stock -= qty_sold
                    records.append({
                        "transaction_id": txn_id, "product_id": prod["product_id"], "date": day,
                        "transaction_type": "SALE", "quantity": qty_sold, "stock_after": stock,
                        "is_anomaly": is_anomaly,
                    })
                    txn_id += 1

            # --- periodic restock when stock runs low ---
            if stock <= prod["reorder_level_baseline"] and rng.random() < 0.8:
                restock_qty = int(rng.integers(40, 120))
                # occasional anomalous bulk restock
                anomaly_restock = False
                if rng.random() < anomaly_rate:
                    restock_qty += int(rng.uniform(80, 150))
                    anomaly_restock = True
                stock += restock_qty
                records.append({
                    "transaction_id": txn_id, "product_id": prod["product_id"], "date": day,
                    "transaction_type": "RESTOCK", "quantity": restock_qty, "stock_after": stock,
                    "is_anomaly": anomaly_restock,
                })
                txn_id += 1

            # --- rare customer returns ---
            if rng.random() < 0.01:
                ret_qty = int(rng.integers(1, 5))
                stock += ret_qty
                records.append({
                    "transaction_id": txn_id, "product_id": prod["product_id"], "date": day,
                    "transaction_type": "RETURN", "quantity": ret_qty, "stock_after": stock,
                    "is_anomaly": False,
                })
                txn_id += 1

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "product_id"]).reset_index(drop=True)


def main():
    products_df = generate_products(n_per_category=10)
    transactions_df = generate_transactions(products_df, n_days=200)

    products_path = DATA_DIR / "products.csv"
    transactions_path = DATA_DIR / "transactions.csv"
    products_df.to_csv(products_path, index=False)
    transactions_df.to_csv(transactions_path, index=False)

    print(f"Products generated:     {len(products_df)}")
    print(f"Transactions generated: {len(transactions_df)}  (target: 3000+)")
    print(f"Injected anomalies:     {int(transactions_df['is_anomaly'].sum())}")
    print(f"Saved to: {products_path} and {transactions_path}")


if __name__ == "__main__":
    main()
