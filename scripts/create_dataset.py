"""
Generates synthetic data/ecommerce_sales.csv for evaluation and demo purposes.
Explicitly labeled as a synthetic evaluation benchmark dataset, not real customer/production data.
"""

import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

# Deterministic random seed
np.random.seed(42)

n_records = 100

order_ids = [f"ORD-{1001 + i}" for i in range(n_records)]
customers = [
    ("CUST-101", "Alice Smith", "North"),
    ("CUST-102", "Bob Jones", "East"),
    ("CUST-103", "Charlie Brown", "West"),
    ("CUST-104", "Diana Prince", "South"),
    ("CUST-105", "Evan Wright", "North"),
    ("CUST-106", "Fiona Gallagher", "East"),
    ("CUST-107", "George Clark", "West"),
    ("CUST-108", "Hannah Abbott", "South"),
    ("CUST-109", "Ian Malcolm", "North"),
    ("CUST-110", "Julia Roberts", "East"),
    ("CUST-111", "Kevin Bacon", "West"),
    ("CUST-112", "Laura Croft", "South"),
    ("CUST-113", "Michael Scott", "North"),
    ("CUST-114", "Nancy Wheeler", "East"),
    ("CUST-115", "Oscar Martinez", "West"),
    ("CUST-116", "Pam Beesly", "South"),
    ("CUST-117", "Quinn Fabray", "North"),
    ("CUST-118", "Ryan Howard", "East"),
    ("CUST-119", "Sarah Connor", "West"),
    ("CUST-120", "Thomas Anderson", "South")
]

products = [
    ("Electronics", "Laptop Pro", 1200.0),
    ("Electronics", "Wireless Mouse", 25.0),
    ("Electronics", "Noise-Cancelling Headphones", 200.0),
    ("Electronics", "Mechanical Keyboard", 110.0),
    ("Electronics", "USB-C Hub", 45.0),
    ("Electronics", "4K Monitor", 400.0),
    ("Furniture", "Ergonomic Chair", 350.0),
    ("Furniture", "Standing Desk", 600.0),
    ("Furniture", "Desk Lamp", 40.0),
    ("Office Supplies", "Notebook Pack", 15.0),
    ("Office Supplies", "Ballpoint Pens (12pk)", 10.0),
    ("Office Supplies", "Stapler & Clips Set", 18.0),
    ("Apparel", "Cotton Hoodie", 55.0),
    ("Apparel", "Running Shoes", 95.0),
    ("Apparel", "Thermal Jacket", 140.0)
]

dates = pd.date_range(start="2023-01-05", end="2023-12-25", periods=n_records).strftime("%Y-%m-%d").tolist()

rows = []
for i in range(n_records):
    cust_id, cust_name, default_region = customers[i % len(customers)]
    category, product_name, base_price = products[np.random.randint(0, len(products))]
    qty = int(np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05]))
    
    # discount with a few nulls
    if i in [12, 45, 78]:
        discount = np.nan
        applied_discount = 0.0
    else:
        discount = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.35, 0.25, 0.2, 0.1, 0.1]))
        applied_discount = discount
        
    revenue = round(qty * base_price * (1.0 - applied_discount), 2)
    # realistic margin between 10% and 40%, with occasional slight negative on heavy discount
    margin_pct = np.random.uniform(0.12, 0.38) if applied_discount < 0.15 else np.random.uniform(-0.05, 0.15)
    profit = round(revenue * margin_pct, 2)
    
    # customer rating (1 to 5) with a few missing values
    if i in [8, 33, 62, 91]:
        rating = np.nan
    else:
        rating = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.40, 0.25]))

    rows.append({
        "order_id": order_ids[i],
        "order_date": dates[i],
        "customer_id": cust_id,
        "customer_name": cust_name,
        "region": default_region,
        "category": category,
        "product_name": product_name,
        "quantity": qty,
        "unit_price": base_price,
        "discount": discount,
        "revenue": revenue,
        "profit": profit,
        "rating": rating
    })

df = pd.DataFrame(rows)
output_path = os.path.join("data", "ecommerce_sales.csv")
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} rows to {output_path}")
print(df.info())
print("Summary stats:")
print(f"Total Revenue: {df['revenue'].sum():.2f}")
print(f"Average Revenue: {df['revenue'].mean():.2f}")
print(f"Total Profit: {df['profit'].sum():.2f}")
print(f"Total Orders: {len(df)}")
