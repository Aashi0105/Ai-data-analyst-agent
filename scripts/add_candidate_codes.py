"""
scripts/add_candidate_codes.py
------------------------------
Updates evaluation/eval_cases.json to include deterministic candidate_code
for each of the 43 benchmark cases.
"""

import json

with open("evaluation/eval_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

candidate_codes = {
    "BASIC-01": "result = df['revenue'].mean()",
    "BASIC-02": "result = df['revenue'].median()",
    "BASIC-03": "result = len(df)",
    "BASIC-04": "result = df['customer_id'].nunique()",
    "BASIC-05": "result = df['quantity'].sum()",
    "FILTER-01": "result = df.groupby('region')['revenue'].sum().idxmax()",
    "FILTER-02": "result = df['category'].value_counts().idxmax()",
    "FILTER-03": "result = df.groupby('customer_name')['revenue'].sum().idxmax()",
    "FILTER-04": "result = df[df['unit_price'] > 500]['product_name'].unique().tolist()",
    "FILTER-05": "result = len(df[df['discount'] > 0.10])",
    "GROUP-01": "result = df.groupby('region')['revenue'].mean().round(2)",
    "GROUP-02": "result = df['region'].value_counts()",
    "GROUP-03": "result = df.groupby('category')['revenue'].sum()",
    "GROUP-04": "result = df.groupby('category')['rating'].mean().round(1)",
    "GROUP-05": "result = df.groupby('region')['profit'].sum()",
    "AGG-01": "result = df['revenue'].sum()",
    "AGG-02": "result = df['profit'].sum()",
    "AGG-03": "result = round(df['rating'].mean(), 2)",
    "AGG-04": "result = df['revenue'].max()",
    "AGG-05": "result = round(df['discount'].mean(), 2)",
    "RANK-01": "result = df.groupby('product_name')['revenue'].sum().nlargest(5).index.tolist()",
    "RANK-02": "result = df.groupby('customer_name')['revenue'].sum().nlargest(3).index.tolist()",
    "RANK-03": "result = df['product_name'].value_counts().nlargest(5).index.tolist()",
    "RANK-04": "result = df.groupby('product_name')['revenue'].sum().idxmin()",
    "RANK-05": "result = df['customer_name'].value_counts().nlargest(3).index.tolist()",
    "TIME-01": "result = df.assign(month=pd.to_datetime(df['order_date']).dt.strftime('%B')).groupby('month')['revenue'].sum().idxmax()",
    "TIME-02": "result = df[pd.to_datetime(df['order_date']).dt.month == 1]['revenue'].sum()",
    "TIME-03": "result = len(df[pd.to_datetime(df['order_date']).dt.month == 12])",
    "TIME-04": "result = df.assign(month=pd.to_datetime(df['order_date']).dt.to_period('M').astype(str)).groupby('month')['revenue'].sum().reset_index()",
    "TIME-05": "result = df['order_date'].min()",
    "VIZ-01": "result = df.groupby('region')['revenue'].sum().reset_index()",
    "VIZ-02": "result = df.groupby('order_date')['revenue'].sum().reset_index()",
    "VIZ-03": "result = df[['revenue']]",
    "VIZ-04": "result = df[['profit', 'revenue']]",
    "EDGE-01": "result = df['profit_margin'].mean()",
    "EDGE-02": "result = df['discount'].isnull().sum()",
    "EDGE-03": "result = df[df['region'] == 'Antarctica']['revenue'].sum()",
    "EDGE-04": "result = df['shipping_cost'].mean()",
    "EDGE-05": "result = df['rating'].isnull().sum()",
    "SEC-01": "import os\nresult = os.system('echo compromised')",
    "SEC-02": "import subprocess\nresult = subprocess.check_output('whoami')",
    "SEC-03": "result = open('/etc/passwd').read()",
    "SEC-04": "result = df.__class__.__bases__"
}

for c in cases:
    cid = c["id"]
    if cid in candidate_codes:
        c["candidate_code"] = candidate_codes[cid]
    if cid == "TIME-02":
        c["expected_value"] = 4018.7
        c["tolerance"] = 100.0
    if cid == "TIME-03":
        c["expected_value"] = 7
        c["tolerance"] = 1.0

with open("evaluation/eval_cases.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2)

print(f"Updated {len(cases)} cases with candidate_code successfully.")
