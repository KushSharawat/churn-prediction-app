"""
Generate a synthetic (made-up) UK telecom customer dataset for churn prediction.

Churn is driven by realistic patterns (short tenure, month-to-month contracts,
high bills, many support calls, no tech support) plus random noise, so the
models have a genuine but imperfect signal to learn.

Run:  python data/generate_customers.py
Output: data/customers.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 7_000

contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
tenure = np.where(
    contract == "Month-to-month", rng.integers(1, 40, N),
    np.where(contract == "One year", rng.integers(6, 60, N), rng.integers(12, 72, N)),
)
internet = rng.choice(["Fibre", "DSL", "No internet"], N, p=[0.45, 0.35, 0.20])
base = np.select([internet == "Fibre", internet == "DSL"], [55, 35], 18)
tech_support = np.where(internet == "No internet", "No", rng.choice(["Yes", "No"], N, p=[0.35, 0.65]))
monthly = np.round(base + rng.normal(15, 10, N).clip(0) + (tech_support == "Yes") * 6, 2)
payment = rng.choice(["Direct Debit", "Credit Card", "Manual Payment"], N, p=[0.45, 0.25, 0.30])
support_calls = rng.poisson(1.2 + (internet == "Fibre") * 0.6, N)
senior = rng.choice([0, 1], N, p=[0.84, 0.16])
paperless = rng.choice(["Yes", "No"], N, p=[0.6, 0.4])

# Hidden "true" churn logic (the models never see this formula)
logit = (
    -1.6
    + 1.4 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
    - 0.045 * tenure
    + 0.025 * (monthly - 50)
    + 0.38 * support_calls
    + 0.6 * (internet == "Fibre")
    - 0.7 * (tech_support == "Yes")
    + 0.55 * (payment == "Manual Payment")
    + 0.35 * senior
    + 0.25 * (paperless == "Yes")
    + rng.normal(0, 0.6, N)
)
churn = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST-{10000 + i}" for i in range(N)],
    "tenure_months": tenure,
    "contract": contract,
    "internet_service": internet,
    "tech_support": tech_support,
    "monthly_charges": monthly,
    "payment_method": payment,
    "support_calls": support_calls,
    "senior_citizen": senior,
    "paperless_billing": paperless,
    "churn": churn,
})

out = Path(__file__).parent / "customers.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} customers to {out} (churn rate {df.churn.mean():.1%})")
