from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "data" / "insurance_claims.csv"

claims = pd.read_csv(
    data_path,
    parse_dates=["claim_date", "label_available_date"],
)

print("DATASET SIZE")
print(f"Rows: {claims.shape[0]:,}")
print(f"Columns: {claims.shape[1]}")

print("\nMISSING VALUES")
print(claims.isna().sum())

print("\nDUPLICATE CLAIM IDs")
print(claims["claim_id"].duplicated().sum())

print("\nFRAUD DISTRIBUTION")
print(claims["is_fraud"].value_counts().sort_index())
print(f"Fraud rate: {claims['is_fraud'].mean():.2%}")

print("\nCLAIM AMOUNT SUMMARY")
print(claims["claim_amount"].describe().round(2))