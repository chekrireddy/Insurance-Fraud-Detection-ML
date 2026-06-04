import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

df = pd.DataFrame({
    "age": np.random.randint(18, 80, n),
    "claim_amount": np.random.randint(1000, 50000, n),
    "premium": np.random.randint(200, 5000, n),
    "num_previous_claims": np.random.randint(0, 10, n),
    "policy_years": np.random.randint(1, 20, n)
})

# Create fraud pattern
fraud_score = (
    (df["claim_amount"] > 30000).astype(int)
    + (df["num_previous_claims"] > 5).astype(int)
)

df["fraud_reported"] = np.where(fraud_score >= 1, 1, 0)

df.to_csv("../data/insurance_claims.csv", index=False)

print("Dataset Generated")
print(df.head())