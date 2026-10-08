from pathlib import Path

import numpy as np
import pandas as pd


# Fixed seed makes the generated data reproducible.
rng = np.random.default_rng(42)
NUMBER_OF_CLAIMS = 5000

# Resolve paths relative to this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Create information available when a claim is reviewed.
claims = pd.DataFrame({
    "claim_id": np.arange(1, NUMBER_OF_CLAIMS + 1),
    "claim_amount": rng.lognormal(
        mean=8.5, sigma=0.8, size=NUMBER_OF_CLAIMS
    ).round(2),
    "policy_age_days": rng.integers(
        1, 3651, size=NUMBER_OF_CLAIMS
    ),
    "report_delay_days": rng.poisson(
        lam=4, size=NUMBER_OF_CLAIMS
    ),
    "previous_claims": rng.poisson(
        lam=0.8, size=NUMBER_OF_CLAIMS
    ),
    "repair_estimate_ratio": np.clip(
        rng.normal(loc=1.0, scale=0.3, size=NUMBER_OF_CLAIMS),
        0.1,
        3.0,
    ).round(2),
})

# Ordered timestamps will let us train on earlier claims
# and evaluate on later claims.
claims["claim_date"] = pd.date_range(
    start="2023-01-01",
    periods=NUMBER_OF_CLAIMS,
    freq="4h",
)

# Invented relationships for this learning exercise.
# These are assumptions, not findings about real insurance fraud.
risk_score = (
    -4.0
    + 0.6 * np.log(claims["claim_amount"] / 5000)
    + 1.4 * (claims["policy_age_days"] < 180)
    + 0.15 * claims["report_delay_days"]
    + 0.45 * claims["previous_claims"]
    + 1.5 * (claims["repair_estimate_ratio"] > 1.4)
)

# Convert the score into a probability between 0 and 1.
fraud_probability = 1 / (1 + np.exp(-risk_score))

# Draw an outcome from that probability.
# Higher risk does not automatically mean fraud.
claims["is_fraud"] = rng.binomial(
    n=1,
    p=fraud_probability,
)

# Assume investigation outcomes become available after 30 days.
claims["label_available_date"] = (
    claims["claim_date"] + pd.Timedelta(days=30)
)

output_path = DATA_DIR / "insurance_claims.csv"
claims.to_csv(output_path, index=False)

print(f"Saved dataset to: {output_path}")
print(f"Total claims: {len(claims):,}")
print(f"Fraudulent claims: {claims['is_fraud'].sum():,}")
print(f"Fraud rate: {claims['is_fraud'].mean():.2%}")
print("\nFirst five claims:")
print(claims.head().to_string(index=False))