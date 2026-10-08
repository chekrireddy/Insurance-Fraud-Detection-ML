from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

claims = pd.read_csv(
    DATA_DIR / "insurance_claims.csv",
    parse_dates=["claim_date", "label_available_date"],
)

claims = claims.sort_values("claim_date").reset_index(drop=True)

# First 60% of time: training.
# Next 20%: validation.
# Final 20%: testing.
validation_start = claims.iloc[int(len(claims) * 0.6)]["claim_date"]
test_start = claims.iloc[int(len(claims) * 0.8)]["claim_date"]

# Training outcomes must be known before validation begins.
train = claims[
    (claims["claim_date"] < validation_start)
    & (claims["label_available_date"] < validation_start)
].copy()

# Validation outcomes must be known before testing begins.
validation = claims[
    (claims["claim_date"] >= validation_start)
    & (claims["claim_date"] < test_start)
    & (claims["label_available_date"] < test_start)
].copy()

test = claims[
    claims["claim_date"] >= test_start
].copy()

# Confirm that future outcomes cannot enter earlier stages.
assert train["label_available_date"].max() < validation_start
assert validation["label_available_date"].max() < test_start

for name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    dataset.to_csv(DATA_DIR / f"{name}.csv", index=False)

    print(
        f"{name}: {len(dataset):,} claims | "
        f"fraud rate: {dataset['is_fraud'].mean():.2%}"
    )

excluded = len(claims) - len(train) - len(validation) - len(test)
print(f"\nClaims excluded for label timing: {excluded}")
print("Saved train.csv, validation.csv, and test.csv")