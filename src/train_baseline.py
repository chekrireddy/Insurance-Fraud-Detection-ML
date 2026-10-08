from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

# Only use information available when scoring a claim.
features = [
    "claim_amount",
    "policy_age_days",
    "report_delay_days",
    "previous_claims",
    "repair_estimate_ratio",
]

X_train = train[features]
y_train = train["is_fraud"]

X_validation = validation[features]
y_validation = validation["is_fraud"]

# Scaling puts features with different units on comparable scales.
# The pipeline learns scaling values from training data only.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)

# Return a score between 0 and 1 for each validation claim.
scores = model.predict_proba(X_validation)[:, 1]

# Start with a conventional threshold; we'll examine it later.
threshold = 0.5
predictions = (scores >= threshold).astype(int)

print("VALIDATION RESULTS")
print(f"Claims evaluated: {len(validation)}")
print(f"Actual fraud cases: {y_validation.sum()}")
print(f"Claims flagged: {predictions.sum()}")

print(
    f"\nPrecision: "
    f"{precision_score(y_validation, predictions, zero_division=0):.3f}"
)
print(
    f"Recall: "
    f"{recall_score(y_validation, predictions, zero_division=0):.3f}"
)
print(
    f"F1: "
    f"{f1_score(y_validation, predictions, zero_division=0):.3f}"
)
print(
    f"Average precision: "
    f"{average_precision_score(y_validation, scores):.3f}"
)

print("\nConfusion matrix:")
print(confusion_matrix(y_validation, predictions, labels=[0, 1]))

# Save preprocessing and model together.
joblib.dump(
    {
        "pipeline": model,
        "features": features,
        "threshold": threshold,
    },
    MODELS_DIR / "baseline_model.joblib",
)

print("\nSaved models/baseline_model.joblib")