from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

validation = pd.read_csv(
    PROJECT_ROOT / "data" / "validation.csv"
)

artifact = joblib.load(
    PROJECT_ROOT / "models" / "baseline_model.joblib"
)

model = artifact["pipeline"]
features = artifact["features"]

actual = validation["is_fraud"]
scores = model.predict_proba(validation[features])[:, 1]

results = []

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    predicted = (scores >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "claims_flagged": int(predicted.sum()),
        "fraud_caught": int(
            ((predicted == 1) & (actual == 1)).sum()
        ),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
    })

comparison = pd.DataFrame(results)

print("VALIDATION THRESHOLD COMPARISON")
print(comparison.round(3).to_string(index=False))