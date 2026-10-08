from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
validation = pd.read_csv(PROJECT_ROOT / "data" / "validation.csv")

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

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_validation)[:, 1]

    # Average precision evaluates ranking across thresholds.
    average_precision = average_precision_score(
        y_validation, scores
    )

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        predictions = (scores >= threshold).astype(int)

        results.append({
            "model": name,
            "threshold": threshold,
            "flagged": int(predictions.sum()),
            "fraud_caught": int(
                ((predictions == 1) & (y_validation == 1)).sum()
            ),
            "precision": precision_score(
                y_validation, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_validation, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_validation, predictions, zero_division=0
            ),
            "average_precision": average_precision,
        })

comparison = pd.DataFrame(results)

# Show the highest validation F1 for each model
# among the six thresholds we tested.
best_rows = comparison.loc[
    comparison.groupby("model")["f1"].idxmax()
].sort_values("f1", ascending=False)

print("BEST VALIDATION F1 PER MODEL")
print(best_rows.round(3).to_string(index=False))

reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(exist_ok=True)

comparison.to_csv(
    reports_dir / "model_comparison.csv",
    index=False,
)

# Save Random Forest separately, preserving the baseline.
forest_row = best_rows[
    best_rows["model"] == "Random Forest"
].iloc[0]

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(
    {
        "pipeline": models["Random Forest"],
        "features": features,
        "threshold": float(forest_row["threshold"]),
    },
    models_dir / "random_forest_model.joblib",
)

print("\nSaved reports/model_comparison.csv")
print("Saved models/random_forest_model.joblib")