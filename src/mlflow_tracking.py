import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Create experiment
mlflow.set_experiment("Insurance_Fraud_Detection")

# Load dataset
df = pd.read_csv("data/insurance_claims.csv")

# Print all available columns so the dataset schema is visible when rerunning.
print("Available columns:")
print(df.columns.tolist())

# Automatically identify the fraud target column used by this dataset.
target_candidates = ["is_fraud", "fraud_reported", "fraudulent", "target"]
target_column = next((column for column in target_candidates if column in df.columns), None)

if target_column is None:
    print("Target column not found. Available columns are:")
    print(df.columns.tolist())
    raise KeyError("Fraud target column not found in dataset.")

print(f"Target column used: {target_column}")

# Target column
y = df[target_column]

# Features
X = df.drop(target_column, axis=1)

# Encode categorical columns
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            model_name.replace(" ", "_")
        )

        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

print("\nMLflow tracking completed successfully!")
