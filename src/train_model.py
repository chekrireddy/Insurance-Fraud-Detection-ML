"""
Insurance Claim Fraud Detection - Model Training Pipeline

Phase 3 trains and compares multiple classification models for fraud detection.
The best model is selected using ROC AUC because fraud detection is an
imbalanced classification problem, where accuracy alone can be misleading.

Outputs:
    models/best_fraud_model.pkl
    reports/model_metrics.txt
    reports/confusion_matrix.png
"""

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


matplotlib.use("Agg")


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "insurance_claims.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_OUTPUT_PATH = MODELS_DIR / "best_fraud_model.pkl"
METRICS_OUTPUT_PATH = REPORTS_DIR / "model_metrics.txt"
CONFUSION_MATRIX_OUTPUT_PATH = REPORTS_DIR / "confusion_matrix.png"

TARGET_COLUMN = "is_fraud"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the fraud detection dataset from CSV.

    Args:
        file_path: Path to the insurance claims CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    logger.info("Loading dataset from %s", file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info("Dataset loaded successfully with shape %s", df.shape)
    return df


# ============================================================================
# PREPROCESSING
# ============================================================================

def validate_target(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate that the target column exists and contains at least two classes.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    unique_targets = df[target_column].dropna().unique()
    if len(unique_targets) < 2:
        raise ValueError("Target column must contain at least two classes.")


def detect_feature_columns(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """
    Automatically detect numerical and categorical feature columns.

    The claim_id column is excluded because it is an identifier, not a predictive
    business feature. Keeping IDs in a model can create noise or accidental
    memorization without helping future predictions.
    """
    excluded_columns = {target_column, "claim_id"}
    feature_df = df.drop(columns=[col for col in excluded_columns if col in df.columns])

    numerical_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info("Detected numerical columns: %s", numerical_columns)
    logger.info("Detected categorical columns: %s", categorical_columns)

    return numerical_columns, categorical_columns


def handle_missing_values(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Handle unexpected null values before model training.

    Numerical nulls are filled with the median. Categorical nulls are filled with
    the mode when available, otherwise with 'Unknown'.
    """
    df = df.copy()

    for column in numerical_columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            logger.warning("Filling nulls in numerical column '%s' with median=%s", column, median_value)
            df[column] = df[column].fillna(median_value)

    for column in categorical_columns:
        if df[column].isnull().any():
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            logger.warning("Filling nulls in categorical column '%s' with value=%s", column, fill_value)
            df[column] = df[column].fillna(fill_value)

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains null values.")

    return df


def encode_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label encode categorical feature columns.

    A separate LabelEncoder is stored for each categorical column so the saved
    model artifact contains the preprocessing needed for future inference.
    """
    df = df.copy()
    label_encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        label_encoders[column] = encoder
        logger.info("Encoded categorical column '%s'", column)

    return df, label_encoders


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], dict[str, LabelEncoder]]:
    """
    Prepare X/y data for model training.
    """
    validate_target(df, target_column)

    numerical_columns, categorical_columns = detect_feature_columns(df, target_column)
    df = handle_missing_values(df, numerical_columns, categorical_columns)
    df, label_encoders = encode_categorical_columns(df, categorical_columns)

    feature_columns = numerical_columns + categorical_columns
    X = df[feature_columns]
    y = df[target_column].astype(int)

    logger.info("Prepared %s features and target '%s'", len(feature_columns), target_column)
    return X, y, numerical_columns, categorical_columns, label_encoders


# ============================================================================
# MODEL TRAINING
# ============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into train and test sets using stratification.
    """
    logger.info("Splitting data into 80% train and 20% test sets")
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models(y_train: pd.Series) -> dict[str, object]:
    """
    Create the model candidates for comparison.

    Class balancing is used where supported to better handle the minority fraud
    class. XGBoost uses scale_pos_weight for the same purpose.
    """
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        ),
    }


def train_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """
    Train all candidate models.
    """
    trained_models = {}

    for model_name, model in models.items():
        logger.info("Training model: %s", model_name)
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        logger.info("Completed training: %s", model_name)

    return trained_models


# ============================================================================
# EVALUATION
# ============================================================================

def get_prediction_probabilities(model: object, X_test: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class prediction probabilities for ROC AUC.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Model {type(model).__name__} does not support predict_proba.")

    return model.predict_proba(X_test)[:, 1]


def evaluate_model(
    model_name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """
    Evaluate a trained model using classification metrics.
    """
    logger.info("Evaluating model: %s", model_name)

    y_pred = model.predict(X_test)
    y_proba = get_prediction_probabilities(model, X_test)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "Confusion_Matrix": matrix,
    }


def evaluate_models(
    trained_models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all trained models and return a comparison DataFrame.
    """
    evaluation_results = [
        evaluate_model(model_name, model, X_test, y_test)
        for model_name, model in trained_models.items()
    ]

    comparison_df = pd.DataFrame(evaluation_results)
    comparison_df = comparison_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return comparison_df


def select_best_model(
    comparison_df: pd.DataFrame,
    trained_models: dict[str, object],
) -> tuple[str, object, dict[str, object]]:
    """
    Select the best model by ROC AUC.
    """
    best_row = comparison_df.iloc[0].to_dict()
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    logger.info("Best model selected: %s with ROC AUC %.4f", best_model_name, best_row["ROC_AUC"])
    return best_model_name, best_model, best_row


# ============================================================================
# OUTPUTS
# ============================================================================

def format_model_comparison(comparison_df: pd.DataFrame) -> str:
    """
    Format model comparison results for console and text report output.
    """
    display_columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    display_df = comparison_df[display_columns].copy()

    for column in display_columns[1:]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    return display_df.to_string(index=False)


def print_model_summary(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_model_metrics: dict[str, object],
) -> None:
    """
    Print the required model comparison summary.
    """
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(format_model_comparison(comparison_df))
    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_model_name}")
    print(f"ROC AUC: {best_model_metrics['ROC_AUC']:.4f}")
    print("=" * 50)


def save_best_model(
    best_model_name: str,
    best_model: object,
    best_model_metrics: dict[str, object],
    numerical_columns: list[str],
    categorical_columns: list[str],
    label_encoders: dict[str, LabelEncoder],
) -> None:
    """
    Save the best model and preprocessing metadata as one production-style artifact.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    model_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "target_column": TARGET_COLUMN,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "label_encoders": label_encoders,
        "best_metrics": {
            key: value
            for key, value in best_model_metrics.items()
            if key != "Confusion_Matrix"
        },
    }

    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)
    logger.info("Best model artifact saved to %s", MODEL_OUTPUT_PATH)


def save_metrics_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_model_metrics: dict[str, object],
) -> None:
    """
    Save model comparison metrics and best-model details to a text report.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    report = [
        "=" * 50,
        "MODEL COMPARISON",
        "=" * 50,
        format_model_comparison(comparison_df),
        "",
        "=" * 50,
        f"BEST MODEL: {best_model_name}",
        f"ROC AUC: {best_model_metrics['ROC_AUC']:.4f}",
        "=" * 50,
        "",
        "Best Model Confusion Matrix:",
        str(best_model_metrics["Confusion_Matrix"]),
        "",
        "Metric Selection Note:",
        "ROC AUC is used as the primary model selection metric because fraud",
        "detection is an imbalanced classification problem. Accuracy alone can",
        "hide poor minority-class performance.",
    ]

    METRICS_OUTPUT_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Metrics report saved to %s", METRICS_OUTPUT_PATH)


def save_confusion_matrix_plot(
    best_model_name: str,
    confusion_matrix_values: np.ndarray,
) -> None:
    """
    Save a confusion matrix heatmap for the best model.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Legitimate", "Predicted Fraud"],
        yticklabels=["Actual Legitimate", "Actual Fraud"],
    )
    plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Confusion matrix image saved to %s", CONFUSION_MATRIX_OUTPUT_PATH)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_training_pipeline() -> None:
    """
    Run the complete Phase 3 model training pipeline.
    """
    df = load_data(DATA_PATH)
    X, y, numerical_columns, categorical_columns, label_encoders = prepare_features_and_target(
        df,
        TARGET_COLUMN,
    )

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale features for Logistic Regression while preserving tree-model inputs.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    models = build_models(y_train)

    trained_models = {
        "Logistic Regression": models["Logistic Regression"].fit(X_train_scaled, y_train),
        "Random Forest": models["Random Forest"].fit(X_train, y_train),
        "XGBoost": models["XGBoost"].fit(X_train, y_train),
    }

    # Evaluate Logistic Regression with scaled features and tree models with raw encoded features.
    evaluation_results = [
        evaluate_model("Logistic Regression", trained_models["Logistic Regression"], X_test_scaled, y_test),
        evaluate_model("Random Forest", trained_models["Random Forest"], X_test, y_test),
        evaluate_model("XGBoost", trained_models["XGBoost"], X_test, y_test),
    ]
    comparison_df = pd.DataFrame(evaluation_results)
    comparison_df = comparison_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

    best_model_name, best_model, best_model_metrics = select_best_model(comparison_df, trained_models)
    if best_model_name == "Logistic Regression":
        best_model = {
            "pipeline_type": "scaled_logistic_regression",
            "scaler": scaler,
            "model": trained_models[best_model_name],
        }

    print_model_summary(comparison_df, best_model_name, best_model_metrics)
    save_best_model(
        best_model_name,
        best_model,
        best_model_metrics,
        numerical_columns,
        categorical_columns,
        label_encoders,
    )
    save_metrics_report(comparison_df, best_model_name, best_model_metrics)
    save_confusion_matrix_plot(best_model_name, best_model_metrics["Confusion_Matrix"])


def main() -> None:
    """
    Entry point with top-level exception handling.
    """
    try:
        logger.info("Starting Phase 3 model training pipeline")
        run_training_pipeline()
        logger.info("Phase 3 model training pipeline completed successfully")
    except Exception as error:
        logger.exception("Model training pipeline failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
