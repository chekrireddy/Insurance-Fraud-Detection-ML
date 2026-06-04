# Insurance-Fraud-Detection-ML

Production-style machine learning project for insurance claim fraud detection.

## Project Status

- Phase 1: Synthetic insurance claims dataset generated
- Phase 2: Exploratory data analysis completed
- Phase 3: Model training pipeline completed

## Repository Structure

```text
data/
  insurance_claims.csv
models/
  best_fraud_model.pkl
reports/
  model_metrics.txt
  confusion_matrix.png
  01_fraud_distribution.png
  02_claim_amount_distribution.png
  03_premium_distribution.png
  04_correlation_heatmap.png
  05_fraud_by_region.png
  06_fraud_by_claim_type.png
  07_fraud_by_claim_range.png
src/
  generate_dataset.py
  eda.py
  train_model.py
requirements.txt
```

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `pip` is not available but the Windows Python launcher is installed:

```powershell
py -m pip install -r requirements.txt
```

## Run Commands

Generate the dataset:

```powershell
py src/generate_dataset.py
```

Run EDA and generate visual reports:

```powershell
py src/eda.py
```

Train and evaluate fraud detection models:

```powershell
py src/train_model.py
```

## Phase 3 Results

The training pipeline compares:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

The best model is selected using ROC AUC because fraud detection is an imbalanced classification problem where accuracy alone can be misleading.

Current best model:

```text
Random Forest
ROC AUC: 0.9988
```

## Next Phase

Phase 4 will add an inference pipeline for scoring new insurance claims with the saved model.
