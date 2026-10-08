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

## Learning baseline — work in progress

This branch adds a personal learning project using 5,000 synthetic claims.
The scripts below use a different schema from the earlier pipeline above;
run them in order and do not mix the two pipelines. The current requirements
file supports the learning baseline; legacy XGBoost/MLflow scripts need their
original dependencies. Existing historical reports describe the earlier model.

```powershell
python -m pip install -r requirements.txt
python src/generate_data.py
python src/check_data.py
python src/split_data.py
python src/train_baseline.py
python src/check_thresholds.py
```

Chronological splitting with a 30-day outcome delay produces 2,820 training,
820 validation, and 1,000 test claims. Another 360 claims are excluded at
split boundaries because their labels would not yet be available.

The baseline script currently uses a 0.50 threshold: validation precision
0.200, recall 0.018, F1 0.033, and average precision 0.160.
Threshold comparison found that 0.10 had the highest validation F1 among
six tested thresholds: precision 0.146, recall 0.473, F1 0.223.
That comparison does not automatically change the saved model threshold.
These results were obtained in the local learning run. The test set remains
unused for model evaluation. Results are synthetic, not GEICO business results.

Generated datasets and models are ignored for new files. Historical data and
model files already tracked in Git remain tracked; .gitignore does not remove them.
