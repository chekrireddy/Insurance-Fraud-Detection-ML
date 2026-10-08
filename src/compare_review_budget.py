"""Compare saved models at a 10% validation review budget."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    validation = pd.read_csv(ROOT / 'data' / 'validation.csv')
    if validation.empty:
        raise ValueError('Validation data must not be empty.')
    review_count = int(np.ceil(len(validation) * 0.10))
    total_fraud = int(validation['is_fraud'].sum())
    results = []
    files = {
        'Logistic Regression': 'baseline_model.joblib',
        'Random Forest': 'random_forest_model.joblib',
    }
    for name, filename in files.items():
        artifact = joblib.load(ROOT / 'models' / filename)
        scores = artifact['pipeline'].predict_proba(
            validation[artifact['features']]
        )[:, 1]
        # Rank highest scores first; ties retain original row order.
        positions = np.argsort(-scores, kind='stable')[:review_count]
        caught = int(validation.iloc[positions]['is_fraud'].sum())
        results.append({
            'model': name,
            'claims_reviewed': review_count,
            'fraud_caught': caught,
            'false_alarms': review_count - caught,
            'precision': caught / review_count,
            'recall': caught / total_fraud if total_fraud else 0.0,
        })
    comparison = pd.DataFrame(results)
    print(f'Validation claims: {len(validation)}')
    print(f'Total fraud cases: {total_fraud}')
    print(f'Review budget: {review_count} claims\n')
    print(comparison.round(3).to_string(index=False))
    (ROOT / 'reports').mkdir(exist_ok=True)
    comparison.to_csv(ROOT / 'reports' / 'review_budget_comparison.csv', index=False)
    print('\nSaved reports/review_budget_comparison.csv')

if __name__ == '__main__':
    main()
