"""
Insurance Claim Fraud Detection - Dataset Generator

This script generates a realistic synthetic dataset of 5,000 insurance claims
with features commonly used in fraud detection. The data includes:
- Customer demographics (age, location)
- Policy information (years held, type)
- Claim details (amount, type, description)
- Fraud label (fraudulent or legitimate)

Author: Senior ML Engineer
Date: June 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
# This ensures the same "random" data is generated every time we run this script
# This is crucial for testing, debugging, and collaboration with team members
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_RECORDS = 5000  # Total number of insurance claims to generate
FRAUD_RATE = 0.15   # Expected fraud rate (15% of claims are fraudulent)

# ============================================================================
# STEP 1: Define Realistic Claim Data
# ============================================================================

print("=" * 70)
print("GEICO Fraud Detection - Dataset Generator")
print("=" * 70)
print(f"\nGenerating {NUM_RECORDS:,} synthetic insurance claim records...")

# Create a base dataset with realistic features
# Think of this as creating empty columns that we'll fill with data

df = pd.DataFrame()

# AGE: Customer age (insurance claim filers are typically 18-80 years old)
# We use normal distribution (bell curve) because most people are middle-aged
df['age'] = np.random.normal(loc=45, scale=15, size=NUM_RECORDS).astype(int)
df['age'] = df['age'].clip(18, 80)  # Ensure age is between 18-80

# LOCATION: Geographic region (important because fraud patterns vary by location)
# Categorical: different regions may have different fraud rates
regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
df['region'] = np.random.choice(regions, size=NUM_RECORDS, p=[0.2, 0.25, 0.25, 0.15, 0.15])

# POLICY_YEARS: How long the customer has held the policy
# New customers are riskier (more fraud) than long-term customers
df['policy_years'] = np.random.exponential(scale=5, size=NUM_RECORDS).astype(int)
df['policy_years'] = df['policy_years'].clip(0, 30)  # Cap at 30 years

# CLAIM_AMOUNT: Dollar amount of the claim (ranging from $500 to $150,000)
# This is often skewed (log-normal) because most claims are small but some are huge
df['claim_amount'] = np.random.lognormal(mean=10, sigma=1.5, size=NUM_RECORDS).astype(int)
df['claim_amount'] = df['claim_amount'].clip(500, 150000)

# PREMIUM_AMOUNT: Annual insurance premium paid by customer
# Higher premiums often indicate higher coverage limits
df['premium_annual'] = np.random.lognormal(mean=6.5, sigma=0.8, size=NUM_RECORDS).astype(int)
df['premium_annual'] = df['premium_annual'].clip(300, 5000)

# CLAIM_TYPE: Category of the claim
claim_types = ['auto', 'home', 'health', 'property', 'other']
df['claim_type'] = np.random.choice(claim_types, size=NUM_RECORDS, p=[0.4, 0.3, 0.15, 0.1, 0.05])

# NUM_PREVIOUS_CLAIMS: How many claims the customer has filed before
# Customers with many previous claims might be higher risk
df['num_previous_claims'] = np.random.poisson(lam=2, size=NUM_RECORDS)

# DAYS_SINCE_POLICY_START: How many days since policy was activated
df['days_since_policy_start'] = (df['policy_years'] * 365).astype(int)

# ============================================================================
# STEP 2: Engineer Fraud Features & Labels
# ============================================================================

print("\nEngineering fraud indicators...")

# FRAUD_FLAG: This is what we're trying to predict!
# We create realistic fraud patterns based on domain knowledge:
fraud_indicators = np.zeros(NUM_RECORDS)

# FRAUD PATTERN 1: High claim-to-premium ratio
# Fraudsters often claim amounts way higher than their premiums warrant
claim_premium_ratio = df['claim_amount'] / df['premium_annual']
fraud_indicators += (claim_premium_ratio > 20).astype(int)  # Very suspicious if claim is 20x premium

# FRAUD PATTERN 2: New customers with high claims
# Scammers sometimes open policies and immediately file large claims
new_customer = df['policy_years'] < 1
high_claim = df['claim_amount'] > 50000
fraud_indicators += (new_customer & high_claim).astype(int)

# FRAUD PATTERN 3: Multiple recent claims (claim frequency)
# Fraudsters often file multiple claims quickly
high_previous_claims = df['num_previous_claims'] > 5
high_claim_amount = df['claim_amount'] > 30000
fraud_indicators += (high_previous_claims & high_claim_amount).astype(int)

# FRAUD PATTERN 4: Certain regions have higher fraud rates
# This is realistic - fraud varies geographically
high_fraud_regions = df['region'].isin(['Southwest', 'West'])
fraud_indicators += high_fraud_regions.astype(int) * 0.3

# FRAUD PATTERN 5: Specific claim types are higher risk
# Auto claims have highest fraud rate
auto_claims = df['claim_type'] == 'auto'
fraud_indicators += auto_claims.astype(int) * 0.2

# Convert fraud indicators to binary fraud labels
# If fraud score > threshold, mark as fraud; otherwise legitimate
# We normalize to target approximately 15% fraud rate
fraud_threshold = np.percentile(fraud_indicators, 100 * (1 - FRAUD_RATE))
df['is_fraud'] = (fraud_indicators >= fraud_threshold).astype(int)

print(f"Fraud rate in dataset: {df['is_fraud'].mean()*100:.2f}%")

# ============================================================================
# STEP 3: Add Temporal Features
# ============================================================================

print("Adding temporal features...")

# CLAIM_DATE: When the claim was filed (past 2 years)
base_date = datetime.now() - timedelta(days=730)
df['claim_date'] = [base_date + timedelta(days=int(x)) for x in np.random.uniform(0, 730, NUM_RECORDS)]
df['claim_date'] = pd.to_datetime(df['claim_date']).dt.date

# MONTH: Extract month from claim date (fraud patterns may vary seasonally)
df['claim_month'] = pd.to_datetime(df['claim_date']).dt.month

# ============================================================================
# STEP 4: Reorder Columns & Add Metadata
# ============================================================================

# Reorder columns for logical flow
columns_order = [
    'age', 'region', 'policy_years', 'claim_type', 'claim_amount',
    'premium_annual', 'num_previous_claims', 'claim_date', 'claim_month',
    'days_since_policy_start', 'is_fraud'
]
df = df[columns_order]

# Add a unique claim ID
df.insert(0, 'claim_id', range(1, NUM_RECORDS + 1))

# ============================================================================
# STEP 5: Save Dataset
# ============================================================================

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(output_dir, 'insurance_claims.csv')
df.to_csv(output_path, index=False)

print(f"\n✓ Dataset saved to: {output_path}")

# ============================================================================
# STEP 6: Print Summary Statistics (Data Exploration Preview)
# ============================================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\nFirst 5 records:")
print(df.head())

print("\n\nData Types:")
print(df.dtypes)

print("\n\nFraud Distribution:")
fraud_counts = df['is_fraud'].value_counts()
print(f"  Legitimate claims: {fraud_counts.get(0, 0):,} ({fraud_counts.get(0, 0)/len(df)*100:.2f}%)")
print(f"  Fraudulent claims: {fraud_counts.get(1, 0):,} ({fraud_counts.get(1, 0)/len(df)*100:.2f}%)")

print("\n\nClaim Amount Statistics (in $):")
print(df['claim_amount'].describe())

print("\n\nPremium Amount Statistics (in $):")
print(df['premium_annual'].describe())

print("\n" + "=" * 70)
print("✓ Dataset generation complete! Ready for Phase 2: EDA")
print("=" * 70)
