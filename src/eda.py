"""
Insurance Claim Fraud Detection - Exploratory Data Analysis (EDA)

This script performs comprehensive exploratory data analysis on the insurance claims dataset.
It includes:
- Dataset shape and structure analysis
- Data type validation
- Missing value detection
- Statistical summaries
- Fraud pattern analysis
- Professional visualizations
- Business insights and recommendations

Author: Senior Data Scientist
Date: June 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Ensure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# PART 1: DATA LOADING & VALIDATION
# ============================================================================

def load_data(filepath):
    """
    Load the insurance claims dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        Exception: If data loading fails
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"[OK] Dataset loaded successfully: {filepath}")
        print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
        return df
    
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        sys.exit(1)


# ============================================================================
# PART 2: DATASET STRUCTURE ANALYSIS
# ============================================================================

def analyze_dataset_shape(df):
    """
    Analyze and print the dataset dimensions and structure.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("=" * 80)
    print("1. DATASET SHAPE ANALYSIS")
    print("=" * 80)
    
    print(f"\nDataset Dimensions:")
    print(f"  * Total Rows: {df.shape[0]:,} (insurance claims)")
    print(f"  * Total Columns: {df.shape[1]} (features)")
    print(f"  * Total Cells: {df.shape[0] * df.shape[1]:,}")
    
    print(f"\nColumn Names ({df.shape[1]} features):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:<30s} (Type: {df[col].dtype})")


def analyze_data_types(df):
    """
    Analyze and print data type distribution.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("2. DATA TYPE ANALYSIS")
    print("=" * 80)
    
    print("\nData Type Distribution:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  * {dtype}: {count} columns")
    
    print("\nDetailed Column Types:")
    print(df.dtypes)


# ============================================================================
# PART 3: DATA QUALITY ANALYSIS
# ============================================================================

def analyze_missing_values(df):
    """
    Analyze and print missing values in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("3. MISSING VALUE ANALYSIS")
    print("=" * 80)
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].reset_index(drop=True)
    
    if len(missing_data) == 0:
        print("\n[OK] No missing values detected in the dataset!")
    else:
        print("\nMissing Values Summary:")
        print(missing_data.to_string(index=False))


def analyze_duplicate_records(df):
    """
    Analyze and print duplicate records in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("4. DUPLICATE RECORD ANALYSIS")
    print("=" * 80)
    
    total_duplicates = df.duplicated().sum()
    print(f"\nTotal Duplicate Rows: {total_duplicates}")
    
    if total_duplicates == 0:
        print("[OK] No duplicate records detected!")
    else:
        print(f"[WARN] {total_duplicates} duplicate records found (consider removing them)")


# ============================================================================
# PART 4: STATISTICAL ANALYSIS
# ============================================================================

def analyze_statistical_summary(df):
    """
    Generate and print comprehensive statistical summary.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("5. STATISTICAL SUMMARY (NUMERICAL FEATURES)")
    print("=" * 80)
    
    print("\nDescriptive Statistics:")
    print(df.describe().to_string())
    
    print("\n\nNumerical Columns Summary:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  * Mean: {df[col].mean():.2f}")
        print(f"  * Median: {df[col].median():.2f}")
        print(f"  * Std Dev: {df[col].std():.2f}")
        print(f"  * Min: {df[col].min():.2f}")
        print(f"  * Max: {df[col].max():.2f}")
        print(f"  * Q1 (25%): {df[col].quantile(0.25):.2f}")
        print(f"  * Q3 (75%): {df[col].quantile(0.75):.2f}")


def analyze_categorical_features(df):
    """
    Analyze and print categorical features distribution.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("6. CATEGORICAL FEATURES ANALYSIS")
    print("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  * {value}: {count:,} ({percentage:.2f}%)")


# ============================================================================
# PART 5: FRAUD ANALYSIS
# ============================================================================

def analyze_fraud_distribution(df):
    """
    Analyze and print fraud vs legitimate claim distribution.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("7. FRAUD DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    fraud_counts = df['is_fraud'].value_counts()
    fraud_percentage = (df['is_fraud'].value_counts() / len(df)) * 100
    
    print(f"\nFraud vs Legitimate Claims:")
    print(f"  * Legitimate Claims (0): {fraud_counts.get(0, 0):,} ({fraud_percentage.get(0, 0):.2f}%)")
    print(f"  * Fraudulent Claims (1): {fraud_counts.get(1, 0):,} ({fraud_percentage.get(1, 0):.2f}%)")
    print(f"  * Total Claims: {len(df):,}")
    print(f"\n  Overall Fraud Rate: {(df['is_fraud'].sum() / len(df)) * 100:.2f}%")


def analyze_claim_amount_by_fraud(df):
    """
    Analyze claim amounts for fraudulent vs legitimate claims.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("8. CLAIM AMOUNT ANALYSIS BY FRAUD STATUS")
    print("=" * 80)
    
    print("\nLegitimate Claims - Claim Amount Statistics:")
    legitimate = df[df['is_fraud'] == 0]['claim_amount']
    print(f"  * Mean: ${legitimate.mean():,.2f}")
    print(f"  * Median: ${legitimate.median():,.2f}")
    print(f"  * Min: ${legitimate.min():,.2f}")
    print(f"  * Max: ${legitimate.max():,.2f}")
    
    print("\nFraudulent Claims - Claim Amount Statistics:")
    fraudulent = df[df['is_fraud'] == 1]['claim_amount']
    print(f"  * Mean: ${fraudulent.mean():,.2f}")
    print(f"  * Median: ${fraudulent.median():,.2f}")
    print(f"  * Min: ${fraudulent.min():,.2f}")
    print(f"  * Max: ${fraudulent.max():,.2f}")
    
    print(f"\n  Difference in Mean: ${fraudulent.mean() - legitimate.mean():,.2f}")
    print(f"  Fraudulent claims are {(fraudulent.mean() / legitimate.mean()):.2f}x higher on average")


def analyze_fraud_by_region(df):
    """
    Analyze fraud rate by geographic region.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("9. FRAUD RATE BY REGION")
    print("=" * 80)
    
    fraud_by_region = df.groupby('region')['is_fraud'].agg(['sum', 'count', 'mean'])
    fraud_by_region.columns = ['Fraudulent_Cases', 'Total_Cases', 'Fraud_Rate']
    fraud_by_region['Fraud_Rate'] = fraud_by_region['Fraud_Rate'] * 100
    fraud_by_region = fraud_by_region.sort_values('Fraud_Rate', ascending=False)
    
    print("\nFraud Statistics by Region:")
    for region in fraud_by_region.index:
        fraudulent = fraud_by_region.loc[region, 'Fraudulent_Cases']
        total = fraud_by_region.loc[region, 'Total_Cases']
        rate = fraud_by_region.loc[region, 'Fraud_Rate']
        print(f"  * {region}: {int(fraudulent)}/{int(total)} fraudulent ({rate:.2f}%)")


def analyze_fraud_by_claim_type(df):
    """
    Analyze fraud rate by claim type.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("10. FRAUD RATE BY CLAIM TYPE")
    print("=" * 80)
    
    fraud_by_type = df.groupby('claim_type')['is_fraud'].agg(['sum', 'count', 'mean'])
    fraud_by_type.columns = ['Fraudulent_Cases', 'Total_Cases', 'Fraud_Rate']
    fraud_by_type['Fraud_Rate'] = fraud_by_type['Fraud_Rate'] * 100
    fraud_by_type = fraud_by_type.sort_values('Fraud_Rate', ascending=False)
    
    print("\nFraud Statistics by Claim Type:")
    for claim_type in fraud_by_type.index:
        fraudulent = fraud_by_type.loc[claim_type, 'Fraudulent_Cases']
        total = fraud_by_type.loc[claim_type, 'Total_Cases']
        rate = fraud_by_type.loc[claim_type, 'Fraud_Rate']
        print(f"  * {claim_type}: {int(fraudulent)}/{int(total)} fraudulent ({rate:.2f}%)")


# ============================================================================
# PART 6: VISUALIZATIONS
# ============================================================================

def create_fraud_count_plot(df):
    """
    Create and save a count plot showing fraud vs legitimate claims.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("\n-> Creating Fraud Distribution visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fraud_labels = {0: 'Legitimate', 1: 'Fraudulent'}
    fraud_counts = df['is_fraud'].value_counts().sort_index()
    
    bars = ax.bar(
        [fraud_labels[i] for i in fraud_counts.index],
        fraud_counts.values,
        color=['#2ecc71', '#e74c3c'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold'
        )
    
    ax.set_xlabel('Claim Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Claims', fontsize=12, fontweight='bold')
    ax.set_title('Fraud vs Legitimate Claims Distribution\n(Insurance Dataset Overview)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '01_fraud_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()


def create_claim_amount_distribution(df):
    """
    Create and save claim amount distribution visualizations.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("-> Creating Claim Amount Distribution visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall distribution
    axes[0].hist(df['claim_amount'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Claim Amount ($)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Overall Claim Amount Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Distribution by fraud status
    legitimate = df[df['is_fraud'] == 0]['claim_amount']
    fraudulent = df[df['is_fraud'] == 1]['claim_amount']
    
    axes[1].hist(legitimate, bins=40, label='Legitimate', color='#2ecc71', alpha=0.6, edgecolor='black')
    axes[1].hist(fraudulent, bins=40, label='Fraudulent', color='#e74c3c', alpha=0.6, edgecolor='black')
    axes[1].set_xlabel('Claim Amount ($)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Claim Amount Distribution by Fraud Status', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '02_claim_amount_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()


def create_premium_distribution(df):
    """
    Create and save premium amount distribution visualizations.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("-> Creating Premium Amount Distribution visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall distribution
    axes[0].hist(df['premium_annual'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Annual Premium ($)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Overall Annual Premium Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Distribution by fraud status
    legitimate = df[df['is_fraud'] == 0]['premium_annual']
    fraudulent = df[df['is_fraud'] == 1]['premium_annual']
    
    axes[1].hist(legitimate, bins=40, label='Legitimate', color='#2ecc71', alpha=0.6, edgecolor='black')
    axes[1].hist(fraudulent, bins=40, label='Fraudulent', color='#e74c3c', alpha=0.6, edgecolor='black')
    axes[1].set_xlabel('Annual Premium ($)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Premium Distribution by Fraud Status', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '03_premium_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()


def create_correlation_heatmap(df):
    """
    Create and save correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("-> Creating Correlation Heatmap visualization...")
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    correlation_matrix = numerical_df.corr()
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix - Numerical Features\n(Positive = moves together, Negative = moves opposite)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '04_correlation_heatmap.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()


def create_fraud_by_region_plot(df):
    """
    Create and save fraud rate by region visualization.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("-> Creating Fraud Rate by Region visualization...")
    
    fraud_by_region = df.groupby('region')['is_fraud'].agg(['sum', 'count', 'mean'])
    fraud_by_region['fraud_rate'] = fraud_by_region['mean'] * 100
    fraud_by_region = fraud_by_region.sort_values('fraud_rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(
        fraud_by_region.index,
        fraud_by_region['fraud_rate'],
        color=['#e74c3c' if x > 20 else '#f39c12' if x > 15 else '#2ecc71' for x in fraud_by_region['fraud_rate']],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels
    for i, (idx, row) in enumerate(fraud_by_region.iterrows()):
        ax.text(
            row['fraud_rate'], i,
            f" {row['fraud_rate']:.2f}% ({int(row['sum'])}/{int(row['count'])})",
            va='center', fontweight='bold'
        )
    
    ax.set_xlabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Region', fontsize=12, fontweight='bold')
    ax.set_title('Fraud Rate by Geographic Region\n(Red = High Risk, Yellow = Medium, Green = Low)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '05_fraud_by_region.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()


def create_fraud_by_claim_type_plot(df):
    """
    Create and save fraud rate by claim type visualization.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("-> Creating Fraud Rate by Claim Type visualization...")
    
    fraud_by_type = df.groupby('claim_type')['is_fraud'].agg(['sum', 'count', 'mean'])
    fraud_by_type['fraud_rate'] = fraud_by_type['mean'] * 100
    fraud_by_type = fraud_by_type.sort_values('fraud_rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(
        fraud_by_type.index,
        fraud_by_type['fraud_rate'],
        color=['#e74c3c' if x > 20 else '#f39c12' if x > 15 else '#2ecc71' for x in fraud_by_type['fraud_rate']],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontweight='bold'
        )
    
    ax.set_xlabel('Claim Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Fraud Rate by Claim Type\n(Red = High Risk, Yellow = Medium, Green = Low)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '06_fraud_by_claim_type.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()


def create_claim_amount_range_analysis(df):
    """
    Create and save fraud rate by claim amount range visualization.
    
    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    print("-> Creating Fraud Rate by Claim Amount Range visualization...")
    
    # Create claim amount bins
    bins = [0, 10000, 25000, 50000, 100000, 150000]
    labels = ['<$10K', '$10K-$25K', '$25K-$50K', '$50K-$100K', '>$100K']
    df['claim_range'] = pd.cut(df['claim_amount'], bins=bins, labels=labels)
    
    fraud_by_range = df.groupby('claim_range', observed=True)['is_fraud'].agg(['sum', 'count', 'mean'])
    fraud_by_range['fraud_rate'] = fraud_by_range['mean'] * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(
        range(len(fraud_by_range)),
        fraud_by_range['fraud_rate'],
        color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels
    for i, (idx, row) in enumerate(fraud_by_range.iterrows()):
        ax.text(
            i, row['fraud_rate'],
            f"{row['fraud_rate']:.2f}%\n({int(row['sum'])}/{int(row['count'])})",
            ha='center', va='bottom', fontweight='bold', fontsize=9
        )
    
    ax.set_xticks(range(len(fraud_by_range)))
    ax.set_xticklabels(fraud_by_range.index)
    ax.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Claim Amount Range', fontsize=12, fontweight='bold')
    ax.set_title('Fraud Rate by Claim Amount Range\n(Higher claims = Higher fraud risk)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = REPORTS_DIR / '07_fraud_by_claim_range.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to: {filepath}")
    plt.close()
    
    # Clean up the temporary column
    df.drop('claim_range', axis=1, inplace=True)


# ============================================================================
# PART 7: BUSINESS INSIGHTS
# ============================================================================

def print_business_insights(df):
    """
    Print key business insights and recommendations.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "=" * 80)
    print("BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n[KEY FINDINGS]\n")
    
    # Insight 1: Fraud Rate
    fraud_rate = (df['is_fraud'].sum() / len(df)) * 100
    print(f"1. FRAUD PREVALENCE")
    print(f"   * Overall fraud rate: {fraud_rate:.2f}%")
    print(f"   * This means approximately 1 in {int(100/fraud_rate)} claims is fraudulent")
    print(f"   * Action: Implement automated fraud detection to flag suspicious claims\n")
    
    # Insight 2: Claim Amount
    legitimate_avg = df[df['is_fraud'] == 0]['claim_amount'].mean()
    fraudulent_avg = df[df['is_fraud'] == 1]['claim_amount'].mean()
    ratio = fraudulent_avg / legitimate_avg
    print(f"2. CLAIM AMOUNT & FRAUD")
    print(f"   * Average legitimate claim: ${legitimate_avg:,.2f}")
    print(f"   * Average fraudulent claim: ${fraudulent_avg:,.2f}")
    print(f"   * Fraudulent claims are {ratio:.2f}x LARGER on average")
    print(f"   * Action: Set alert threshold for claims > ${fraudulent_avg:,.2f}\n")
    
    # Insight 3: Region Analysis
    fraud_by_region = df.groupby('region')['is_fraud'].mean() * 100
    highest_fraud_region = fraud_by_region.idxmax()
    highest_fraud_rate = fraud_by_region.max()
    print(f"3. GEOGRAPHIC RISK FACTORS")
    print(f"   * Highest fraud region: {highest_fraud_region} ({highest_fraud_rate:.2f}%)")
    print(f"   * Action: Deploy additional investigative resources in {highest_fraud_region}\n")
    
    # Insight 4: Claim Type Analysis
    fraud_by_type = df.groupby('claim_type')['is_fraud'].mean() * 100
    highest_fraud_type = fraud_by_type.idxmax()
    highest_fraud_type_rate = fraud_by_type.max()
    print(f"4. CLAIM TYPE RISK")
    print(f"   * Highest fraud claim type: {highest_fraud_type} ({highest_fraud_type_rate:.2f}%)")
    print(f"   * Action: Tighten approval process for {highest_fraud_type} claims\n")
    
    # Insight 5: Policy Age
    new_customer_fraud = df[df['policy_years'] < 1]['is_fraud'].mean() * 100
    established_fraud = df[df['policy_years'] >= 5]['is_fraud'].mean() * 100
    print(f"5. CUSTOMER TENURE & FRAUD")
    print(f"   * Fraud rate for new customers (<1 year): {new_customer_fraud:.2f}%")
    print(f"   * Fraud rate for established customers (5+ years): {established_fraud:.2f}%")
    print(f"   * Difference: {new_customer_fraud - established_fraud:.2f} percentage points")
    print(f"   * Action: Implement stricter verification for new customers\n")
    
    # Insight 6: Policy History
    high_previous_claims_fraud = df[df['num_previous_claims'] > 5]['is_fraud'].mean() * 100
    low_previous_claims_fraud = df[df['num_previous_claims'] <= 2]['is_fraud'].mean() * 100
    print(f"6. CLAIM HISTORY PATTERN")
    print(f"   * Fraud rate for customers with >5 previous claims: {high_previous_claims_fraud:.2f}%")
    print(f"   * Fraud rate for customers with <=2 previous claims: {low_previous_claims_fraud:.2f}%")
    print(f"   * Action: Flag repeat claimers with high claim amounts\n")
    
    # Insight 7: Premium vs Claim Ratio
    df['claim_to_premium_ratio'] = df['claim_amount'] / df['premium_annual']
    high_ratio_fraud = df[df['claim_to_premium_ratio'] > 20]['is_fraud'].mean() * 100
    low_ratio_fraud = df[df['claim_to_premium_ratio'] <= 10]['is_fraud'].mean() * 100
    print(f"7. CLAIM-TO-PREMIUM RATIO INDICATOR")
    print(f"   * Fraud rate when claim > 20x premium: {high_ratio_fraud:.2f}%")
    print(f"   * Fraud rate when claim <= 10x premium: {low_ratio_fraud:.2f}%")
    print(f"   * Action: Create alert rule for claims >15x premium amount\n")
    
    print("=" * 80)
    print("[OK] EDA ANALYSIS COMPLETE")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to orchestrate the entire EDA analysis.
    """
    print("\n")
    print("=" * 80)
    print("=" + " " * 78 + "=")
    print("=" + "  INSURANCE FRAUD DETECTION - EXPLORATORY DATA ANALYSIS (EDA)".center(78) + "=")
    print("=" + " " * 78 + "=")
    print("=" * 80)
    
    # Load data
    data_path = DATA_DIR / 'insurance_claims.csv'
    df = load_data(str(data_path))
    
    # Run all analyses
    analyze_dataset_shape(df)
    analyze_data_types(df)
    analyze_missing_values(df)
    analyze_duplicate_records(df)
    analyze_statistical_summary(df)
    analyze_categorical_features(df)
    analyze_fraud_distribution(df)
    analyze_claim_amount_by_fraud(df)
    analyze_fraud_by_region(df)
    analyze_fraud_by_claim_type(df)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print(f"\nSaving charts to: {REPORTS_DIR}\n")
    
    create_fraud_count_plot(df)
    create_claim_amount_distribution(df)
    create_premium_distribution(df)
    create_correlation_heatmap(df)
    create_fraud_by_region_plot(df)
    create_fraud_by_claim_type_plot(df)
    create_claim_amount_range_analysis(df)
    
    # Print business insights
    print_business_insights(df)
    
    print("\n[OK] All visualizations saved successfully!")
    print(f"[OK] Reports location: {REPORTS_DIR}")


if __name__ == '__main__':
    main()
