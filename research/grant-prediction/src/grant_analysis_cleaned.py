"""
Social Grant Eligibility and Food Insecurity Analysis - KwaZulu-Natal Province
================================================================================

This script analyzes the relationship between food insecurity and social grant
receipt among households in KwaZulu-Natal using the 2015 Living Conditions Survey.

Author: Oluhle Jawe
Date: 2026
Dataset: 2015 Living Conditions Survey (LCS)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import warnings
import os

folder = os.environ.get('LCS_DATA_DIR', '.')
os.chdir(folder)
print(f"Working directory set to: {os.getcwd()}")
# Configure display and warning settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================

def load_data():
    """Load all required datasets from the 2015 Living Conditions Survey."""
    print("Loading datasets...")
    
    datasets = {
        'household': pd.read_csv('LCS2015HOUSEHOLD.csv'),
        'assets': pd.read_csv('LCS2015HOUSEHOLDASSETS.csv'),
        'person_income': pd.read_csv('LCS2015PERSONINCOME.csv'),
        'persons': pd.read_csv('LCS2015PERSONSFINAL.csv'),
        'total': pd.read_csv('LCS2015TOTALLCS.csv')
    }
    
    print(f"Loaded {len(datasets)} datasets successfully")
    return datasets


def filter_kzn_data(datasets):
    """
    Filter all datasets to KwaZulu-Natal province only (province_code = 5).
    
    Parameters:
    -----------
    datasets : dict
        Dictionary containing all LCS datasets
        
    Returns:
    --------
    dict : KZN-filtered datasets
    """
    print("\nFiltering data for KwaZulu-Natal province (code = 5)...")
    
    # Filter household data
    kzn_hh = datasets['household'][datasets['household']['province_code'] == 5].copy()
    
    # Get list of KZN household IDs
    kzn_uqnos = kzn_hh['UQNO'].unique()
    
    # Filter related datasets
    kzn_data = {
        'household': kzn_hh,
        'persons': datasets['persons'][datasets['persons']['UQNO'].isin(kzn_uqnos)].copy(),
        'total': datasets['total'][datasets['total']['UQNO'].isin(kzn_uqnos)].copy(),
        'assets': datasets['assets'][datasets['assets']['UQNO'].isin(kzn_uqnos)].copy(),
        'person_income': datasets['person_income'][datasets['person_income']['UQNO'].isin(kzn_uqnos)].copy()
    }
    
    print(f"KZN households: {len(kzn_hh):,}")
    print(f"KZN persons: {len(kzn_data['persons']):,}")
    
    return kzn_data


# ==============================================================================
# 2. FOOD INSECURITY INDEX CREATION
# ==============================================================================

def create_food_insecurity_index(kzn_hh):
    """
    Create a food insecurity index based on six indicators:
    1. Q224ANOMONEY - No money to buy food
    2. Q225ASIZE - Reduced meal size
    3. Q226ASKIP - Skipped meals
    4. Q227ALESS - Ate less than needed
    5. adult_hungry - Adults went hungry often/always (Q222ADULT = 3 or 4)
    6. child_hungry - Children went hungry often/always (Q223CHILD = 3 or 4)
    
    Households scoring >= 2 are classified as food insecure.
    
    Parameters:
    -----------
    kzn_hh : DataFrame
        KZN household data
        
    Returns:
    --------
    DataFrame : Household data with food insecurity indicators and classification
    """
    print("\nCreating food insecurity index...")
    
    # Create binary indicators for first four questions (1 = Yes)
    fi_binary_cols = ['Q224ANOMONEY', 'Q225ASIZE', 'Q226ASKIP', 'Q227ALESS']
    for col in fi_binary_cols:
        kzn_hh[col] = (kzn_hh[col] == 1).astype(int)
    
    # Create hunger indicators (Often=3 or Always=4)
    kzn_hh['adult_hungry'] = kzn_hh['Q222ADULT'].apply(
        lambda x: 1 if x in [3, 4] else 0
    )
    kzn_hh['child_hungry'] = kzn_hh['Q223CHILD'].apply(
        lambda x: 1 if x in [3, 4] else 0
    )
    
    # Calculate composite food insecurity score
    fi_indicators = fi_binary_cols + ['adult_hungry', 'child_hungry']
    kzn_hh['fi_score'] = kzn_hh[fi_indicators].sum(axis=1)
    
    # Classify as food insecure if score >= 2
    kzn_hh['food_insecure'] = (kzn_hh['fi_score'] >= 2).astype(int)
    
    # Report statistics
    food_insecure_count = kzn_hh['food_insecure'].sum()
    total_count = len(kzn_hh)
    
    print(f"Food insecure households: {food_insecure_count:,} ({food_insecure_count/total_count*100:.1f}%)")
    print(f"Food secure households: {total_count - food_insecure_count:,} ({(total_count - food_insecure_count)/total_count*100:.1f}%)")
    
    return kzn_hh


# ==============================================================================
# 3. SOCIAL GRANT IDENTIFICATION
# ==============================================================================

def identify_grant_recipients(kzn_hh, kzn_total):
    """
    Identify households receiving social grants based on COICOP expenditure codes.
    
    Grant-related COICOP codes:
    50331000, 50332000, 50332100, 50333100, 50333200, 50333300, 50333400, 50333500
    
    Parameters:
    -----------
    kzn_hh : DataFrame
        KZN household data
    kzn_total : DataFrame
        KZN expenditure data
        
    Returns:
    --------
    DataFrame : Household data with grant receipt indicator
    """
    print("\nIdentifying grant recipients...")
    
    # Define grant-related COICOP codes
    grant_codes = [50331000, 50332000, 50332100, 50333100, 
                   50333200, 50333300, 50333400, 50333500]
    
    # Flag grant-related expenditures
    kzn_total['is_grant'] = kzn_total['Coicop'].isin(grant_codes)
    
    # Aggregate to household level (any grant = receives grants)
    grant_hh = kzn_total.groupby('UQNO')['is_grant'].any().reset_index()
    grant_hh.rename(columns={'is_grant': 'hh_receives_grant'}, inplace=True)
    
    # Merge with household data
    kzn_hh = kzn_hh.merge(grant_hh, on='UQNO', how='left')
    kzn_hh['hh_receives_grant'] = kzn_hh['hh_receives_grant'].fillna(False)
    
    # Report statistics
    grant_count = kzn_hh['hh_receives_grant'].sum()
    total_count = len(kzn_hh)
    
    print(f"Households receiving grants: {grant_count:,} ({grant_count/total_count*100:.1f}%)")
    print(f"Households not receiving grants: {total_count - grant_count:,} ({(total_count - grant_count)/total_count*100:.1f}%)")
    
    return kzn_hh


# ==============================================================================
# 4. DESCRIPTIVE ANALYSIS
# ==============================================================================

def analyze_grant_coverage(kzn_hh):
    """
    Analyze grant coverage among food-insecure households.
    
    Parameters:
    -----------
    kzn_hh : DataFrame
        KZN household data with food insecurity and grant indicators
    """
    print("\n" + "="*80)
    print("GRANT COVERAGE ANALYSIS AMONG FOOD-INSECURE HOUSEHOLDS")
    print("="*80)
    
    # Filter to food-insecure households
    food_insecure_hh = kzn_hh[kzn_hh['food_insecure'] == 1]
    
    # Calculate grant coverage
    with_grants = food_insecure_hh['hh_receives_grant'].sum()
    without_grants = len(food_insecure_hh) - with_grants
    
    print(f"\nTotal food-insecure households: {len(food_insecure_hh):,}")
    print(f"  - WITH grants: {with_grants:,} ({with_grants/len(food_insecure_hh)*100:.1f}%)")
    print(f"  - WITHOUT grants: {without_grants:,} ({without_grants/len(food_insecure_hh)*100:.1f}%)")
    print(f"\nCOVERAGE GAP: {without_grants:,} food-insecure households receive NO grants")
    
    # Create cross-tabulation
    print("\n" + "-"*80)
    print("Cross-tabulation: Food Insecurity x Grant Receipt")
    print("-"*80)
    crosstab = pd.crosstab(
        kzn_hh['food_insecure'], 
        kzn_hh['hh_receives_grant'],
        margins=True
    )
    crosstab.index = ['Food Secure', 'Food Insecure', 'Total']
    crosstab.columns = ['No Grant', 'Has Grant', 'Total']
    print(crosstab)
    
    return food_insecure_hh


# ==============================================================================
# 5. STATISTICAL MODELING
# ==============================================================================

def prepare_modeling_data(kzn_hh, kzn_persons):
    """
    Prepare data for logistic regression modeling.
    
    Parameters:
    -----------
    kzn_hh : DataFrame
        Household data
    kzn_persons : DataFrame
        Person-level data
        
    Returns:
    --------
    tuple : (X, y, weights, feature_names)
    """
    print("\nPreparing data for statistical modeling...")
    
    # Filter to food-insecure households only
    analysis_df = kzn_hh[kzn_hh['food_insecure'] == 1].copy()
    
    # Create target variable (1 = excluded from grants, 0 = receives grants)
    analysis_df['grant_excluded'] = (~analysis_df['hh_receives_grant']).astype(int)
    
    # Calculate household-level aggregates from person data
    hh_size = kzn_persons.groupby('UQNO').size().reset_index(name='hhsize')
    analysis_df = analysis_df.merge(hh_size, on='UQNO', how='left')
    
    # Select and prepare predictor variables
    # Note: Actual variable selection would depend on data exploration
    # This is a simplified example based on the notebook
    
    predictor_cols = ['hhsize']  # Add other relevant predictors
    
    # Handle missing values
    analysis_df = analysis_df.dropna(subset=predictor_cols + ['grant_excluded'])
    
    print(f"Modeling dataset: {len(analysis_df):,} observations")
    print(f"Target variable (grant_excluded) distribution:")
    print(analysis_df['grant_excluded'].value_counts())
    
    return analysis_df


def fit_logistic_regression(analysis_df, predictor_cols):
    """
    Fit a weighted regularized logistic regression model.
    
    Parameters:
    -----------
    analysis_df : DataFrame
        Analysis dataset with predictors and target
    predictor_cols : list
        List of predictor variable names
        
    Returns:
    --------
    statsmodels results object
    """
    print("\nFitting regularized logistic regression model...")
    
    # Prepare features and target
    X = analysis_df[predictor_cols]
    y = analysis_df['grant_excluded']
    
    # Use survey weights if available
    if 'weight' in analysis_df.columns:
        weights = analysis_df['weight']
    else:
        weights = np.ones(len(analysis_df))
    
    # Train-test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to numpy arrays
    X_arr = X_train.astype(float).values
    y_arr = y_train.astype(float).values
    w_arr = w_train.astype(float).values
    
    # Remove constant columns
    constant_cols = [
        i for i in range(X_arr.shape[1]) 
        if np.allclose(X_arr[:, i], X_arr[0, i])
    ]
    if constant_cols:
        X_arr = np.delete(X_arr, constant_cols, axis=1)
        print(f"Removed {len(constant_cols)} constant columns")
    
    # Add intercept
    X_const = sm.add_constant(X_arr)
    
    # Fit regularized logistic regression (L1 penalty)
    try:
        model = sm.Logit(y_arr, X_const).fit_regularized(
            method='l1', 
            alpha=0.01, 
            disp=0
        )
        
        print("Model fitted successfully")
        
        # Extract and display significant predictors
        display_model_results(model, X_train.columns, constant_cols)
        
        return model
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None


def display_model_results(model, original_columns, removed_cols):
    """
    Display model results with odds ratios and significance.
    
    Parameters:
    -----------
    model : statsmodels results
        Fitted model
    original_columns : Index
        Original feature names
    removed_cols : list
        Indices of removed constant columns
    """
    print("\n" + "="*80)
    print("MODEL RESULTS - PREDICTORS OF GRANT EXCLUSION")
    print("="*80)
    
    # Get final variable names
    final_vars = [
        col for i, col in enumerate(original_columns) 
        if i not in removed_cols
    ]
    
    # Extract model parameters
    params = model.params
    pvals = model.pvalues
    conf_int = model.conf_int()
    
    # Build results DataFrame
    results_df = pd.DataFrame({
        'Variable': ['Intercept'] + final_vars,
        'Coefficient': params,
        'P-value': pvals,
        'CI_Lower': conf_int.iloc[:, 0] if hasattr(conf_int, 'iloc') else conf_int[:, 0],
        'CI_Upper': conf_int.iloc[:, 1] if hasattr(conf_int, 'iloc') else conf_int[:, 1]
    })
    
    # Calculate odds ratios
    results_df['Odds_Ratio'] = np.exp(results_df['Coefficient'])
    results_df['OR_Lower'] = np.exp(results_df['CI_Lower'])
    results_df['OR_Upper'] = np.exp(results_df['CI_Upper'])
    
    # Filter to significant predictors (p < 0.05)
    sig_results = results_df[results_df['P-value'] < 0.05].copy()
    sig_results = sig_results.sort_values('P-value')
    
    if len(sig_results) > 0:
        print("\nSignificant Predictors (p < 0.05):")
        print("-" * 80)
        print(sig_results[['Variable', 'Odds_Ratio', 'P-value', 'OR_Lower', 'OR_Upper']].to_string(index=False))
        
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("Odds Ratio > 1: Increases likelihood of grant exclusion")
        print("Odds Ratio < 1: Decreases likelihood of grant exclusion")
        print("95% Confidence Interval: (OR_Lower, OR_Upper)")
    else:
        print("\nNo significant predictors found at p < 0.05 level")
    
    return results_df


# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================

def create_visualizations(kzn_hh):
    """
    Create visualizations of key findings.
    
    Parameters:
    -----------
    kzn_hh : DataFrame
        Household data with all indicators
    """
    print("\nCreating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Food insecurity distribution
    fi_counts = kzn_hh['food_insecure'].value_counts()
    axes[0, 0].bar(['Food Secure', 'Food Insecure'], fi_counts.values, color=['#2E86AB', '#A23B72'])
    axes[0, 0].set_title('Food Insecurity Status - KZN Households', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Households', fontsize=12)
    for i, v in enumerate(fi_counts.values):
        axes[0, 0].text(i, v + 50, f'{v:,}\n({v/len(kzn_hh)*100:.1f}%)', 
                       ha='center', fontsize=10)
    
    # 2. Grant coverage overall
    grant_counts = kzn_hh['hh_receives_grant'].value_counts()
    axes[0, 1].bar(['No Grant', 'Receives Grant'], grant_counts.values, color=['#F18F01', '#06A77D'])
    axes[0, 1].set_title('Social Grant Coverage - All KZN Households', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Number of Households', fontsize=12)
    for i, v in enumerate(grant_counts.values):
        axes[0, 1].text(i, v + 50, f'{v:,}\n({v/len(kzn_hh)*100:.1f}%)', 
                       ha='center', fontsize=10)
    
    # 3. Grant coverage among food-insecure households
    fi_hh = kzn_hh[kzn_hh['food_insecure'] == 1]
    fi_grant_counts = fi_hh['hh_receives_grant'].value_counts()
    axes[1, 0].bar(['Excluded from Grants', 'Receives Grants'], fi_grant_counts.values, 
                   color=['#C73E1D', '#06A77D'])
    axes[1, 0].set_title('Grant Coverage - Food Insecure Households Only', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Households', fontsize=12)
    for i, v in enumerate(fi_grant_counts.values):
        axes[1, 0].text(i, v + 20, f'{v:,}\n({v/len(fi_hh)*100:.1f}%)', 
                       ha='center', fontsize=10)
    
    # 4. Food insecurity score distribution
    score_counts = kzn_hh['fi_score'].value_counts().sort_index()
    axes[1, 1].bar(score_counts.index, score_counts.values, color='#5D5D5D')
    axes[1, 1].axvline(x=1.5, color='red', linestyle='--', linewidth=2, 
                      label='Food Insecurity Threshold (≥2)')
    axes[1, 1].set_title('Distribution of Food Insecurity Scores', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Food Insecurity Score', fontsize=12)
    axes[1, 1].set_ylabel('Number of Households', fontsize=12)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('grant_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'grant_analysis_visualizations.png'")
    
    return fig


# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("SOCIAL GRANT ELIGIBILITY AND FOOD INSECURITY ANALYSIS")
    print("KwaZulu-Natal Province - 2015 Living Conditions Survey")
    print("="*80)
    
    # 1. Load data
    datasets = load_data()
    
    # 2. Filter to KZN
    kzn_data = filter_kzn_data(datasets)
    
    # 3. Create food insecurity index
    kzn_hh = create_food_insecurity_index(kzn_data['household'])
    
    # 4. Identify grant recipients
    kzn_hh = identify_grant_recipients(kzn_hh, kzn_data['total'])
    
    # 5. Analyze grant coverage
    food_insecure_hh = analyze_grant_coverage(kzn_hh)
    
    # 6. Create visualizations
    create_visualizations(kzn_hh)
    
    # 7. Statistical modeling (simplified - full model requires more variables)
    # analysis_df = prepare_modeling_data(kzn_hh, kzn_data['persons'])
    # model = fit_logistic_regression(analysis_df, predictor_cols)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Outputs:")
    print("  1. Detailed console output with statistics")
    print("  2. Visualizations: grant_analysis_visualizations.png")
    print("  3. Analysis report: See separate Word document")
    
    return kzn_hh


if __name__ == "__main__":
    # Execute analysis
    kzn_hh = main()
    
    # Optional: Save processed data
    # kzn_hh.to_csv('kzn_household_analysis.csv', index=False)
    # print("\nProcessed data saved to 'kzn_household_analysis.csv'")
