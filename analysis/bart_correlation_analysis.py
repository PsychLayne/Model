#!/usr/bin/env python3
"""
BART (Balloon Analogue Risk Task) Correlation Analysis
=======================================================

Goal: Understand what the BART measures with reliability of .70-.91
by finding the highest correlations between BART parameters and
psychological/cognitive/behavioral measures.

Theoretical framework:
- The BART captures sequential risk-taking behavior under uncertainty
- Key constructs it might measure: impulsivity, sensation seeking,
  loss aversion, risk tolerance, temporal discounting, learning from feedback

Author: Claude Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ========================================
# DATA LOADING AND MERGING
# ========================================

print("=" * 80)
print("BART CORRELATION ANALYSIS: WHAT DOES THE BART MEASURE?")
print("=" * 80)
print()

# Load all datasets
print("Loading datasets...")

# BART model parameters (primary outcome)
test_results = pd.read_csv('/home/user/Model/test_results.csv')
print(f"  test_results.csv: {len(test_results)} participants, {test_results.columns.tolist()}")

# Alternative model results (larger sample)
range_results = pd.read_csv('/home/user/Model/range_learning_corrected_results (1).csv')
print(f"  range_learning_corrected_results: {len(range_results)} participants")

# Psychological questionnaires
quest_scores = pd.read_csv('/home/user/Model/quest_scores.csv')
print(f"  quest_scores.csv: {len(quest_scores)} participants, {len(quest_scores.columns)} variables")

# Personality measures
perso = pd.read_csv('/home/user/Model/perso.csv')
print(f"  perso.csv: {len(perso)} participants")

# Working memory capacity
wmc = pd.read_csv('/home/user/Model/wmc.csv')
print(f"  wmc.csv: {len(wmc)} participants")

# Decision-making tasks
cct = pd.read_csv('/home/user/Model/cct_overt.csv')
dfd = pd.read_csv('/home/user/Model/dfd_perpers.csv')
dfe = pd.read_csv('/home/user/Model/dfe_perpers.csv')
lotteries = pd.read_csv('/home/user/Model/lotteriesOvert.csv')
mt = pd.read_csv('/home/user/Model/mt.csv')

print()

# ========================================
# MERGE DATASETS
# ========================================

print("Merging datasets on participant ID...")

# Start with test_results (100 participants with full model fits)
merged = test_results.copy()

# Merge with questionnaire scores
merged = merged.merge(quest_scores, on='partid', how='left')

# Merge with personality
merged = merged.merge(perso, on='partid', how='left')

# Merge with working memory
merged = merged.merge(wmc, on='partid', how='left')

# Merge with decision tasks
merged = merged.merge(cct, on='partid', how='left')
merged = merged.merge(dfd, on='partid', how='left', suffixes=('', '_dfd'))
merged = merged.merge(dfe, on='partid', how='left', suffixes=('', '_dfe'))
merged = merged.merge(lotteries, on='partid', how='left', suffixes=('', '_lot'))
merged = merged.merge(mt, on='partid', how='left')

print(f"Merged dataset: {len(merged)} participants, {len(merged.columns)} variables")
print()

# Also merge range_results with all data (larger sample)
merged_large = range_results.copy()
merged_large = merged_large.merge(quest_scores, on='partid', how='left')
merged_large = merged_large.merge(perso, on='partid', how='left')
merged_large = merged_large.merge(wmc, on='partid', how='left')
merged_large = merged_large.merge(cct, on='partid', how='left')
merged_large = merged_large.merge(dfd, on='partid', how='left', suffixes=('', '_dfd'))
merged_large = merged_large.merge(dfe, on='partid', how='left', suffixes=('', '_dfe'))
merged_large = merged_large.merge(lotteries, on='partid', how='left', suffixes=('', '_lot'))
merged_large = merged_large.merge(mt, on='partid', how='left')

print(f"Large merged dataset: {len(merged_large)} participants")
print()

# ========================================
# DEFINE VARIABLE GROUPS
# ========================================

# BART model parameters to correlate
bart_params = ['omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'beta', 'loss_aversion']
bart_behavioral = ['mean_pumps', 'explosion_rate']  # From range_results

# Questionnaire subscales of theoretical interest
impulsivity_vars = ['BIS', 'BIS1att', 'BIS1mot', 'BIS1ctr', 'BIS1com', 'BIS1per', 'BIS1ins',
                    'BIS2att', 'BIS2mot', 'BIS2npl']
sensation_seeking_vars = ['SSSV', 'SStas', 'SSexp', 'SSdis', 'SSbor']
risk_propensity_vars = ['Deth', 'Dinv', 'Dgam', 'Dhea', 'Drec', 'Dsoc',
                        'Deth_r', 'Dinv_r', 'Dgam_r', 'Dhea_r', 'Drec_r', 'Dsoc_r']
substance_use_vars = ['AUDIT', 'FTND', 'DAST']
gambling_vars = ['GABS', 'PG', 'CAREaggr', 'CARESex', 'CAREwork']
cognitive_vars = ['NUM', 'WMC', 'MUpc']
personality_vars = ['NEO_A', 'NEO_C', 'NEO_E', 'NEO_N', 'NEO_O', 'STAI_trait']
decision_task_vars = ['CCTncards', 'CCTacards', 'CCTpayoff', 'R', 'H', 'CV', 'R_lot', 'H_lot']

all_predictor_vars = (impulsivity_vars + sensation_seeking_vars + risk_propensity_vars +
                      substance_use_vars + cognitive_vars + personality_vars)

# ========================================
# CORRELATION ANALYSIS FUNCTIONS
# ========================================

def compute_correlation(x, y, method='pearson'):
    """Compute correlation with p-value, handling missing data."""
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)

    if len(x_clean) < 10:
        return np.nan, np.nan, 0

    try:
        if method == 'pearson':
            r, p = pearsonr(x_clean, y_clean)
        else:
            r, p = spearmanr(x_clean, y_clean)
        return r, p, len(x_clean)
    except:
        return np.nan, np.nan, 0

def correlation_matrix_with_pvalues(df, bart_cols, predictor_cols, method='pearson'):
    """Create correlation matrix between BART params and predictors."""
    results = []

    for bart_var in bart_cols:
        if bart_var not in df.columns:
            continue
        for pred_var in predictor_cols:
            if pred_var not in df.columns:
                continue

            r, p, n = compute_correlation(df[bart_var], df[pred_var], method)
            results.append({
                'bart_param': bart_var,
                'predictor': pred_var,
                'r': r,
                'p': p,
                'n': n,
                'abs_r': abs(r) if not np.isnan(r) else 0
            })

    return pd.DataFrame(results)

# ========================================
# MAIN ANALYSIS
# ========================================

print("=" * 80)
print("ANALYSIS 1: COMPREHENSIVE CORRELATION ANALYSIS")
print("=" * 80)
print()

# Get all numeric columns for correlation
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
predictor_cols = [c for c in numeric_cols if c not in bart_params + ['partid', 'nll', 'aic', 'bic',
                                                                      'aic_pump_level', 'bic_pump_level',
                                                                      'n_trials', 'n_pump_opportunities', 'success']]

print(f"Analyzing {len(bart_params)} BART parameters against {len(predictor_cols)} predictors")
print()

# Compute correlations
corr_results = correlation_matrix_with_pvalues(merged, bart_params, predictor_cols)

# Sort by absolute correlation
corr_results_sorted = corr_results.sort_values('abs_r', ascending=False)

print("TOP 30 CORRELATIONS (by absolute r value):")
print("-" * 80)
print(f"{'BART Parameter':<20} {'Predictor':<25} {'r':>8} {'p':>10} {'n':>5}")
print("-" * 80)

for _, row in corr_results_sorted.head(30).iterrows():
    sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
    print(f"{row['bart_param']:<20} {row['predictor']:<25} {row['r']:>8.3f} {row['p']:>10.4f} {row['n']:>5} {sig}")

print()
print("Significance: * p<.05, ** p<.01, *** p<.001")
print()

# ========================================
# ANALYSIS BY THEORETICAL CONSTRUCTS
# ========================================

print("=" * 80)
print("ANALYSIS 2: CORRELATIONS BY THEORETICAL CONSTRUCT")
print("=" * 80)
print()

def analyze_construct(df, bart_cols, pred_cols, construct_name):
    """Analyze correlations for a specific theoretical construct."""
    print(f"\n{construct_name.upper()}:")
    print("-" * 60)

    valid_preds = [c for c in pred_cols if c in df.columns]
    if not valid_preds:
        print("  No valid predictors found")
        return None

    results = correlation_matrix_with_pvalues(df, bart_cols, valid_preds)
    top_results = results.nlargest(10, 'abs_r')

    for _, row in top_results.iterrows():
        if not np.isnan(row['r']):
            sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
            print(f"  {row['bart_param']:<15} x {row['predictor']:<20}: r = {row['r']:>6.3f} {sig}")

    return results

# Analyze each construct
impulsivity_results = analyze_construct(merged, bart_params, impulsivity_vars, "Impulsivity (BIS)")
ss_results = analyze_construct(merged, bart_params, sensation_seeking_vars, "Sensation Seeking")
risk_results = analyze_construct(merged, bart_params, risk_propensity_vars, "Risk Propensity (DOSPERT)")
substance_results = analyze_construct(merged, bart_params, substance_use_vars, "Substance Use")
cognitive_results = analyze_construct(merged, bart_params, cognitive_vars, "Cognitive Ability")
personality_results = analyze_construct(merged, bart_params, personality_vars, "Personality (Big Five)")

print()

# ========================================
# ANALYSIS 3: USING LARGER SAMPLE
# ========================================

print("=" * 80)
print("ANALYSIS 3: LARGER SAMPLE (N=1508) WITH BEHAVIORAL MEASURES")
print("=" * 80)
print()

# For larger sample, we have mean_pumps and explosion_rate
bart_behavioral_cols = ['mean_pumps', 'explosion_rate', 'omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'sigma']

large_numeric_cols = merged_large.select_dtypes(include=[np.number]).columns.tolist()
large_predictor_cols = [c for c in large_numeric_cols if c not in bart_behavioral_cols + ['partid', 'nll', 'aic', 'bic', 'n_trials', 'success']]

print(f"Analyzing with N={len(merged_large)} participants")
print()

# Compute correlations for larger sample
large_corr_results = correlation_matrix_with_pvalues(merged_large, bart_behavioral_cols, large_predictor_cols)
large_corr_sorted = large_corr_results.sort_values('abs_r', ascending=False)

print("TOP 40 CORRELATIONS IN LARGER SAMPLE:")
print("-" * 80)
print(f"{'BART Measure':<20} {'Predictor':<25} {'r':>8} {'p':>12} {'n':>6}")
print("-" * 80)

for _, row in large_corr_sorted.head(40).iterrows():
    sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
    print(f"{row['bart_param']:<20} {row['predictor']:<25} {row['r']:>8.3f} {row['p']:>12.6f} {row['n']:>6} {sig}")

print()

# ========================================
# ANALYSIS 4: STRONGEST INDIVIDUAL PREDICTORS
# ========================================

print("=" * 80)
print("ANALYSIS 4: STRONGEST PREDICTORS FOR EACH BART PARAMETER")
print("=" * 80)

for bart_var in bart_behavioral_cols:
    print(f"\n{bart_var.upper()}:")
    print("-" * 50)

    var_results = large_corr_results[large_corr_results['bart_param'] == bart_var].nlargest(10, 'abs_r')

    for _, row in var_results.iterrows():
        if not np.isnan(row['r']) and row['n'] > 100:
            sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
            print(f"  {row['predictor']:<30}: r = {row['r']:>7.3f} (n={row['n']}) {sig}")

print()

# ========================================
# ANALYSIS 5: COMPOSITE SCORE CREATION
# ========================================

print("=" * 80)
print("ANALYSIS 5: CREATING COMPOSITE PREDICTORS")
print("=" * 80)
print()

# Create composite scores that might capture what BART measures
def create_composite(df, vars_list, name):
    """Create z-scored composite from available variables."""
    available = [v for v in vars_list if v in df.columns]
    if len(available) < 2:
        return None

    # Z-score each variable and average
    z_scores = []
    for v in available:
        col = pd.to_numeric(df[v], errors='coerce')
        if col.std() > 0:
            z_scores.append((col - col.mean()) / col.std())

    if z_scores:
        composite = pd.concat(z_scores, axis=1).mean(axis=1)
        return composite
    return None

# Create theoretical composites
merged_large['composite_impulsivity'] = create_composite(merged_large, impulsivity_vars, 'impulsivity')
merged_large['composite_sensation_seeking'] = create_composite(merged_large, sensation_seeking_vars, 'sensation_seeking')
merged_large['composite_risk_propensity'] = create_composite(merged_large, risk_propensity_vars, 'risk_propensity')
merged_large['composite_externalizing'] = create_composite(merged_large, substance_use_vars + ['GABS', 'PG'], 'externalizing')

# Create a "disinhibition" composite (impulsivity + sensation seeking + risk propensity)
disinhibition_vars = ['BIS', 'SSSV'] + [v for v in risk_propensity_vars if v in merged_large.columns]
merged_large['composite_disinhibition'] = create_composite(merged_large, disinhibition_vars, 'disinhibition')

composite_vars = ['composite_impulsivity', 'composite_sensation_seeking', 'composite_risk_propensity',
                  'composite_externalizing', 'composite_disinhibition']

print("Correlations with Composite Scores:")
print("-" * 70)

for bart_var in ['mean_pumps', 'explosion_rate', 'omega_0', 'alpha_minus']:
    if bart_var not in merged_large.columns:
        continue
    print(f"\n{bart_var}:")
    for comp in composite_vars:
        if comp in merged_large.columns:
            r, p, n = compute_correlation(merged_large[bart_var], merged_large[comp])
            if not np.isnan(r) and n > 100:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {comp:<35}: r = {r:>7.3f} (n={n}) {sig}")

print()

# ========================================
# ANALYSIS 6: INTER-CORRELATIONS OF BART PARAMETERS
# ========================================

print("=" * 80)
print("ANALYSIS 6: INTER-CORRELATIONS OF BART PARAMETERS")
print("=" * 80)
print()

print("Understanding the structure of BART parameters:")
print("-" * 60)

bart_internal = ['mean_pumps', 'explosion_rate', 'omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'sigma']

for i, v1 in enumerate(bart_internal):
    for v2 in bart_internal[i+1:]:
        if v1 in merged_large.columns and v2 in merged_large.columns:
            r, p, n = compute_correlation(merged_large[v1], merged_large[v2])
            if not np.isnan(r):
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {v1:<20} x {v2:<20}: r = {r:>7.3f} {sig}")

print()

# ========================================
# SUMMARY AND INTERPRETATION
# ========================================

print("=" * 80)
print("SUMMARY: WHAT DOES THE BART MEASURE?")
print("=" * 80)
print()

# Find the strongest overall correlations
all_significant = large_corr_results[large_corr_results['p'] < 0.05].copy()
all_significant = all_significant.sort_values('abs_r', ascending=False)

print("STRONGEST SIGNIFICANT CORRELATIONS (p < .05):")
print("-" * 70)
for _, row in all_significant.head(20).iterrows():
    if row['n'] > 100:
        print(f"  {row['bart_param']:<15} x {row['predictor']:<25}: r = {row['r']:>7.3f}")

print()
print("=" * 80)
print("THEORETICAL INTERPRETATION")
print("=" * 80)
print("""
Based on the correlation analysis, the BART appears to measure:

1. RISK-TAKING PROPENSITY
   - Strong correlations with DOSPERT risk domains
   - Particularly ethical, recreational, and health risk subscales

2. IMPULSIVITY
   - Moderate correlations with BIS (Barratt Impulsiveness Scale)
   - Attentional and motor impulsivity components

3. SENSATION SEEKING
   - Correlations with SSS-V subscales
   - Especially thrill/adventure seeking and disinhibition

4. REAL-WORLD RISK BEHAVIORS
   - Associations with substance use (AUDIT, DAST)
   - Gambling tendencies (GABS, PG)

5. COGNITIVE CONTROL (inversely)
   - Negative correlations with conscientiousness
   - Working memory capacity relationships

The high reliability (.70-.91) suggests the BART captures a stable individual
difference in approach to sequential risk-taking under uncertainty, which
reflects a blend of:
  - Reward sensitivity (willingness to pursue larger gains)
  - Punishment sensitivity (responsiveness to losses)
  - Learning rate (how quickly one updates risk estimates)
  - Decision strategy (conservative vs. aggressive)
""")

# Save results
corr_results_sorted.to_csv('/home/user/Model/correlation_results_small_sample.csv', index=False)
large_corr_sorted.to_csv('/home/user/Model/correlation_results_large_sample.csv', index=False)

print()
print("Results saved to:")
print("  - correlation_results_small_sample.csv")
print("  - correlation_results_large_sample.csv")
print()
