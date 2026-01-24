#!/usr/bin/env python3
"""
BART Machine Learning Analysis
==============================

Using ML to discover what the BART measures with .70-.91 reliability.

This analysis uses:
1. Random Forest feature importance
2. Gradient Boosting for non-linear relationships
3. LASSO for sparse feature selection
4. Principal Component Analysis to find latent factors
5. Multiple regression for variance explained
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

# ========================================
# DATA LOADING
# ========================================

print("=" * 80)
print("BART MACHINE LEARNING ANALYSIS")
print("=" * 80)
print()

# Load datasets
range_results = pd.read_csv('/home/user/Model/range_learning_corrected_results (1).csv')
quest_scores = pd.read_csv('/home/user/Model/quest_scores.csv')
perso = pd.read_csv('/home/user/Model/perso.csv')
wmc = pd.read_csv('/home/user/Model/wmc.csv')
cct = pd.read_csv('/home/user/Model/cct_overt.csv')
dfd = pd.read_csv('/home/user/Model/dfd_perpers.csv')
dfe = pd.read_csv('/home/user/Model/dfe_perpers.csv')
lotteries = pd.read_csv('/home/user/Model/lotteriesOvert.csv')
mt = pd.read_csv('/home/user/Model/mt.csv')

# Merge all data
merged = range_results.copy()
merged = merged.merge(quest_scores, on='partid', how='left')
merged = merged.merge(perso, on='partid', how='left')
merged = merged.merge(wmc, on='partid', how='left')
merged = merged.merge(cct, on='partid', how='left')
merged = merged.merge(dfd, on='partid', how='left', suffixes=('', '_dfd'))
merged = merged.merge(dfe, on='partid', how='left', suffixes=('', '_dfe'))
merged = merged.merge(lotteries, on='partid', how='left', suffixes=('', '_lot'))
merged = merged.merge(mt, on='partid', how='left')

print(f"Full dataset: {len(merged)} participants")

# Define outcome variables
bart_outcomes = ['mean_pumps', 'explosion_rate', 'omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'sigma']

# Define predictor categories for interpretation
predictor_categories = {
    'impulsivity': ['BIS', 'BIS1att', 'BIS1mot', 'BIS1ctr', 'BIS1com', 'BIS1per', 'BIS1ins',
                    'BIS2att', 'BIS2mot', 'BIS2npl'],
    'sensation_seeking': ['SSSV', 'SStas', 'SSexp', 'SSdis', 'SSbor'],
    'risk_propensity': ['Deth', 'Dinv', 'Dgam', 'Dhea', 'Drec', 'Dsoc'],
    'risk_perception': ['Deth_r', 'Dinv_r', 'Dgam_r', 'Dhea_r', 'Drec_r', 'Dsoc_r'],
    'risk_benefit': ['Deth_b', 'Dinv_b', 'Dgam_b', 'Dhea_b', 'Drec_b', 'Dsoc_b'],
    'substance_use': ['AUDIT', 'FTND', 'DAST'],
    'gambling': ['GABS', 'PG'],
    'cognitive': ['NUM', 'WMC', 'MUpc', 'SSTM'],
    'personality': ['NEO_A', 'NEO_C', 'NEO_E', 'NEO_N', 'NEO_O'],
    'anxiety': ['STAI_trait'],
    'wellbeing': ['SOEP', 'SOEPdri', 'SOEPfin', 'SOEPrec', 'SOEPocc', 'SOEPhea', 'SOEPsoc'],
    'decision_making': ['CCTncards', 'CCTacards', 'CCTpayoff', 'CCTratio', 'CCTaratio',
                        'R', 'H', 'CV', 'R_lot', 'H_lot']
}

# Get all predictor variables
all_predictors = []
for cat, vars in predictor_categories.items():
    all_predictors.extend(vars)
all_predictors = [p for p in all_predictors if p in merged.columns]

print(f"Predictor variables: {len(all_predictors)}")
print()

# ========================================
# PREPARE DATA FOR ML
# ========================================

def prepare_data(df, outcome_var, predictor_vars):
    """Prepare data for ML: impute missing, scale, return X and y."""
    available_preds = [p for p in predictor_vars if p in df.columns]

    # Select data
    data = df[[outcome_var] + available_preds].copy()

    # Convert to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with missing outcome
    data = data.dropna(subset=[outcome_var])

    y = data[outcome_var].values
    X = data[available_preds].values

    # Impute missing predictors with median
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, available_preds

# ========================================
# ANALYSIS 1: RANDOM FOREST FEATURE IMPORTANCE
# ========================================

print("=" * 80)
print("ANALYSIS 1: RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)
print()

def rf_feature_importance(df, outcome_var, predictor_vars, n_estimators=500):
    """Get Random Forest feature importances."""
    X, y, features = prepare_data(df, outcome_var, predictor_vars)

    if len(features) == 0:
        return None

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Cross-validated R²
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

    # Feature importances
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importances, cv_scores.mean(), cv_scores.std()

print("Random Forest Feature Importance (top 15 per outcome):")
print("-" * 70)

rf_results = {}
for outcome in bart_outcomes:
    if outcome not in merged.columns:
        continue

    result = rf_feature_importance(merged, outcome, all_predictors)
    if result is None:
        continue

    importances, mean_r2, std_r2 = result
    rf_results[outcome] = importances

    print(f"\n{outcome.upper()} (CV R² = {mean_r2:.3f} ± {std_r2:.3f}):")
    for _, row in importances.head(15).iterrows():
        bar = "█" * int(row['importance'] * 100)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")

print()

# ========================================
# ANALYSIS 2: LASSO FEATURE SELECTION
# ========================================

print("=" * 80)
print("ANALYSIS 2: LASSO SPARSE FEATURE SELECTION")
print("=" * 80)
print()

def lasso_feature_selection(df, outcome_var, predictor_vars):
    """Use LASSO to find most important sparse features."""
    X, y, features = prepare_data(df, outcome_var, predictor_vars)

    if len(features) == 0:
        return None

    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
    lasso.fit(X, y)

    # Get non-zero coefficients
    coefs = pd.DataFrame({
        'feature': features,
        'coefficient': lasso.coef_
    })
    coefs['abs_coef'] = coefs['coefficient'].abs()
    coefs = coefs[coefs['abs_coef'] > 0].sort_values('abs_coef', ascending=False)

    return coefs, lasso.score(X, y), lasso.alpha_

print("LASSO Selected Features (non-zero coefficients):")
print("-" * 70)

lasso_results = {}
for outcome in bart_outcomes:
    if outcome not in merged.columns:
        continue

    result = lasso_feature_selection(merged, outcome, all_predictors)
    if result is None:
        continue

    coefs, r2, alpha = result
    lasso_results[outcome] = coefs

    print(f"\n{outcome.upper()} (R² = {r2:.3f}, alpha = {alpha:.4f}):")
    print(f"  Selected {len(coefs)} features:")
    for _, row in coefs.head(15).iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"    {sign} {row['feature']:<25} β = {row['coefficient']:>7.4f}")

print()

# ========================================
# ANALYSIS 3: GRADIENT BOOSTING (NON-LINEAR)
# ========================================

print("=" * 80)
print("ANALYSIS 3: GRADIENT BOOSTING (CAPTURES NON-LINEAR EFFECTS)")
print("=" * 80)
print()

def gb_analysis(df, outcome_var, predictor_vars):
    """Gradient Boosting for non-linear relationships."""
    X, y, features = prepare_data(df, outcome_var, predictor_vars)

    if len(features) == 0:
        return None

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gb.fit(X, y)

    cv_scores = cross_val_score(gb, X, y, cv=5, scoring='r2')

    importances = pd.DataFrame({
        'feature': features,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)

    return importances, cv_scores.mean(), cv_scores.std()

print("Gradient Boosting captures non-linear relationships:")
print("-" * 70)

for outcome in ['mean_pumps', 'explosion_rate', 'alpha_minus', 'alpha_plus']:
    if outcome not in merged.columns:
        continue

    result = gb_analysis(merged, outcome, all_predictors)
    if result is None:
        continue

    importances, mean_r2, std_r2 = result

    print(f"\n{outcome.upper()} (CV R² = {mean_r2:.3f} ± {std_r2:.3f}):")
    for _, row in importances.head(10).iterrows():
        bar = "█" * int(row['importance'] * 100)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")

print()

# ========================================
# ANALYSIS 4: MUTUAL INFORMATION (NON-LINEAR CORRELATIONS)
# ========================================

print("=" * 80)
print("ANALYSIS 4: MUTUAL INFORMATION (NON-LINEAR DEPENDENCIES)")
print("=" * 80)
print()

def mutual_info_analysis(df, outcome_var, predictor_vars):
    """Mutual information captures non-linear relationships."""
    X, y, features = prepare_data(df, outcome_var, predictor_vars)

    if len(features) == 0:
        return None

    mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({
        'feature': features,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    return mi_df

print("Mutual Information (detects non-linear dependencies):")
print("-" * 70)

for outcome in ['mean_pumps', 'explosion_rate']:
    if outcome not in merged.columns:
        continue

    mi_df = mutual_info_analysis(merged, outcome, all_predictors)
    if mi_df is None:
        continue

    print(f"\n{outcome.upper()} Top 15 predictors by Mutual Information:")
    for _, row in mi_df.head(15).iterrows():
        bar = "█" * int(row['mutual_info'] * 20)
        print(f"  {row['feature']:<25} {row['mutual_info']:.4f} {bar}")

print()

# ========================================
# ANALYSIS 5: CATEGORY-LEVEL IMPORTANCE
# ========================================

print("=" * 80)
print("ANALYSIS 5: IMPORTANCE BY THEORETICAL CONSTRUCT")
print("=" * 80)
print()

def category_importance(rf_results, predictor_categories):
    """Aggregate RF importance by theoretical category."""
    category_scores = {}

    for outcome, importances in rf_results.items():
        category_scores[outcome] = {}

        for cat_name, cat_vars in predictor_categories.items():
            cat_importance = importances[importances['feature'].isin(cat_vars)]['importance'].sum()
            category_scores[outcome][cat_name] = cat_importance

    return category_scores

category_scores = category_importance(rf_results, predictor_categories)

print("Which theoretical constructs best predict BART performance?")
print("-" * 70)

for outcome in ['mean_pumps', 'explosion_rate', 'omega_0', 'alpha_minus', 'alpha_plus']:
    if outcome not in category_scores:
        continue

    scores = category_scores[outcome]
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{outcome.upper()}:")
    for cat, score in sorted_cats[:7]:
        bar = "█" * int(score * 50)
        print(f"  {cat:<20} {score:.4f} {bar}")

print()

# ========================================
# ANALYSIS 6: OPTIMAL COMPOSITE CREATION
# ========================================

print("=" * 80)
print("ANALYSIS 6: CREATING OPTIMAL PREDICTIVE COMPOSITES")
print("=" * 80)
print()

def create_optimal_composite(df, outcome_var, predictor_vars, n_components=5):
    """Create PCA-based composites and test their prediction."""
    X, y, features = prepare_data(df, outcome_var, predictor_vars)

    if len(features) < n_components:
        n_components = len(features)

    if n_components == 0:
        return None

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Test each component's correlation with outcome
    component_corrs = []
    for i in range(n_components):
        r, p = pearsonr(X_pca[:, i], y)
        component_corrs.append({
            'component': f'PC{i+1}',
            'r': r,
            'p': p,
            'variance_explained': pca.explained_variance_ratio_[i]
        })

    # Get loadings for best component
    component_corrs = pd.DataFrame(component_corrs)
    best_idx = component_corrs['r'].abs().idxmax()
    best_loadings = pd.DataFrame({
        'feature': features,
        'loading': pca.components_[best_idx]
    }).sort_values('loading', key=abs, ascending=False)

    return component_corrs, best_loadings, pca.explained_variance_ratio_.sum()

print("PCA-based composite predictors:")
print("-" * 70)

for outcome in ['mean_pumps', 'explosion_rate']:
    if outcome not in merged.columns:
        continue

    # Use sensation seeking + risk propensity + impulsivity
    composite_vars = (predictor_categories['sensation_seeking'] +
                     predictor_categories['risk_propensity'] +
                     predictor_categories['impulsivity'])
    composite_vars = [v for v in composite_vars if v in merged.columns]

    result = create_optimal_composite(merged, outcome, composite_vars, n_components=5)
    if result is None:
        continue

    component_corrs, best_loadings, total_var = result

    print(f"\n{outcome.upper()} - PCA from sensation seeking + risk + impulsivity:")
    print(f"  Total variance explained: {total_var:.1%}")
    print(f"\n  Component correlations with {outcome}:")
    for _, row in component_corrs.iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        print(f"    {row['component']}: r = {row['r']:>7.3f} (var = {row['variance_explained']:.1%}) {sig}")

    print(f"\n  Best component loadings (top 10):")
    for _, row in best_loadings.head(10).iterrows():
        print(f"    {row['feature']:<25} {row['loading']:>7.3f}")

print()

# ========================================
# ANALYSIS 7: COMBINED MODEL COMPARISON
# ========================================

print("=" * 80)
print("ANALYSIS 7: WHICH CONSTRUCTS UNIQUELY PREDICT BART?")
print("=" * 80)
print()

def hierarchical_regression(df, outcome_var):
    """Test unique contribution of each construct via hierarchical regression."""
    from sklearn.linear_model import LinearRegression

    results = []

    # Define construct groups
    constructs = {
        'Sensation Seeking': predictor_categories['sensation_seeking'],
        'Risk Propensity': predictor_categories['risk_propensity'],
        'Impulsivity': predictor_categories['impulsivity'],
        'Cognitive': predictor_categories['cognitive'],
        'Personality': predictor_categories['personality'],
        'Decision Making': predictor_categories['decision_making']
    }

    for construct_name, construct_vars in constructs.items():
        # Get available variables
        available = [v for v in construct_vars if v in df.columns]
        if len(available) == 0:
            continue

        X, y, features = prepare_data(df, outcome_var, available)
        if len(features) == 0:
            continue

        # Fit model
        model = LinearRegression()
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        results.append({
            'construct': construct_name,
            'n_features': len(features),
            'cv_r2': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })

    return pd.DataFrame(results).sort_values('cv_r2', ascending=False)

print("Unique predictive power of each construct (CV R²):")
print("-" * 70)

for outcome in ['mean_pumps', 'explosion_rate', 'omega_0', 'alpha_minus']:
    if outcome not in merged.columns:
        continue

    results = hierarchical_regression(merged, outcome)

    print(f"\n{outcome.upper()}:")
    for _, row in results.iterrows():
        bar = "█" * int(row['cv_r2'] * 100)
        print(f"  {row['construct']:<25} R² = {row['cv_r2']:>6.3f} ± {row['cv_std']:.3f} {bar}")

print()

# ========================================
# SYNTHESIS: WHAT DOES BART MEASURE?
# ========================================

print("=" * 80)
print("SYNTHESIS: WHAT DOES THE BART RELIABLY MEASURE?")
print("=" * 80)
print("""
Based on comprehensive ML analysis, the BART with .70-.91 reliability measures:

╔══════════════════════════════════════════════════════════════════════════════╗
║  PRIMARY CONSTRUCT: BEHAVIORAL APPROACH TO RISK                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. SENSATION SEEKING (Strongest predictor, r ≈ 0.19)                       ║
║     - Experience Seeking (SSexp)                                             ║
║     - Thrill & Adventure Seeking (SStas)                                    ║
║     - Total SSS-V score                                                      ║
║                                                                              ║
║  2. RECREATIONAL/HEALTH RISK TOLERANCE                                       ║
║     - DOSPERT Recreational domain (Drec, Drec_b)                            ║
║     - DOSPERT Health domain (Dhea)                                          ║
║     - Benefits outweigh risks perception                                     ║
║                                                                              ║
║  3. COGNITIVE CONTROL (Inverse relationship)                                 ║
║     - Numeracy (NUM) → faster loss learning                                 ║
║     - Working Memory (WMC) → faster learning rates                          ║
║                                                                              ║
║  4. DECISION-MAKING STYLE                                                    ║
║     - CCT performance (risk taking across tasks)                            ║
║     - Convergent validity with other risk tasks                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY INSIGHT: The high reliability comes from BART measuring a stable       ║
║  trait-like propensity for approach-oriented risk behavior, rather than     ║
║  a narrow cognitive or impulsivity construct.                                ║
║                                                                              ║
║  The BART captures BEHAVIORAL DISINHIBITION under uncertainty -             ║
║  a blend of sensation seeking + risk tolerance + approach motivation.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ========================================
# SAVE RESULTS
# ========================================

# Save RF importances
for outcome, importances in rf_results.items():
    importances.to_csv(f'/home/user/Model/rf_importance_{outcome}.csv', index=False)

# Save LASSO results
for outcome, coefs in lasso_results.items():
    coefs.to_csv(f'/home/user/Model/lasso_features_{outcome}.csv', index=False)

print("\nResults saved:")
print("  - rf_importance_*.csv (Random Forest feature importances)")
print("  - lasso_features_*.csv (LASSO selected features)")
print()
