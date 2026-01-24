#!/usr/bin/env python3
"""
BART Partial Correlations: Can We Unmask Stronger Personality Effects?

Testing whether controlling for potential confounds reveals larger
Big Five correlations with model parameters.

Potential suppressors:
1. Cognitive ability (NUM, WMC) - might suppress personality effects
2. Other personality traits - unique effects after controlling others
3. Age/demographics
4. General risk-taking tendency (mean_pumps)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def partial_corr(x, y, covariates, df):
    """
    Compute partial correlation between x and y, controlling for covariates.
    Returns r, p, n
    """
    # Get complete cases
    all_vars = [x, y] + covariates
    mask = df[all_vars].notna().all(axis=1)
    data = df.loc[mask, all_vars].copy()

    if len(data) < 30:
        return np.nan, np.nan, 0

    # Residualize x and y on covariates
    from sklearn.linear_model import LinearRegression

    X_cov = data[covariates].values

    # Residualize x
    reg_x = LinearRegression().fit(X_cov, data[x].values)
    resid_x = data[x].values - reg_x.predict(X_cov)

    # Residualize y
    reg_y = LinearRegression().fit(X_cov, data[y].values)
    resid_y = data[y].values - reg_y.predict(X_cov)

    # Correlation of residuals
    r, p = pearsonr(resid_x, resid_y)

    return r, p, len(data)

print("=" * 85)
print("PARTIAL CORRELATIONS: UNMASKING PERSONALITY EFFECTS")
print("=" * 85)

# Load data
range_results = pd.read_csv('/home/user/Model/range_learning_corrected_results (1).csv')
perso = pd.read_csv('/home/user/Model/perso.csv')
quest_scores = pd.read_csv('/home/user/Model/quest_scores.csv')
wmc = pd.read_csv('/home/user/Model/wmc.csv')

# Merge
merged = range_results.merge(perso, on='partid', how='inner')
merged = merged.merge(quest_scores, on='partid', how='left')
merged = merged.merge(wmc, on='partid', how='left')

print(f"\nSample: N = {len(merged)}")

# Variables
model_params = ['omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'sigma']
big_five = ['NEO_E', 'NEO_A', 'NEO_C', 'NEO_N', 'NEO_O']
big_five_names = {'NEO_E': 'Extraversion', 'NEO_A': 'Agreeableness',
                  'NEO_C': 'Conscientiousness', 'NEO_N': 'Neuroticism',
                  'NEO_O': 'Openness'}

# ============================================================================
# TEST 1: CONTROL FOR COGNITIVE ABILITY
# ============================================================================

print("\n" + "=" * 85)
print("TEST 1: CONTROLLING FOR COGNITIVE ABILITY (NUM, WMC)")
print("=" * 85)

print("""
HYPOTHESIS: Cognitive ability might suppress personality effects.
If smarter people have different personality-behavior relationships,
partialing out NUM and WMC might reveal stronger effects.
""")

cog_controls = ['NUM', 'WMC']

# Check which controls are available
available_cog = [c for c in cog_controls if c in merged.columns and merged[c].notna().sum() > 100]
print(f"Controlling for: {available_cog}")

print(f"\n{'Parameter':<12} {'Trait':<15} {'Zero-order r':>14} {'Partial r':>12} {'Change':>10}")
print("-" * 70)

improvements = []

for param in model_params:
    for trait in big_five:
        if trait not in merged.columns or param not in merged.columns:
            continue

        # Zero-order correlation
        mask = ~(merged[param].isna() | merged[trait].isna())
        if mask.sum() < 50:
            continue
        r_zero, p_zero = pearsonr(merged.loc[mask, param], merged.loc[mask, trait])

        # Partial correlation
        r_part, p_part, n = partial_corr(param, trait, available_cog, merged)

        if np.isnan(r_part):
            continue

        change = r_part - r_zero
        change_pct = (abs(r_part) - abs(r_zero)) / abs(r_zero) * 100 if r_zero != 0 else 0

        sig_zero = "*" if p_zero < 0.05 else ""
        sig_part = "*" if p_part < 0.05 else ""

        # Only show if there's a notable change or significance
        if abs(change) > 0.02 or p_zero < 0.05 or p_part < 0.05:
            arrow = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"{param:<12} {big_five_names[trait]:<15} {r_zero:>10.3f}{sig_zero:<3} {r_part:>10.3f}{sig_part:<3} {arrow} {change:>+.3f}")

            improvements.append({
                'param': param, 'trait': trait,
                'r_zero': r_zero, 'r_partial': r_part,
                'change': change, 'change_pct': change_pct
            })

# ============================================================================
# TEST 2: CONTROL FOR OTHER BIG FIVE TRAITS
# ============================================================================

print("\n" + "=" * 85)
print("TEST 2: UNIQUE PERSONALITY EFFECTS (CONTROLLING OTHER BIG FIVE)")
print("=" * 85)

print("""
HYPOTHESIS: Each trait's effect might be cleaner after removing shared
variance with other traits. E.g., Extraversion effect on α⁺ controlling
for Neuroticism, Conscientiousness, etc.
""")

print(f"\n{'Parameter':<12} {'Trait':<15} {'Zero-order r':>14} {'Unique r':>12} {'Change':>10}")
print("-" * 70)

for param in ['alpha_plus', 'omega_0', 'alpha_minus']:  # Focus on key params
    for trait in big_five:
        if trait not in merged.columns or param not in merged.columns:
            continue

        # Other Big Five as controls
        other_traits = [t for t in big_five if t != trait and t in merged.columns]

        # Zero-order
        mask = ~(merged[param].isna() | merged[trait].isna())
        if mask.sum() < 50:
            continue
        r_zero, p_zero = pearsonr(merged.loc[mask, param], merged.loc[mask, trait])

        # Partial (controlling other traits)
        r_part, p_part, n = partial_corr(param, trait, other_traits, merged)

        if np.isnan(r_part):
            continue

        change = r_part - r_zero
        sig_zero = "*" if p_zero < 0.05 else ""
        sig_part = "*" if p_part < 0.05 else ""

        if abs(r_zero) > 0.05 or abs(r_part) > 0.05:
            arrow = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"{param:<12} {big_five_names[trait]:<15} {r_zero:>10.3f}{sig_zero:<3} {r_part:>10.3f}{sig_part:<3} {arrow} {change:>+.3f}")

# ============================================================================
# TEST 3: CONTROL FOR GENERAL PUMPING TENDENCY
# ============================================================================

print("\n" + "=" * 85)
print("TEST 3: CONTROLLING FOR MEAN PUMPS (GENERAL RISK TENDENCY)")
print("=" * 85)

print("""
HYPOTHESIS: Model parameters might show stronger personality correlations
after removing variance associated with overall pumping tendency.
This isolates the PROCESS (how you learn) from the OUTCOME (how much you pump).
""")

print(f"\n{'Parameter':<12} {'Trait':<15} {'Zero-order r':>14} {'Partial r':>12} {'Change':>10}")
print("-" * 70)

for param in model_params:
    if param == 'mean_pumps':
        continue

    for trait in big_five:
        if trait not in merged.columns or param not in merged.columns:
            continue

        # Zero-order
        mask = ~(merged[param].isna() | merged[trait].isna())
        if mask.sum() < 50:
            continue
        r_zero, p_zero = pearsonr(merged.loc[mask, param], merged.loc[mask, trait])

        # Partial (controlling mean_pumps)
        r_part, p_part, n = partial_corr(param, trait, ['mean_pumps'], merged)

        if np.isnan(r_part):
            continue

        change = r_part - r_zero
        sig_zero = "*" if p_zero < 0.05 else ""
        sig_part = "*" if p_part < 0.05 else ""

        if abs(r_zero) > 0.08 or abs(r_part) > 0.08:
            arrow = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"{param:<12} {big_five_names[trait]:<15} {r_zero:>10.3f}{sig_zero:<3} {r_part:>10.3f}{sig_part:<3} {arrow} {change:>+.3f}")

# ============================================================================
# TEST 4: COMBINED CONTROLS (COGNITIVE + MEAN PUMPS)
# ============================================================================

print("\n" + "=" * 85)
print("TEST 4: FULL MODEL - CONTROLLING COGNITIVE + MEAN PUMPS")
print("=" * 85)

print("""
HYPOTHESIS: The cleanest personality effects should emerge when we control
for both cognitive ability AND general pumping tendency.
""")

full_controls = ['NUM', 'WMC', 'mean_pumps']
full_controls = [c for c in full_controls if c in merged.columns]

print(f"Controlling for: {full_controls}")

print(f"\n{'Parameter':<12} {'Trait':<15} {'Zero-order r':>14} {'Full partial r':>14} {'Change':>10}")
print("-" * 75)

best_improvements = []

for param in model_params:
    for trait in big_five:
        if trait not in merged.columns or param not in merged.columns:
            continue

        # Zero-order
        mask = ~(merged[param].isna() | merged[trait].isna())
        if mask.sum() < 50:
            continue
        r_zero, p_zero = pearsonr(merged.loc[mask, param], merged.loc[mask, trait])

        # Full partial
        controls_for_param = [c for c in full_controls if c != param]
        r_part, p_part, n = partial_corr(param, trait, controls_for_param, merged)

        if np.isnan(r_part):
            continue

        change = r_part - r_zero
        sig_zero = "**" if p_zero < 0.01 else "*" if p_zero < 0.05 else ""
        sig_part = "**" if p_part < 0.01 else "*" if p_part < 0.05 else ""

        arrow = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
        print(f"{param:<12} {big_five_names[trait]:<15} {r_zero:>10.3f}{sig_zero:<4} {r_part:>12.3f}{sig_part:<4} {arrow} {change:>+.3f}")

        best_improvements.append({
            'param': param, 'trait': big_five_names[trait],
            'r_zero': r_zero, 'r_partial': r_part,
            'p_zero': p_zero, 'p_partial': p_part,
            'change': change, 'abs_change': abs(r_part) - abs(r_zero)
        })

# ============================================================================
# TEST 5: ANXIETY AS NEUROTICISM PROXY (LARGER N)
# ============================================================================

print("\n" + "=" * 85)
print("TEST 5: STAI TRAIT ANXIETY (LARGER SAMPLE)")
print("=" * 85)

print("""
The Big Five sample is small (N ≈ 287). STAI has more cases.
STAI correlates highly with NEO_N, so it's a proxy for Neuroticism.
""")

if 'STAI_trait' in merged.columns:
    for param in model_params:
        mask = ~(merged[param].isna() | merged['STAI_trait'].isna())
        if mask.sum() < 100:
            continue

        # Zero-order
        r_zero, p_zero = pearsonr(merged.loc[mask, param], merged.loc[mask, 'STAI_trait'])

        # Partial (controlling cognitive)
        r_part, p_part, n = partial_corr(param, 'STAI_trait', ['NUM', 'WMC'], merged)

        if not np.isnan(r_part):
            change = r_part - r_zero
            sig_zero = "**" if p_zero < 0.01 else "*" if p_zero < 0.05 else ""
            sig_part = "**" if p_part < 0.01 else "*" if p_part < 0.05 else ""

            print(f"{param:<12} STAI (N={n:>4})  r = {r_zero:>7.3f}{sig_zero:<3} → {r_part:>7.3f}{sig_part:<3} (Δ = {change:>+.3f})")

# ============================================================================
# SUMMARY: WHICH CONTROLS HELP MOST?
# ============================================================================

print("\n" + "=" * 85)
print("SUMMARY: WHAT WORKS?")
print("=" * 85)

print("""
FINDINGS:
""")

# Sort by improvement
best_df = pd.DataFrame(best_improvements)
best_df = best_df.sort_values('abs_change', ascending=False)

print("LARGEST IMPROVEMENTS (controlling cognitive + mean_pumps):")
print("-" * 70)
for _, row in best_df.head(10).iterrows():
    direction = "↑" if row['change'] > 0 else "↓"
    print(f"  {row['param']:<12} × {row['trait']:<15}: {row['r_zero']:>6.3f} → {row['r_partial']:>6.3f} ({direction} {row['abs_change']:>+.3f})")

# Check if any become significant
newly_sig = best_df[(best_df['p_zero'] >= 0.05) & (best_df['p_partial'] < 0.05)]
if len(newly_sig) > 0:
    print("\nNEWLY SIGNIFICANT AFTER CONTROLS:")
    for _, row in newly_sig.iterrows():
        print(f"  {row['param']} × {row['trait']}: {row['r_zero']:.3f} (ns) → {row['r_partial']:.3f}*")

# Check strongest partial correlations
print("\nSTRONGEST PARTIAL CORRELATIONS:")
print("-" * 70)
strongest = best_df.nlargest(5, 'r_partial')
for _, row in strongest.iterrows():
    sig = "**" if row['p_partial'] < 0.01 else "*" if row['p_partial'] < 0.05 else ""
    print(f"  {row['param']:<12} × {row['trait']:<15}: r = {row['r_partial']:.3f}{sig}")

print("""
─────────────────────────────────────────────────────────────────────────────────────

INTERPRETATION:

The partial correlations tell us whether controlling for confounds
INCREASES the personality-parameter relationship. If so, those
confounds were acting as suppressors.

Key questions:
1. Does controlling for cognitive ability increase personality effects?
2. Does controlling for mean_pumps isolate "process" from "outcome"?
3. Are there personality effects hidden by shared method variance?

─────────────────────────────────────────────────────────────────────────────────────
""")

# Save results
best_df.to_csv('/home/user/Model/partial_correlation_results.csv', index=False)
print("\nResults saved to: partial_correlation_results.csv")
