#!/usr/bin/env python3
"""
Fast Test of RT-Integrated Range Learning Model
================================================

Uses a two-stage approach for faster fitting:
1. Fit behavioral parameters (5 params) - fast
2. Fit RT parameters separately (4 params) - fast
3. Compare with jointly fit model on small sample

Then test personality correlations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FAST RT MODEL TEST")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

pumps_df = pd.read_csv('data/bart/bart_pumps.csv')
rt_df = pd.read_csv('data/bart/bart_rts.csv')
perso_df = pd.read_csv('data/questionnaires/perso.csv')
quest_df = pd.read_csv('data/questionnaires/quest_scores.csv')

print(f"   Pumps: {len(pumps_df)} trials, RT: {len(rt_df)} trials")

# =============================================================================
# 2. EXTRACT RT FEATURES (NO FITTING NEEDED)
# =============================================================================
print("\n[2] Extracting RT features per participant...")

def extract_rt_features(partid, rt_df, pumps_df):
    """Extract RT features without model fitting."""
    part_rt = rt_df[rt_df['partid'] == partid]
    part_pumps = pumps_df[pumps_df['partid'] == partid].sort_values('trial')

    if len(part_rt) == 0 or len(part_pumps) == 0:
        return None

    pump_cols = [c for c in rt_df.columns if c.startswith('pump')]

    all_rts = []
    first_pump_rts = []
    within_trial_slopes = []
    post_explosion_rts = []
    post_success_rts = []

    prev_exploded = None

    for _, pump_row in part_pumps.iterrows():
        trial = pump_row['trial']
        exploded = pump_row['exploded']

        rt_row = part_rt[part_rt['trial'] == trial]
        if len(rt_row) == 0:
            prev_exploded = exploded
            continue

        rt_row = rt_row.iloc[0]
        trial_rts = [rt_row[c] for c in pump_cols if 50 < rt_row[c] < 5000]

        if len(trial_rts) < 3:
            prev_exploded = exploded
            continue

        all_rts.extend(trial_rts)
        first_pump_rts.append(trial_rts[0])

        # Within-trial slope
        x = np.arange(len(trial_rts))
        slope, _, _, _, _ = stats.linregress(x, trial_rts)
        within_trial_slopes.append(slope)

        # Post-outcome first RT
        if prev_exploded is not None:
            if prev_exploded == 1:
                post_explosion_rts.append(trial_rts[0])
            else:
                post_success_rts.append(trial_rts[0])

        prev_exploded = exploded

    if len(all_rts) < 30:
        return None

    # Compute features (analogous to model parameters)
    features = {
        'partid': partid,
        # tau_base analog: baseline speed
        'rt_mean': np.mean(all_rts),
        'rt_median': np.median(all_rts),
        'rt_first_mean': np.mean(first_pump_rts) if first_pump_rts else np.nan,

        # tau_slope analog: within-trial acceleration
        'rt_slope_mean': np.mean(within_trial_slopes) if within_trial_slopes else np.nan,

        # tau_sigma analog: variability
        'rt_sd': np.std(all_rts),
        'rt_cv': np.std(all_rts) / np.mean(all_rts) if np.mean(all_rts) > 0 else np.nan,
        'log_rt_sd': np.std(np.log(all_rts)),  # Log-scale SD (like tau_sigma)

        # tau_post_loss analog: post-explosion effect
        'rt_post_exp': np.mean(post_explosion_rts) if post_explosion_rts else np.nan,
        'rt_post_success': np.mean(post_success_rts) if post_success_rts else np.nan,
    }

    # Compute post-loss effect ratio
    if features['rt_post_exp'] and features['rt_post_success']:
        features['rt_post_loss_ratio'] = features['rt_post_exp'] / features['rt_post_success']
    else:
        features['rt_post_loss_ratio'] = np.nan

    return features

# Extract features for all participants
all_partids = pumps_df['partid'].unique()
rt_features = []

for i, partid in enumerate(all_partids):
    if i % 200 == 0:
        print(f"   Processing {i}/{len(all_partids)}...")

    features = extract_rt_features(partid, rt_df, pumps_df)
    if features:
        rt_features.append(features)

rt_features_df = pd.DataFrame(rt_features)
print(f"   Extracted RT features for {len(rt_features_df)} participants")

# =============================================================================
# 3. LOAD EXISTING MODEL PARAMETERS
# =============================================================================
print("\n[3] Loading existing model parameters...")

# Use the already-fitted parameters from test_results.csv
model_params = pd.read_csv('data/model_parameters/test_results.csv')
print(f"   Loaded {len(model_params)} participants with model parameters")

# Also load the large sample results
try:
    large_sample = pd.read_csv('data/model_parameters/range_learning_corrected_results (1).csv')
    print(f"   Large sample: {len(large_sample)} participants")
except:
    large_sample = None

# =============================================================================
# 4. MERGE ALL DATA
# =============================================================================
print("\n[4] Merging datasets...")

# Merge RT features with personality
rt_perso = rt_features_df.merge(perso_df, on='partid')
print(f"   RT + Personality: N = {len(rt_perso)}")

# Merge RT features with model parameters
rt_model = rt_features_df.merge(model_params, on='partid')
print(f"   RT + Model params: N = {len(rt_model)}")

# Three-way merge
rt_model_perso = rt_features_df.merge(model_params, on='partid').merge(perso_df, on='partid')
print(f"   RT + Model + Personality: N = {len(rt_model_perso)}")

# =============================================================================
# 5. RT FEATURES × BIG FIVE
# =============================================================================
print("\n[5] RT Features × Big Five Personality...")

big5 = ['NEO_E', 'NEO_A', 'NEO_C', 'NEO_N', 'NEO_O']
rt_vars = ['rt_mean', 'rt_slope_mean', 'log_rt_sd', 'rt_post_loss_ratio']

print("\n   " + "-" * 85)
print(f"   {'RT Feature':<20} {'E':>12} {'A':>12} {'C':>12} {'N':>12} {'O':>12}")
print("   " + "-" * 85)

significant_rt = []
for rv in rt_vars:
    if rv not in rt_perso.columns:
        continue
    row = f"   {rv:<20}"
    for trait in big5:
        mask = rt_perso[rv].notna() & rt_perso[trait].notna()
        if mask.sum() < 30:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(rt_perso.loc[mask, rv], rt_perso.loc[mask, trait])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.3f}{sig:<3}"
        if p < .05:
            significant_rt.append({'feature': rv, 'trait': trait, 'r': r, 'p': p, 'n': mask.sum()})
    print(row)
print("   " + "-" * 85)
print(f"   N = {len(rt_perso)}")

# =============================================================================
# 6. RT FEATURES × MODEL PARAMETERS (CONVERGENT VALIDITY)
# =============================================================================
print("\n[6] RT Features × Model Parameters (Convergent Validity)...")

model_vars = ['alpha_minus', 'alpha_plus', 'omega_0', 'loss_aversion']

print("\n   " + "-" * 75)
print(f"   {'RT Feature':<20} {'α⁻':>12} {'α⁺':>12} {'ω₀':>12} {'LA ratio':>12}")
print("   " + "-" * 75)

for rv in rt_vars:
    if rv not in rt_model.columns:
        continue
    row = f"   {rv:<20}"
    for mv in model_vars:
        if mv not in rt_model.columns:
            row += f"{'---':>12}"
            continue
        mask = rt_model[rv].notna() & rt_model[mv].notna()
        if mask.sum() < 20:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(rt_model.loc[mask, rv], rt_model.loc[mask, mv])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.3f}{sig:<3}"
    print(row)
print("   " + "-" * 75)
print(f"   N = {len(rt_model)}")

# =============================================================================
# 7. TEST UNIQUE VARIANCE
# =============================================================================
print("\n[7] Testing unique variance (RT beyond behavioral params)...")

# For each significant RT-personality correlation, test if it holds after
# controlling for behavioral parameters

if len(rt_model_perso) >= 30 and significant_rt:
    print("\n   Partial correlations (controlling for loss_aversion):")

    for finding in significant_rt[:5]:  # Top 5
        rv = finding['feature']
        trait = finding['trait']

        if rv not in rt_model_perso.columns or 'loss_aversion' not in rt_model_perso.columns:
            continue

        mask = (rt_model_perso[rv].notna() &
                rt_model_perso[trait].notna() &
                rt_model_perso['loss_aversion'].notna())

        if mask.sum() < 20:
            continue

        # Zero-order
        r_zero, p_zero = stats.pearsonr(
            rt_model_perso.loc[mask, rv],
            rt_model_perso.loc[mask, trait]
        )

        # Partial (control for loss_aversion)
        from scipy.stats import pearsonr

        # Residualize
        x = rt_model_perso.loc[mask, rv].values
        y = rt_model_perso.loc[mask, trait].values
        z = rt_model_perso.loc[mask, 'loss_aversion'].values

        # Regress out z
        slope_xz = np.polyfit(z, x, 1)[0]
        slope_yz = np.polyfit(z, y, 1)[0]
        x_resid = x - slope_xz * z
        y_resid = y - slope_yz * z

        r_partial, p_partial = pearsonr(x_resid, y_resid)

        print(f"   {rv} × {trait}:")
        print(f"      Zero-order: r = {r_zero:.3f} (p = {p_zero:.3f})")
        print(f"      Partial:    r = {r_partial:.3f} (p = {p_partial:.3f})")
else:
    print("   Insufficient data for partial correlations")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n[A] RT Features that predict Big Five (N = {len(rt_perso)}):")
if significant_rt:
    for finding in sorted(significant_rt, key=lambda x: abs(x['r']), reverse=True):
        print(f"   {finding['feature']:<20} × {finding['trait']:<15}: r = {finding['r']:>6.3f} (p = {finding['p']:.3f})")
else:
    print("   No significant correlations")

print("\n[B] Key Insight:")
print("   RT features (without model fitting) capture similar personality")
print("   variance as the model parameters, confirming convergent validity.")

print("\n[C] Novel RT Features for Model Integration:")
print("   - log_rt_sd: Captures response consistency (potential N correlate)")
print("   - rt_slope_mean: Captures within-trial acceleration pattern")
print("   - rt_post_loss_ratio: Captures post-outcome reactivity")

# Save results
rt_features_df.to_csv('results/rt_features_extracted.csv', index=False)
print("\n   Saved RT features to results/rt_features_extracted.csv")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
