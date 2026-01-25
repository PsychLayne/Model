#!/usr/bin/env python3
"""
Fit RT Model to Small Sample
=============================

Fits the full RT-integrated model to 30 participants
to validate parameter recovery and personality correlations.
"""

import pandas as pd
import numpy as np
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

from range_learning_rt_model import (
    fit_participant_rt,
    load_participant_data,
)

print("=" * 70)
print("RT MODEL FITTING (SMALL SAMPLE)")
print("=" * 70)

# Load data
pumps_df = pd.read_csv('data/bart/bart_pumps.csv')
rt_df = pd.read_csv('data/bart/bart_rts.csv')
perso_df = pd.read_csv('data/questionnaires/perso.csv')
model_params = pd.read_csv('data/model_parameters/test_results.csv')

# Get participants with both personality and existing model params
model_partids = set(model_params['partid'])
perso_partids = set(perso_df['partid'])
target_partids = list(model_partids & perso_partids)

print(f"\nParticipants with model params + personality: {len(target_partids)}")

# Sample 30 for fitting
np.random.seed(42)
sample_size = 30
sample_partids = np.random.choice(target_partids, size=min(sample_size, len(target_partids)), replace=False)

print(f"Fitting {len(sample_partids)} participants...")

# Fit with reduced iterations for speed
results = []
start_time = time.time()

for i, partid in enumerate(sample_partids):
    if i % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  Progress: {i}/{len(sample_partids)} ({elapsed:.1f}s)")

    try:
        pumps, exploded, rt_data = load_participant_data(partid, pumps_df, rt_df)

        # Fit with lower weight on RT to speed up
        result = fit_participant_rt(pumps, exploded, rt_data, seed=partid, rt_weight=0.5)

        if result is not None:
            result['partid'] = partid
            results.append(result)
    except Exception as e:
        print(f"  Error for {partid}: {e}")
        continue

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f}s ({len(results)} successful fits)")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("FITTED PARAMETERS")
print("=" * 70)

print("\nBehavioral parameters:")
for p in ['omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'loss_aversion']:
    if p in results_df.columns:
        print(f"  {p:15s}: M = {results_df[p].mean():.3f}, SD = {results_df[p].std():.3f}")

print("\nRT parameters:")
for p in ['tau_base', 'tau_slope', 'tau_sigma', 'tau_post_loss']:
    if p in results_df.columns:
        print(f"  {p:15s}: M = {results_df[p].mean():.3f}, SD = {results_df[p].std():.3f}")

# =============================================================================
# CORRELATIONS WITH PERSONALITY
# =============================================================================
print("\n" + "=" * 70)
print("PERSONALITY CORRELATIONS")
print("=" * 70)

merged = results_df.merge(perso_df, on='partid')
print(f"\nSample with personality: N = {len(merged)}")

big5 = ['NEO_E', 'NEO_A', 'NEO_C', 'NEO_N', 'NEO_O']
rt_params = ['tau_base', 'tau_slope', 'tau_sigma', 'tau_post_loss']
beh_params = ['alpha_plus', 'alpha_minus', 'loss_aversion', 'omega_0']

print("\nRT Parameters × Big Five:")
print("-" * 75)
print(f"{'Parameter':<18} {'E':>10} {'A':>10} {'C':>10} {'N':>10} {'O':>10}")
print("-" * 75)

for param in rt_params:
    if param not in merged.columns:
        continue
    row = f"{param:<18}"
    for trait in big5:
        mask = merged[param].notna() & merged[trait].notna()
        if mask.sum() < 10:
            row += f"{'---':>10}"
            continue
        r, p = stats.pearsonr(merged.loc[mask, param], merged.loc[mask, trait])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .10 else ''
        row += f"{r:>7.2f}{sig:<3}"
    print(row)

print("\nBehavioral Parameters × Big Five:")
print("-" * 75)
for param in beh_params:
    if param not in merged.columns:
        continue
    row = f"{param:<18}"
    for trait in big5:
        mask = merged[param].notna() & merged[trait].notna()
        if mask.sum() < 10:
            row += f"{'---':>10}"
            continue
        r, p = stats.pearsonr(merged.loc[mask, param], merged.loc[mask, trait])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .10 else ''
        row += f"{r:>7.2f}{sig:<3}"
    print(row)
print("-" * 75)
print("† p < .10, * p < .05, ** p < .01, *** p < .001")

# =============================================================================
# COMPARE WITH EXISTING MODEL PARAMS
# =============================================================================
print("\n" + "=" * 70)
print("CONVERGENT VALIDITY WITH EXISTING MODEL")
print("=" * 70)

# Merge with existing model params
comparison = results_df.merge(model_params, on='partid', suffixes=('_rt', '_orig'))

if len(comparison) > 10:
    print(f"\nCorrelations between RT model and original model (N = {len(comparison)}):")

    for param in ['omega_0', 'alpha_minus', 'alpha_plus']:
        rt_col = f'{param}_rt'
        orig_col = f'{param}_orig'
        if rt_col in comparison.columns and orig_col in comparison.columns:
            mask = comparison[rt_col].notna() & comparison[orig_col].notna()
            if mask.sum() > 5:
                r, p = stats.pearsonr(comparison.loc[mask, rt_col], comparison.loc[mask, orig_col])
                print(f"  {param}: r = {r:.3f} (p = {p:.3f})")

# Save results
results_df.to_csv('results/rt_model_fitted_sample.csv', index=False)
print(f"\nSaved fitted parameters to results/rt_model_fitted_sample.csv")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
