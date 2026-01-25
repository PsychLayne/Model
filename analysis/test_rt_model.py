#!/usr/bin/env python3
"""
Test RT-Integrated Range Learning Model
========================================

Fits the RT-integrated model to participants and compares
personality correlations with the original model.

Key Questions:
1. Do RT parameters (tau_*) correlate with Big Five?
2. Do RT parameters capture variance beyond behavioral parameters?
3. Does the combined model improve personality prediction?
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import time

# Import the RT model
from range_learning_rt_model import (
    fit_participant_rt,
    fit_behavior_only,
    load_participant_data,
)

print("=" * 70)
print("RT-INTEGRATED RANGE LEARNING MODEL TEST")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

pumps_df = pd.read_csv('data/bart/bart_pumps.csv')
rt_df = pd.read_csv('data/bart/bart_rts.csv')
perso_df = pd.read_csv('data/questionnaires/perso.csv')
quest_df = pd.read_csv('data/questionnaires/quest_scores.csv')

print(f"   Pumps data: {len(pumps_df)} trials")
print(f"   RT data: {len(rt_df)} trials")
print(f"   Personality data: {len(perso_df)} participants")

# Get participants with personality data
perso_partids = set(perso_df['partid'].unique())
pumps_partids = set(pumps_df['partid'].unique())
overlap_partids = list(perso_partids & pumps_partids)
print(f"   Participants with both BART and personality: {len(overlap_partids)}")

# =============================================================================
# 2. FIT MODEL TO SAMPLE
# =============================================================================
print("\n[2] Fitting RT-integrated model to sample...")

# Sample for testing (use all 287 with personality data, but limit for speed)
np.random.seed(42)
sample_size = min(100, len(overlap_partids))  # Start with 100 for speed
sample_partids = np.random.choice(overlap_partids, size=sample_size, replace=False)

print(f"   Fitting {sample_size} participants...")

results_rt = []
results_behavior = []
start_time = time.time()

for i, partid in enumerate(sample_partids):
    if i % 20 == 0:
        elapsed = time.time() - start_time
        print(f"   Progress: {i}/{sample_size} ({elapsed:.1f}s elapsed)")

    try:
        pumps, exploded, rt_data = load_participant_data(partid, pumps_df, rt_df)

        # Fit RT-integrated model
        result_rt = fit_participant_rt(pumps, exploded, rt_data, seed=partid)
        if result_rt is not None:
            result_rt['partid'] = partid
            results_rt.append(result_rt)

        # Fit behavior-only model for comparison
        result_beh = fit_behavior_only(pumps, exploded, seed=partid)
        if result_beh is not None:
            result_beh['partid'] = partid
            results_behavior.append(result_beh)

    except Exception as e:
        continue

elapsed = time.time() - start_time
print(f"   Completed in {elapsed:.1f}s")
print(f"   RT model: {len(results_rt)} successful fits")
print(f"   Behavior model: {len(results_behavior)} successful fits")

# Convert to DataFrames
rt_results_df = pd.DataFrame(results_rt)
beh_results_df = pd.DataFrame(results_behavior)

# =============================================================================
# 3. PARAMETER SUMMARY
# =============================================================================
print("\n[3] Parameter summary (RT-integrated model)...")

print("\n   Behavioral parameters:")
for param in ['omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'beta', 'loss_aversion']:
    if param in rt_results_df.columns:
        print(f"   {param:15s}: M = {rt_results_df[param].mean():>7.3f}, SD = {rt_results_df[param].std():>7.3f}")

print("\n   RT parameters:")
for param in ['tau_base', 'tau_slope', 'tau_sigma', 'tau_post_loss']:
    if param in rt_results_df.columns:
        print(f"   {param:15s}: M = {rt_results_df[param].mean():>7.3f}, SD = {rt_results_df[param].std():>7.3f}")

# =============================================================================
# 4. CORRELATE WITH PERSONALITY
# =============================================================================
print("\n[4] Correlating parameters with Big Five personality...")

# Merge with personality
rt_perso = rt_results_df.merge(perso_df, on='partid')
beh_perso = beh_results_df.merge(perso_df, on='partid')

print(f"   RT model with personality: N = {len(rt_perso)}")
print(f"   Behavior model with personality: N = {len(beh_perso)}")

big5 = ['NEO_E', 'NEO_A', 'NEO_C', 'NEO_N', 'NEO_O']
big5_names = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

# RT parameters
rt_params = ['tau_base', 'tau_slope', 'tau_sigma', 'tau_post_loss']

# Behavioral parameters
beh_params = ['alpha_plus', 'alpha_minus', 'omega_0', 'rho_0', 'loss_aversion']

print("\n   RT Parameters × Big Five:")
print("   " + "-" * 85)
print(f"   {'Parameter':<18} {'E':>12} {'A':>12} {'C':>12} {'N':>12} {'O':>12}")
print("   " + "-" * 85)

rt_significant = []
for param in rt_params:
    if param not in rt_perso.columns:
        continue
    row = f"   {param:<18}"
    for trait in big5:
        mask = rt_perso[param].notna() & rt_perso[trait].notna()
        if mask.sum() < 20:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(rt_perso.loc[mask, param], rt_perso.loc[mask, trait])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.3f}{sig:<3}"
        if p < .05:
            rt_significant.append({'param': param, 'trait': trait, 'r': r, 'p': p})
    print(row)
print("   " + "-" * 85)

print("\n   Behavioral Parameters × Big Five (from RT-integrated model):")
print("   " + "-" * 85)
print(f"   {'Parameter':<18} {'E':>12} {'A':>12} {'C':>12} {'N':>12} {'O':>12}")
print("   " + "-" * 85)

beh_significant = []
for param in beh_params:
    if param not in rt_perso.columns:
        continue
    row = f"   {param:<18}"
    for trait in big5:
        mask = rt_perso[param].notna() & rt_perso[trait].notna()
        if mask.sum() < 20:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(rt_perso.loc[mask, param], rt_perso.loc[mask, trait])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.3f}{sig:<3}"
        if p < .05:
            beh_significant.append({'param': param, 'trait': trait, 'r': r, 'p': p})
    print(row)
print("   " + "-" * 85)

# =============================================================================
# 5. COMPARE WITH BEHAVIOR-ONLY MODEL
# =============================================================================
print("\n[5] Comparing with behavior-only model...")

print("\n   Behavioral Parameters × Big Five (behavior-only model):")
print("   " + "-" * 85)
print(f"   {'Parameter':<18} {'E':>12} {'A':>12} {'C':>12} {'N':>12} {'O':>12}")
print("   " + "-" * 85)

for param in beh_params:
    if param not in beh_perso.columns:
        continue
    row = f"   {param:<18}"
    for trait in big5:
        mask = beh_perso[param].notna() & beh_perso[trait].notna()
        if mask.sum() < 20:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(beh_perso.loc[mask, param], beh_perso.loc[mask, trait])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.3f}{sig:<3}"
    print(row)
print("   " + "-" * 85)

# =============================================================================
# 6. CORRELATE RT PARAMS WITH QUESTIONNAIRES
# =============================================================================
print("\n[6] RT parameters × Key questionnaire scales...")

rt_quest = rt_results_df.merge(quest_df, on='partid')
print(f"   Sample size: N = {len(rt_quest)}")

key_scales = ['BIS', 'SSSV', 'NUM', 'STAI_trait'] if 'STAI_trait' in quest_df.columns else ['BIS', 'SSSV', 'NUM']

# Check STAI in perso
if 'STAI_trait' in perso_df.columns:
    rt_stai = rt_results_df.merge(perso_df[['partid', 'STAI_trait']], on='partid')
else:
    rt_stai = None

print("\n   RT Parameters × Questionnaires:")
print("   " + "-" * 60)
scales_available = [s for s in ['BIS', 'SSSV', 'NUM'] if s in rt_quest.columns]
header = f"   {'Parameter':<18}" + "".join([f"{s:>12}" for s in scales_available])
print(header)
print("   " + "-" * 60)

for param in rt_params:
    if param not in rt_quest.columns:
        continue
    row = f"   {param:<18}"
    for scale in scales_available:
        mask = rt_quest[param].notna() & rt_quest[scale].notna()
        if mask.sum() < 20:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(rt_quest.loc[mask, param], rt_quest.loc[mask, scale])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.3f}{sig:<3}"
    print(row)
print("   " + "-" * 60)

# =============================================================================
# 7. KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

print("\n[A] Significant RT Parameter Correlations with Big Five:")
if rt_significant:
    for finding in sorted(rt_significant, key=lambda x: abs(x['r']), reverse=True):
        print(f"   {finding['param']} × {finding['trait']}: r = {finding['r']:.3f} (p = {finding['p']:.3f})")
else:
    print("   No significant correlations found")

print("\n[B] Significant Behavioral Parameter Correlations (RT model):")
if beh_significant:
    for finding in sorted(beh_significant, key=lambda x: abs(x['r']), reverse=True):
        print(f"   {finding['param']} × {finding['trait']}: r = {finding['r']:.3f} (p = {finding['p']:.3f})")
else:
    print("   No significant correlations found")

print("\n[C] Model Fit Comparison:")
if 'aic' in rt_results_df.columns and 'aic' in beh_results_df.columns:
    # Match participants
    common = set(rt_results_df['partid']) & set(beh_results_df['partid'])
    rt_aic = rt_results_df[rt_results_df['partid'].isin(common)].set_index('partid')['aic']
    beh_aic = beh_results_df[beh_results_df['partid'].isin(common)].set_index('partid')['aic']

    # Note: RT model has more parameters, so AIC comparison isn't fair
    # Just report raw values
    print(f"   RT model mean AIC: {rt_results_df['aic'].mean():.1f}")
    print(f"   Behavior model mean AIC: {beh_results_df['aic'].mean():.1f}")
    print("   (Note: RT model has 9 params vs 5, so higher AIC expected)")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n[7] Saving results...")

rt_results_df.to_csv('results/rt_model_parameters.csv', index=False)
print("   Saved to results/rt_model_parameters.csv")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
