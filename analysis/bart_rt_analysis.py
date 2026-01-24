#!/usr/bin/env python3
"""
BART Reaction Time Analysis
===========================
Tests whether RT patterns correlate with personality and model parameters.

Key RT metrics to extract:
1. Mean RT - overall response speed
2. RT variability (SD, CV) - response consistency
3. RT trend - speeding up or slowing down within trials
4. First pump RT - initial deliberation
5. Late pump RT - decision near threshold
6. RT change after explosions - emotional recovery
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BART REACTION TIME ANALYSIS")
print("Testing RT patterns as personality predictors")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# RT data
rt_df = pd.read_csv('data/bart/bart_rts.csv')
print(f"   RT data: {len(rt_df)} trial-rows")

# Pump data (to know which trials exploded)
pumps_df = pd.read_csv('data/bart/bart_pumps.csv')
print(f"   Pump data: {len(pumps_df)} trials")

# Personality data
perso_df = pd.read_csv('data/questionnaires/perso.csv')
print(f"   Personality data: {len(perso_df)} participants")

# Model parameters
model_df = pd.read_csv('data/model_parameters/test_results.csv')
print(f"   Model parameters: {len(model_df)} participants")

# Questionnaire data
quest_df = pd.read_csv('data/questionnaires/quest_scores.csv')
print(f"   Questionnaire data: {len(quest_df)} participants")

# =============================================================================
# 2. EXTRACT RT METRICS PER PARTICIPANT
# =============================================================================
print("\n[2] Extracting RT metrics per participant...")

def extract_rt_metrics(participant_rt_df):
    """Extract comprehensive RT metrics for one participant."""
    metrics = {}

    all_rts = []
    first_pump_rts = []
    late_pump_rts = []  # Last 5 pumps before stopping
    rt_trends = []  # Slope within each trial

    for _, row in participant_rt_df.iterrows():
        # Get non-zero RTs for this trial
        rt_cols = [c for c in row.index if c.startswith('pump')]
        trial_rts = [row[c] for c in rt_cols if row[c] > 0]

        if len(trial_rts) < 3:
            continue

        all_rts.extend(trial_rts)
        first_pump_rts.append(trial_rts[0])

        # Late pumps (last 5 or fewer)
        late_pump_rts.extend(trial_rts[-5:])

        # RT trend within trial (slope)
        if len(trial_rts) >= 5:
            x = np.arange(len(trial_rts))
            slope, _, _, _, _ = stats.linregress(x, trial_rts)
            rt_trends.append(slope)

    if len(all_rts) < 10:
        return None

    # Basic metrics
    metrics['rt_mean'] = np.mean(all_rts)
    metrics['rt_sd'] = np.std(all_rts)
    metrics['rt_cv'] = metrics['rt_sd'] / metrics['rt_mean'] if metrics['rt_mean'] > 0 else np.nan
    metrics['rt_median'] = np.median(all_rts)

    # First pump RT (deliberation)
    metrics['rt_first_mean'] = np.mean(first_pump_rts) if first_pump_rts else np.nan
    metrics['rt_first_sd'] = np.std(first_pump_rts) if len(first_pump_rts) > 1 else np.nan

    # Late pump RT (near threshold)
    metrics['rt_late_mean'] = np.mean(late_pump_rts) if late_pump_rts else np.nan

    # RT trend (negative = speeding up)
    metrics['rt_trend_mean'] = np.mean(rt_trends) if rt_trends else np.nan
    metrics['rt_trend_sd'] = np.std(rt_trends) if len(rt_trends) > 1 else np.nan

    # Skewness (positive = occasional long RTs)
    metrics['rt_skew'] = stats.skew(all_rts) if len(all_rts) > 10 else np.nan

    return metrics

# Extract metrics for each participant
rt_metrics_list = []
for partid in rt_df['partid'].unique():
    part_rt = rt_df[rt_df['partid'] == partid]
    metrics = extract_rt_metrics(part_rt)
    if metrics:
        metrics['partid'] = partid
        rt_metrics_list.append(metrics)

rt_metrics_df = pd.DataFrame(rt_metrics_list)
print(f"   Extracted RT metrics for {len(rt_metrics_df)} participants")

# Show summary
print("\n   RT Metrics Summary:")
for col in ['rt_mean', 'rt_sd', 'rt_cv', 'rt_first_mean', 'rt_late_mean', 'rt_trend_mean']:
    if col in rt_metrics_df.columns:
        print(f"   {col:20s}: M = {rt_metrics_df[col].mean():.1f}, SD = {rt_metrics_df[col].std():.1f}")

# =============================================================================
# 3. ADD POST-EXPLOSION RT METRICS
# =============================================================================
print("\n[3] Computing post-explosion RT changes...")

# Merge RT with pump data to know which trials exploded
rt_pump_merged = rt_df.merge(pumps_df[['partid', 'trial', 'exploded', 'pumps']], on=['partid', 'trial'])

def compute_post_explosion_rt(part_rt_pump):
    """Compute RT changes after explosions vs successes."""
    part_rt_pump = part_rt_pump.sort_values('trial')

    post_exp_first_rts = []
    post_success_first_rts = []

    trials = part_rt_pump['trial'].values
    exploded = part_rt_pump['exploded'].values

    for i in range(1, len(trials)):
        # Get first pump RT for current trial
        row = part_rt_pump.iloc[i]
        current_first_rt = row['pump1']
        if current_first_rt <= 0:
            continue

        # Was previous trial an explosion?
        if exploded[i-1] == 1:
            post_exp_first_rts.append(current_first_rt)
        else:
            post_success_first_rts.append(current_first_rt)

    result = {}
    result['rt_post_explosion'] = np.mean(post_exp_first_rts) if post_exp_first_rts else np.nan
    result['rt_post_success'] = np.mean(post_success_first_rts) if post_success_first_rts else np.nan

    if result['rt_post_explosion'] and result['rt_post_success']:
        result['rt_explosion_effect'] = result['rt_post_explosion'] - result['rt_post_success']
    else:
        result['rt_explosion_effect'] = np.nan

    return result

post_exp_metrics = []
for partid in rt_pump_merged['partid'].unique():
    part_data = rt_pump_merged[rt_pump_merged['partid'] == partid]
    metrics = compute_post_explosion_rt(part_data)
    metrics['partid'] = partid
    post_exp_metrics.append(metrics)

post_exp_df = pd.DataFrame(post_exp_metrics)
rt_metrics_df = rt_metrics_df.merge(post_exp_df, on='partid', how='left')

print(f"   RT after explosion: M = {rt_metrics_df['rt_post_explosion'].mean():.1f} ms")
print(f"   RT after success:   M = {rt_metrics_df['rt_post_success'].mean():.1f} ms")
print(f"   Explosion effect:   M = {rt_metrics_df['rt_explosion_effect'].mean():.1f} ms (+ = slower after loss)")

# =============================================================================
# 4. CORRELATE RT WITH PERSONALITY (BIG FIVE)
# =============================================================================
print("\n[4] Correlating RT metrics with Big Five personality...")

# Merge RT metrics with personality
rt_perso = rt_metrics_df.merge(perso_df, on='partid', how='inner')
print(f"   Merged sample: N = {len(rt_perso)}")

rt_vars = ['rt_mean', 'rt_sd', 'rt_cv', 'rt_first_mean', 'rt_late_mean',
           'rt_trend_mean', 'rt_skew', 'rt_explosion_effect']
big5_vars = ['NEO_E', 'NEO_A', 'NEO_C', 'NEO_N', 'NEO_O']
big5_names = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

print("\n   RT × Big Five Correlations:")
print("   " + "-" * 85)
print(f"   {'RT Metric':<25} {'E':>10} {'A':>10} {'C':>10} {'N':>10} {'O':>10}")
print("   " + "-" * 85)

significant_findings = []
for rt_var in rt_vars:
    if rt_var not in rt_perso.columns:
        continue
    row = f"   {rt_var:<25}"
    for big5_var in big5_vars:
        if big5_var not in rt_perso.columns:
            row += f"{'---':>10}"
            continue
        mask = rt_perso[rt_var].notna() & rt_perso[big5_var].notna()
        if mask.sum() < 30:
            row += f"{'---':>10}"
            continue
        r, p = stats.pearsonr(rt_perso.loc[mask, rt_var], rt_perso.loc[mask, big5_var])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>7.2f}{sig:<3}"
        if p < .05:
            significant_findings.append({
                'rt_var': rt_var,
                'personality': big5_var,
                'r': r,
                'p': p,
                'n': mask.sum()
            })
    print(row)

print("   " + "-" * 85)
print("   * p<.05, ** p<.01, *** p<.001")

# =============================================================================
# 5. CORRELATE RT WITH MODEL PARAMETERS
# =============================================================================
print("\n[5] Correlating RT metrics with model parameters...")

# Merge RT metrics with model parameters
rt_model = rt_metrics_df.merge(model_df, on='partid', how='inner')
print(f"   Merged sample: N = {len(rt_model)}")

model_params = ['omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'sigma', 'mean_pumps']

print("\n   RT × Model Parameter Correlations:")
print("   " + "-" * 95)
print(f"   {'RT Metric':<25} {'omega_0':>12} {'rho_0':>12} {'alpha_minus':>12} {'alpha_plus':>12} {'mean_pumps':>12}")
print("   " + "-" * 95)

for rt_var in rt_vars:
    if rt_var not in rt_model.columns:
        continue
    row = f"   {rt_var:<25}"
    for param in model_params:
        if param not in rt_model.columns:
            row += f"{'---':>12}"
            continue
        mask = rt_model[rt_var].notna() & rt_model[param].notna()
        if mask.sum() < 20:
            row += f"{'---':>12}"
            continue
        r, p = stats.pearsonr(rt_model.loc[mask, rt_var], rt_model.loc[mask, param])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>9.2f}{sig:<3}"
    print(row)

print("   " + "-" * 95)

# =============================================================================
# 6. CORRELATE RT WITH QUESTIONNAIRES (IMPULSIVITY, SENSATION SEEKING)
# =============================================================================
print("\n[6] Correlating RT metrics with key questionnaire measures...")

# Merge with questionnaires
rt_quest = rt_metrics_df.merge(quest_df, on='partid', how='inner')
print(f"   Merged sample: N = {len(rt_quest)}")

key_scales = ['BIS', 'SSSV', 'SSexp', 'SStas', 'NUM']
scale_names = {'BIS': 'Impulsivity', 'SSSV': 'Sensation Seeking',
               'SSexp': 'Experience Seeking', 'SStas': 'Thrill/Adventure',
               'NUM': 'Numeracy'}

print("\n   RT × Key Scales Correlations:")
print("   " + "-" * 80)
print(f"   {'RT Metric':<25} {'BIS':>10} {'SSSV':>10} {'SSexp':>10} {'SStas':>10} {'NUM':>10}")
print("   " + "-" * 80)

for rt_var in rt_vars:
    if rt_var not in rt_quest.columns:
        continue
    row = f"   {rt_var:<25}"
    for scale in key_scales:
        if scale not in rt_quest.columns:
            row += f"{'---':>10}"
            continue
        mask = rt_quest[rt_var].notna() & rt_quest[scale].notna()
        if mask.sum() < 30:
            row += f"{'---':>10}"
            continue
        r, p = stats.pearsonr(rt_quest.loc[mask, rt_var], rt_quest.loc[mask, scale])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        row += f"{r:>7.2f}{sig:<3}"
    print(row)

print("   " + "-" * 80)

# =============================================================================
# 7. CORRELATE RT WITH ANXIETY (STAI)
# =============================================================================
print("\n[7] RT × Anxiety (STAI) correlations...")

if 'STAI_trait' in rt_perso.columns:
    print("\n   RT × STAI_trait (Trait Anxiety):")
    for rt_var in rt_vars:
        if rt_var not in rt_perso.columns:
            continue
        mask = rt_perso[rt_var].notna() & rt_perso['STAI_trait'].notna()
        if mask.sum() < 30:
            continue
        r, p = stats.pearsonr(rt_perso.loc[mask, rt_var], rt_perso.loc[mask, 'STAI_trait'])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        if abs(r) > 0.08 or p < 0.1:
            print(f"   {rt_var:<25} r = {r:>6.3f} {sig:<3} (p = {p:.3f}, n = {mask.sum()})")

# =============================================================================
# 8. SUMMARY OF SIGNIFICANT FINDINGS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Significant RT-Personality Correlations")
print("=" * 70)

if significant_findings:
    sig_df = pd.DataFrame(significant_findings)
    sig_df = sig_df.sort_values('p')
    print("\n   Rank | RT Variable              | Personality    |    r    |   p   |  n")
    print("   " + "-" * 75)
    for i, row in sig_df.iterrows():
        print(f"   {sig_df.index.get_loc(i)+1:4d} | {row['rt_var']:<24} | {row['personality']:<14} | {row['r']:>6.3f} | {row['p']:.3f} | {row['n']}")
else:
    print("\n   No significant RT-Big Five correlations found.")

# =============================================================================
# 9. THEORETICAL INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Key Questions Answered:
-----------------------

1. Does RT variability correlate with Neuroticism?
   → Test: rt_cv × NEO_N (anxious people show more variable responding?)

2. Does deliberation time correlate with Conscientiousness?
   → Test: rt_first_mean × NEO_C (careful people think longer before starting?)

3. Does post-explosion RT change correlate with emotional stability?
   → Test: rt_explosion_effect × NEO_N (neurotic people slow down more after loss?)

4. Does RT correlate with impulsivity?
   → Test: rt_mean × BIS (impulsive people respond faster?)

5. Do RT patterns predict model parameters?
   → Test: rt_trend × learning rates (RT patterns reveal learning style?)
""")

# Save results
rt_metrics_df.to_csv('results/rt_metrics.csv', index=False)
print("\n   Saved RT metrics to results/rt_metrics.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
