"""
BART Model - Big 5 Personality Correlation Analysis

This script:
1. Runs the Range Learning model on participants from bart_big5 folder
2. Correlates model parameters with Big 5 personality scores

Note: Using max_pumps = 128 (treating all balloons as 128 max)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "bart_big5"
SELF_REPORTS_FILE = os.path.join(DATA_DIR, "self_reports.csv")
N_PARTICIPANTS = 30  # Number of participants to analyze

# Model parameters for 128 max pumps
MAX_PUMPS = 128
RHO_MIN = 0.1
RHO_MAX = 2.0

# Bounds for 128-pump task
BOUNDS = {
    'omega_0': (1, 128),       # Initial range estimate
    'rho_0': (0.1, 2.0),       # Initial confidence multiplier
    'alpha_minus': (0.001, 0.5),  # Confidence penalty after explosion
    'alpha_plus': (0.001, 0.5),   # Confidence boost after cash-out
    'beta': (0.01, 2.0),       # Response sensitivity
}


def _compute_log_pump_prob(i, target, beta):
    """Compute log P(pump at opportunity i) using logistic function."""
    delta = i - target
    beta_delta = beta * delta

    if beta_delta > 700:
        return -beta_delta
    elif beta_delta < -700:
        return 0.0
    else:
        return -np.log1p(np.exp(beta_delta))


def _compute_log_stop_prob(i, target, beta):
    """Compute log P(stop at opportunity i)."""
    delta = i - target
    beta_delta = beta * delta

    if beta_delta > 700:
        return 0.0
    elif beta_delta < -700:
        return beta_delta
    else:
        return beta_delta - np.log1p(np.exp(beta_delta))


def range_learning_negll(params, pumps, exploded):
    """
    Negative log-likelihood for the Range Learning model.
    """
    omega_0, rho_0, alpha_minus, alpha_plus, beta = params

    if omega_0 <= 0 or beta <= 0:
        return 1e10
    if not (RHO_MIN <= rho_0 <= RHO_MAX):
        return 1e10
    if alpha_minus <= 0 or alpha_plus <= 0:
        return 1e10

    omega = omega_0
    rho = rho_0
    total_nll = 0.0

    for t in range(len(pumps)):
        n_pumps = int(pumps[t])
        exploded_t = exploded[t]

        target = omega * rho

        if n_pumps < 1:
            continue

        trial_ll = 0.0
        for i in range(1, n_pumps + 1):
            trial_ll += _compute_log_pump_prob(i, target, beta)

        if not exploded_t:
            trial_ll += _compute_log_stop_prob(n_pumps + 1, target, beta)

        total_nll -= trial_ll

        if exploded_t:
            rho = np.clip(rho - alpha_minus, RHO_MIN, RHO_MAX)
        else:
            omega = max(omega, n_pumps)
            rho = np.clip(rho + alpha_plus, RHO_MIN, RHO_MAX)

    if not np.isfinite(total_nll):
        return 1e10

    return total_nll


def fit_participant(pumps, exploded, seed=None, n_starts=3):
    """
    Fit the Range Learning model to a single participant.
    """
    pumps = np.asarray(pumps)
    exploded = np.asarray(exploded)

    if len(pumps) < 5:
        return None

    bounds_list = [
        BOUNDS['omega_0'],
        BOUNDS['rho_0'],
        BOUNDS['alpha_minus'],
        BOUNDS['alpha_plus'],
        BOUNDS['beta'],
    ]

    rng = np.random.RandomState(seed)

    best_result = None
    best_nll = np.inf

    for start_idx in range(n_starts):
        try:
            start_seed = rng.randint(0, 2**31 - 1)

            result = differential_evolution(
                range_learning_negll,
                bounds=bounds_list,
                args=(pumps, exploded),
                maxiter=200,
                polish=True,
                workers=1,
                tol=0.01,
                seed=start_seed,
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result

        except Exception:
            continue

    if best_result is None:
        return None

    total_pump_opps = sum(
        int(p) + (1 if not e else 0)
        for p, e in zip(pumps, exploded)
    )

    n_params = 5
    n_trials = len(pumps)
    nll = best_result.fun
    aic = 2 * nll + 2 * n_params
    bic = 2 * nll + n_params * np.log(total_pump_opps)

    omega_0, rho_0, alpha_minus, alpha_plus, beta = best_result.x

    return {
        'omega_0': omega_0,
        'rho_0': rho_0,
        'alpha_minus': alpha_minus,
        'alpha_plus': alpha_plus,
        'beta': beta,
        'nll': nll,
        'aic': aic,
        'bic': bic,
        'n_trials': n_trials,
        'n_pump_opportunities': total_pump_opps,
        'success': best_result.success,
        'loss_aversion': alpha_minus / alpha_plus if alpha_plus > 0 else np.nan,
    }


def parse_bart_file(filepath):
    """
    Parse a BART data file to extract pumps and explosions per trial.
    """
    df = pd.read_csv(filepath)

    df = df.copy()

    current_pumps = 0
    trial_data = []

    for idx, row in df.iterrows():
        level_name = row['Eprime.LevelName']

        if pd.isna(level_name):
            continue

        if 'PumpList' in level_name:
            current_pumps += 1
        elif 'BalloonList' in level_name:
            trial_type = row['type']
            is_explosion = 1 if trial_type == 'exp' else 0

            trial_data.append({
                'pumps': current_pumps,
                'exploded': is_explosion
            })
            current_pumps = 0

    if len(trial_data) == 0:
        return None, None

    trial_df = pd.DataFrame(trial_data)
    return trial_df['pumps'].values, trial_df['exploded'].values


def compute_big5_scores(row):
    """
    Compute Big 5 personality scores from a self-reports row.
    """
    try:
        scores = {
            'Openness': np.mean([row['B5_O_1'], row['B5_O_2'], row['B5_O_3']]),
            'Conscientiousness': np.mean([row['B5_C_1'], row['B5_C_2'], row['B5_C_3']]),
            'Extraversion': np.mean([row['B5_E_1'], row['B5_E_2'], row['B5_E_3']]),
            'Agreeableness': np.mean([row['B5_A_1'], row['B5_A_2'], row['B5_A_3']]),
            'Neuroticism': np.mean([row['B5_N_1'], row['B5_N_2'], row['B5_N_3']])
        }

        if any(np.isnan(v) for v in scores.values()):
            return None

        return scores
    except (KeyError, TypeError):
        return None


def main():
    print("=" * 70)
    print("BART MODEL - BIG 5 PERSONALITY CORRELATION ANALYSIS")
    print("(Using max_pumps = 128, treating all balloons uniformly)")
    print("=" * 70)

    # Load self-reports
    print("\nLoading self-reports data...")
    self_reports = pd.read_csv(SELF_REPORTS_FILE)
    print(f"  Found {len(self_reports)} participants in self-reports")

    # Get BART files and sort numerically
    bart_files = glob.glob(os.path.join(DATA_DIR, "*_BART.csv"))

    # Extract numeric IDs and sort
    file_info = []
    for f in bart_files:
        basename = os.path.basename(f)
        pid = basename.replace('_BART.csv', '')
        try:
            numeric_id = int(pid)
            file_info.append((numeric_id, f))
        except ValueError:
            continue

    file_info.sort(key=lambda x: x[0])

    print(f"  Found {len(file_info)} BART data files with numeric IDs")
    print(f"  ID range: {file_info[0][0]} to {file_info[-1][0]}")

    # Map numeric IDs to self_reports rows (1-indexed to 0-indexed)
    id_to_row = {}
    for i, row in self_reports.iterrows():
        id_to_row[i + 1] = i

    print(f"\nProcessing {min(N_PARTICIPANTS, len(file_info))} participants...")
    print("-" * 70)

    # Store results
    results = []

    # Process participants
    count = 0
    for numeric_id, filepath in file_info:
        if count >= N_PARTICIPANTS:
            break

        # Check if we have a mapping for this ID
        if numeric_id not in id_to_row:
            continue

        row_idx = id_to_row[numeric_id]
        self_report_row = self_reports.iloc[row_idx]

        # Parse BART data
        pumps, exploded = parse_bart_file(filepath)

        if pumps is None or len(pumps) < 5:
            continue

        # Fit the model
        print(f"  Fitting participant {numeric_id}...", end=" ", flush=True)
        result = fit_participant(pumps, exploded, seed=42)

        if result is None:
            print("FAILED")
            continue

        # Get Big 5 scores
        big5 = compute_big5_scores(self_report_row)

        if big5 is None:
            print(f"OK - LA={result['loss_aversion']:.2f} (no Big5 data)")
            continue

        print(f"OK - LA={result['loss_aversion']:.2f}, ω₀={result['omega_0']:.1f}")

        # Combine results
        participant_result = {
            'id': numeric_id,
            'n_trials': result['n_trials'],
            'omega_0': result['omega_0'],
            'rho_0': result['rho_0'],
            'alpha_minus': result['alpha_minus'],
            'alpha_plus': result['alpha_plus'],
            'beta': result['beta'],
            'loss_aversion': result['loss_aversion'],
            'mean_pumps': np.mean(pumps),
            'explosion_rate': np.mean(exploded),
            **big5
        }
        results.append(participant_result)
        count += 1

    print("-" * 70)
    print(f"Successfully processed {len(results)} participants with complete data")

    if len(results) < 5:
        print("\nInsufficient participants for correlation analysis.")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print descriptive statistics
    print("\n" + "=" * 70)
    print("MODEL PARAMETER DESCRIPTIVE STATISTICS")
    print("=" * 70)

    model_params = ['omega_0', 'rho_0', 'alpha_minus', 'alpha_plus', 'beta', 'loss_aversion']
    for param in model_params:
        values = results_df[param]
        print(f"\n{param}:")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  SD:   {values.std():.4f}")
        print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

    # Big 5 descriptives
    print("\n" + "=" * 70)
    print("BIG 5 PERSONALITY DESCRIPTIVE STATISTICS")
    print("=" * 70)

    big5_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    for trait in big5_traits:
        values = results_df[trait]
        print(f"\n{trait}:")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  SD:   {values.std():.2f}")
        print(f"  Range: [{values.min():.2f}, {values.max():.2f}]")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: MODEL PARAMETERS × BIG 5")
    print("=" * 70)

    print("\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "", "O", "C", "E", "A", "N"
    ))
    print("-" * 70)

    correlation_results = []

    for param in model_params:
        row_str = f"{param:<20}"
        for trait in big5_traits:
            r, p = stats.pearsonr(results_df[param], results_df[trait])

            sig = ""
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            elif p < 0.10:
                sig = "†"

            row_str += f" {r:>7.3f}{sig}"

            correlation_results.append({
                'parameter': param,
                'trait': trait,
                'r': r,
                'p': p
            })
        print(row_str)

    print("\n† p < .10, * p < .05, ** p < .01, *** p < .001")
    print(f"N = {len(results_df)}")

    # Highlight significant correlations
    print("\n" + "=" * 70)
    print("SIGNIFICANT CORRELATIONS (p < .05)")
    print("=" * 70)

    corr_df = pd.DataFrame(correlation_results)
    sig_corr = corr_df[corr_df['p'] < 0.05].sort_values('p')

    if len(sig_corr) == 0:
        print("\nNo significant correlations at p < .05")
    else:
        for _, row in sig_corr.iterrows():
            print(f"\n{row['parameter']} × {row['trait']}:")
            print(f"  r = {row['r']:.3f}, p = {row['p']:.4f}")

    # Marginally significant
    print("\n" + "=" * 70)
    print("MARGINALLY SIGNIFICANT CORRELATIONS (.05 < p < .10)")
    print("=" * 70)

    marginal_corr = corr_df[(corr_df['p'] >= 0.05) & (corr_df['p'] < 0.10)].sort_values('p')

    if len(marginal_corr) == 0:
        print("\nNo marginally significant correlations")
    else:
        for _, row in marginal_corr.iterrows():
            print(f"\n{row['parameter']} × {row['trait']}:")
            print(f"  r = {row['r']:.3f}, p = {row['p']:.4f}")

    # Behavioral measures vs Big 5
    print("\n" + "=" * 70)
    print("BEHAVIORAL MEASURES × BIG 5 CORRELATIONS")
    print("=" * 70)

    behavioral = ['mean_pumps', 'explosion_rate']

    print("\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "", "O", "C", "E", "A", "N"
    ))
    print("-" * 70)

    for param in behavioral:
        row_str = f"{param:<20}"
        for trait in big5_traits:
            r, p = stats.pearsonr(results_df[param], results_df[trait])
            sig = ""
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            elif p < 0.10:
                sig = "†"
            row_str += f" {r:>7.3f}{sig}"
        print(row_str)

    print("\n† p < .10, * p < .05, ** p < .01, *** p < .001")

    # Save results
    results_df.to_csv('bart_big5_model_results.csv', index=False)
    print(f"\nResults saved to bart_big5_model_results.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
