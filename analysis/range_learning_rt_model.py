#!/usr/bin/env python3
"""
RT-Integrated Range Learning Model
===================================

Extends the canonical Range Learning model by jointly modeling
reaction times alongside pump decisions.

New RT Parameters:
- tau_base: Baseline RT (ms) - individual response speed
- tau_slope: Within-trial RT change - positive = slowing, negative = speeding
- tau_sigma: RT variability (log-scale) - response consistency
- tau_post_loss: RT multiplicative change after explosions - emotional recovery

Theoretical Motivation:
- tau_base may capture Extraversion (fast responders are more extraverted)
- tau_sigma may capture Neuroticism (anxious people show inconsistent RTs)
- tau_post_loss may capture emotional reactivity/recovery

Author: Range Learning Project
Date: 2025-01-25
"""

import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

__version__ = "1.0.0"
__date__ = "2025-01-25"

# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

# Original behavioral bounds
BEHAVIOR_BOUNDS = {
    'omega_0':     (1, 128),
    'rho_0':       (0.1, 2.0),
    'alpha_minus': (0.001, 0.5),
    'alpha_plus':  (0.001, 0.5),
    'beta':        (0.01, 2.0),
}

# New RT bounds
RT_BOUNDS = {
    'tau_base':      (100, 600),      # Baseline RT in ms
    'tau_slope':     (-5.0, 5.0),     # RT change per pump (ms)
    'tau_sigma':     (0.1, 1.5),      # Log-RT standard deviation
    'tau_post_loss': (0.5, 2.0),      # Multiplicative RT change after loss
}

# Combined bounds for full model
FULL_BOUNDS = {**BEHAVIOR_BOUNDS, **RT_BOUNDS}

RHO_MIN = 0.1
RHO_MAX = 2.0
MAX_PUMPS = 128

# =============================================================================
# OPTIMIZER SETTINGS
# =============================================================================

OPTIMIZER_SETTINGS = {
    'maxiter': 200,
    'polish': True,
    'workers': 1,
    'tol': 0.01,
    'n_starts': 3,  # Fewer starts since more parameters
}


# =============================================================================
# LIKELIHOOD FUNCTIONS
# =============================================================================

def _compute_log_pump_prob(i: int, target: float, beta: float) -> float:
    """Log P(pump at opportunity i) - from original model."""
    delta = i - target
    beta_delta = beta * delta

    if beta_delta > 700:
        return -beta_delta
    elif beta_delta < -700:
        return 0.0
    else:
        return -np.log1p(np.exp(beta_delta))


def _compute_log_stop_prob(i: int, target: float, beta: float) -> float:
    """Log P(stop at opportunity i) - from original model."""
    delta = i - target
    beta_delta = beta * delta

    if beta_delta > 700:
        return 0.0
    elif beta_delta < -700:
        return beta_delta
    else:
        return beta_delta - np.log1p(np.exp(beta_delta))


def behavior_negll(params: np.ndarray,
                   pumps: np.ndarray,
                   exploded: np.ndarray) -> float:
    """
    Behavioral negative log-likelihood (original Range Learning).
    """
    omega_0, rho_0, alpha_minus, alpha_plus, beta = params[:5]

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


def rt_negll(params: np.ndarray,
             rt_data: List[List[float]],
             exploded: np.ndarray) -> float:
    """
    RT negative log-likelihood.

    Model: log(RT_it) ~ Normal(mu_it, tau_sigma)
    where mu_it = log(tau_base + tau_slope * i) + log(tau_post_loss) * prev_explosion

    Parameters
    ----------
    params : array
        [tau_base, tau_slope, tau_sigma, tau_post_loss]
    rt_data : list of lists
        RT values for each pump in each trial (0 = no data)
    exploded : array
        Whether each trial exploded

    Returns
    -------
    float
        Negative log-likelihood
    """
    tau_base, tau_slope, tau_sigma, tau_post_loss = params

    # Parameter validation
    if tau_base <= 0 or tau_sigma <= 0 or tau_post_loss <= 0:
        return 1e10

    total_nll = 0.0
    n_obs = 0
    prev_exploded = 0

    for t, trial_rts in enumerate(rt_data):
        # Filter valid RTs (> 50ms, < 5000ms to remove outliers)
        valid_rts = [(i+1, rt) for i, rt in enumerate(trial_rts)
                     if 50 < rt < 5000]

        if len(valid_rts) == 0:
            prev_exploded = exploded[t] if t < len(exploded) else 0
            continue

        for i, rt in valid_rts:
            # Expected RT (in ms)
            expected_rt = tau_base + tau_slope * i

            # Apply post-loss effect
            if prev_exploded:
                expected_rt *= tau_post_loss

            # Ensure positive
            expected_rt = max(expected_rt, 50)

            # Log-normal likelihood
            log_rt = np.log(rt)
            log_expected = np.log(expected_rt)

            # Log-likelihood of observation
            ll = -0.5 * ((log_rt - log_expected) / tau_sigma) ** 2
            ll -= np.log(tau_sigma)
            ll -= np.log(rt)  # Jacobian for log-normal
            ll -= 0.5 * np.log(2 * np.pi)

            total_nll -= ll
            n_obs += 1

        prev_exploded = exploded[t] if t < len(exploded) else 0

    if n_obs == 0 or not np.isfinite(total_nll):
        return 1e10

    return total_nll


def combined_negll(params: np.ndarray,
                   pumps: np.ndarray,
                   exploded: np.ndarray,
                   rt_data: List[List[float]],
                   rt_weight: float = 1.0) -> float:
    """
    Combined negative log-likelihood for behavior + RT.

    Parameters
    ----------
    params : array
        [omega_0, rho_0, alpha_minus, alpha_plus, beta,
         tau_base, tau_slope, tau_sigma, tau_post_loss]
    pumps, exploded : arrays
        Behavioral data
    rt_data : list of lists
        RT data
    rt_weight : float
        Weight for RT likelihood (1.0 = equal weight)

    Returns
    -------
    float
        Combined negative log-likelihood
    """
    behavior_params = params[:5]
    rt_params = params[5:9]

    nll_behavior = behavior_negll(behavior_params, pumps, exploded)
    nll_rt = rt_negll(rt_params, rt_data, exploded)

    return nll_behavior + rt_weight * nll_rt


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_participant_rt(pumps: np.ndarray,
                       exploded: np.ndarray,
                       rt_data: List[List[float]],
                       seed: Optional[int] = None,
                       rt_weight: float = 1.0) -> Optional[Dict]:
    """
    Fit the RT-integrated Range Learning model to a single participant.

    Parameters
    ----------
    pumps : array
        Number of pumps on each trial
    exploded : array
        Whether each trial exploded
    rt_data : list of lists
        RT values for each pump position in each trial
    seed : int, optional
        Random seed
    rt_weight : float
        Weight for RT component (1.0 = equal weight)

    Returns
    -------
    dict or None
        Fitted parameters and diagnostics
    """
    pumps = np.asarray(pumps)
    exploded = np.asarray(exploded)

    if len(pumps) < 5:
        return None

    # Set up bounds
    bounds_list = [
        FULL_BOUNDS['omega_0'],
        FULL_BOUNDS['rho_0'],
        FULL_BOUNDS['alpha_minus'],
        FULL_BOUNDS['alpha_plus'],
        FULL_BOUNDS['beta'],
        FULL_BOUNDS['tau_base'],
        FULL_BOUNDS['tau_slope'],
        FULL_BOUNDS['tau_sigma'],
        FULL_BOUNDS['tau_post_loss'],
    ]

    rng = np.random.RandomState(seed)

    best_result = None
    best_nll = np.inf

    for start_idx in range(OPTIMIZER_SETTINGS['n_starts']):
        try:
            start_seed = rng.randint(0, 2**31 - 1)

            result = differential_evolution(
                combined_negll,
                bounds=bounds_list,
                args=(pumps, exploded, rt_data, rt_weight),
                maxiter=OPTIMIZER_SETTINGS['maxiter'],
                polish=OPTIMIZER_SETTINGS['polish'],
                workers=OPTIMIZER_SETTINGS['workers'],
                tol=OPTIMIZER_SETTINGS['tol'],
                seed=start_seed,
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result

        except Exception as e:
            continue

    if best_result is None:
        return None

    # Extract parameters
    omega_0, rho_0, alpha_minus, alpha_plus, beta = best_result.x[:5]
    tau_base, tau_slope, tau_sigma, tau_post_loss = best_result.x[5:9]

    # Compute information criteria
    n_params = 9
    n_trials = len(pumps)
    total_pump_opps = sum(int(p) + (1 if not e else 0) for p, e in zip(pumps, exploded))
    n_rt_obs = sum(len([rt for rt in trial if 50 < rt < 5000]) for trial in rt_data)

    nll = best_result.fun
    n_total_obs = total_pump_opps + n_rt_obs
    aic = 2 * nll + 2 * n_params
    bic = 2 * nll + n_params * np.log(n_total_obs)

    # Compute derived measures
    loss_aversion = alpha_minus / alpha_plus if alpha_plus > 0 else np.nan

    # RT-derived personality proxies
    # tau_base: lower = faster = more extraverted?
    # tau_sigma: higher = more variable = more neurotic?
    # tau_post_loss: higher = more reactive = more neurotic?

    return {
        # Behavioral parameters
        'omega_0': omega_0,
        'rho_0': rho_0,
        'alpha_minus': alpha_minus,
        'alpha_plus': alpha_plus,
        'beta': beta,
        'loss_aversion': loss_aversion,

        # RT parameters
        'tau_base': tau_base,
        'tau_slope': tau_slope,
        'tau_sigma': tau_sigma,
        'tau_post_loss': tau_post_loss,

        # Model fit
        'nll': nll,
        'aic': aic,
        'bic': bic,
        'n_trials': n_trials,
        'n_pump_opportunities': total_pump_opps,
        'n_rt_observations': n_rt_obs,
        'success': best_result.success,
    }


def fit_behavior_only(pumps: np.ndarray,
                      exploded: np.ndarray,
                      seed: Optional[int] = None) -> Optional[Dict]:
    """
    Fit original Range Learning model (behavior only) for comparison.
    """
    pumps = np.asarray(pumps)
    exploded = np.asarray(exploded)

    if len(pumps) < 5:
        return None

    bounds_list = [
        BEHAVIOR_BOUNDS['omega_0'],
        BEHAVIOR_BOUNDS['rho_0'],
        BEHAVIOR_BOUNDS['alpha_minus'],
        BEHAVIOR_BOUNDS['alpha_plus'],
        BEHAVIOR_BOUNDS['beta'],
    ]

    rng = np.random.RandomState(seed)

    best_result = None
    best_nll = np.inf

    for start_idx in range(OPTIMIZER_SETTINGS['n_starts']):
        try:
            start_seed = rng.randint(0, 2**31 - 1)

            result = differential_evolution(
                behavior_negll,
                bounds=bounds_list,
                args=(pumps, exploded),
                maxiter=OPTIMIZER_SETTINGS['maxiter'],
                polish=OPTIMIZER_SETTINGS['polish'],
                workers=OPTIMIZER_SETTINGS['workers'],
                tol=OPTIMIZER_SETTINGS['tol'],
                seed=start_seed,
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result

        except Exception:
            continue

    if best_result is None:
        return None

    omega_0, rho_0, alpha_minus, alpha_plus, beta = best_result.x

    n_params = 5
    n_trials = len(pumps)
    total_pump_opps = sum(int(p) + (1 if not e else 0) for p, e in zip(pumps, exploded))

    nll = best_result.fun
    aic = 2 * nll + 2 * n_params
    bic = 2 * nll + n_params * np.log(total_pump_opps)

    return {
        'omega_0': omega_0,
        'rho_0': rho_0,
        'alpha_minus': alpha_minus,
        'alpha_plus': alpha_plus,
        'beta': beta,
        'loss_aversion': alpha_minus / alpha_plus if alpha_plus > 0 else np.nan,
        'nll': nll,
        'aic': aic,
        'bic': bic,
        'n_trials': n_trials,
        'n_pump_opportunities': total_pump_opps,
        'success': best_result.success,
    }


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_participant_data(partid: int,
                          pumps_df,
                          rt_df) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    """
    Load and align behavioral and RT data for a participant.

    Returns
    -------
    pumps : array
    exploded : array
    rt_data : list of lists (RTs for each pump in each trial)
    """
    # Get behavioral data
    part_pumps = pumps_df[pumps_df['partid'] == partid].sort_values('trial')
    pumps = part_pumps['pumps'].values
    exploded = part_pumps['exploded'].values
    trials = part_pumps['trial'].values

    # Get RT data
    part_rt = rt_df[rt_df['partid'] == partid]
    pump_cols = [c for c in rt_df.columns if c.startswith('pump')]

    rt_data = []
    for trial in trials:
        trial_rt_row = part_rt[part_rt['trial'] == trial]
        if len(trial_rt_row) == 0:
            rt_data.append([])
        else:
            row = trial_rt_row.iloc[0]
            rts = [row[c] for c in pump_cols]
            rt_data.append(rts)

    return pumps, exploded, rt_data


# =============================================================================
# MAIN FITTING FUNCTION
# =============================================================================

def fit_all_participants(pumps_df, rt_df,
                         participant_ids: Optional[List] = None,
                         verbose: bool = True) -> Dict:
    """
    Fit RT-integrated model to all participants.

    Returns dict with 'results' (list of dicts) and 'summary' stats.
    """
    if participant_ids is None:
        participant_ids = pumps_df['partid'].unique()

    results = []

    for i, partid in enumerate(participant_ids):
        if verbose and i % 20 == 0:
            print(f"  Fitting participant {i+1}/{len(participant_ids)}...")

        try:
            pumps, exploded, rt_data = load_participant_data(partid, pumps_df, rt_df)

            result = fit_participant_rt(pumps, exploded, rt_data, seed=partid)

            if result is not None:
                result['partid'] = partid
                results.append(result)

        except Exception as e:
            if verbose:
                print(f"  Error for {partid}: {e}")
            continue

    if verbose:
        print(f"  Successfully fit {len(results)}/{len(participant_ids)} participants")

    return {
        'results': results,
        'n_success': len(results),
        'n_total': len(participant_ids),
    }


if __name__ == "__main__":
    print("RT-Integrated Range Learning Model")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"\nBehavioral Parameters: {list(BEHAVIOR_BOUNDS.keys())}")
    print(f"RT Parameters: {list(RT_BOUNDS.keys())}")
    print(f"Total parameters: {len(FULL_BOUNDS)}")
