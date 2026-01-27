"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RANGE LEARNING MODEL (PUMP-LEVEL)                         ║
║                      Canonical Implementation                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is the SINGLE SOURCE OF TRUTH for the Pump-Level Range Learning model.
All analysis scripts should import from this file.

DO NOT copy model code into other scripts. Import from here:

    from range_learning_pump_level_model import (
        fit_participant,
        simulate_participant,
        range_learning_negll,
        BOUNDS,
        RHO_MIN,
        RHO_MAX,
    )

Model Description
-----------------
The Range Learning model captures how people learn safe behavior limits in the
Balloon Analogue Risk Task (BART). Key mechanisms:

  - ω (omega): Learned safe range, updated via "ratchet" (only increases)
  - ρ (rho): Confidence multiplier, adjusts up/down based on outcomes
  - Target each trial = ω × ρ

After each trial:
  - Cash-out: ω = max(ω, pumps), ρ increases by α⁺
  - Explosion: ω unchanged, ρ decreases by α⁻

Loss aversion emerges when α⁻ > α⁺ (stronger response to losses than gains).

Pump-Level Decision Model
-------------------------
Unlike the original formulation (which uses Gaussian noise around the target),
this version models each pump as a binary decision using a logistic function:

    P(pump at i) = 1 / (1 + exp(β × (i - target)))

This formulation:
  - Models the actual decision structure (repeated pump/stop choices)
  - Handles explosions naturally (no counterfactual inference needed)
  - Enables valid comparison with other pump-level models (e.g., BSR)
  - Provides better recovery of the loss aversion ratio

Validation
----------
  - Parameter recovery: mean r = 0.81 (N=100 simulations)
  - Loss aversion ratio recovery: r = 0.96 (+21% vs. original formulation)
  - Classification accuracy (LA > 1): 96%

Comparison with Original Formulation
------------------------------------
  - Original (σ): Per-balloon likelihood, Gaussian noise
  - Pump-level (β): Per-pump likelihood, logistic response function
  
Both yield consistent loss aversion estimates (~2.0), but the pump-level
version provides substantially better recovery of individual differences.

Version History
---------------
  v1.0 (2026-01-23): Initial canonical version with validated implementation

Author: Range Learning Project
"""

import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.0.0"
__date__ = "2026-01-23"


# =============================================================================
# CANONICAL PARAMETER BOUNDS
# =============================================================================
# These bounds have been validated through parameter recovery (mean r = 0.81).
# DO NOT MODIFY without re-running parameter recovery.
#
# CRITICAL: α⁻ and α⁺ have EQUAL bounds to ensure any observed asymmetry
# reflects genuine behavioral differences, not constraint artifacts.
# =============================================================================

BOUNDS = {
    'omega_0':     (1, 128),      # Initial range estimate (full pump range)
    'rho_0':       (0.1, 2.0),    # Initial confidence multiplier
    'alpha_minus': (0.001, 0.5),  # Confidence penalty after explosion
    'alpha_plus':  (0.001, 0.5),  # Confidence boost after cash-out (EQUAL to α⁻)
    'beta':        (0.01, 2.0),   # Response sensitivity (logistic steepness)
}

# Runtime bounds for rho (consistent with fitting bounds)
RHO_MIN = 0.1
RHO_MAX = 2.0

# BART task parameter
MAX_PUMPS = 128


# =============================================================================
# OPTIMIZER SETTINGS
# =============================================================================
# Differential evolution is used for reliable global optimization.
# These settings balance accuracy and runtime.
# =============================================================================

OPTIMIZER_SETTINGS = {
    'method': 'differential_evolution',
    'maxiter': 200,
    'polish': True,      # Refine with L-BFGS-B at the end
    'workers': 1,        # Parallelism handled at participant level
    'tol': 0.01,
    'n_starts': 5,       # Number of random restarts
}


# =============================================================================
# CORE MODEL FUNCTIONS
# =============================================================================

def _compute_log_pump_prob(i: int, target: float, beta: float) -> float:
    """
    Compute log P(pump at opportunity i) using logistic function.
    
    P(pump) = 1 / (1 + exp(β × (i - target)))
    log P(pump) = -log(1 + exp(β × (i - target)))
    
    Uses numerically stable computation for extreme values.
    """
    delta = i - target
    beta_delta = beta * delta
    
    if beta_delta > 700:
        return -beta_delta
    elif beta_delta < -700:
        return 0.0
    else:
        return -np.log1p(np.exp(beta_delta))


def _compute_log_stop_prob(i: int, target: float, beta: float) -> float:
    """
    Compute log P(stop at opportunity i) = log(1 - P(pump)).
    
    P(stop) = exp(β × (i - target)) / (1 + exp(β × (i - target)))
    log P(stop) = β × (i - target) - log(1 + exp(β × (i - target)))
    
    Uses numerically stable computation for extreme values.
    """
    delta = i - target
    beta_delta = beta * delta
    
    if beta_delta > 700:
        return 0.0
    elif beta_delta < -700:
        return beta_delta
    else:
        return beta_delta - np.log1p(np.exp(beta_delta))


def range_learning_negll(params: np.ndarray, 
                         pumps: np.ndarray, 
                         exploded: np.ndarray) -> float:
    """
    Negative log-likelihood for the Pump-Level Range Learning model.
    
    Parameters
    ----------
    params : array-like
        [omega_0, rho_0, alpha_minus, alpha_plus, beta]
    pumps : array-like
        Number of pumps on each trial
    exploded : array-like
        Whether balloon exploded (1) or cashed out (0) on each trial
    
    Returns
    -------
    float
        Negative log-likelihood (to be minimized)
    
    Notes
    -----
    Likelihood model:
      - Each pump i: P(pump) = logistic(target - i, β)
      - Cash-out at n: ∏_{i=1}^{n} P(pump_i) × P(stop_{n+1})
      - Explosion at n: ∏_{i=1}^{n} P(pump_i)  [no stopping term]
    
    The explosion case has no stopping probability because the participant
    did not choose to stop—the balloon exploded. This is more principled than
    the original formulation which uses a survival function to infer the
    counterfactual "would have pumped at least n times."
    """
    omega_0, rho_0, alpha_minus, alpha_plus, beta = params
    
    # Parameter validation
    if omega_0 <= 0 or beta <= 0:
        return 1e10
    if not (RHO_MIN <= rho_0 <= RHO_MAX):
        return 1e10
    if alpha_minus <= 0 or alpha_plus <= 0:
        return 1e10
    
    # Initialize state
    omega = omega_0
    rho = rho_0
    total_nll = 0.0
    
    for t in range(len(pumps)):
        n_pumps = int(pumps[t])
        exploded_t = exploded[t]
        
        # Target for this trial
        target = omega * rho
        
        # Skip if no pumps (safety check)
        if n_pumps < 1:
            continue
        
        # Likelihood for pumping decisions (pumps 1 through n_pumps)
        trial_ll = 0.0
        for i in range(1, n_pumps + 1):
            trial_ll += _compute_log_pump_prob(i, target, beta)
        
        # If cashed out, add stopping probability at n_pumps + 1
        if not exploded_t:
            trial_ll += _compute_log_stop_prob(n_pumps + 1, target, beta)
        # If exploded, no stopping probability (they didn't choose to stop)
        
        total_nll -= trial_ll
        
        # Update state for next trial
        if exploded_t:
            # Explosion: omega unchanged, rho decreases
            rho = np.clip(rho - alpha_minus, RHO_MIN, RHO_MAX)
        else:
            # Cash-out: omega ratchets up, rho increases
            omega = max(omega, n_pumps)
            rho = np.clip(rho + alpha_plus, RHO_MIN, RHO_MAX)
    
    # Handle numerical issues
    if not np.isfinite(total_nll):
        return 1e10
    
    return total_nll


def fit_participant(pumps: np.ndarray, 
                    exploded: np.ndarray,
                    seed: Optional[int] = None,
                    n_starts: Optional[int] = None) -> Optional[Dict]:
    """
    Fit the Pump-Level Range Learning model to a single participant's data.
    
    Parameters
    ----------
    pumps : array-like
        Number of pumps on each trial
    exploded : array-like
        Whether balloon exploded (1) or cashed out (0) on each trial
    seed : int, optional
        Random seed for reproducibility
    n_starts : int, optional
        Number of random restarts (default: OPTIMIZER_SETTINGS['n_starts'])
    
    Returns
    -------
    dict or None
        Fitted parameters and diagnostics, or None if fitting fails.
        
        Keys:
          - omega_0, rho_0, alpha_minus, alpha_plus, beta: fitted values
          - nll: negative log-likelihood at optimum
          - aic, bic: information criteria
          - n_trials: number of trials
          - n_pump_opportunities: total pump decisions modeled
          - success: whether optimizer converged
          - loss_aversion: α⁻/α⁺ ratio
    
    Example
    -------
    >>> result = fit_participant(pumps, exploded)
    >>> if result is not None:
    >>>     print(f"Loss aversion: {result['loss_aversion']:.2f}")
    """
    pumps = np.asarray(pumps)
    exploded = np.asarray(exploded)
    
    if len(pumps) < 5:
        return None
    
    if n_starts is None:
        n_starts = OPTIMIZER_SETTINGS['n_starts']
    
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
    
    # Count total pump opportunities for BIC calculation
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


def simulate_participant(omega_0: float,
                         rho_0: float,
                         alpha_minus: float,
                         alpha_plus: float,
                         beta: float,
                         n_trials: int = 30,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate BART behavior using the Pump-Level Range Learning model.
    
    Parameters
    ----------
    omega_0 : float
        Initial range estimate
    rho_0 : float
        Initial confidence multiplier
    alpha_minus : float
        Confidence penalty after explosion
    alpha_plus : float
        Confidence boost after cash-out
    beta : float
        Response sensitivity (logistic steepness)
    n_trials : int
        Number of trials to simulate
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    pumps : ndarray
        Number of pumps on each trial
    exploded : ndarray
        Whether balloon exploded (1) or cashed out (0)
    
    Example
    -------
    >>> pumps, exploded = simulate_participant(
    ...     omega_0=50, rho_0=1.0, alpha_minus=0.15, alpha_plus=0.08, beta=0.3
    ... )
    """
    rng = np.random.RandomState(seed)
    
    omega = omega_0
    rho = np.clip(rho_0, RHO_MIN, RHO_MAX)
    
    pumps = np.zeros(n_trials, dtype=int)
    exploded = np.zeros(n_trials, dtype=int)
    
    for t in range(n_trials):
        target = omega * rho
        
        # Sample burst point from BART hazard function
        burst_point = _sample_burst_point(rng, MAX_PUMPS)
        
        # Simulate pump decisions
        n_pumps = 0
        for i in range(1, MAX_PUMPS + 1):
            # Check for explosion first
            if i == burst_point:
                n_pumps = i
                exploded[t] = 1
                break
            
            # Decide whether to pump using logistic function
            delta = i - target
            p_pump = 1 / (1 + np.exp(beta * delta))
            
            if rng.random() < p_pump:
                n_pumps = i
            else:
                # Chose to stop
                n_pumps = i - 1 if i > 1 else 1
                break
        
        pumps[t] = max(n_pumps, 1)  # At least 1 pump
        
        # Update state for next trial
        if exploded[t]:
            rho = np.clip(rho - alpha_minus, RHO_MIN, RHO_MAX)
        else:
            omega = max(omega, pumps[t])
            rho = np.clip(rho + alpha_plus, RHO_MIN, RHO_MAX)
    
    return pumps, exploded


def get_state_after_trials(params: Dict,
                           pumps: np.ndarray,
                           exploded: np.ndarray) -> Tuple[float, float]:
    """
    Compute the model state (omega, rho) after observing a sequence of trials.
    
    Useful for out-of-sample prediction: fit on early trials, then get state
    to predict later trials.
    
    Parameters
    ----------
    params : dict
        Fitted parameters (must include omega_0, rho_0, alpha_minus, alpha_plus)
    pumps : array-like
        Observed pumps
    exploded : array-like
        Observed outcomes
    
    Returns
    -------
    omega : float
        Current range estimate
    rho : float
        Current confidence multiplier
    """
    omega = params['omega_0']
    rho = np.clip(params['rho_0'], RHO_MIN, RHO_MAX)
    
    for t in range(len(pumps)):
        if exploded[t] == 0:
            omega = max(omega, pumps[t])
            rho = np.clip(rho + params['alpha_plus'], RHO_MIN, RHO_MAX)
        else:
            rho = np.clip(rho - params['alpha_minus'], RHO_MIN, RHO_MAX)
    
    return omega, rho


def compute_target_sequence(params: Dict,
                            pumps: np.ndarray,
                            exploded: np.ndarray) -> np.ndarray:
    """
    Compute the sequence of targets the model would produce for given data.
    
    Useful for visualizing model predictions against actual behavior.
    
    Parameters
    ----------
    params : dict
        Fitted parameters
    pumps : array-like
        Observed pumps
    exploded : array-like
        Observed outcomes
    
    Returns
    -------
    targets : ndarray
        Target value for each trial
    """
    omega = params['omega_0']
    rho = np.clip(params['rho_0'], RHO_MIN, RHO_MAX)
    
    targets = np.zeros(len(pumps))
    
    for t in range(len(pumps)):
        targets[t] = omega * rho
        
        if exploded[t] == 0:
            omega = max(omega, pumps[t])
            rho = np.clip(rho + params['alpha_plus'], RHO_MIN, RHO_MAX)
        else:
            rho = np.clip(rho - params['alpha_minus'], RHO_MIN, RHO_MAX)
    
    return targets


def compute_loss_aversion_ratio(params: Dict) -> float:
    """
    Compute the loss aversion ratio from fitted parameters.
    
    Parameters
    ----------
    params : dict
        Fitted parameters (must include alpha_minus, alpha_plus)
    
    Returns
    -------
    float
        α⁻/α⁺ ratio (values > 1 indicate loss aversion)
    """
    if params['alpha_plus'] > 0:
        return params['alpha_minus'] / params['alpha_plus']
    else:
        return np.nan


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _sample_burst_point(rng: np.random.RandomState, max_pumps: int = 128) -> int:
    """
    Sample a burst point from the BART hazard function.
    
    P(burst on pump k | survived to k) = 1/(max_pumps - k + 1)
    This creates a linearly increasing hazard rate.
    """
    for k in range(1, max_pumps + 1):
        p_burst = 1.0 / (max_pumps - k + 1)
        if rng.random() < p_burst:
            return k
    return max_pumps


# =============================================================================
# VALIDATION INFO
# =============================================================================

VALIDATION_INFO = {
    'parameter_recovery': {
        'n_simulations': 100,
        'n_trials': 30,
        'mean_r': 0.814,
        'individual_r': {
            'omega_0': 0.844,
            'rho_0': 0.763,
            'alpha_minus': 0.875,
            'alpha_plus': 0.880,
            'beta': 0.707,
        },
        'loss_aversion_ratio_r': 0.964,
        'classification_accuracy': 0.96,
        'date': '2026-01-23',
    },
    'empirical_results': {
        'n_participants': 1507,
        'mean_omega_0': 61.90,
        'mean_rho_0': 1.097,
        'mean_alpha_minus': 0.1619,
        'mean_alpha_plus': 0.0790,
        'mean_beta': 0.226,
        'loss_aversion_ratio': 2.05,
        'pct_loss_averse': 90.2,
    },
    'model_comparison': {
        'vs_bsr_aic_wins': 0.750,
        'vs_bsr_bic_wins': 0.510,
        'mean_aic_diff': -6.82,
        'aic_ttest_p': '<1e-92',
    },
    'comparison_with_original': {
        'original_loss_aversion_r': 0.794,
        'pump_level_loss_aversion_r': 0.964,
        'improvement': '+21%',
    }
}


# =============================================================================
# MODULE INFO
# =============================================================================

def print_info():
    """Print model information and validation summary."""
    print("=" * 70)
    print("RANGE LEARNING MODEL (PUMP-LEVEL) - CANONICAL IMPLEMENTATION")
    print("=" * 70)
    print(f"\nVersion: {__version__} ({__date__})")
    
    print("\n*** PARAMETER BOUNDS ***")
    for param, bounds in BOUNDS.items():
        print(f"  {param:<15} {bounds}")
    
    print(f"\n*** RUNTIME BOUNDS ***")
    print(f"  ρ bounds: [{RHO_MIN}, {RHO_MAX}]")
    
    print(f"\n*** PARAMETER RECOVERY ***")
    pr = VALIDATION_INFO['parameter_recovery']
    print(f"  Mean r:              {pr['mean_r']:.3f}")
    print(f"  Loss aversion r:     {pr['loss_aversion_ratio_r']:.3f}")
    print(f"  Classification (LA): {pr['classification_accuracy']*100:.0f}%")
    
    print(f"\n*** COMPARISON WITH ORIGINAL FORMULATION ***")
    comp = VALIDATION_INFO['comparison_with_original']
    print(f"  Original LA recovery:   r = {comp['original_loss_aversion_r']:.3f}")
    print(f"  Pump-level LA recovery: r = {comp['pump_level_loss_aversion_r']:.3f}")
    print(f"  Improvement:            {comp['improvement']}")
    
    print(f"\n*** EMPIRICAL RESULTS (N={VALIDATION_INFO['empirical_results']['n_participants']}) ***")
    er = VALIDATION_INFO['empirical_results']
    print(f"  Mean α⁻:              {er['mean_alpha_minus']:.4f}")
    print(f"  Mean α⁺:              {er['mean_alpha_plus']:.4f}")
    print(f"  Loss aversion ratio:  {er['loss_aversion_ratio']:.2f}")
    print(f"  % Loss averse:        {er['pct_loss_averse']:.1f}%")
    
    print(f"\n*** MODEL COMPARISON (vs BSR Model 3) ***")
    mc = VALIDATION_INFO['model_comparison']
    print(f"  AIC: RL wins {mc['vs_bsr_aic_wins']*100:.1f}%")
    print(f"  BIC: RL wins {mc['vs_bsr_bic_wins']*100:.1f}%")
    print(f"  Mean AIC diff: {mc['mean_aic_diff']:.2f} (negative = RL better)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_info()
