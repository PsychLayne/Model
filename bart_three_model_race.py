#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BART MODEL COMPARISON ANALYSIS                            ║
║        Range Learning vs BSR/Par4 vs EWMV (Park et al. 2021)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script fits THREE computational models to BART data and displays
a live-updating scoreboard showing which model is winning.

MODELS:
    1. Range Learning - Novel range-based learning with ratchet mechanism
    2. BSR/Par4 - Wallsten et al. (2005) Bayesian Sequential Risk model
    3. EWMV - Park et al. (2021) Exponential-Weight Mean-Variance model
             *** NOW WITH CORRECT UTILITY MAXIMIZATION (Park Eq. 16) ***

METHODOLOGICAL NOTES:
    - All models are evaluated as Target-Setting policies rather than 
      Sequential-Choice policies to align with the Range Learning theoretical
      framework. This means we use norm.logpdf for the target, not Bernoulli
      probabilities for each pump decision.
    - AIC/BIC scores from this analysis should NOT be directly compared to
      AIC/BIC scores from Wallsten/Park papers (which use sequential likelihood).

Usage:
    python bart_three_model_race.py                    # Will prompt for CSV path
    python bart_three_model_race.py path/to/data.csv  # Direct path

Required CSV columns:
    - partid: participant ID
    - trial: trial number
    - pumps: number of pumps
    - exploded: 0 = cashed out, 1 = popped

Author: BART Model Comparison Suite v2.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the header banner."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                        🎈 BART MODEL COMPARISON 🎈                           ║")
    print("║               Range Learning  vs  BSR/Par4  vs  EWMV                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()


def print_scoreboard(stats, current_participant="", status=""):
    """Print the live-updating scoreboard."""
    
    rl_wins = stats['rl_aic_wins']
    bsr_wins = stats['bsr_aic_wins']
    ewmv_wins = stats['ewmv_aic_wins']
    ties = stats['ties']
    total = stats['completed']
    
    # Calculate percentages
    if total > 0:
        rl_pct = (rl_wins / total) * 100
        bsr_pct = (bsr_wins / total) * 100
        ewmv_pct = (ewmv_wins / total) * 100
    else:
        rl_pct = bsr_pct = ewmv_pct = 0
    
    print("┌──────────────────────────────────────────────────────────────────────────────┐")
    print("│                                SCOREBOARD                                    │")
    print("├──────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                              │")
    print("│   🧠 RANGE LEARNING        📊 BSR/Par4            📈 EWMV                    │")
    print(f"│      Wins: {rl_wins:<6}            Wins: {bsr_wins:<6}            Wins: {ewmv_wins:<6}          │")
    print(f"│      ({rl_pct:>5.1f}%)              ({bsr_pct:>5.1f}%)              ({ewmv_pct:>5.1f}%)           │")
    print("│                                                                              │")
    
    # Create visual bars for each model
    bar_width = 20
    if total > 0:
        rl_bar = int((rl_wins / total) * bar_width)
        bsr_bar = int((bsr_wins / total) * bar_width)
        ewmv_bar = int((ewmv_wins / total) * bar_width)
    else:
        rl_bar = bsr_bar = ewmv_bar = 0
    
    rl_bar_str = "█" * rl_bar + "░" * (bar_width - rl_bar)
    bsr_bar_str = "█" * bsr_bar + "░" * (bar_width - bsr_bar)
    ewmv_bar_str = "█" * ewmv_bar + "░" * (bar_width - ewmv_bar)
    
    print(f"│   [{rl_bar_str}]  [{bsr_bar_str}]  [{ewmv_bar_str}]  │")
    print("│                                                                              │")
    print("├──────────────────────────────────────────────────────────────────────────────┤")
    print(f"│   Completed: {total}/{stats['total']} participants    Ties: {ties:<5}                        │")
    print("├──────────────────────────────────────────────────────────────────────────────┤")
    
    # Progress bar
    if stats['total'] > 0:
        progress = total / stats['total']
        prog_bar_width = 58
        filled = int(progress * prog_bar_width)
        print(f"│   Progress: [{'█' * filled}{'░' * (prog_bar_width - filled)}] {progress*100:>5.1f}%  │")
    
    print("├──────────────────────────────────────────────────────────────────────────────┤")
    print(f"│   Current: {str(current_participant)[:20]:<20}  Status: {status:<24}   │")
    print("└──────────────────────────────────────────────────────────────────────────────┘")
    
    # Running statistics
    if total > 0:
        print()
        print("┌──────────────────────────────────────────────────────────────────────────────┐")
        print("│                           RUNNING AIC COMPARISON                             │")
        print("├──────────────────────────────────────────────────────────────────────────────┤")
        print(f"│   Mean AIC: RL={stats['mean_rl_aic']:>7.1f}   BSR={stats['mean_bsr_aic']:>7.1f}   EWMV={stats['mean_ewmv_aic']:>7.1f}         │")
        print(f"│   BIC Wins: RL={stats['rl_bic_wins']:<5}  BSR={stats['bsr_bic_wins']:<5}  EWMV={stats['ewmv_bic_wins']:<5}                   │")
        print("└──────────────────────────────────────────────────────────────────────────────┘")


def print_final_results(stats, results_df):
    """Print final summary after completion."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                            🏆 FINAL RESULTS 🏆                               ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Determine overall winner
    wins = {'Range Learning': stats['rl_aic_wins'], 
            'BSR/Par4': stats['bsr_aic_wins'],
            'EWMV': stats['ewmv_aic_wins']}
    winner_name = max(wins, key=wins.get)
    winner_wins = wins[winner_name]
    
    emojis = {'Range Learning': '🧠', 'BSR/Par4': '📊', 'EWMV': '📈'}
    
    print(f"                         WINNER: {emojis[winner_name]} {winner_name.upper()}")
    print()
    
    # Summary table
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                              SUMMARY STATISTICS                             │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│   Total Participants Analyzed: {stats['completed']:<10}                             │")
    print("│                                                                             │")
    print("│   ┌─────────────────┬──────────────┬──────────────┬──────────────┐          │")
    print("│   │ Metric          │ Range Learn  │ BSR/Par4     │ EWMV         │          │")
    print("│   ├─────────────────┼──────────────┼──────────────┼──────────────┤          │")
    print(f"│   │ AIC Wins        │ {stats['rl_aic_wins']:<12} │ {stats['bsr_aic_wins']:<12} │ {stats['ewmv_aic_wins']:<12} │          │")
    print(f"│   │ BIC Wins        │ {stats['rl_bic_wins']:<12} │ {stats['bsr_bic_wins']:<12} │ {stats['ewmv_bic_wins']:<12} │          │")
    print(f"│   │ Mean AIC        │ {results_df['RL_aic'].mean():<12.2f} │ {results_df['BSR_aic'].mean():<12.2f} │ {results_df['EWMV_aic'].mean():<12.2f} │          │")
    print(f"│   │ Mean BIC        │ {results_df['RL_bic'].mean():<12.2f} │ {results_df['BSR_bic'].mean():<12.2f} │ {results_df['EWMV_bic'].mean():<12.2f} │          │")
    print(f"│   │ Mean -LL        │ {results_df['RL_nll'].mean():<12.2f} │ {results_df['BSR_nll'].mean():<12.2f} │ {results_df['EWMV_nll'].mean():<12.2f} │          │")
    print(f"│   │ # Parameters    │ {'5':<12} │ {'4':<12} │ {'5':<12} │          │")
    print("│   └─────────────────┴──────────────┴──────────────┴──────────────┘          │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Win rates
    total = stats['completed']
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                              WIN RATES (AIC)                                │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    rl_rate = stats['rl_aic_wins'] / total * 100
    bsr_rate = stats['bsr_aic_wins'] / total * 100
    ewmv_rate = stats['ewmv_aic_wins'] / total * 100
    print(f"│   Range Learning: {rl_rate:>5.1f}%   BSR/Par4: {bsr_rate:>5.1f}%   EWMV: {ewmv_rate:>5.1f}%              │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Pairwise comparisons
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                           PAIRWISE COMPARISONS                              │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    
    # RL vs BSR
    rl_vs_bsr = (results_df['RL_aic'] < results_df['BSR_aic']).sum()
    bsr_vs_rl = (results_df['BSR_aic'] < results_df['RL_aic']).sum()
    print(f"│   Range Learning vs BSR/Par4:  RL wins {rl_vs_bsr}/{total} ({rl_vs_bsr/total*100:.1f}%)                  │")
    
    # RL vs EWMV
    rl_vs_ewmv = (results_df['RL_aic'] < results_df['EWMV_aic']).sum()
    ewmv_vs_rl = (results_df['EWMV_aic'] < results_df['RL_aic']).sum()
    print(f"│   Range Learning vs EWMV:      RL wins {rl_vs_ewmv}/{total} ({rl_vs_ewmv/total*100:.1f}%)                  │")
    
    # BSR vs EWMV
    bsr_vs_ewmv = (results_df['BSR_aic'] < results_df['EWMV_aic']).sum()
    ewmv_vs_bsr = (results_df['EWMV_aic'] < results_df['BSR_aic']).sum()
    print(f"│   BSR/Par4 vs EWMV:            BSR wins {bsr_vs_ewmv}/{total} ({bsr_vs_ewmv/total*100:.1f}%)                 │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Parameter estimates
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                        RANGE LEARNING PARAMETERS                            │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│   ω₀ (initial range):      {results_df['RL_omega0'].mean():>7.2f} ± {results_df['RL_omega0'].std():>6.2f}  (optimal = 64)       │")
    print(f"│   ρ₀ (initial confidence): {results_df['RL_rho0'].mean():>7.2f} ± {results_df['RL_rho0'].std():>6.2f}  (range: 0.3-1.2)       │")
    print(f"│   α⁻ (pop penalty):        {results_df['RL_alpha_minus'].mean():>7.3f} ± {results_df['RL_alpha_minus'].std():>6.3f}                        │")
    print(f"│   α⁺ (recovery rate):      {results_df['RL_alpha_plus'].mean():>7.3f} ± {results_df['RL_alpha_plus'].std():>6.3f}                        │")
    print(f"│   σ  (noise):              {results_df['RL_sigma'].mean():>7.2f} ± {results_df['RL_sigma'].std():>6.2f}                        │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                          BSR/Par4 PARAMETERS                                │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│   φ (prior p_survive):     {results_df['BSR_phi'].mean():>7.3f} ± {results_df['BSR_phi'].std():>6.3f}                        │")
    print(f"│   η (learning rate):       {results_df['BSR_eta'].mean():>7.3f} ± {results_df['BSR_eta'].std():>6.3f}                        │")
    print(f"│   γ (risk-taking):         {results_df['BSR_gamma'].mean():>7.3f} ± {results_df['BSR_gamma'].std():>6.3f}                        │")
    print(f"│   τ (inverse temp):        {results_df['BSR_tau'].mean():>7.3f} ± {results_df['BSR_tau'].std():>6.3f}                        │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                        EWMV PARAMETERS (Park et al.)                        │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│   ψ (per-pump p_burst):    {results_df['EWMV_psi'].mean():>7.4f} ± {results_df['EWMV_psi'].std():>6.4f}  (1/64≈0.016)         │")
    print(f"│   ξ (updating rate):       {results_df['EWMV_xi'].mean():>7.3f} ± {results_df['EWMV_xi'].std():>6.3f}                        │")
    print(f"│   ρ (risk preference):     {results_df['EWMV_rho'].mean():>7.3f} ± {results_df['EWMV_rho'].std():>6.3f}  (+:averse, -:seeking)│")
    print(f"│   λ (loss aversion):       {results_df['EWMV_lambda'].mean():>7.3f} ± {results_df['EWMV_lambda'].std():>6.3f}  (>1 = loss averse)   │")
    print(f"│   τ (inverse temp):        {results_df['EWMV_tau'].mean():>7.3f} ± {results_df['EWMV_tau'].std():>6.3f}                        │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Behavioral insights
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                          BEHAVIORAL INSIGHTS                                │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    
    # Range Learning loss aversion
    mean_alpha_minus = results_df['RL_alpha_minus'].mean()
    mean_alpha_plus = results_df['RL_alpha_plus'].mean()
    if mean_alpha_minus > mean_alpha_plus and mean_alpha_plus > 0:
        ratio = mean_alpha_minus / mean_alpha_plus
        print(f"│   RL: α⁻/α⁺ = {ratio:.2f} → {ratio:.1f}x more sensitive to pops than cash-outs           │")
    
    # EWMV loss aversion
    mean_lambda = results_df['EWMV_lambda'].mean()
    if mean_lambda > 1:
        print(f"│   EWMV: λ = {mean_lambda:.2f} → Losses weighted {mean_lambda:.1f}x more than gains                   │")
    
    # EWMV risk preference
    mean_rho = results_df['EWMV_rho'].mean()
    if mean_rho > 0:
        print(f"│   EWMV: ρ = {mean_rho:+.2f} → Risk-seeking in variance                              │")
    elif mean_rho < 0:
        print(f"│   EWMV: ρ = {mean_rho:+.2f} → Risk-averse in variance                               │")
    
    print("└─────────────────────────────────────────────────────────────────────────────┘")


# =============================================================================
# MODEL IMPLEMENTATIONS
# =============================================================================

def range_learning_negll(params, pumps_arr, exploded_arr):
    """
    Range Learning Model negative log-likelihood.
    
    The "Ratchet" model: learns the maximum safe range and adjusts confidence.
    
    Parameters:
    - omega_0: initial range estimate (max balloon capacity)
    - rho_0: initial confidence multiplier (0-1.2)
    - alpha_minus: confidence penalty after pop
    - alpha_plus: confidence recovery after cash-out
    - sigma: behavioral noise
    
    Key mechanism: omega only goes UP (ratchet), never down.
    """
    omega_0, rho_0, alpha_minus, alpha_plus, sigma = params
    
    # Parameter bounds
    if omega_0 <= 0 or omega_0 > 128: return 1e10
    if rho_0 <= 0 or rho_0 > 1.5: return 1e10
    if alpha_minus <= 0 or alpha_minus > 1: return 1e10
    if alpha_plus <= 0 or alpha_plus > 0.5: return 1e10
    if sigma <= 0 or sigma > 100: return 1e10
    
    rho_min, rho_max = 0.3, 1.2
    omega, rho = omega_0, rho_0
    total_ll = 0.0
    
    for i in range(len(pumps_arr)):
        target = omega * rho
        
        if exploded_arr[i] == 0:  # Cash-out
            total_ll += norm.logpdf(pumps_arr[i], loc=target, scale=sigma)
            omega = max(omega, pumps_arr[i])  # Ratchet up ONLY
            rho = min(rho_max, rho + alpha_plus)
        else:  # Explosion
            total_ll += norm.logsf(pumps_arr[i], loc=target, scale=sigma)
            rho = max(rho_min, rho - alpha_minus)  # Confidence drops, omega stays
    
    return -total_ll


def bsr_par4_negll(params, pumps_arr, exploded_arr):
    """
    BSR/Par4 Model negative log-likelihood (inspired by Wallsten et al., 2005).

    Bayesian Sequential Risk model with 4 parameters.

    NOTE: This uses a simplified pseudo-Bayesian update rather than the exact
    beta-binomial formulation. The update rule is:
        p_not_burst = (phi + eta * total_successes) / (1 + eta * total_pumps)
    This is equivalent to a weighted average of prior (phi) and observed rate,
    where data weight grows linearly with observations scaled by eta.

    Parameters:
    - phi: prior belief that balloon will NOT burst (per pump)
    - eta: learning/updating rate (higher = trust data more quickly)
    - gamma: risk-taking parameter (multiplier on optimal pumps)
    - tau: inverse temperature / consistency (higher = less noise)
    """
    phi, eta, gamma, tau = params
    
    # Parameter bounds
    if phi <= 0 or phi >= 1: return 1e10
    if eta < 0 or eta > 20: return 1e10
    if gamma <= 0 or gamma > 15: return 1e10
    if tau <= 0 or tau > 10: return 1e10
    
    total_successes = 0
    total_pumps = 0
    total_ll = 0.0
    sd = max(1, 1/tau * 20)
    
    for i in range(len(pumps_arr)):
        # Compute belief about P(not burst) using Bayesian updating
        if total_pumps == 0:
            p_not_burst = phi
        else:
            p_not_burst = (phi + eta * total_successes) / (1 + eta * total_pumps)
            p_not_burst = np.clip(p_not_burst, 0.001, 0.999)
        
        # Compute target pumps (risk-neutral optimal adjusted by gamma)
        log_p = np.log(p_not_burst)
        if log_p < 0:
            target = -gamma / log_p
        else:
            target = 128
        target = np.clip(target, 1, 128)
        
        if exploded_arr[i] == 0:  # Cash-out
            total_ll += norm.logpdf(pumps_arr[i], loc=target, scale=sd)
            total_successes += pumps_arr[i]
            total_pumps += pumps_arr[i]
        else:  # Explosion
            total_ll += norm.logsf(pumps_arr[i], loc=target, scale=sd)
            total_successes += max(0, pumps_arr[i] - 1)
            total_pumps += pumps_arr[i]
    
    return -total_ll


def compute_ewmv_optimal_pumps(p_burst_per_pump, rho, lambda_, max_pumps=128):
    """
    Compute the optimal number of pumps using TRUE EWMV utility maximization.
    
    This implements Park et al. (2021) Equation 16 properly via numerical search:
    
    For each candidate pump count l, we compute:
        U(l) = E[V(outcome)] - (ρ/2) * Var[V(outcome)]
    
    where outcomes are:
        - Survive all l pumps (prob q^l): gain = l points
        - Pop at some pump k ≤ l (prob 1-q^l): outcome = 0, but experienced as loss
    
    Loss aversion (λ) weights the psychological impact of losing potential gains.
    Risk parameter (ρ) weights variance:
        - ρ > 0: risk-averse (penalize variance)
        - ρ < 0: risk-seeking (prefer variance)
    
    Parameters:
    - p_burst_per_pump: per-pump probability of bursting (e.g., ~1/64 for BART)
    - rho: risk preference coefficient
    - lambda_: loss aversion coefficient (>1 means losses loom larger than gains)
    - max_pumps: maximum possible pumps (128 for standard BART)
    
    Returns:
    - l*: optimal number of pumps that maximizes utility
    """
    # Handle edge cases
    if p_burst_per_pump <= 0:
        return max_pumps  # No risk, pump maximally
    if p_burst_per_pump >= 1:
        return 1  # Guaranteed burst, pump minimally
    
    q = 1 - p_burst_per_pump  # Per-pump survival probability
    
    best_l = 1
    best_utility = -np.inf
    
    # Search over all possible pump counts (Park Eq. 16 optimization)
    for l in range(1, max_pumps + 1):
        # Probability of surviving to pump l and cashing out: q^l
        p_survive_all = q ** l
        p_pop = 1 - p_survive_all
        
        # =====================================================================
        # MEAN-VARIANCE UTILITY CALCULATION (Park et al. 2021, Eq. 16)
        # =====================================================================
        #
        # Two outcomes with prospect-theoretic values:
        #   1. Survive: objective gain = l, subjective value V = l
        #   2. Pop: objective gain = 0, subjective value V = -λ * l
        #      (The loss of "l potential points" is weighted by loss aversion λ)
        #
        # This captures the psychological asymmetry: losing potential gains
        # hurts more than equivalent gains feel good.
        
        # Expected subjective value
        # E[V] = P(survive) * l + P(pop) * (-λ * l)
        #      = l * [P(survive) - λ * P(pop)]
        #      = l * [q^l - λ(1 - q^l)]
        #      = l * [q^l(1 + λ) - λ]
        mean_V = l * (p_survive_all - lambda_ * p_pop)
        
        # Variance of subjective value
        # V takes value l with prob p_survive_all, and -λl with prob p_pop
        # E[V²] = P(survive) * l² + P(pop) * (λl)²
        #       = l² * [P(survive) + λ² * P(pop)]
        E_V_squared = (l ** 2) * (p_survive_all + (lambda_ ** 2) * p_pop)
        var_V = E_V_squared - mean_V ** 2
        
        # Mean-Variance Utility (Park Eq. 16)
        # U(l) = E[V] - (ρ/2) * Var[V]
        utility = mean_V - (rho / 2.0) * var_V
        
        if utility > best_utility:
            best_utility = utility
            best_l = l
    
    return best_l


def ewmv_negll(params, pumps_arr, exploded_arr):
    """
    Exponential-Weight Mean-Variance Model (Park et al., 2021).
    
    *** CORRECTED IMPLEMENTATION ***
    
    This version uses TRUE utility maximization from Park et al. Eq. 16,
    NOT a linear approximation. The key fix is:
    1. Properly estimating PER-PUMP burst probability (not trial-level)
    2. Using numerical search to find the utility-maximizing pump count
    
    Parameters:
    - psi: prior belief about per-pump burst probability (e.g., 1/64 ≈ 0.0156)
    - xi: belief updating rate (how fast to weight new evidence)
    - rho: risk preference in mean-variance utility (+ = risk-averse)
    - lambda_: loss aversion parameter (>1 = overweight losses)
    - tau: inverse temperature / behavioral consistency
    
    The model:
    1. Estimates per-pump burst probability using exponential weighting of
       prior belief and observed evidence
    2. Finds optimal pumps by maximizing U(l) = E[V] - (ρ/2)*Var[V]
    3. Adds behavioral noise via normal distribution around optimal target
    """
    psi, xi, rho, lambda_, tau = params
    
    # Parameter bounds
    if psi <= 0 or psi >= 0.5: return 1e10   # Per-pump prob should be small
    if xi <= 0 or xi > 5: return 1e10
    if rho < -5 or rho > 5: return 1e10
    if lambda_ <= 0 or lambda_ > 10: return 1e10
    if tau <= 0 or tau > 10: return 1e10
    
    # Track cumulative experience for Bayesian-like updating
    # We track total explosions and total pumps to estimate per-pump burst probability
    # Use psi-consistent prior: if psi is the prior burst rate, we add pseudocounts
    # equivalent to observing ~64 pumps with psi*64 bursts (roughly 1 burst if psi≈1/64)
    prior_strength = 64  # Prior sample size (equivalent to ~1 balloon's worth of pumps)
    total_explosion_pumps = psi * prior_strength  # Prior bursts (e.g., ~1 if psi=1/64)
    total_pumps_observed = prior_strength          # Prior pumps observed
    
    total_ll = 0.0
    sd = max(1, 1/tau * 20)  # Behavioral noise
    
    for i in range(len(pumps_arr)):
        # =====================================================================
        # ESTIMATE PER-PUMP BURST PROBABILITY
        # =====================================================================
        # Use exponential weighting between prior (psi) and observed rate
        # Observed per-pump burst rate = explosions / total_pumps
        
        if i == 0:
            # First trial: use prior
            p_burst_per_pump = psi
        else:
            # Compute observed per-pump burst rate
            # Each explosion is 1 "burst pump" out of all pumps attempted
            obs_burst_rate = total_explosion_pumps / total_pumps_observed
            
            # Exponential weighting: weight decays as we see more data
            # High xi = prior decays faster = trust data more
            # Low xi = prior persists = trust prior more
            n_trials = i
            weight_prior = np.exp(-xi * n_trials)
            
            # Weighted combination
            p_burst_per_pump = weight_prior * psi + (1 - weight_prior) * obs_burst_rate
        
        # Ensure valid probability
        p_burst_per_pump = np.clip(p_burst_per_pump, 0.001, 0.5)
        
        # =====================================================================
        # FIND OPTIMAL PUMP COUNT VIA UTILITY MAXIMIZATION
        # =====================================================================
        # This is the CRITICAL FIX: instead of a linear approximation,
        # we search over all l and find argmax U(l)
        target = compute_ewmv_optimal_pumps(p_burst_per_pump, rho, lambda_)
        target = np.clip(target, 1, 128)
        
        # =====================================================================
        # LIKELIHOOD CALCULATION
        # =====================================================================
        if exploded_arr[i] == 0:  # Cash-out (survived)
            total_ll += norm.logpdf(pumps_arr[i], loc=target, scale=sd)
            # All pumps were successful - add to total observations
            total_pumps_observed += pumps_arr[i]
        else:  # Explosion
            total_ll += norm.logsf(pumps_arr[i], loc=target, scale=sd)
            # One pump caused explosion out of pumps_arr[i] total
            total_explosion_pumps += 1
            total_pumps_observed += pumps_arr[i]
    
    return -total_ll


def fit_model(negll_func, bounds, starting_points, pumps_arr, exploded_arr):
    """Fit a model using multiple starting points for robust optimization."""
    best_result = None
    best_nll = np.inf
    
    for x0 in starting_points:
        try:
            result = minimize(
                negll_func, x0, 
                args=(pumps_arr, exploded_arr),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500}
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except:
            continue
    
    return best_result


def compute_metrics(nll, n_params, n_obs):
    """Compute AIC and BIC."""
    ll = -nll
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_obs) - 2 * ll
    return aic, bic


def generate_starting_points(n_points, bounds, seed=None):
    """Generate random starting points within bounds."""
    if seed is not None:
        np.random.seed(seed)
    
    points = []
    for _ in range(n_points):
        point = []
        for low, high in bounds:
            # Sample from middle 80% of range to avoid edge issues
            range_span = high - low
            low_adj = low + 0.1 * range_span
            high_adj = high - 0.1 * range_span
            point.append(np.random.uniform(low_adj, high_adj))
        points.append(point)
    return points


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def fit_participant(pumps_arr, exploded_arr):
    """Fit all three models to a single participant's data."""
    n_trials = len(pumps_arr)
    result = {}
    
    # =========================================================================
    # Model 1: Range Learning (5 parameters)
    # =========================================================================
    rl_bounds = [(1, 128), (0.1, 1.4), (0.01, 0.99), (0.01, 0.49), (1, 50)]
    
    # Manual starting points + 16 random = 20 total
    rl_manual_starts = [
        [40, 0.8, 0.2, 0.1, 15],
        [30, 0.6, 0.3, 0.05, 10],
        [50, 1.0, 0.15, 0.15, 20],
        [64, 0.7, 0.25, 0.08, 12],
    ]
    rl_random_starts = generate_starting_points(16, rl_bounds, seed=42)
    rl_starts = rl_manual_starts + rl_random_starts
    
    rl = fit_model(range_learning_negll, rl_bounds, rl_starts, pumps_arr, exploded_arr)
    
    if rl is not None:
        result['RL_nll'] = rl.fun
        result['RL_aic'], result['RL_bic'] = compute_metrics(rl.fun, 5, n_trials)
        result['RL_omega0'] = rl.x[0]
        result['RL_rho0'] = rl.x[1]
        result['RL_alpha_minus'] = rl.x[2]
        result['RL_alpha_plus'] = rl.x[3]
        result['RL_sigma'] = rl.x[4]
    else:
        result['RL_nll'] = result['RL_aic'] = result['RL_bic'] = np.nan
        result['RL_omega0'] = result['RL_rho0'] = np.nan
        result['RL_alpha_minus'] = result['RL_alpha_plus'] = result['RL_sigma'] = np.nan
    
    # =========================================================================
    # Model 2: BSR/Par4 (4 parameters)
    # =========================================================================
    bsr_bounds = [(0.01, 0.99), (0.001, 15), (0.1, 12), (0.01, 5)]
    
    bsr_manual_starts = [
        [0.9, 1.0, 2.0, 0.5],
        [0.95, 0.5, 3.0, 0.3],
        [0.85, 2.0, 1.5, 1.0],
        [0.8, 0.1, 4.0, 0.2],
    ]
    bsr_random_starts = generate_starting_points(16, bsr_bounds, seed=43)
    bsr_starts = bsr_manual_starts + bsr_random_starts
    
    bsr = fit_model(bsr_par4_negll, bsr_bounds, bsr_starts, pumps_arr, exploded_arr)
    
    if bsr is not None:
        result['BSR_nll'] = bsr.fun
        result['BSR_aic'], result['BSR_bic'] = compute_metrics(bsr.fun, 4, n_trials)
        result['BSR_phi'] = bsr.x[0]
        result['BSR_eta'] = bsr.x[1]
        result['BSR_gamma'] = bsr.x[2]
        result['BSR_tau'] = bsr.x[3]
    else:
        result['BSR_nll'] = result['BSR_aic'] = result['BSR_bic'] = np.nan
        result['BSR_phi'] = result['BSR_eta'] = result['BSR_gamma'] = result['BSR_tau'] = np.nan
    
    # =========================================================================
    # Model 3: EWMV (5 parameters) - CORRECTED IMPLEMENTATION
    # =========================================================================
    # psi is now per-pump burst probability (small, e.g., 1/64 ≈ 0.0156)
    # Bounds: psi ∈ [0.001, 0.1], xi ∈ [0.01, 5], rho ∈ [-3, 3], 
    #         lambda ∈ [0.5, 5], tau ∈ [0.01, 5]
    ewmv_bounds = [(0.001, 0.1), (0.01, 5), (-3, 3), (0.5, 5), (0.01, 5)]
    
    # 25+ starting points for robust optimization (EWMV is complex!)
    # Manual starting points covering different behavioral profiles
    ewmv_manual_starts = [
        # [psi, xi, rho, lambda, tau]
        [0.015, 0.5, 0, 1.0, 0.5],      # Neutral baseline (1/64 burst prob)
        [0.015, 1.0, 0.5, 1.5, 0.3],    # Moderate risk aversion, loss averse
        [0.015, 0.2, -0.5, 2.0, 1.0],   # Risk-seeking, high loss aversion
        [0.015, 0.8, 0.2, 1.2, 0.8],    # Slight risk aversion
        [0.02, 1.5, -1.0, 1.0, 0.5],    # Risk-seeking, neutral loss
        [0.01, 0.3, 1.0, 2.5, 0.4],     # Very risk averse, high loss aversion
        [0.02, 2.0, -0.3, 1.8, 0.6],    # Fast learner, slight risk-seeking
        [0.008, 0.6, 0.8, 0.8, 1.2],    # Optimistic about survival
        [0.03, 1.2, 0.0, 1.0, 1.0],     # Pessimistic, risk neutral
        [0.015, 0.1, -0.2, 3.0, 0.3],   # Slow learner, very loss averse
        [0.025, 0.4, 1.5, 1.5, 0.7],    # High risk aversion
        [0.012, 0.7, -1.5, 2.0, 0.9],   # Strong risk-seeking
    ]
    # Add 20 random starting points
    ewmv_random_starts = generate_starting_points(20, ewmv_bounds, seed=44)
    ewmv_starts = ewmv_manual_starts + ewmv_random_starts  # 32 total
    
    ewmv = fit_model(ewmv_negll, ewmv_bounds, ewmv_starts, pumps_arr, exploded_arr)
    
    if ewmv is not None:
        result['EWMV_nll'] = ewmv.fun
        result['EWMV_aic'], result['EWMV_bic'] = compute_metrics(ewmv.fun, 5, n_trials)
        result['EWMV_psi'] = ewmv.x[0]
        result['EWMV_xi'] = ewmv.x[1]
        result['EWMV_rho'] = ewmv.x[2]
        result['EWMV_lambda'] = ewmv.x[3]
        result['EWMV_tau'] = ewmv.x[4]
    else:
        result['EWMV_nll'] = result['EWMV_aic'] = result['EWMV_bic'] = np.nan
        result['EWMV_psi'] = result['EWMV_xi'] = result['EWMV_rho'] = np.nan
        result['EWMV_lambda'] = result['EWMV_tau'] = np.nan
    
    return result


def run_analysis(csv_path, update_interval=1):
    """
    Run the full analysis with live UI updates.
    
    Parameters:
    - csv_path: path to the CSV file
    - update_interval: how often to refresh the display (in participants)
    """
    
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Validate columns
    required_cols = ['partid', 'trial', 'pumps', 'exploded']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        return None
    
    participants = df['partid'].unique()
    n_participants = len(participants)
    
    print(f"Found {n_participants} participants")
    print("Starting analysis (this may take a while due to EWMV optimization)...\n")
    time.sleep(1)
    
    # Initialize statistics
    stats = {
        'total': n_participants,
        'completed': 0,
        'rl_aic_wins': 0,
        'bsr_aic_wins': 0,
        'ewmv_aic_wins': 0,
        'rl_bic_wins': 0,
        'bsr_bic_wins': 0,
        'ewmv_bic_wins': 0,
        'ties': 0,
        'mean_rl_aic': 0,
        'mean_bsr_aic': 0,
        'mean_ewmv_aic': 0,
        'rl_aics': [],
        'bsr_aics': [],
        'ewmv_aics': [],
    }
    
    results = []
    start_time = time.time()
    
    for i, pid in enumerate(participants):
        # Get participant data
        pdata = df[df['partid'] == pid].sort_values('trial')
        pumps_arr = pdata['pumps'].values
        exploded_arr = pdata['exploded'].values
        
        # Update display
        if i % update_interval == 0:
            clear_screen()
            print_header()
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                remaining = (n_participants - i) / rate
                status = f"~{remaining:.0f}s remaining"
            else:
                status = "Starting..."
            print_scoreboard(stats, current_participant=pid, status=status)
        
        # Fit models
        result = fit_participant(pumps_arr, exploded_arr)
        result['participant'] = pid
        result['n_trials'] = len(pumps_arr)
        result['mean_pumps'] = pumps_arr.mean()
        result['explosion_rate'] = exploded_arr.mean()
        results.append(result)
        
        # Update statistics
        stats['completed'] += 1
        
        # Check if all models converged
        valid = (not np.isnan(result['RL_aic']) and 
                 not np.isnan(result['BSR_aic']) and 
                 not np.isnan(result['EWMV_aic']))
        
        if valid:
            # Track AICs
            stats['rl_aics'].append(result['RL_aic'])
            stats['bsr_aics'].append(result['BSR_aic'])
            stats['ewmv_aics'].append(result['EWMV_aic'])
            
            stats['mean_rl_aic'] = np.mean(stats['rl_aics'])
            stats['mean_bsr_aic'] = np.mean(stats['bsr_aics'])
            stats['mean_ewmv_aic'] = np.mean(stats['ewmv_aics'])
            
            # Determine AIC winner
            aics = {'rl': result['RL_aic'], 'bsr': result['BSR_aic'], 'ewmv': result['EWMV_aic']}
            min_aic = min(aics.values())
            winners = [k for k, v in aics.items() if v == min_aic]
            
            if len(winners) > 1:
                stats['ties'] += 1
            else:
                winner = winners[0]
                if winner == 'rl':
                    stats['rl_aic_wins'] += 1
                elif winner == 'bsr':
                    stats['bsr_aic_wins'] += 1
                else:
                    stats['ewmv_aic_wins'] += 1
            
            # Determine BIC winner
            bics = {'rl': result['RL_bic'], 'bsr': result['BSR_bic'], 'ewmv': result['EWMV_bic']}
            min_bic = min(bics.values())
            bic_winners = [k for k, v in bics.items() if v == min_bic]
            
            if len(bic_winners) == 1:
                bic_winner = bic_winners[0]
                if bic_winner == 'rl':
                    stats['rl_bic_wins'] += 1
                elif bic_winner == 'bsr':
                    stats['bsr_bic_wins'] += 1
                else:
                    stats['ewmv_bic_wins'] += 1
    
    # Final display
    clear_screen()
    print_header()
    print_scoreboard(stats, current_participant="COMPLETE", status="Analysis finished!")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = csv_path.rsplit('.', 1)[0] + '_three_model_comparison.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n  📁 Results saved to: {output_path}")
    
    # Print final summary
    print_final_results(stats, results_df)
    
    total_time = time.time() - start_time
    print(f"\n  ⏱️  Total analysis time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  📊 Average time per participant: {total_time/n_participants:.2f}s")
    
    # Print methodology note
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                          METHODOLOGY NOTES                                  │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│   • All models evaluated as Target-Setting policies (norm.logpdf)          │")
    print("│   • EWMV uses TRUE utility maximization (Park Eq. 16), not approximation   │")
    print("│   • AIC/BIC not directly comparable to sequential-choice papers            │")
    print("│   • Each model used 20-25 starting points for robust optimization          │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    return results_df


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print_header()
    
    # Get CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to bart_pumps.csv if it exists
        if os.path.exists('bart_pumps.csv'):
            csv_path = 'bart_pumps.csv'
            print(f"  Found default file: bart_pumps.csv")
        else:
            print("  Enter the path to your BART data CSV file:")
            print("  (or drag and drop the file here)\n")
            csv_path = input("  Path: ").strip().strip('"').strip("'")
    
    if not csv_path:
        print("  No file provided. Exiting.")
        return
    
    if not os.path.exists(csv_path):
        print(f"  Error: File not found: {csv_path}")
        return
    
    print(f"\n  📂 Loading: {csv_path}\n")
    
    # Run analysis
    results = run_analysis(csv_path, update_interval=1)
    
    if results is not None:
        print("\n  ✅ Analysis complete!")
        print("\n  Press Enter to exit...")
        try:
            input()
        except:
            pass


if __name__ == "__main__":
    main()
