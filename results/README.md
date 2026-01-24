# Results Directory

This directory contains output files from the BART analyses.

---

## Files

### correlation_results_large_sample.csv
**Source**: `bart_correlation_analysis.py`

Correlations between all BART measures and psychological variables for the full sample (N = 1,507).

| Column | Description |
|--------|-------------|
| bart_param | BART measure (mean_pumps, explosion_rate, omega_0, etc.) |
| predictor | Psychological variable |
| r | Pearson correlation |
| p | p-value |
| n | Sample size |
| abs_r | Absolute value of r |

---

### correlation_results_small_sample.csv
**Source**: `bart_correlation_analysis.py`

Correlations for the subset with full model parameters (N = 100).

Same structure as large sample file.

---

### bart_with_composites.csv
**Source**: `bart_final_synthesis.py`

Dataset with novel composite scores added.

| Column | Description |
|--------|-------------|
| partid | Participant ID |
| mean_pumps | Average pumps per trial |
| explosion_rate | Proportion of explosions |
| omega_0, rho_0, alpha_minus, alpha_plus, sigma | Model parameters |
| COMPOSITE_approach | Z-scored average of sensation seeking + risk propensity |
| COMPOSITE_cognitive | Z-scored average of cognitive measures |
| COMPOSITE_externalizing | Z-scored average of substance use + gambling |
| COMPOSITE_optimal | LASSO-weighted optimal predictor |

**Key finding**: COMPOSITE_optimal achieves r = .30 with mean_pumps (highest correlation found).

---

### behavioral_dynamics.csv
**Source**: `bart_novel_methods.py`

Trial-by-trial behavioral dynamics computed for each participant.

| Column | Description |
|--------|-------------|
| partid | Participant ID |
| n_trials | Number of trials completed |
| mean_pumps | Average pumps |
| sd_pumps | Standard deviation of pumps |
| explosion_rate | Proportion of explosions |
| post_explosion_change | Average change in pumps after an explosion |
| post_success_change | Average change in pumps after successful cash-out |
| resilience | -post_explosion_change (higher = bounces back more) |
| cv | Coefficient of variation (behavioral consistency) |
| max_consecutive_losses | Longest streak of explosions |

**Key finding**: Behavioral variability (cv) does NOT correlate with impulsivity (r = .02).

---

### extreme_groups_results.csv
**Source**: `bart_novel_methods.py`

Effect sizes comparing HIGH (top 25%) vs. LOW (bottom 25%) pumpers.

| Column | Description |
|--------|-------------|
| measure | Psychological variable |
| category | Construct category (APPROACH, IMPULSIVITY, AVOIDANCE, COGNITIVE, PATHOLOGY) |
| var | Variable name |
| low_mean | Mean for low pumpers |
| high_mean | Mean for high pumpers |
| t | t-statistic |
| p | p-value |
| d | Cohen's d effect size |

**Key findings**:
- Sensation Seeking: d = 0.48 (large)
- Impulsivity (BIS): d = 0.12 (negligible)
- Numeracy: d = 0.42 (high pumpers are SMARTER)

---

### parameter_personality_matrix.csv
**Source**: `bart_personality_decomposition.py`

Correlation matrix between model parameters and Big Five personality traits.

| Column | Description |
|--------|-------------|
| Parameter | Model parameter name |
| Extraversion, Agreeableness, etc. | Correlation with each Big Five trait |
| *_p | p-value for each correlation |
| *_n | Sample size |

**Key findings**:
- α⁺ × Extraversion: r = .19**
- α⁺ × Conscientiousness: r = .16**
- ω₀ × Openness: r = .13*

---

### partial_correlation_results.csv
**Source**: `bart_partial_correlations.py`

Partial correlations controlling for cognitive ability and mean_pumps.

| Column | Description |
|--------|-------------|
| param | Model parameter |
| trait | Big Five trait |
| r_zero | Zero-order correlation |
| r_partial | Partial correlation (after controls) |
| p_zero, p_partial | p-values |
| change | r_partial - r_zero |
| abs_change | Change in absolute correlation |

**Key finding**: α⁺ effects survive all controls (robust effects).

---

### rt_metrics.csv
**Source**: `bart_rt_analysis.py`

Reaction time metrics extracted for each participant.

| Column | Description |
|--------|-------------|
| partid | Participant ID |
| rt_mean | Average RT across all pumps (ms) |
| rt_sd | Within-person RT standard deviation |
| rt_cv | Coefficient of variation (consistency) |
| rt_first_mean | Average first-pump RT (deliberation time) |
| rt_late_mean | Average RT for last 5 pumps (near threshold) |
| rt_trend_mean | Within-trial RT slope (speeding/slowing) |
| rt_skew | RT distribution skewness |
| rt_post_explosion | RT after explosion trials |
| rt_post_success | RT after successful cash-outs |
| rt_explosion_effect | RT change after explosions |

**Key findings**:
- rt_mean × Neuroticism: r = .15* (neurotic people respond slower)
- rt_cv × Extraversion: r = -.14* (extraverts are more consistent)
- rt_mean × loss_aversion: r = .64*** (convergent validity)

---

### rt_convergent_validity_findings.md
**Source**: `bart_rt_analysis.py` + theoretical synthesis

Detailed writeup of the convergent validity finding between RT and loss aversion ratio.

**Key finding**: RT and loss aversion ratio predict Neuroticism equally well (r ≈ .14-.15) and correlate r = .64 with each other, providing construct validity evidence that the Range Learning model extracts psychologically meaningful parameters.

---

## Summary Statistics

| File | Key Metric | Value |
|------|------------|-------|
| correlation_results | Best single predictor | SSSV (r = .19) |
| bart_with_composites | Best composite | COMPOSITE_optimal (r = .30) |
| extreme_groups | Effect size ratio | Approach/Impulsivity = 2.8x |
| parameter_personality | Best parameter-trait | α⁺ × Extraversion (r = .19) |
| rt_metrics | RT × Loss Aversion convergence | r = .64*** |

---

## Interpretation

These results collectively support the conclusion that **the BART measures Approach-Oriented Risk Tendency**, characterized by:

1. Strong correlations with sensation seeking and risk propensity
2. Weak correlations with impulsivity and pathology
3. Different model parameters mapping to different personality traits
4. High pumpers being more cognitively able, not less
5. **Convergent validity**: RT patterns and model parameters predict the same personality traits, validating that the model extracts psychologically meaningful parameters
