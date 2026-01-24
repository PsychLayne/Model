# Analysis Scripts

This directory contains Python scripts for analyzing what the BART measures.

## Overview

These analyses address the central question: **What does the BART measure with .70-.91 reliability?**

Our conclusion: The BART measures **Approach-Oriented Risk Tendency** - a stable motivational trait reflecting how strongly individuals pursue rewards despite uncertainty.

---

## Scripts (in logical order)

### 1. bart_correlation_analysis.py
**Purpose**: Initial comprehensive correlation analysis

**What it does**:
- Merges all datasets (BART parameters + questionnaires + personality)
- Computes correlations between BART measures and all psychological variables
- Organizes results by theoretical construct

**Key findings**:
- Sensation Seeking (SSSV) is strongest correlate with mean_pumps (r = .19)
- Impulsivity (BIS) correlations are weak (r ≈ .05)
- CCT shows convergent validity (r ≈ .16)

---

### 2. bart_deep_analysis.py
**Purpose**: Construct-level analysis and initial ML models

**What it does**:
- Computes average correlations by theoretical construct
- Runs Random Forest feature importance
- Runs LASSO feature selection
- Ranks constructs by predictive power

**Key findings**:
- Sensation Seeking construct has highest average |r| with BART
- Impulsivity construct has lowest average |r|
- LASSO selects CCTratio, NUM, SSexp as key predictors

---

### 3. bart_ml_analysis.py
**Purpose**: Full machine learning pipeline

**What it does**:
- Random Forest with cross-validation
- Gradient Boosting for non-linear effects
- Mutual information analysis
- Category-level feature importance

**Key findings**:
- CV R² ≈ .05-.06 (modest but significant)
- Approach-related variables dominate feature importance
- Cognitive variables predict learning rates, not pumping

---

### 4. bart_final_synthesis.py
**Purpose**: Create optimal composite scores and synthesize findings

**What it does**:
- Creates COMPOSITE_approach (sensation seeking + risk propensity)
- Creates COMPOSITE_optimal (LASSO-weighted combination)
- Tests multiple regression combinations
- Explains each model parameter's psychological meaning

**Key findings**:
- COMPOSITE_optimal achieves r = .30 with mean_pumps (highest found)
- Different parameters capture different constructs
- Provides interpretive framework for each parameter

---

### 5. bart_theoretical_deep_dive.py
**Purpose**: Seven lines of evidence for approach-tendency theory

**What it does**:
- Tests what BART correlates WITH (approach constructs)
- Tests what BART does NOT correlate with (impulsivity, pathology)
- Examines risk perception paradox
- Tests cognitive ability dissociation
- Creates BAS/BIS composites and tests RST predictions

**Key findings**:
- BAS composite correlates r = .21 with mean_pumps
- BIS composite shows r = .03 (non-significant)
- High pumpers KNOW the risk but choose to approach anyway
- Smarter people pump MORE (not less)

---

### 6. bart_novel_methods.py
**Purpose**: Two novel methodological tests of approach-tendency theory

**Method 1: Post-Outcome Behavioral Dynamics**
- Analyzes trial-by-trial behavior after explosions vs. successes
- Tests whether high sensation seekers show "resilience" (bouncing back)
- Examines behavioral consistency vs. impulsivity

**Method 2: Extreme Groups Psychological Profiling**
- Compares top 25% vs. bottom 25% pumpers on all measures
- Calculates effect sizes for each psychological construct
- Direct falsification test of impulsivity theory

**Key findings**:
- Behavioral variability is INDEPENDENT of impulsivity (r = .02)
- HIGH pumpers: d = 0.48 for sensation seeking, d = 0.12 for impulsivity
- Effect size ratio: Approach/Impulsivity = 2.8x
- HIGH pumpers have HIGHER cognitive ability (d = 0.42 for numeracy)

---

### 7. bart_personality_decomposition.py
**Purpose**: Test discriminant validity of model parameters across Big Five

**What it does**:
- Creates correlation matrix: parameters × Big Five
- Tests whether different parameters correlate with different traits
- Evaluates theoretical coherence of mappings

**Key findings**:
- α⁺ → Extraversion (r = .19), Conscientiousness (r = .16)
- ω₀ → Openness (r = .13)
- Different parameters show discriminant validity
- This is rare in behavioral tasks

---

### 8. bart_partial_correlations.py
**Purpose**: Test whether controlling confounds increases personality effects

**What it does**:
- Partial correlations controlling for cognitive ability
- Partial correlations controlling for mean_pumps
- Partial correlations controlling for other Big Five traits
- Tests suppression effects

**Key findings**:
- Controlling for confounds doesn't dramatically increase correlations
- α⁺ effects survive all controls (robust)
- Sample size (N ≈ 287 for Big Five) is the limiting factor

---

### 9. bart_raw_vs_model.py
**Purpose**: Critical test - do model parameters add value over raw BART?

**What it does**:
- Compares correlation profiles: raw BART vs. model parameters
- Tests unique variance explained by parameters
- Examines cognitive ability as decisive case

**Key findings**:
- Raw BART: mean_pumps × NUM = +.15 (smart → pump more)
- Model: alpha_minus × NUM = -.17 (smart → learn faster) - OPPOSITE SIGN
- Model parameters reveal structure raw BART cannot
- Different parameters → different psychological systems

---

## Running the Analyses

```bash
# All scripts can be run independently
python3 analysis/bart_correlation_analysis.py
python3 analysis/bart_deep_analysis.py
# etc.

# Required packages
pip install pandas numpy scipy scikit-learn
```

---

## Summary of Evidence

| Analysis | Key Evidence |
|----------|--------------|
| Correlations | SSSV r=.19, BIS r=.05 |
| ML Feature Importance | Approach variables dominate |
| Extreme Groups | SS d=.48, Impulsivity d=.12 |
| BAS vs BIS | BAS r=.21, BIS r=.03 |
| Cognitive Ability | Smart people pump MORE |
| Behavioral Dynamics | Variability ≠ Impulsivity |
| Parameter Mapping | α⁺→E, α⁻→Cognitive, ω₀→O |
| Raw vs Model | Opposite signs for cognitive ability |

**Conclusion**: The BART measures Approach-Oriented Risk Tendency, not impulsivity.
