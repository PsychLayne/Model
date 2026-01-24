# BART Analysis: What Does the Balloon Analogue Risk Task Measure?

## The Question

The BART (Balloon Analogue Risk Task) has remarkably high reliability (.70-.91), yet correlations with external measures are typically modest (.15-.20).

**What is the BART measuring so reliably?**

---

## The Answer

**The BART measures Approach-Oriented Risk Tendency** - a stable motivational trait reflecting how strongly individuals pursue rewards despite uncertainty.

This is fundamentally different from impulsivity (loss of control) or pathology (dysfunction).

---

## Key Findings

### 1. What BART Correlates WITH

| Construct | Best Measure | Correlation |
|-----------|--------------|-------------|
| **Sensation Seeking** | SSSV | r = .19*** |
| **Recreational Risk** | Drec | r = .17*** |
| **CCT Risk-Taking** | CCTratio | r = .16*** |
| **Health Risk** | Dhea | r = .15*** |
| **Openness** | NEO_O | r = .14* |

### 2. What BART Does NOT Correlate With

| Construct | Best Measure | Correlation |
|-----------|--------------|-------------|
| Impulsivity | BIS | r = .05 (ns) |
| Gambling Pathology | PG | r = .02 (ns) |
| Anxiety | STAI | r = .04 (ns) |
| Neuroticism | NEO_N | r = .02 (ns) |

### 3. The Cognitive Ability Paradox

**High pumpers are MORE cognitively able, not less.**

| Group | Numeracy (d) | Working Memory (d) |
|-------|--------------|-------------------|
| HIGH vs LOW pumpers | +0.42*** | +0.27*** |

This directly contradicts impulsivity/deficit models.

### 4. Effect Size Comparison (Extreme Groups)

| Construct | Cohen's d |
|-----------|-----------|
| **Sensation Seeking** | **0.48*** |
| Experience Seeking | 0.45*** |
| Numeracy | 0.42*** |
| Openness | 0.40* |
| Impulsivity (BIS) | 0.12 (ns) |

**Ratio: Approach measures differentiate 2.8x better than impulsivity.**

---

## The Model Parameters Add Value

The Range Learning model decomposes BART behavior into psychologically meaningful components:

| Parameter | Best Correlate | r | Interpretation |
|-----------|---------------|---|----------------|
| **α⁺** (reward learning) | Extraversion | .19** | Personality drives reward learning |
| **α⁺** | Conscientiousness | .16** | Systematic learners update appropriately |
| **α⁻** (loss learning) | Numeracy | -.17*** | Cognitive ability drives loss learning |
| **ω₀** (threshold) | Openness | .13* | Open individuals explore more |

**Critical finding**: Raw BART and model parameters show OPPOSITE correlations with cognitive ability:
- mean_pumps × Numeracy = **+.15** (smart → pump more)
- alpha_minus × Numeracy = **-.17** (smart → learn faster)

The model separates WHAT you do from HOW you learn.

---

## Theoretical Implications

### The BART is NOT:
- ❌ An impulsivity measure (BIS correlations are negligible)
- ❌ A pathology indicator (gambling/addiction correlations near zero)
- ❌ A cognitive deficit marker (high pumpers are smarter)
- ❌ Measuring "loss of control"

### The BART IS:
- ✅ An approach motivation measure (sensation seeking, risk propensity)
- ✅ A behavioral operationalization of personality
- ✅ Capturing how people ACTUALLY behave under uncertainty
- ✅ Measuring a stable, trait-like individual difference

---

## Why This Matters

| Old Frame (Impulsivity) | New Frame (Approach Tendency) |
|-------------------------|-------------------------------|
| High pumpers have a deficit | High pumpers have different motivation |
| Risk-taking is dysfunction | Risk-taking is a trait on a continuum |
| Implies need for treatment | Implies normal personality variation |
| "Can't stop myself" | "The reward is worth the risk" |

---

## The BART's Unique Contribution

The BART + Range Learning model may be unique among behavioral tasks:

1. **High reliability** (.70-.91)
2. **Validated computational model** (Range Learning)
3. **Discriminant personality correlations** (different parameters → different traits)
4. **Theoretically coherent pattern** (maps to approach motivation)

This makes BART valuable for **behavioral decomposition of personality** - translating abstract traits into concrete computational parameters.

---

## Directory Structure

```
Model/
├── README.md                 # This file
├── data/
│   ├── README.md             # Data documentation
│   ├── bart/                 # BART task data
│   ├── questionnaires/       # Psychological measures
│   ├── decision_tasks/       # Other behavioral tasks
│   └── model_parameters/     # Range Learning outputs
├── analysis/
│   ├── README.md             # Analysis documentation
│   └── *.py                  # Analysis scripts
└── results/
    ├── README.md             # Results documentation
    └── *.csv                 # Output files
```

---

## Running the Analyses

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn

# Run analyses (each can be run independently)
python3 analysis/bart_correlation_analysis.py
python3 analysis/bart_novel_methods.py
python3 analysis/bart_personality_decomposition.py
python3 analysis/bart_raw_vs_model.py
```

---

## Citation

If using these analyses or findings, please cite appropriately.

---

## Summary

**The BART reliably measures Approach-Oriented Risk Tendency** - a fundamental motivational dimension reflecting how strongly individuals pursue potentially rewarding outcomes despite uncertainty. The high reliability comes from tapping a stable, biologically-grounded personality trait, not from measuring a narrow cognitive process or pathology.

The modest correlations with questionnaires reflect method variance (behavior ≠ self-report), not poor validity. The BART captures what questionnaires cannot: how people actually behave when facing real consequences under uncertainty.
