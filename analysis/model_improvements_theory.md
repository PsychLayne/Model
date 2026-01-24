# Theorycrafting: Improving Personality Extraction from BART

## Current State

The Range Learning model successfully extracts personality-relevant parameters from BART behavior:

| Parameter | Best Personality Correlate | r |
|-----------|---------------------------|---|
| α⁺ (reward learning) | Extraversion | .19** |
| α⁺ | Conscientiousness | .16** |
| Loss Aversion Ratio | Neuroticism | .14* |
| ω₀ (threshold) | Openness | .13* |

**Convergent validity**: Loss aversion ratio correlates r = .64 with simple RT metrics, and both predict Neuroticism at r ≈ .14-.15.

**Current gaps**:
- Agreeableness shows no significant correlations
- Correlations top out around r = .20
- RT data is unused despite containing personality signal

---

## Part 1: Improvements to the Range Learning Model

### 1.1 Integrate Reaction Time as a Parameter

**Current**: The model ignores RT entirely.

**Proposed**: Add RT-based parameters that capture response hesitation.

```
New Parameters:
- τ₀ (tau_0): Baseline response time
- τ_trend: Within-trial RT acceleration/deceleration
- τ_var: RT variability (response consistency)
```

**Rationale**:
- RT already correlates with personality (Extraversion, Neuroticism)
- RT correlates r = .44 with alpha_minus, suggesting it captures learning-relevant information
- RT variability might capture Neuroticism uniquely (anxious hesitation)

**Expected personality gains**:
| New Parameter | Expected Correlate | Rationale |
|--------------|-------------------|-----------|
| τ₀ | Neuroticism (+) | Anxious hesitation |
| τ_var | Neuroticism (+) | Inconsistent responding under anxiety |
| τ_trend | Conscientiousness (-) | Systematic people show consistent patterns |

---

### 1.2 Separate Post-Outcome Learning Rates

**Current**: α⁺ for successes, α⁻ for explosions.

**Proposed**: Separate learning rates by outcome SEQUENCE, not just outcome type.

```
New Parameters:
- α⁺_streak: Learning rate during success streaks
- α⁺_recovery: Learning rate for first success after explosion
- α⁻_first: Learning rate for first explosion after successes
- α⁻_streak: Learning rate during loss streaks
```

**Rationale**:
- Resilience (bouncing back after loss) is personality-relevant
- Streak effects might capture hot hand beliefs or gambler's fallacy tendencies
- Recovery patterns might relate to emotional regulation (Neuroticism)

**Expected personality gains**:
| New Parameter | Expected Correlate | Rationale |
|--------------|-------------------|-----------|
| α⁺_recovery | Extraversion (+) | Quick emotional recovery |
| α⁻_streak | Neuroticism (+) | Rumination amplifies loss learning |

---

### 1.3 Dynamic Threshold Parameter

**Current**: ω₀ is static (initial belief only).

**Proposed**: Allow threshold to drift across trials.

```
New Parameters:
- ω₀: Initial threshold (keep)
- ω_drift: Rate of threshold change across task
- ω_volatility: Trial-to-trial threshold fluctuation
```

**Rationale**:
- Some people become more conservative over time (fatigue, loss accumulation)
- Others become bolder (confidence building, habituation)
- Drift direction might reflect approach vs. avoidance motivation

**Expected personality gains**:
| New Parameter | Expected Correlate | Rationale |
|--------------|-------------------|-----------|
| ω_drift (+) | Openness (+) | Exploration increases with experience |
| ω_drift (-) | Neuroticism (+) | Anxiety leads to increasing caution |
| ω_volatility | Conscientiousness (-) | Systematic people maintain stable thresholds |

---

### 1.4 Risk Perception Integration

**Current**: Model captures behavior but not subjective perception.

**Data available**: bart_riskperc.csv contains explicit risk perception ratings.

**Proposed**: Model the gap between perceived and objective risk.

```
New Parameters:
- π_bias: Systematic over/underestimation of risk
- π_sensitivity: How strongly behavior responds to perceived risk
- π_updating: How quickly risk perception changes with experience
```

**Rationale**:
- Some people KNOW the risk but approach anyway (high BAS)
- Others underestimate risk (cognitive issue, not motivational)
- Separating perception from behavior disentangles cognition from motivation

**Expected personality gains**:
| New Parameter | Expected Correlate | Rationale |
|--------------|-------------------|-----------|
| π_bias (-) | Openness (+) | Open individuals underestimate novel risks |
| π_sensitivity | Neuroticism (+) | Anxious people respond more to perceived danger |
| Behavior given perception | Extraversion | Approach tendency independent of perception |

---

## Part 2: Novel Model Architectures

### 2.1 Dual-Process Model

**Concept**: Separate impulsive (fast, automatic) from reflective (slow, deliberative) systems.

```
Architecture:
- System 1 (Impulsive): Fast RT responses, habitual pumping
- System 2 (Reflective): Slow RT responses, deliberate decisions
- ω_conflict: Weight given to each system

Behavioral signatures:
- Fast RT + high pumps = impulsive approach
- Slow RT + high pumps = deliberate risk-taking
- Fast RT + low pumps = impulsive avoidance
- Slow RT + low pumps = deliberate caution
```

**Expected personality mapping**:
| System Weight | Expected Correlate | Rationale |
|--------------|-------------------|-----------|
| System 1 dominance | Impulsivity (BIS) | Finally capture impulsivity! |
| System 2 dominance | Conscientiousness | Deliberate decision-making |
| System conflict | Neuroticism | Ambivalence, anxiety |

**Why this might work**: Current correlations with BIS are near zero. This might be because impulsivity isn't about WHAT you choose but HOW you choose it. A dual-process model could finally separate impulsive risk-taking from deliberate risk-taking.

---

### 2.2 Hierarchical Bayesian Personality Model

**Concept**: Estimate personality traits DIRECTLY as latent variables that generate BART behavior.

```
Architecture:
Level 3: Personality traits (E, A, C, N, O) - latent
Level 2: Model parameters (α⁺, α⁻, ω₀, etc.) - generated from personality
Level 1: Trial behavior - generated from parameters

Structural equations:
α⁺ ~ β₁*Extraversion + β₂*Conscientiousness + ε
α⁻ ~ β₃*Neuroticism + β₄*Numeracy + ε
ω₀ ~ β₅*Openness + β₆*Sensation_Seeking + ε
```

**Advantages**:
- Directly estimates personality-parameter mappings
- Shrinks noisy individual estimates toward group patterns
- Can handle missing personality data
- Quantifies uncertainty in personality predictions

**Implementation**: Stan or PyMC, requires personality data for subset of participants to train mappings.

---

### 2.3 Reinforcement Learning with Individual Difference Priors

**Concept**: Standard RL model but with informative priors based on personality.

```
Architecture:
- Base RL model (e.g., Q-learning or actor-critic)
- Prior on learning rate: α ~ f(Extraversion, Conscientiousness)
- Prior on risk sensitivity: ρ ~ f(Neuroticism, Sensation_Seeking)
- Prior on exploration: ε ~ f(Openness)

Training:
1. Estimate population-level personality → parameter mappings
2. For new participant, use their personality as prior
3. Update parameters based on their BART behavior
4. Posterior parameters = personality-informed individual estimate
```

**Advantages**:
- Uses personality questionnaires as informative priors, not just validation
- Should improve parameter estimation for short tasks
- Explicit personality → behavior model

---

### 2.4 State-Space Model with Mood Dynamics

**Concept**: Allow internal states (mood, arousal, fatigue) to fluctuate and influence behavior.

```
Architecture:
Hidden states:
- s_arousal(t): Current arousal level
- s_mood(t): Current positive/negative affect
- s_fatigue(t): Accumulated fatigue

State dynamics:
- s_arousal(t+1) = s_arousal(t) + δ_explosion*exploded(t) + δ_success*success(t) + noise
- States influence parameters: α⁺(t) = α⁺_base + γ*s_arousal(t)

Personality enters as:
- Individual differences in state dynamics (how much explosions affect mood)
- Individual differences in state → behavior mappings
```

**Expected personality mapping**:
| State Dynamic | Expected Correlate | Rationale |
|--------------|-------------------|-----------|
| Arousal reactivity | Extraversion | BAS sensitivity |
| Mood reactivity to loss | Neuroticism | Emotional instability |
| Fatigue rate | Conscientiousness (-) | Sustained effort |

---

## Part 3: The Agreeableness Problem

Agreeableness consistently shows NO correlation with any BART measure. This might be because:

1. **BART is non-social**: Agreeableness is about interpersonal warmth, cooperation, trust. BART has no social component.

2. **Possible solutions**:
   - **Social BART variant**: Other players' outcomes visible, or competitive framing
   - **Cooperative BART**: Shared balloon where both players must agree to pump
   - **Trust BART**: Partner controls explosion probability, participant decides pumping

3. **Alternative**: Accept that BART cannot measure Agreeableness (it's a solo task) and focus on the four traits it CAN measure.

---

## Part 4: Implementation Priorities

Ranked by expected impact and feasibility:

### Tier 1: High Impact, Feasible Now

| Improvement | Expected Gain | Data Needed | Complexity |
|-------------|---------------|-------------|------------|
| **Add RT parameters** | +Neuroticism, +Extraversion | Already have | Low |
| **Dynamic threshold** | +Openness, +Neuroticism | Already have | Medium |

### Tier 2: High Impact, Requires Development

| Improvement | Expected Gain | Data Needed | Complexity |
|-------------|---------------|-------------|------------|
| **Dual-process model** | +Impulsivity (BIS) | RT patterns | High |
| **Risk perception integration** | Cognition vs motivation separation | bart_riskperc.csv | Medium |

### Tier 3: Ambitious, Requires New Data Collection

| Improvement | Expected Gain | Data Needed | Complexity |
|-------------|---------------|-------------|------------|
| **Hierarchical Bayesian** | All traits, better estimates | Large sample with both | Very High |
| **Social BART variant** | +Agreeableness | New task design | High |

---

## Part 5: Predicted Correlation Improvements

**Current best correlations**: r ≈ .15-.20 for individual parameters

**Predicted with improvements**:

| Trait | Current Best | After RT Integration | After Full Model |
|-------|-------------|---------------------|------------------|
| Extraversion | .19 (α⁺) | .22-.25 | .25-.30 |
| Conscientiousness | .16 (α⁺) | .18-.20 | .22-.25 |
| Neuroticism | .14 (LA ratio) | .18-.22 | .22-.28 |
| Openness | .13 (ω₀) | .15-.18 | .20-.25 |
| Agreeableness | .00 | .00 | .00 (without social variant) |

**Rationale for upper bounds**:
- Method variance (behavior ≠ self-report) creates ceiling around r = .30-.35
- Similar to other behavior-personality correlations in literature
- Hierarchical models might push slightly higher by reducing measurement error

---

## Part 6: Validation Strategy

For any new model, validate using:

1. **Convergent validity**: Does the new parameter correlate with its expected personality trait?
2. **Discriminant validity**: Does it NOT correlate with other traits (or correlate less)?
3. **Incremental validity**: Does it explain variance BEYOND current parameters?
4. **RT convergence**: Does the parameter correlate with simple RT metrics (like loss aversion does)?
5. **Cross-sample replication**: Does the pattern hold in independent samples?

---

## Summary

The Range Learning model is a strong foundation, but personality extraction can be improved by:

1. **Integrating RT** (immediate, low-hanging fruit)
2. **Adding dynamic parameters** (threshold drift, sequence-dependent learning)
3. **Separating cognition from motivation** (risk perception integration)
4. **Novel architectures** (dual-process, hierarchical Bayesian) for deeper personality modeling

The ceiling for BART-personality correlations is likely r ≈ .30-.35 due to method variance, but current correlations of r ≈ .15-.20 can likely be pushed toward this ceiling with model improvements.

**Recommended first step**: Implement RT-integrated Range Learning model and test whether Neuroticism and Extraversion correlations increase.
