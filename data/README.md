# Data Directory

This directory contains the raw data files for the BART (Balloon Analogue Risk Task) analysis.

## Directory Structure

```
data/
├── bart/                    # BART task data
├── questionnaires/          # Psychological questionnaires
├── decision_tasks/          # Other decision-making tasks
└── model_parameters/        # Computational model outputs
```

---

## bart/

Trial-level BART data from N = 1,507 participants.

| File | Rows | Description |
|------|------|-------------|
| `bart_pumps.csv` | 45,119 | Individual pump decisions per trial |
| `bart_riskperc.csv` | 17,115 | Risk perception ratings for balloon sizes |
| `bart_rts.csv` | 43,594 | Reaction times for each pump (127 columns) |

### bart_pumps.csv Columns
- `partid`: Participant ID
- `trial`: Trial number (1-30)
- `block`: Block number (1-3)
- `pumps`: Number of pumps on this trial
- `exploded`: Whether balloon exploded (0/1)
- `payoff`: Points earned (0 if exploded)

---

## questionnaires/

Psychological assessments and individual differences measures.

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `quest_scores.csv` | 1,507 | 68 | Aggregated questionnaire scale scores |
| `quest_proc.csv` | 1,507 | 642 | Raw item-level responses |
| `quest_codebook.pdf` | - | - | Documentation for all variables |
| `perso.csv` | 1,507 | 8 | Big Five personality (NEO) + STAI anxiety |
| `wmc.csv` | 1,507 | 12 | Working memory capacity measures |

### Key Scales in quest_scores.csv
- **BIS**: Barratt Impulsiveness Scale (total and subscales)
- **SSSV**: Sensation Seeking Scale (total and subscales: SStas, SSexp, SSdis, SSbor)
- **DOSPERT**: Domain-Specific Risk-Taking (Deth, Dinv, Dgam, Dhea, Drec, Dsoc + _r and _b variants)
- **AUDIT/FTND/DAST**: Substance use measures
- **GABS/PG**: Gambling measures
- **NUM**: Numeracy

### perso.csv Columns
- `NEO_A/C/E/N/O`: Big Five personality traits
- `STAI_trait`: State-Trait Anxiety Inventory (trait subscale)

---

## decision_tasks/

Other behavioral decision-making tasks for convergent validity.

| File | Rows | Description |
|------|------|-------------|
| `cct_overt.csv` | 1,507 | Columbia Card Task (CCT) - risk-taking task |
| `dfd_perpers.csv` | 1,507 | Decisions from Description task |
| `dfe_perpers.csv` | 1,507 | Decisions from Experience task |
| `lotteriesOvert.csv` | 1,507 | Lottery choice preferences |
| `mt.csv` | 1,507 | Memory task performance |

### cct_overt.csv (Key for convergent validity)
- `CCTncards`: Total cards turned over
- `CCTratio`: Risk-taking ratio
- `CCTpayoff`: Total earnings

---

## model_parameters/

Computational model outputs from Range Learning model fitting.

| File | Rows | Description |
|------|------|-------------|
| `range_learning_corrected_results (1).csv` | 1,507 | Full sample model parameters |
| `test_results.csv` | 100 | Subset with additional parameters |

### Model Parameters
- `omega_0`: Initial belief about burst probability (threshold)
- `rho_0`: Risk sensitivity parameter
- `alpha_minus`: Learning rate from negative outcomes (explosions)
- `alpha_plus`: Learning rate from positive outcomes (successful pumps)
- `sigma`: Decision noise parameter
- `mean_pumps`: Average pumps per trial
- `explosion_rate`: Proportion of trials ending in explosion

---

## Sample Sizes

| Dataset | N |
|---------|---|
| BART trials | 45,119 (from 1,507 participants) |
| Full questionnaires | 1,507 |
| Big Five personality | ~287 (subset) |
| Model parameters | 1,507 |

---

## Data Quality Notes

1. Big Five personality data (perso.csv) has smaller N (~287) than other measures
2. Some participants have missing data on specific scales
3. All participant IDs are anonymized (format: 64XXXXXX)
