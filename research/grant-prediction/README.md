# Predicting social-grant exclusion among food-insecure KwaZulu-Natal households

Honours analysis of the Statistics South Africa Living Conditions Survey (LCS) **2014/2015**, restricted to KwaZulu-Natal. The modelling question is: among households already classified as food-insecure, which ones receive no social grant?

This page only states numbers that sit next to a file in this folder. A single headline classification score was **not** locked in the written assignments, so it is marked `[METRIC NEEDED]`.

## Locked facts

| Item | Value | Where it is in this work |
|---|---|---|
| **Target** | `grant_excluded` = 1 if a **food-insecure** household has **no** grant-coded expenditure, else 0 | `notebooks/GrantDeterminer.ipynb`; `src/social_grant.py`; `reports/analysis-report.md` |
| **Years** | LCS 2014/2015 (files named `LCS2015*.csv`) | report, notebook, scripts |
| **Geography** | KwaZulu-Natal (`province_code == 5`) | same |
| **Sample (KZN)** | 3,686 households | report; notebook descriptive cells |
| **Modelling n** | 1,253 food-insecure households (901 grant recipients, 352 excluded; 28% excluded) | report; notebook `value_counts` |
| **Split** | Stratified 80/20, `random_state=42`. Train 1,002 (721 / 281); test 251 (180 / 71). SMOTE on train only → 1,442 | notebook cell outputs |
| **Baseline** | `[METRIC NEEDED]` | ROC plots draw a random-guess line at AUC = 0.5. No majority-class or dummy classifier was fitted or reported |
| **Models** | L1 logistic regression (interpretable primary in the concept note), decision tree, random forest (`RandomizedSearchCV`), AdaBoost | Assignment 4 methods; notebook; `src/social_grant.py` |
| **Metric the write-ups said they would use** | ROC-AUC and weighted F1 (also precision, recall, accuracy) | Assignment 1 and Assignment 4 methods. **No assignment text published a final number** |
| **Headline metric** | `[METRIC NEEDED]` | Two executed pipelines disagree (see below). Do not quote one number as *the* result until you pick a run |

### Food-insecurity definition (same in report and code)

Six binary indicators; household is food-insecure if the sum is **≥ 2**:

1. `Q224ANOMONEY` — no money to buy food
2. `Q225ASIZE` — reduced meal size
3. `Q226ASKIP` — skipped meals
4. `Q227ALESS` — ate less than needed
5. `adult_hungry` — adults hungry often/always (`Q222ADULT` in {3, 4})
6. `child_hungry` — children hungry often/always (`Q223CHILD` in {3, 4})

Grant receipt is any household expenditure with COICOP codes `50331000`, `50332000`, `50332100`, `50333100`, `50333200`, `50333300`, `50333400`, `50333500`.

## Computed scores (not locked in a write-up)

Keep these labelled as *computed*, not as the official result.

**Executed notebook** (`notebooks/GrantDeterminer.ipynb`) — unweighted test set, n = 251, 52 predictors after dummy encoding (UQNO and weights dropped from `X`):

| Model | Accuracy | ROC-AUC | Excluded-class F1 (from classification report) |
|---|---|---|---|
| Logistic regression | 0.8167 | 0.8826 | 0.70 |
| Decision tree | 0.7928 | 0.7489 | 0.64 |
| Random forest | 0.8606 | 0.8937 | 0.74 |
| AdaBoost | 0.8287 | 0.9049 | 0.70 |
| Tuned random forest | 0.8526 | 0.8931 | 0.73 |

Tuned forest inner CV ROC-AUC on SMOTE-balanced folds: 0.9679 (optimistic relative to the test AUC).

**Weighted pipeline figure** (`reports/figures/weighted_metrics_table.png`, produced by `src/social_grant.py`; wider feature set than the notebook):

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic regression | 0.7689 | 0.801 | 0.6606 | 0.6823 | 0.6713 |
| Decision tree | 0.7829 | 0.7498 | 0.7041 | 0.6424 | 0.6718 |
| Random forest | 0.8656 | 0.9109 | 0.8614 | 0.7286 | 0.7895 |
| AdaBoost | 0.8556 | 0.9185 | 0.8051 | 0.7687 | 0.7865 |

The ROC figure for that run labels AdaBoost weighted AUC = 0.919 and random forest 0.911, with a random-guess diagonal at 0.5.

## Odds ratios that *were* written in the analysis report

From `reports/analysis-report.md` (L1-penalised logistic regression among food-insecure households):

| Predictor | Odds ratio | Report interpretation |
|---|---|---|
| Household size | 0.55 | Larger households less likely excluded |
| Head employed | 3.74 | Employment raises exclusion risk (working poor) |
| Female household head | 0.30 | Female-headed households less likely excluded |
| Log income | 0.60 | Higher income associated with exclusion |
| `Q532WELFARE_5` | 8.61 | Specific welfare-access category |

These are associations, not causal effects.

## Data (not in git)

Stats SA LCS 2014/2015 household, person, income, assets, and COICOP total files. Licensed survey microdata. **Not republished here.**

Required local files (set `LCS_DATA_DIR` to the folder that holds them):

- `LCS2015HOUSEHOLD.csv`
- `LCS2015HOUSEHOLDASSETS.csv`
- `LCS2015PERSONINCOME.csv`
- `LCS2015PERSONSFINAL.csv`
- `LCS2015TOTALLCS.csv`

Household key in the source files is `UQNO`. That identifier is excluded from the modelling matrix in the scripts; notebook outputs that printed raw `UQNO` values were redacted before this copy was staged.

## Layout

```
research/grant-prediction/
  README.md
  requirements.txt
  .gitignore
  notebooks/GrantDeterminer.ipynb   executed analysis (metrics in cell outputs)
  src/social_grant.py               weighted CV + SHAP pipeline
  src/grant_analysis_cleaned.py     descriptive coverage + logistic helper
  reports/analysis-report.md        findings written up from the local Word report
  reports/figures/                  ROC, SHAP, coverage, odds-ratio plots
```

## Run

```bash
pip install -r requirements.txt
set LCS_DATA_DIR=C:\path\to\LCS2015   # PowerShell: $env:LCS_DATA_DIR = "..."
python src/social_grant.py
```

## What this model must not be used for

- Ranking households as “deserving” of a grant
- Auto-rejecting or auto-approving SASSA applications
- Causal claims (“employment causes exclusion”)

## Do not commit

- Any `LCS2015*.csv` or `LCS Dataset.zip`
- Files that print household `UQNO` values, person numbers, names, or ID numbers
- Student-number assignment coversheets
- `grant_env/`
