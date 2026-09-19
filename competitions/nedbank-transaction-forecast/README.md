# Nedbank transaction-volume forecast (Zindi)

Public write-up of local work on Nedbank’s anonymised customer transaction forecast challenge on Zindi. Training files are **not** in this repo.

## Problem

Predict `next_3m_txn_count`: how many bank transactions each customer makes in **November 2015–January 2016**, from history that ends on **31 October 2015**.

- Train customers: 8,360 (`Train.csv`)
- Test customers: 3,584 (`Test.csv`)
- Inputs: `transactions_features.parquet`, `financials_features.parquet`, `demographics_clean.parquet`
- Metric: **RMSLE** (RMSE on `log1p`)
- Submission column `next_3m_txn_count` must be **`log1p` of the raw count**, not the raw count

A challenge note in `reports/NedbankChallenge.md` records a public-board score of `0.3505610` as a target to beat. That is **not** this repo’s score, and no leaderboard place is claimed here.

## What is actually in this folder

| Path | Role |
|---|---|
| `notebooks/StarterNotebook.ipynb` | Host starter: load files, mean baseline, write `submission.csv` |
| `notebooks/nedbank_v17.ipynb` | Optuna-tuned CatBoost + LightGBM + XGBoost + ExtraTrees blend |
| `notebooks/nedbank_v18.ipynb` | v17 successor: Ridge instead of ExtraTrees; heavier Optuna |
| `notebooks/nedbank_v26.ipynb` | Fixed-param CatBoost + LightGBM + XGBoost + Ridge, inverse-error blend |
| `notebooks/nedbank_v28.ipynb` | v26 plus TabMLP, scipy blend, full-data tree refit (source). Stored output is the 4-model blend |
| `src/nedbank_v26.py` | Script export of v26 |
| `src/evaluate.py` | Local RMSLE scorer (`submission` vs `reference`) |
| `src/make_v18.py` | Rewrites v17 cells into `nedbank_v18.ipynb` |
| `reports/` | Challenge spec, package readme, experiment log |

Notebooks were copied **code + markdown only**. Outputs that printed customer rows were stripped.

Not copied: `Train.csv`, `Test.csv`, `SampleSubmission.csv`, `VariableDefinitions.csv`, `*.parquet`, any `submission*.csv`, `.venv`. `ML-Nedbank/nedbank_v28.ipynb` is an extra copy of an earlier v28 and was left out.

## Results I can cite from files

Best **recorded** local CV in this workspace: **RMSLE 0.377504**, 5-fold customer-stratified OOF, CatBoost / LightGBM / XGBoost / Ridge inverse-error blend, stored in `nedbank_v28.ipynb` output (see `reports/experiment-log.md`).

| Version | Split | Model | RMSLE |
|---|---|---|---|
| v12 (`metrics_v12.json`; notebook missing) | OOF, 10 seeds | CatBoost + LightGBM simple blend | 0.385615 |
| v17 | 5-fold StratifiedKFold × 3 seeds; 3-fold Optuna tune | CatBoost + LightGBM + XGBoost + ExtraTrees | **[METRIC NEEDED]** (tune widgets ~0.398) |
| v18 | 5-fold × 3 seeds; 3-fold Optuna | CatBoost + LightGBM + XGBoost + RidgeCV | **[METRIC NEEDED]** (tune widgets ~0.398) |
| v26 | 5-fold StratifiedKFold, seed 42 | CatBoost + LightGBM + XGBoost + Ridge | **[METRIC NEEDED]** |
| v28 stored output | 5-fold StratifiedKFold, seed 42 | same 4 models, inverse-error blend | **0.377504** |
| v28 current source | same folds | + TabMLP (seeds 42/123/777) + scipy blend + 50/50 full refit | **[METRIC NEEDED]** |
| Public LB | ~30% of test | — | **not recorded** |
| Private LB | ~70% of test | — | **not recorded** |

## Method (v26 / v28, the latest complete pipeline)

1. **Time barrier.** Drop transactions and financial snapshots after 2015-10-31. Do not use the target window as a feature.
2. **Customer features.** Recency, tenure, rolling 1–12 month counts and amounts, debit/credit mix, statement-balance stats, monthly slopes, Nov–Jan holiday windows from prior years, year-on-year pre-holiday ratios, type/batch/reversal count pivots, product-level financial snapshots, age/income with missingness flags.
3. **Categoricals.** One-hot if cardinality ≤ 25; frequency encoding on all object columns; out-of-fold target encoding up to cardinality 120.
4. **CV.** `StratifiedKFold` (5 folds) on quantile bins of `log1p(target)`. This is a customer split, not a later calendar holdout. Train vs test IDs are the official files.
5. **Models.** Trees on `y_log`. v26 blends with inverse CV error. v28 source also trains a tabular MLP and a non-negative scipy blend, then mixes fold-average tree preds 50/50 with a full-data refit.
6. **Submit.** `expm1` of the blended log pred, clip at 0, then write `log1p` again for Zindi.

v17/v18 add Optuna (Cat/LGB/XGB) and multi-seed OOF before a constrained SLSQP blend. v18 drops ExtraTrees for RidgeCV.

There is **no implemented dumb baseline score** in the later notebooks. The starter writes the train-mean as a submission but does not print RMSLE.

## Layout

```
competitions/nedbank-transaction-forecast/
  README.md
  notebooks/     stripped feature + model notebooks
  src/           evaluate.py, v26 script, v18 notebook builder
  reports/       challenge spec + experiment log
```

## Do not commit

- `Train.csv`, `Test.csv`, parquet extracts, or `VariableDefinitions.csv`
- submission files with customer-level predictions
- API keys, Zindi tokens, `.venv`

## How to rerun (data stays local)

Put the official files next to the notebook, or point `DATA_PATH` at that folder. Then run `notebooks/nedbank_v28.ipynb` (or `src/nedbank_v26.py`). Optional local score, only if you have a reference file:

```bash
python src/evaluate.py submission_v28.csv PublicReference.csv
```
