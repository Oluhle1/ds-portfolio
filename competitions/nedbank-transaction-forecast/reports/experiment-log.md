# Experiment log

Numbers below are copied from files in this workspace. If a notebook never printed a final CV score, the cell is `[METRIC NEEDED]`. Public and private Zindi leaderboard scores are not stored here.

Target everywhere: `next_3m_txn_count` (count of transactions in Nov 2015–Jan 2016). Models train on `log1p(y)`. Submissions write `log1p(raw_pred)`. Metric: RMSLE = RMSE on `log1p`.

Feature cutoff used in v17–v28: **2015-10-31**. Competition train/test split is given (`Train.csv` / `Test.csv`). Local CV is **StratifiedKFold on binned log-target**, not a later time holdout.

## Recorded scores

| Source | What was scored | RMSLE |
|---|---|---|
| `submissions/metrics_v12.json` | v12 simple-blend OOF (10 seeds; CatBoost OOF 0.38675, LightGBM OOF 0.38743) | **0.385615** |
| `submissions/metrics_v12.json` | v12 stacked OOF | 0.756039 (worse; not used) |
| `nedbank_v17.ipynb` | 3-fold Optuna tune (widget: Cat 0.398544, LGB 0.398406, XGB 0.398467) | tune only; final 5-fold blend **[METRIC NEEDED]** |
| `nedbank_v18.ipynb` | same Optuna widget values as v17 | tune only; final 5-fold blend **[METRIC NEEDED]** |
| `nedbank_v26.ipynb` / `src/nedbank_v26.py` | 5-fold inverse-error blend | **[METRIC NEEDED]** |
| `nedbank_v28.ipynb` stored output | 5-fold OOF CatBoost / LightGBM / XGBoost / Ridge | 0.379145 / 0.377591 / 0.378796 / 0.420336 |
| `nedbank_v28.ipynb` stored output | inverse-error blend CV | **0.377504** |
| `nedbank_v28.ipynb` current source | TabMLP + scipy constrained blend + full-data tree refit | **[METRIC NEEDED]** (those columns are not in the stored fold table) |
| `StarterNotebook.ipynb` | train-mean baseline | **[METRIC NEEDED]** |
| Public / private Zindi leaderboard | — | **not recorded in this repo** |

v28 note: the saved cell output prints `fold N complete` and a four-column fold table (cat / lgb / xgb / ridge), and `Saved: /content/submission_v26.csv`. The **source** of that same notebook later adds TabMLP, scipy blending, and `submission_v28.csv`. Treat **0.377504** as the recorded 4-model blend, not as proof that the MLP path was scored.

v12 generating notebook is not in this workspace; only the metrics JSON is.

## v28 fold table (stored output)

| fold | cat | lgb | xgb | ridge |
|---:|---:|---:|---:|---:|
| 1 | 0.383928 | 0.382416 | 0.386516 | 0.421226 |
| 2 | 0.373937 | 0.371024 | 0.372556 | 0.410340 |
| 3 | 0.402131 | 0.395228 | 0.395384 | 0.456195 |
| 4 | 0.347661 | 0.350140 | 0.349753 | 0.384450 |
| 5 | 0.385936 | 0.387519 | 0.388049 | 0.426248 |

Blend weights in that output: cat 0.256, lgb 0.257, xgb 0.256, ridge 0.231.
