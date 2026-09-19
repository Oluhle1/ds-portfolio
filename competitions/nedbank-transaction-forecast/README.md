# Nedbank transaction-volume forecast (Zindi / N*ovation)

Zindi: [Nedbank Transaction Volume Forecasting Challenge](https://zindi.africa/). Handle **Oluhle**. Certificate dated 3 May 2026. Final board closed; points posted 18 May 2026.

## Official result (final leaderboard + certificate)

| | |
|---|---|
| Place | **89 / 251** (certificate: ranked top 50%) |
| Public RMSLE | **0.38759** |
| Private RMSLE | **0.38256** |
| Submissions | 92 |
| Track | Zindi data science / ML |
| Finale / prize | Not claimed here |

Winner on the same final board (edwardrycroft): public 0.37499, private 0.35819. Gap to first on private is about 0.024 RMSLE. Do not round that into “close to first.”

## Local CV (training log, not the leaderboard)

Device: CPU. Five folds. SciPy blend used for the run that produced test preds.

| Model | 5-fold CV RMSLE |
|---|---|
| LightGBM | 0.37706 |
| XGBoost | 0.37747 |
| CatBoost | 0.37914 |
| MLP | 0.40649 |
| Ridge | 0.42034 |
| SciPy blend | 0.37403 |

Blend weights: LightGBM 0.4119, XGBoost 0.2529, CatBoost 0.1803, MLP 0.1548, Ridge 0.

CV 0.374 vs private 0.383 is the honest generalisation gap. Fold 3 (~0.40) vs fold 4 (~0.35) already showed the CV was uneven.

## Still open

- Split type for those five folds: `[KFold vs time / group]` — say which in an interview
- Dumb baseline RMSLE: `[METRIC NEEDED]`

## Claims that are now allowed

- Completed the Nedbank/Zindi 3-month transaction-count forecast (RMSLE)
- Finished **89th of 251**, private RMSLE **0.383**, public **0.388**
- Local blend CV RMSLE 0.374; private score was worse than CV

## Claims that are still forbidden

- Finalist, prize, or Nedbank hire
- “Top of the board” / “near first”
- Publishing Train/Test customer files

## Layout

Notebooks and scripts only under `notebooks/` and `src/`. No raw CSVs.
