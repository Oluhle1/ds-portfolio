# Nedbank transaction-volume forecast (Zindi / N*ovation)

Public write-up of work on Nedbank’s anonymised customer transaction forecast challenge hosted with Zindi as part of the N*ovation Data and Analytics Masters pathway.

## What the challenge was

Predict each customer’s bank-transaction **count** over a future three-month window from anonymised behavioural history (transaction history, monthly snapshots, demographics). Host metric: **RMSLE**. Training data was Nedbank/Zindi-provided and is **not republished here**.

## Locked local results (from training log)

Device: CPU. Five folds. Blend chosen by SciPy weight search on CV.

| Model | 5-fold CV RMSLE |
|---|---|
| LightGBM | 0.37706 |
| XGBoost | 0.37747 |
| CatBoost | 0.37914 |
| MLP | 0.40649 |
| Ridge | 0.42034 |
| **SciPy blend (used)** | **0.37403** |
| Inverse-weight blend (not used) | 0.37659 |

Blend weights: LightGBM 0.4119, XGBoost 0.2529, CatBoost 0.1803, MLP 0.1548, Ridge **0.0**.

Full-data refit boosting rounds before test preds: CatBoost 1905, LightGBM 730, XGBoost 785.

Per-fold RMSLE (same log):

| Fold | CatBoost | LightGBM | XGBoost | Ridge | MLP |
|---|---|---|---|---|---|
| 1 | 0.38393 | 0.38129 | 0.38418 | 0.42123 | 0.41476 |
| 2 | 0.37394 | 0.37305 | 0.37128 | 0.41034 | 0.39422 |
| 3 | 0.40213 | 0.39334 | 0.39401 | 0.45620 | 0.43963 |
| 4 | 0.34766 | 0.35099 | 0.34938 | 0.38445 | 0.37597 |
| 5 | 0.38594 | 0.38521 | 0.38687 | 0.42625 | 0.40512 |

Fold 3 is the hard fold; fold 4 is the easy fold. That spread is part of the result, not noise to hide.

## Still not locked — do not invent

- Split type: `[KFold vs time / group split]` — say which, or the CV number is weaker than it looks
- Dumb baseline RMSLE (per-customer historical 3-month mean): `[METRIC NEEDED]`
- Public leaderboard RMSLE: `[METRIC NEEDED]`
- Private / final RMSLE: `[METRIC NEEDED]`
- Zindi handle / team name: `[ ]`
- Finale invite: `[yes / no]`
- Track: assume Zindi DS/ML unless you say Otinga DE

CV 0.374 is **not** a competition place and **not** a production SLA.

## What the log already supports in an interview

- Ensemble of tree boosters plus a small MLP; linear ridge added no weight
- Blend beat every single model on the same CV (0.374 vs best single 0.377)
- Refit on full train after CV, then write test predictions — standard competition close
- You should be ready to explain whether folds were time-safe. If they were random KFold on customers with a future window target, say that out loud

## Layout

```
competitions/nedbank-transaction-forecast/
  README.md
  notebooks/     feature + model notebooks (no raw CSVs)
  src/
  reports/
```

## Do not commit

Train/Test extracts, submission CSVs with customer rows, Zindi tokens.
