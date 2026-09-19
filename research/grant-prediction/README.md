# Grant-outcome modelling

Predict whether / how a funding application resolves, using only information that would be known at decision time.

## Status

**Draft repo.** Notebooks and locked metrics are not in git yet. Do not cite an AUC, F1, or accuracy from this page until those files land.

## Problem (fill)

- Target: `[award yes/no | amount band | time-to-decision]`
- Population: `[years, geography, funder types, n = ?]`
- Decision point: features must be available *before* the outcome is known

## Method (fill after notebooks land)

- Baseline: `[majority class / logistic / simple tree]`
- Model family: `[ ]`
- Split: `[by year / by funder / random — say which, and why leakage was checked]`
- Metric matched to the decision: `[PR-AUC | Brier | log-loss | …]` — not raw accuracy if the classes are unbalanced

## What this model must not be used for

- Ranking people as “deserving” of funding
- Auto-rejecting applications
- Any claim about causality (“this feature caused the award”)

## Layout

```
research/grant-prediction/
  README.md          this file
  notebooks/         analysis notebooks (add here)
  src/               reusable scripts (add here)
  reports/           short PDF or markdown methods note (add here)
```

Keep raw application text and personal identifiers out of git. If the source data cannot be shared, put a data card here instead of the files.
