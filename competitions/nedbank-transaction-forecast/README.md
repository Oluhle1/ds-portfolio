# Nedbank transaction-volume forecast (Zindi / N*ovation)

Public write-up of work on Nedbank’s anonymised customer transaction forecast challenge hosted with Zindi as part of the N*ovation Data and Analytics Masters pathway.

## What the challenge was

Predict each customer’s bank-transaction **count** over a future three-month window from anonymised behavioural history (transaction history, monthly snapshots, demographics). Evaluation: **RMSLE**. Training data was Nedbank/Zindi-provided and is **not republished here**.

Sources for the public problem statement (not my score): Nedbank / Zindi challenge materials and press on the 2026 Masters challenge.

## My entry — fill before you show this to a hiring manager

- Track: `[Data Science / ML on Zindi | Data Engineering on Otinga | both]`
- Handle / team: `[ ]`
- Local validation RMSLE: `[ ]` (say the split)
- Public leaderboard RMSLE: `[ ]`
- Private / final: `[ ]` if known
- Finale invite: `[yes / no]`

Until those lines are filled, this folder is an approach note, not a results claim.

## Approach I can already describe without a notebook

Banking forecast work on this problem usually dies on three things: leakage across time, high-cardinality text fields, and a right-skewed count target. The write-up should show, in this order:

1. Time split that does not peek into the three-month target window
2. Features known at forecast origin only (recency, frequency, monetary aggregates, calendar seasonality, product mix)
3. A dumb baseline (per-customer historical mean of 3-month counts, or last-3-month run-rate)
4. One model that beats that baseline on RMSLE, with an error plot by customer activity band
5. What still broke (cold-start customers, sparse months, text fields I did not use)

## Layout

```
competitions/nedbank-transaction-forecast/
  README.md
  notebooks/     feature + model notebooks (no raw CSVs)
  src/           feature builders
  reports/       1–2 page methods note for interviews
```

## Do not commit

- `Train.csv`, `Test.csv`, or any Nedbank customer extract
- submission files that still contain customer-level predictions if the host forbids it
- API keys, Zindi tokens

Add those names to `.gitignore` even if you keep local copies.
