# Analysis report: social-grant coverage and food insecurity in KwaZulu-Natal

Source: local `GrantDeterminer_Report.docx` in the honours working folder. Classification scores in the notebook and weighted-metrics figure are **not** restated here as official results; see the project README.

## Data

Five files from the 2015 Living Conditions Survey, filtered to KwaZulu-Natal (`province_code = 5`): **3,686** households.

## Food insecurity

Six-indicator score (no money for food, reduced meal size, skipped meals, ate less than needed, adults hungry often/always, children hungry often/always). Score **≥ 2** → food-insecure.

- Food-insecure: **1,253 (34%)**
- Food-secure: **2,433 (66%)**

## Grant receipt

Any household with at least one COICOP expenditure in `50331000`–`50333500` is a grant recipient.

- All KZN households receiving grants: **2,216 (60%)**
- Food-insecure **with** grants: **901 (72%)**
- Food-insecure **without** grants: **352 (28%)**

The coverage gap the model targets is those 352 households.

## Model used in this write-up

Weighted regularized logistic regression (L1, alpha = 0.01) of `grant_excluded` among food-insecure households. Survey weights applied. Predictors: household demographics, education, income, settlement type, welfare-access indicators.

## Predictors reported as significant (p < 0.05)

| Variable | Odds ratio | Interpretation in the report |
|---|---|---|
| Household size | 0.55 | Larger households less likely excluded |
| Head employed | 3.74 | Employment increases exclusion risk |
| Female household head | 0.30 | Female-headed households less likely excluded |
| Log income | 0.60 | Higher income associated with exclusion |
| `Q532WELFARE_5` | 8.61 | Specific welfare-access indicator |

The report’s policy reading: working-poor households (employed heads who are still food-insecure) face higher exclusion; female-headed and larger households have better coverage (possibly child-support eligibility). These are associations, not causal effects.

## Classification metrics

Not reported in the Word document. Planned in the honours concept notes: ROC-AUC and weighted F1. Executed numbers live in `notebooks/GrantDeterminer.ipynb` and `reports/figures/weighted_metrics_table.png`. A single headline score is still `[METRIC NEEDED]`.
