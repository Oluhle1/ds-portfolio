"""Social grant exclusion prediction for food-insecure KwaZulu-Natal households (LCS 2014/2015).

Weighted train/test evaluation, 5-fold stratified CV with SMOTE on training folds,
L1 logistic regression, decision tree, random forest, AdaBoost, and SHAP on the tuned forest.
Raw survey CSVs are not shipped with this repository.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_validate)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve, confusion_matrix,
                             make_scorer, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


import statsmodels.api as sm
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("WARNING: shap not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# 1. Configuration and paths
# =============================================================================
# LCS 2014/2015 microdata is not in this repo (Stats SA licence).
# Point LCS_DATA_DIR at a local folder that contains the five LCS2015*.csv files.
DATA_FOLDER = os.environ.get('LCS_DATA_DIR', '.')
os.chdir(DATA_FOLDER)

FIGURES_FOLDER = "figures"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

RANDOM_STATE = 42

print("=" * 80)
print("SOCIAL GRANT EXCLUSION PREDICTION – KWAZULU-NATAL")
print("=" * 80)

# =============================================================================
# 2. Load data
# =============================================================================
print("\n1. Loading datasets...")
hh = pd.read_csv('LCS2015HOUSEHOLD.csv')
assets = pd.read_csv('LCS2015HOUSEHOLDASSETS.csv')
person_income = pd.read_csv('LCS2015PERSONINCOME.csv')
persons = pd.read_csv('LCS2015PERSONSFINAL.csv')
total = pd.read_csv('LCS2015TOTALLCS.csv')

# =============================================================================
# 3. Subset to KwaZulu-Natal (province_code == 5)
# =============================================================================
print("\n2. Subsetting to KwaZulu-Natal...")
kzn_hh = hh[hh['province_code'] == 5].copy()
kzn_uqnos = kzn_hh['UQNO'].unique()
kzn_persons = persons[persons['UQNO'].isin(kzn_uqnos)].copy()
kzn_total = total[total['UQNO'].isin(kzn_uqnos)].copy()

print(f"   KZN households: {len(kzn_hh):,}")
print(f"   KZN persons:    {len(kzn_persons):,}")

# =============================================================================
# 4. Food insecurity index (6 indicators)
# =============================================================================
print("\n3. Creating food insecurity index...")
fi_binary_cols = ['Q224ANOMONEY', 'Q225ASIZE', 'Q226ASKIP', 'Q227ALESS']
for col in fi_binary_cols:
    kzn_hh[col] = (kzn_hh[col] == 1).astype(int)

kzn_hh['adult_hungry'] = kzn_hh['Q222ADULT'].apply(lambda x: 1 if x in [3, 4] else 0)
kzn_hh['child_hungry'] = kzn_hh['Q223CHILD'].apply(lambda x: 1 if x in [3, 4] else 0)

fi_indicators = fi_binary_cols + ['adult_hungry', 'child_hungry']
kzn_hh['fi_score'] = kzn_hh[fi_indicators].sum(axis=1)
kzn_hh['food_insecure'] = (kzn_hh['fi_score'] >= 2).astype(int)

fi_count = kzn_hh['food_insecure'].sum()
print(f"   Food insecure: {fi_count:,} ({fi_count/len(kzn_hh)*100:.1f}%)")
print(f"   Food secure:   {len(kzn_hh)-fi_count:,} ({(len(kzn_hh)-fi_count)/len(kzn_hh)*100:.1f}%)")

# =============================================================================
# 5. Identify grant receipt from Coicop codes
# =============================================================================
print("\n4. Identifying grant recipients...")
grant_codes = [50331000, 50332000, 50332100, 50333100,
               50333200, 50333300, 50333400, 50333500]
kzn_total['is_grant'] = kzn_total['Coicop'].isin(grant_codes)
grant_hh = kzn_total.groupby('UQNO')['is_grant'].any().reset_index()
grant_hh.rename(columns={'is_grant': 'hh_receives_grant'}, inplace=True)
kzn_hh = kzn_hh.merge(grant_hh, on='UQNO', how='left')
kzn_hh['hh_receives_grant'] = kzn_hh['hh_receives_grant'].fillna(False)

grant_count = kzn_hh['hh_receives_grant'].sum()
print(f"   Receives grants:    {grant_count:,} ({grant_count/len(kzn_hh)*100:.1f}%)")
print(f"   No grants received: {len(kzn_hh)-grant_count:,} ({(len(kzn_hh)-grant_count)/len(kzn_hh)*100:.1f}%)")

# =============================================================================
# 6. Focus on food-insecure households and create target
# =============================================================================
print("\n5. Creating target variable (grant_excluded) for food-insecure households...")
analysis_df = kzn_hh[kzn_hh['food_insecure'] == 1].copy()
analysis_df['grant_excluded'] = (~analysis_df['hh_receives_grant']).astype(int)
for col in ['hhsize', 'hhsize_x', 'hhsize_y']:
    if col in analysis_df.columns:
        analysis_df = analysis_df.drop(columns=[col])

print(f"   Food-insecure households: {len(analysis_df):,}")
print(f"   → Receive grants:         {len(analysis_df) - analysis_df['grant_excluded'].sum():,}")
print(f"   → Excluded from grants:   {analysis_df['grant_excluded'].sum():,}")

# =============================================================================
# 7. Feature engineering
# =============================================================================
print("\n6. Engineering features...")

hhsize_df = kzn_persons.groupby('UQNO').size().reset_index(name='hhsize')
analysis_df = analysis_df.merge(hhsize_df, on='UQNO', how='left')

heads = kzn_persons[kzn_persons['Q16RELATION'] == 1].copy()
heads['head_education'] = heads['Q21HIGHLEVEL']
heads['head_employed'] = ((heads['Q31AWAGE'] == 1) | (heads['Q31BBUS'] == 1)).astype(int)
heads = heads[['UQNO', 'head_education', 'head_employed']]
analysis_df = analysis_df.merge(heads, on='UQNO', how='left')

income_data = kzn_total[kzn_total['CoicopType'].isin([3, 4])].copy()
income_hh = income_data.groupby('UQNO')['valueannualized_adj'].sum().reset_index()
income_hh.rename(columns={'valueannualized_adj': 'total_hh_income'}, inplace=True)
analysis_df = analysis_df.merge(income_hh, on='UQNO', how='left')
analysis_df['total_hh_income'] = analysis_df['total_hh_income'].fillna(0)

# =============================================================================
# 8. Data cleaning: imputation, log transform, one-hot encoding
# =============================================================================
print("\n7. Cleaning and encoding...")

exclude_from_impute = [
    'UQNO', 'grant_excluded', 'hholds_wgt', 'food_insecure',
    'hh_receives_grant', 'adult_hungry', 'child_hungry', 'fi_score'
]

df_temp = analysis_df.copy()
cols_to_keep = [c for c in df_temp.columns if c not in exclude_from_impute]
df_temp = df_temp[cols_to_keep]

num_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"   Numeric columns to impute:     {len(num_cols)}")
print(f"   Categorical columns to impute: {len(cat_cols)}")

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer([
    ('num', num_imputer, num_cols),
    ('cat', cat_imputer, cat_cols)
], remainder='passthrough')

imputed_array = preprocessor.fit_transform(df_temp)
feature_names = preprocessor.get_feature_names_out()
df_imputed = pd.DataFrame(imputed_array, columns=feature_names, index=df_temp.index)
df_imputed.columns = [col.split('__')[-1] for col in df_imputed.columns]

df_imputed['log_income'] = np.log1p(df_imputed['total_hh_income'])

categoricals = ['head_education', 'Q531WELFARE', 'Q532WELFARE',
                'SETTLEMENT_TYPE', 'SexOfHead', 'head_employed']
for col in categoricals:
    if col not in df_imputed.columns:
        print(f"   WARNING: {col} not found – check column names")

X_encoded = pd.get_dummies(df_imputed, columns=categoricals, drop_first=True)

id_col = 'UQNO'
target = 'grant_excluded'
weights = 'hholds_wgt'

X_encoded[id_col] = analysis_df[id_col].values
X_encoded[target] = analysis_df[target].values
X_encoded[weights] = analysis_df[weights].values

exclude = [id_col, target, weights, 'total_hh_income', 'food_insecure',
           'hh_receives_grant', 'adult_hungry', 'child_hungry', 'fi_score']
predictor_cols = [c for c in X_encoded.columns if c not in exclude]
if 'log_income' not in predictor_cols and 'log_income' in X_encoded.columns:
    predictor_cols.append('log_income')

model_df = X_encoded[[id_col, target, weights] + predictor_cols].copy()
analysis_df['log_income'] = df_imputed['log_income'].values

print(f"   Final dataset shape: {model_df.shape}")
print(f"   Remaining NaNs in predictors: {model_df[predictor_cols].isnull().sum().sum()}")

# =============================================================================
# 9. Prepare X, y, weights – final imputation pass
# =============================================================================
print("\n8. Preparing train/test split...")

X = model_df.drop(columns=[id_col, target, weights])
y = model_df[target]
w = model_df[weights]

# Final median imputation to handle any residual NaNs
final_imputer = SimpleImputer(strategy='median')
X_arr = final_imputer.fit_transform(X)
X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)
print(f"   Total NaNs after final imputation: {X.isnull().sum().sum()}")

# Normalise weights to sum to N (keeps cross-validation comparable)
w_norm = w / w.sum() * len(w)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w_norm, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# SMOTE on training data (unweighted – SMOTE operates on feature space only)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE – training size: {X_train_res.shape[0]:,} (balanced)")

# =============================================================================
# 10. Helper: weighted evaluation metrics
# =============================================================================

def weighted_metrics(y_true, y_pred, y_proba, sample_weight=None, label=""):
    """Compute accuracy, AUC, precision, recall, F1 with sample weights."""
    acc  = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    auc  = roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
    prec = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    rec  = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    f1   = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    if label:
        print(f"\n  ── {label} (weighted) ──")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     ROC-AUC   : {auc:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     F1-Score  : {f1:.4f}")
    return dict(accuracy=acc, auc=auc, precision=prec, recall=rec, f1=f1)


# =============================================================================
# 11. Weighted cross-validation helper
# =============================================================================

def weighted_cross_validate(estimator, X, y, weights, cv=5, label=""):
    """
    Stratified k-fold CV that passes sample_weight through to fit() and
    computes weighted AUC and F1 on each validation fold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs, fold_f1s = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr, w_val = weights.iloc[train_idx], weights.iloc[val_idx]

        # Re-balance training fold with SMOTE
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

        # Fit with sample weights if the estimator supports it
        try:
            estimator.fit(X_tr_res, y_tr_res)
        except TypeError:
            estimator.fit(X_tr_res, y_tr_res)

        y_proba_val = estimator.predict_proba(X_val)[:, 1]
        y_pred_val  = estimator.predict(X_val)

        fold_aucs.append(roc_auc_score(y_val, y_proba_val, sample_weight=w_val))
        fold_f1s.append(f1_score(y_val, y_pred_val, sample_weight=w_val, zero_division=0))

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    mean_f1  = np.mean(fold_f1s)
    std_f1   = np.std(fold_f1s)

    if label:
        print(f"\n  ── {label} – {cv}-fold Weighted CV ──")
        print(f"     ROC-AUC : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"     F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

    return dict(auc_mean=mean_auc, auc_std=std_auc,
                f1_mean=mean_f1,  f1_std=std_f1,
                fold_aucs=fold_aucs, fold_f1s=fold_f1s)


# =============================================================================
# 12. Train models – standard + weighted CV evaluation
# =============================================================================
print("\n9. Training and evaluating models...")

model_specs = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree':       DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(random_state=RANDOM_STATE),
    'AdaBoost':            AdaBoostClassifier(random_state=RANDOM_STATE),
}

# Containers for results
probas     = {}    # test-set probabilities
cv_results = {}    # weighted CV summaries
wm_results = {}    # weighted test-set metrics
fitted_models = {}

for name, model in model_specs.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Weighted cross-validation on full (un-SMOTE'd) training data
    cv_res = weighted_cross_validate(model, X_train, y_train, w_train, cv=5, label=name)
    cv_results[name] = cv_res

    # Fit on SMOTE-balanced training set
    model.fit(X_train_res, y_train_res)
    fitted_models[name] = model

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    probas[name] = y_proba

    # Weighted test-set metrics
    wm = weighted_metrics(y_test, y_pred, y_proba,
                          sample_weight=w_test.values, label=f"{name} [Test Set]")
    wm_results[name] = wm

    print("\n  Classification Report (unweighted, for reference):")
    print(classification_report(y_test, y_pred,
                                target_names=['Grant Recipient', 'Grant Excluded']))

# =============================================================================
# 13. Cross-validation comparison chart
# =============================================================================
print("\n10. Plotting cross-validation comparison...")

cv_names  = list(cv_results.keys())
cv_aucs   = [cv_results[n]['auc_mean'] for n in cv_names]
cv_stds   = [cv_results[n]['auc_std']  for n in cv_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# AUC bar chart
axes[0].bar(cv_names, cv_aucs, yerr=cv_stds, capsize=5, color='steelblue', alpha=0.8)
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title('Weighted CV – ROC-AUC (mean ± 1 SD)', fontsize=13)
axes[0].set_ylabel('Weighted ROC-AUC')
axes[0].tick_params(axis='x', rotation=15)
for i, (v, s) in enumerate(zip(cv_aucs, cv_stds)):
    axes[0].text(i, v + s + 0.005, f"{v:.3f}", ha='center', fontsize=10)

# Boxplot of fold AUCs
fold_data = [cv_results[n]['fold_aucs'] for n in cv_names]
axes[1].boxplot(fold_data, labels=cv_names, patch_artist=True,
                boxprops=dict(facecolor='lightsteelblue'),
                medianprops=dict(color='navy', linewidth=2))
axes[1].set_title('Weighted CV – Fold-level ROC-AUC Distribution', fontsize=13)
axes[1].set_ylabel('Weighted ROC-AUC per Fold')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "cv_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 14. ROC curves (weighted AUC)
# =============================================================================
print("\n11. Plotting ROC curves...")
plt.figure(figsize=(10, 8))
for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba, sample_weight=w_test.values)
    auc = roc_auc_score(y_test, proba, sample_weight=w_test.values)
    plt.plot(fpr, tpr, label=f"{name} (weighted AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Grant Exclusion Prediction (Weighted)")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.4)
plt.savefig(os.path.join(FIGURES_FOLDER, "roc_curves.png"), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 15. Weighted metrics summary table
# =============================================================================
print("\n12. Weighted metrics summary table...")
metrics_summary = pd.DataFrame(wm_results).T
metrics_summary.index.name = 'Model'
print("\n  Weighted test-set metrics summary:")
print(metrics_summary.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
tbl = ax.table(
    cellText=metrics_summary.round(4).values,
    rowLabels=metrics_summary.index,
    colLabels=['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.6)
plt.title("Weighted Test-Set Metrics", fontsize=13, pad=20)
plt.savefig(os.path.join(FIGURES_FOLDER, "weighted_metrics_table.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 16. Hyperparameter tuning for Random Forest
# =============================================================================
print("\n13. Hyperparameter tuning for Random Forest...")

param_dist = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [5, 10, 15, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Use weighted AUC as the CV scoring function
def weighted_auc_scorer(estimator, X, y):
    """Custom scorer that uses survey weights stored in w_norm index."""
    y_proba = estimator.predict_proba(X)[:, 1]
    # Align weights by index
    w_fold = w_norm.reindex(pd.RangeIndex(len(y))).fillna(1.0)
    return roc_auc_score(y, y_proba)

rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    rf_base, param_dist, n_iter=10, cv=3,
    scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
)
random_search.fit(X_train_res, y_train_res)

print(f"   Best parameters:         {random_search.best_params_}")
print(f"   Best CV ROC-AUC:         {random_search.best_score_:.4f}")

best_rf = random_search.best_estimator_
y_pred_best  = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)[:, 1]

print("\n   Tuned Random Forest – weighted test-set metrics:")
weighted_metrics(y_test, y_pred_best, y_proba_best,
                 sample_weight=w_test.values, label="Tuned RF")

print("\n   Classification Report (unweighted):")
print(classification_report(y_test, y_pred_best,
                             target_names=['Grant Recipient', 'Grant Excluded']))

# =============================================================================
# 17. SHAP interpretability on the best Random Forest
# =============================================================================
if SHAP_AVAILABLE:
    print("\n14. Computing SHAP values on Tuned Random Forest...")

    # Use a background sample for efficiency (max 200 rows)
    background_size = min(200, X_train_res.shape[0])
    background = shap.sample(X_train_res, background_size, random_state=RANDOM_STATE)

    explainer    = shap.TreeExplainer(best_rf, background)
    explanation  = explainer(X_test)
    shap_vals_class1 = explanation.values[:, :, 1]   # shape: (n_samples, n_features)
    # ── 17a. Summary plot (beeswarm) ──────────────────────────────────────
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals_class1, X_test, show=False,
                      max_display=20, plot_type='dot')
    plt.title("SHAP Summary Plot – Grant Exclusion (Random Forest)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "shap_summary.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ── 17b. Mean |SHAP| bar chart (global importance) ───────────────────
    mean_shap = pd.Series(
        np.abs(shap_vals_class1).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 7))
    mean_shap[::-1].plot(kind='barh', color='steelblue', alpha=0.85)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 Features – Global SHAP Importance (Grant Exclusion)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "shap_importance.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ── 17c. Dependence plots for top 3 features ─────────────────────────
    top3_features = mean_shap.index[:3].tolist()
    for feat in top3_features:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_vals_class1, X_test,
                             interaction_index='auto', show=False)
        plt.title(f"SHAP Dependence: {feat}", fontsize=12)
        plt.tight_layout()
        safe_name = feat.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(FIGURES_FOLDER, f"shap_dep_{safe_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    # ── 17d. Waterfall plot for a single high-risk household ──────────────
    # Find the test observation with the highest predicted exclusion probability
    high_risk_idx = np.argmax(y_proba_best)
    print(f"\n   High-risk observation index (test set): {high_risk_idx}")
    print(f"   Predicted probability of exclusion: {y_proba_best[high_risk_idx]:.4f}")
    print(f"   True label: {'Excluded' if y_test.iloc[high_risk_idx] == 1 else 'Recipient'}")

    # Compute Explanation object for waterfall
   
    exp = shap.Explanation(
    values=shap_vals_class1[high_risk_idx],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[high_risk_idx].values,
    feature_names=X_test.columns.tolist()
    )
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(exp, show=False, max_display=15)
    plt.title("SHAP Waterfall – Highest-Risk Excluded Household", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "shap_waterfall.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Store for summary section
    shap_top10 = mean_shap.head(10).reset_index()
    shap_top10.columns = ['Variable', 'Mean_SHAP']
    print("\n   Top 10 features by mean |SHAP|:")
    print(shap_top10.to_string(index=False))

else:
    print("\n   SHAP not available – skipping SHAP analysis.")
    shap_top10 = pd.DataFrame({'Variable': [], 'Mean_SHAP': []})

# =============================================================================
# 18. Weighted logistic regression (odds ratios) with L1 selection
# =============================================================================
print("\n15. Weighted logistic regression for odds ratios...")

selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                              max_iter=1000, random_state=RANDOM_STATE)
selector.fit(X_train, y_train, sample_weight=w_train.values)

coefs = selector.coef_[0]
selected_mask = coefs != 0
selected_features  = X_train.columns[selected_mask].tolist()
coefs_selected     = coefs[selected_mask]

coef_df = pd.DataFrame({'Variable': selected_features, 'Coef': coefs_selected})
coef_df['Odds_Ratio'] = np.exp(coef_df['Coef'])

variances = X_train[selected_features].var()
coef_df   = coef_df[~(variances < 1e-12).values]
coef_df   = coef_df.reindex(coef_df['Coef'].abs().sort_values(ascending=False).index)

top10_or = coef_df.head(10)
print("\n   Top 10 predictors by |coefficient| (L1 logistic regression):")
print(top10_or[['Variable', 'Odds_Ratio', 'Coef']].to_string(index=False))
sig_df = top10_or

# =============================================================================
# 19. Odds-ratio visualisation
# =============================================================================
print("\n16. Plotting odds ratios...")
top_or = coef_df.head(15).sort_values('Odds_Ratio')
colors = ['#C73E1D' if v > 1 else '#06A77D' for v in top_or['Odds_Ratio']]

plt.figure(figsize=(10, 7))
plt.barh(top_or['Variable'], top_or['Odds_Ratio'], color=colors, alpha=0.85)
plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.xlabel("Odds Ratio")
plt.title("Odds Ratios – Grant Exclusion (L1 Logistic Regression, Weighted)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "odds_ratios.png"), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 20. EDA visualisations
# =============================================================================
print("\n17. Generating exploratory visualisations...")

# Plot 1: Food insecurity score distribution
plt.figure(figsize=(8, 5))
sns.histplot(kzn_hh['fi_score'], bins=range(0, 7), discrete=True, stat='count')
plt.axvline(x=1.5, color='red', linestyle='--', label='Threshold (≥ 2)')
plt.title('Distribution of Food Insecurity Scores – KwaZulu-Natal')
plt.xlabel('Food Insecurity Score')
plt.ylabel('Number of Households')
plt.legend()
plt.savefig(os.path.join(FIGURES_FOLDER, "fi_score_dist.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Grant coverage among food-insecure households
fi_hh = kzn_hh[kzn_hh['food_insecure'] == 1]
fi_grant_counts = fi_hh['hh_receives_grant'].value_counts()
plt.figure(figsize=(6, 5))
fi_grant_counts.plot(kind='bar', color=['#C73E1D', '#06A77D'])
plt.xticks(ticks=[0, 1], labels=['Excluded from Grants', 'Receives Grants'], rotation=0)
plt.title('Grant Coverage – Food-Insecure Households (KZN)')
plt.ylabel('Number of Households')
for i, v in enumerate(fi_grant_counts.values):
    plt.text(i, v + 20, f'{v:,}\n({v/len(fi_hh)*100:.1f}%)', ha='center')
plt.savefig(os.path.join(FIGURES_FOLDER, "grant_coverage_fi.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Boxplot of log income by grant exclusion
plt.figure(figsize=(6, 5))
sns.boxplot(x='grant_excluded', y='log_income', data=analysis_df)
plt.title('Log Household Income by Grant Exclusion Status')
plt.xlabel('Grant Excluded  (0 = Receives grant, 1 = Excluded)')
plt.ylabel('Log Annualised Income (Adjusted)')
plt.savefig(os.path.join(FIGURES_FOLDER, "log_income_boxplot.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Weighted exclusion proportion by settlement type
prop_settlement = analysis_df.groupby('SETTLEMENT_TYPE').apply(
    lambda g: (g['grant_excluded'] * g['hholds_wgt']).sum() / g['hholds_wgt'].sum()
)
plt.figure(figsize=(8, 5))
prop_settlement.sort_values(ascending=False).plot(kind='bar', color='steelblue', alpha=0.85)
plt.title('Weighted Proportion of Grant-Excluded Food-Insecure HHs\nby Settlement Type')
plt.ylabel('Weighted Proportion Excluded')
plt.xlabel('Settlement Type')
plt.xticks(rotation=15)
plt.savefig(os.path.join(FIGURES_FOLDER, "settlement_type_prop.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 21. Final summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print(f"• Food-insecure KZN households analysed: {len(analysis_df):,}")
print(f"• Grant exclusion rate (unweighted): "
      f"{analysis_df['grant_excluded'].mean()*100:.1f}%")

print("\n• Weighted cross-validation results (5-fold, SMOTE per fold):")
for name in cv_results:
    r = cv_results[name]
    print(f"   {name:25s} → AUC {r['auc_mean']:.3f} ± {r['auc_std']:.3f}"
          f"  |  F1 {r['f1_mean']:.3f} ± {r['f1_std']:.3f}")

print("\n• Weighted test-set metrics:")
for name, m in wm_results.items():
    print(f"   {name:25s} → AUC {m['auc']:.3f}  F1 {m['f1']:.3f}  "
          f"Prec {m['precision']:.3f}  Rec {m['recall']:.3f}")

print("\n• Top 10 predictors of exclusion (odds ratios from L1 logistic regression):")
has_ci = 'OR_lower' in sig_df.columns and 'OR_upper' in sig_df.columns
for _, row in sig_df.iterrows():
    if row['Variable'] == 'Intercept':
        continue
    if has_ci:
        print(f"   – {row['Variable']}: OR = {row['Odds_Ratio']:.2f} "
              f"(95% CI [{row['OR_lower']:.2f}, {row['OR_upper']:.2f}])")
    else:
        print(f"   – {row['Variable']}: OR = {row['Odds_Ratio']:.2f}")

if SHAP_AVAILABLE and not shap_top10.empty:
    print("\n• Top 10 features by SHAP importance (Tuned Random Forest):")
    for _, row in shap_top10.iterrows():
        print(f"   – {row['Variable']}: mean |SHAP| = {row['Mean_SHAP']:.4f}")

print("\n• All figures saved to:", os.path.abspath(FIGURES_FOLDER))
print("=" * 80)