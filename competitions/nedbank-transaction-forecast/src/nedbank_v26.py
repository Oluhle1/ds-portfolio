#!/usr/bin/env python
# coding: utf-8

# # Nedbank Transaction Forecasting - v26
# 
# This notebook builds a full leakage-safe solution for the **Nedbank Transaction Forecasting Challenge** directly from the provided files.
# 
# It follows the challenge brief in `NedbankChallenge.md`:
# 
# - Predict the raw target `next_3m_txn_count`
# - Use only data available up to **2015-10-31**
# - Validate with **RMSLE / RMSE on log1p(target)**
# - Save the final submission with columns `UniqueID` and `next_3m_txn_count`
# - Submit **`np.log1p(raw_predictions)`**, not raw counts

# In[1]:


get_ipython().run_line_magic('pip', 'install -q lightgbm xgboost catboost pyarrow fastparquet')


# In[2]:


import gc
import json
import random
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

SEED = 42
CUTOFF = pd.Timestamp("2015-10-31")
N_FOLDS = 5
LOW_CARD_THRESHOLD = 25
TARGET_ENCODE_MAX_CARD = 120

random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path.cwd()
OUTPUT_PATH = DATA_PATH

print("DATA_PATH :", DATA_PATH)
print("CUTOFF    :", CUTOFF.date())
print("SEED      :", SEED)


# ## Load Data
# 
# The challenge package provides:
# 
# - `Train.csv`: train customer IDs and labels
# - `Test.csv`: customer IDs to score
# - `transactions_features.parquet`: transaction history up to October 2015
# - `financials_features.parquet`: monthly product-level financial snapshots
# - `demographics_clean.parquet`: one row per customer

# In[3]:


train = pd.read_csv(DATA_PATH / "Train.csv")
test = pd.read_csv(DATA_PATH / "Test.csv")
sample_submission = pd.read_csv(DATA_PATH / "SampleSubmission.csv")
txn = pd.read_parquet(DATA_PATH / "transactions_features.parquet")
fin = pd.read_parquet(DATA_PATH / "financials_features.parquet")
demo = pd.read_parquet(DATA_PATH / "demographics_clean.parquet")
data_dict = pd.read_csv(DATA_PATH / "VariableDefinitions.csv")

print("train:", train.shape)
print("test :", test.shape)
print("txn  :", txn.shape)
print("fin  :", fin.shape)
print("demo :", demo.shape)
display(train.head())
display(data_dict.head(10))


# ## Helper Functions
# 
# The evaluation metric is RMSLE, which is equivalent here to RMSE on `log1p(y)`.

# In[4]:


def rmsle_from_raw(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def rmse_on_logs(y_log_true, y_log_pred):
    y_log_pred = np.clip(np.asarray(y_log_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.asarray(y_log_true) - y_log_pred) ** 2)))


def safe_ratio(num, den, offset=1.0):
    return num / (den + offset)


def make_target_bins(y_log, n_bins=12):
    y_log = pd.Series(y_log)
    unique_values = y_log.nunique()
    if unique_values < 2:
        return np.zeros(len(y_log), dtype=int)
    return pd.qcut(y_log, q=min(n_bins, unique_values), labels=False, duplicates="drop")


def build_folds(X, y_log, n_splits=N_FOLDS, seed=SEED):
    bins = make_target_bins(y_log, n_bins=max(n_splits * 2, 8))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, bins))


def recent_group_stat(df, group_col, value_col, lookback, agg="mean"):
    tail = df.groupby(group_col).tail(lookback)
    grouped = tail.groupby(group_col)[value_col]
    return getattr(grouped, agg)()


def group_slope(df, value_col, lookback):
    tail = df.groupby("UniqueID").tail(lookback).copy()
    tail["t"] = tail.groupby("UniqueID").cumcount().astype(np.float32)

    def fit_line(group):
        if len(group) < 2:
            return 0.0
        return float(np.polyfit(group["t"], group[value_col], 1)[0])

    return tail.groupby("UniqueID").apply(fit_line)


def cross_fit_target_encode(train_df, test_df, col, target_col, n_splits=N_FOLDS, smoothing=40, seed=SEED):
    global_mean = train_df[target_col].mean()
    train_encoded = np.zeros(len(train_df), dtype=np.float32)
    bins = make_target_bins(np.log1p(train_df[target_col].values), n_bins=max(n_splits * 2, 8))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fit_idx, hold_idx in splitter.split(train_df, bins):
        fit_part = train_df.iloc[fit_idx]
        stats = fit_part.groupby(col)[target_col].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        train_encoded[hold_idx] = train_df.iloc[hold_idx][col].map(stats["enc"]).fillna(global_mean).astype(np.float32)

    full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    full_stats["enc"] = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
    test_encoded = test_df[col].map(full_stats["enc"]).fillna(global_mean).astype(np.float32).values
    return train_encoded, test_encoded


def build_submission(unique_ids, raw_predictions, path):
    raw_predictions = np.clip(np.asarray(raw_predictions, dtype=float), 0, None)
    submission = pd.DataFrame(
        {
            "UniqueID": unique_ids,
            "next_3m_txn_count": np.log1p(raw_predictions),
        }
    )
    submission.to_csv(path, index=False)
    return submission


# ## Feature Engineering
# 
# The feature pipeline uses only information available before the prediction window:
# 
# - Recency, volume, dispersion, balance, and debit-credit behaviour from transactions
# - Month-level trends and rolling windows
# - Holiday season proxies from prior November-January periods
# - Product and recent-history features from financial snapshots
# - Cleaned demographics plus missingness indicators

# In[ ]:


txn = txn.copy()
txn["TransactionDate"] = pd.to_datetime(txn["TransactionDate"], errors="coerce")
txn = txn[txn["TransactionDate"].notna() & (txn["TransactionDate"] <= CUTOFF)].copy()
txn["TransactionAmount"] = pd.to_numeric(txn["TransactionAmount"], errors="coerce").fillna(0.0)
txn["StatementBalance"] = pd.to_numeric(txn["StatementBalance"], errors="coerce")
txn["AbsAmount"] = txn["TransactionAmount"].abs()
txn["IsDebit"] = np.where(txn["IsDebitCredit"].eq("D"), 1, np.where(txn["TransactionAmount"] < 0, 1, 0)).astype(np.int8)
txn["IsCredit"] = np.where(txn["IsDebitCredit"].eq("C"), 1, np.where(txn["TransactionAmount"] > 0, 1, 0)).astype(np.int8)
txn = txn.sort_values(["UniqueID", "TransactionDate"]).reset_index(drop=True)

fin = fin.copy()
fin["RunDate"] = pd.to_datetime(fin["RunDate"], errors="coerce")
fin = fin[fin["RunDate"].notna() & (fin["RunDate"] <= CUTOFF)].copy()
for col in ["NetInterestIncome", "NetInterestRevenue"]:
    fin[col] = pd.to_numeric(fin[col], errors="coerce")
fin = fin.sort_values(["UniqueID", "RunDate", "Product"]).reset_index(drop=True)

demo = demo.copy()
demo["BirthDate"] = pd.to_datetime(demo["BirthDate"], errors="coerce")
demo["Age"] = ((CUTOFF - demo["BirthDate"]).dt.days / 365.25).clip(lower=18, upper=95)
demo["Age_missing"] = demo["Age"].isna().astype(np.int8)
demo["Age"] = demo["Age"].fillna(demo["Age"].median())
demo["AnnualGrossIncome"] = pd.to_numeric(demo["AnnualGrossIncome"], errors="coerce")
demo["AnnualGrossIncome_missing"] = demo["AnnualGrossIncome"].isna().astype(np.int8)
demo["AnnualGrossIncome"] = demo["AnnualGrossIncome"].fillna(demo["AnnualGrossIncome"].median())
demo["AnnualGrossIncome_log"] = np.log1p(demo["AnnualGrossIncome"].clip(lower=0))
demo["Income_per_Age"] = safe_ratio(demo["AnnualGrossIncome"], demo["Age"])


def build_transaction_features(transactions, cutoff):
    tx = transactions.copy()
    tx["year_month"] = tx["TransactionDate"].dt.to_period("M")

    grp = tx.groupby("UniqueID")
    feats = pd.DataFrame(index=grp.size().index)
    feats.index.name = "UniqueID"

    first_txn = grp["TransactionDate"].min()
    last_txn = grp["TransactionDate"].max()
    tenure = ((cutoff.to_period("M") - first_txn.dt.to_period("M")).apply(lambda x: x.n) + 1).clip(lower=1)

    feats["txn_count_all"] = grp.size()
    feats["txn_active_days_all"] = grp["TransactionDate"].nunique()
    feats["account_count_all"] = grp["AccountID"].nunique()
    feats["days_since_first_txn"] = (cutoff - first_txn).dt.days.clip(lower=0)
    feats["days_since_last_txn"] = (cutoff - last_txn).dt.days.clip(lower=0)
    feats["tenure_months"] = tenure.astype(np.float32)
    feats["active_day_rate"] = safe_ratio(feats["txn_active_days_all"], feats["days_since_first_txn"] + 1)

    amount_stats = grp["AbsAmount"].agg(["sum", "mean", "median", "std", "max"])
    amount_stats.columns = [f"abs_amount_{c}_all" for c in amount_stats.columns]
    feats = feats.join(amount_stats)

    signed_stats = grp["TransactionAmount"].agg(["sum", "mean", "std", "min", "max"])
    signed_stats.columns = [f"signed_amount_{c}_all" for c in signed_stats.columns]
    feats = feats.join(signed_stats)

    balance_stats = grp["StatementBalance"].agg(["last", "mean", "std", "min", "max"])
    balance_stats.columns = [f"statement_balance_{c}" for c in balance_stats.columns]
    feats = feats.join(balance_stats)

    feats["debit_share_all"] = grp["IsDebit"].mean()
    feats["credit_share_all"] = grp["IsCredit"].mean()
    feats["txn_type_nunique_all"] = grp["TransactionTypeDescription"].nunique()
    feats["txn_batch_nunique_all"] = grp["TransactionBatchDescription"].nunique()
    feats["reversal_type_nunique_all"] = grp["ReversalTypeDescription"].nunique()

    tx["prev_txn_date"] = grp["TransactionDate"].shift(1)
    tx["gap_days"] = (tx["TransactionDate"] - tx["prev_txn_date"]).dt.days
    gap_stats = grp["gap_days"].agg(["mean", "std", "max"])
    gap_stats.columns = [f"gap_days_{c}" for c in gap_stats.columns]
    feats = feats.join(gap_stats)

    for days, label in [(30, "1m"), (60, "2m"), (90, "3m"), (180, "6m"), (365, "12m")]:
        window = tx[tx["TransactionDate"] > cutoff - pd.Timedelta(days=days)]
        wgrp = window.groupby("UniqueID")
        feats[f"txn_count_{label}"] = wgrp.size()
        feats[f"txn_active_days_{label}"] = wgrp["TransactionDate"].nunique()
        feats[f"txn_account_count_{label}"] = wgrp["AccountID"].nunique()
        feats[f"abs_amount_sum_{label}"] = wgrp["AbsAmount"].sum()
        feats[f"abs_amount_mean_{label}"] = wgrp["AbsAmount"].mean()
        feats[f"abs_amount_std_{label}"] = wgrp["AbsAmount"].std()
        feats[f"signed_amount_sum_{label}"] = wgrp["TransactionAmount"].sum()
        feats[f"debit_share_{label}"] = wgrp["IsDebit"].mean()
        feats[f"credit_share_{label}"] = wgrp["IsCredit"].mean()
        feats[f"txn_type_nunique_{label}"] = wgrp["TransactionTypeDescription"].nunique()

    monthly = (
        tx.groupby(["UniqueID", "year_month"])
        .agg(
            monthly_txn_count=("TransactionAmount", "size"),
            monthly_abs_sum=("AbsAmount", "sum"),
            monthly_abs_mean=("AbsAmount", "mean"),
            monthly_signed_sum=("TransactionAmount", "sum"),
            monthly_active_days=("TransactionDate", "nunique"),
            monthly_debit_share=("IsDebit", "mean"),
            monthly_credit_count=("IsCredit", "sum"),
        )
        .reset_index()
        .sort_values(["UniqueID", "year_month"])
    )

    count_stats = monthly.groupby("UniqueID")["monthly_txn_count"].agg(["mean", "std", "max", "min", "last"])
    count_stats.columns = [f"monthly_txn_count_{c}" for c in count_stats.columns]
    feats = feats.join(count_stats)

    amount_monthly_stats = monthly.groupby("UniqueID")["monthly_abs_sum"].agg(["mean", "std", "max", "min", "last"])
    amount_monthly_stats.columns = [f"monthly_abs_sum_{c}" for c in amount_monthly_stats.columns]
    feats = feats.join(amount_monthly_stats)

    active_stats = monthly.groupby("UniqueID")["monthly_active_days"].agg(["mean", "std", "max", "last"])
    active_stats.columns = [f"monthly_active_days_{c}" for c in active_stats.columns]
    feats = feats.join(active_stats)

    feats["active_months"] = monthly.groupby("UniqueID").size()
    feats["active_month_ratio"] = safe_ratio(feats["active_months"], feats["tenure_months"])
    feats["monthly_txn_count_cv"] = safe_ratio(feats["monthly_txn_count_std"], feats["monthly_txn_count_mean"])
    feats["monthly_abs_sum_cv"] = safe_ratio(feats["monthly_abs_sum_std"], feats["monthly_abs_sum_mean"])
    feats["monthly_active_days_cv"] = safe_ratio(feats["monthly_active_days_std"], feats["monthly_active_days_mean"])

    feats["monthly_txn_count_last2_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_txn_count", 2, "mean")
    feats["monthly_txn_count_last3_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_txn_count", 3, "mean")
    feats["monthly_txn_count_last6_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_txn_count", 6, "mean")
    feats["monthly_abs_sum_last3_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_abs_sum", 3, "mean")
    feats["monthly_abs_sum_last6_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_abs_sum", 6, "mean")
    feats["monthly_txn_count_slope_3m"] = group_slope(monthly, "monthly_txn_count", 3)
    feats["monthly_txn_count_slope_6m"] = group_slope(monthly, "monthly_txn_count", 6)
    feats["monthly_txn_count_slope_12m"] = group_slope(monthly, "monthly_txn_count", 12)
    feats["monthly_abs_sum_slope_3m"] = group_slope(monthly, "monthly_abs_sum", 3)
    feats["monthly_abs_sum_slope_6m"] = group_slope(monthly, "monthly_abs_sum", 6)

    holiday_windows = {
        "holiday_2013_2014": ("2013-11-01", "2014-01-31"),
        "holiday_2014_2015": ("2014-11-01", "2015-01-31"),
    }
    for prefix, (start, end) in holiday_windows.items():
        window = tx[(tx["TransactionDate"] >= start) & (tx["TransactionDate"] <= end)]
        wgrp = window.groupby("UniqueID")
        feats[f"{prefix}_count"] = wgrp.size()
        feats[f"{prefix}_abs_sum"] = wgrp["AbsAmount"].sum()
        feats[f"{prefix}_avg_ticket"] = wgrp["AbsAmount"].mean()
        feats[f"{prefix}_active_days"] = wgrp["TransactionDate"].nunique()
        feats[f"{prefix}_debit_share"] = wgrp["IsDebit"].mean()

    for col in ["TransactionTypeDescription", "TransactionBatchDescription", "ReversalTypeDescription"]:
        pivot_all = tx.pivot_table(index="UniqueID", columns=col, values="TransactionAmount", aggfunc="size", fill_value=0)
        pivot_all.columns = [f"{col.lower()}_cnt_{str(v).lower().replace(' ', '_').replace('&', 'and').replace('/', '_')}" for v in pivot_all.columns]
        feats = feats.join(pivot_all)

    count_pivot = monthly.pivot(index="UniqueID", columns="year_month", values="monthly_txn_count")
    amount_pivot = monthly.pivot(index="UniqueID", columns="year_month", values="monthly_abs_sum")
    active_pivot = monthly.pivot(index="UniqueID", columns="year_month", values="monthly_active_days")

    selected_periods = [
        pd.Period("2014-08"),
        pd.Period("2014-09"),
        pd.Period("2014-10"),
        pd.Period("2014-11"),
        pd.Period("2014-12"),
        pd.Period("2015-01"),
        pd.Period("2015-08"),
        pd.Period("2015-09"),
        pd.Period("2015-10"),
    ]
    for period in selected_periods:
        key = f"{period.year}_{period.month:02d}"
        feats[f"txn_count_{key}"] = count_pivot.get(period)
        feats[f"abs_sum_{key}"] = amount_pivot.get(period)
        feats[f"active_days_{key}"] = active_pivot.get(period)

    aug_2015 = feats["txn_count_2015_08"]
    sep_2015 = feats["txn_count_2015_09"]
    oct_2015 = feats["txn_count_2015_10"]
    aug_2014 = feats["txn_count_2014_08"]
    sep_2014 = feats["txn_count_2014_09"]
    oct_2014 = feats["txn_count_2014_10"]

    feats["preholiday_2015_count"] = sep_2015 + oct_2015
    feats["preholiday_2014_count"] = sep_2014 + oct_2014
    feats["preholiday_2015_abs_sum"] = feats["abs_sum_2015_09"] + feats["abs_sum_2015_10"]
    feats["preholiday_2014_abs_sum"] = feats["abs_sum_2014_09"] + feats["abs_sum_2014_10"]

    feats["holiday_count_yoy_delta"] = feats["holiday_2014_2015_count"] - feats["holiday_2013_2014_count"]
    feats["holiday_count_yoy_ratio"] = safe_ratio(feats["holiday_2014_2015_count"], feats["holiday_2013_2014_count"])
    feats["holiday_abs_sum_yoy_ratio"] = safe_ratio(feats["holiday_2014_2015_abs_sum"], feats["holiday_2013_2014_abs_sum"])
    feats["preholiday_count_yoy_ratio"] = safe_ratio(feats["preholiday_2015_count"], feats["preholiday_2014_count"])
    feats["oct_2015_to_oct_2014_ratio"] = safe_ratio(oct_2015, oct_2014)
    feats["sep_2015_to_sep_2014_ratio"] = safe_ratio(sep_2015, sep_2014)
    feats["aug_2015_to_aug_2014_ratio"] = safe_ratio(aug_2015, aug_2014)
    feats["oct_sep_count_ratio"] = safe_ratio(oct_2015, sep_2015)
    feats["oct_vs_last3_mean"] = safe_ratio(oct_2015, feats["monthly_txn_count_last3_mean"])
    feats["weighted_recent_count"] = 0.2 * aug_2015 + 0.3 * sep_2015 + 0.5 * oct_2015
    feats["weighted_recent_amount"] = (
        0.2 * feats["abs_sum_2015_08"] + 0.3 * feats["abs_sum_2015_09"] + 0.5 * feats["abs_sum_2015_10"]
    )
    feats["recent_count_vs_12m_rate"] = safe_ratio(feats["txn_count_3m"], feats["txn_count_12m"] / 4.0)
    feats["recent_amount_vs_12m_rate"] = safe_ratio(feats["abs_amount_sum_3m"], feats["abs_amount_sum_12m"] / 4.0)
    feats["reactivated_in_oct_2015"] = ((sep_2015 == 0) & (oct_2015 > 0)).astype(np.int8)
    feats["active_in_both_sep_oct_2015"] = ((sep_2015 > 0) & (oct_2015 > 0)).astype(np.int8)
    feats["slowing_before_nov"] = ((sep_2015 > 0) & (oct_2015 < sep_2015)).astype(np.int8)
    feats["accelerating_before_nov"] = (oct_2015 > sep_2015).astype(np.int8)

    for col in feats.columns:
        if pd.api.types.is_numeric_dtype(feats[col]):
            feats[col] = feats[col].astype(np.float32)

    return feats.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_financial_features(financials, cutoff):
    ff = financials.copy().sort_values(["UniqueID", "RunDate", "Product"])
    num_cols = ["NetInterestIncome", "NetInterestRevenue"]

    latest = ff.groupby("UniqueID").tail(1).set_index("UniqueID")
    feats = latest[num_cols].add_prefix("fin_latest_")

    first_run = ff.groupby("UniqueID")["RunDate"].min()
    last_run = ff.groupby("UniqueID")["RunDate"].max()
    feats["fin_days_since_last_run"] = (cutoff - last_run).dt.days
    feats["fin_history_days"] = (last_run - first_run).dt.days.clip(lower=0)
    feats["fin_product_nunique"] = ff.groupby("UniqueID")["Product"].nunique()
    feats["fin_account_nunique"] = ff.groupby("UniqueID")["AccountID"].nunique()

    last3 = ff.groupby("UniqueID").tail(3).groupby("UniqueID")[num_cols].mean().add_prefix("fin_mean_3_")
    last6 = ff.groupby("UniqueID").tail(6).groupby("UniqueID")[num_cols].mean().add_prefix("fin_mean_6_")
    std3 = ff.groupby("UniqueID").tail(3).groupby("UniqueID")[num_cols].std().add_prefix("fin_std_3_")
    feats = feats.join(last3).join(last6).join(std3)

    def latest_delta(source, lag, prefix):
        rows = []
        for uid, group in source.groupby("UniqueID"):
            group = group.reset_index(drop=True)
            row = {"UniqueID": uid}
            for col in num_cols:
                row[f"{prefix}{col}"] = group.loc[len(group) - 1, col] - group.loc[len(group) - 1 - lag, col] if len(group) > lag else np.nan
            rows.append(row)
        return pd.DataFrame(rows).set_index("UniqueID")

    feats = feats.join(latest_delta(ff, 1, "fin_delta_1_"))
    feats = feats.join(latest_delta(ff, 3, "fin_delta_3_"))

    latest_product = ff.groupby(["UniqueID", "Product"]).tail(1)
    for col in num_cols:
        pivot = latest_product.pivot_table(index="UniqueID", columns="Product", values=col, aggfunc="last")
        pivot.columns = [f"fin_prod_latest_{col}_{str(v).lower()}" for v in pivot.columns]
        feats = feats.join(pivot)

    product_obs = ff.pivot_table(index="UniqueID", columns="Product", values="RunDate", aggfunc="size", fill_value=0)
    product_obs.columns = [f"fin_prod_obs_{str(v).lower()}" for v in product_obs.columns]
    feats = feats.join(product_obs)

    for col in feats.columns:
        if pd.api.types.is_numeric_dtype(feats[col]):
            feats[col] = feats[col].astype(np.float32)
            if feats[col].min() >= 0:
                feats[f"{col}_log"] = np.log1p(feats[col]).astype(np.float32)

    return feats.replace([np.inf, -np.inf], np.nan).fillna(0)


txn_features = build_transaction_features(txn, CUTOFF)
fin_features = build_financial_features(fin, CUTOFF)

print("transaction features:", txn_features.shape)
print("financial features  :", fin_features.shape)


# ## Assemble Modelling Tables
# 
# Demographic categoricals are handled with a mix of:
# 
# - one-hot encoding for low-cardinality columns
# - frequency encoding for all categorical columns
# - leakage-safe target encoding for medium/high-cardinality columns

# In[ ]:


train_master = (
    train.merge(txn_features.reset_index(), on="UniqueID", how="left")
         .merge(fin_features.reset_index(), on="UniqueID", how="left")
         .merge(demo, on="UniqueID", how="left")
)
test_master = (
    test.merge(txn_features.reset_index(), on="UniqueID", how="left")
        .merge(fin_features.reset_index(), on="UniqueID", how="left")
        .merge(demo, on="UniqueID", how="left")
)

for frame in [train_master, test_master]:
    datetime_cols = frame.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    for col in datetime_cols:
        frame[col] = (CUTOFF - frame[col]).dt.days.astype(np.float32)

categorical_cols = [
    col for col in train_master.columns
    if col not in ["UniqueID", "next_3m_txn_count"] and train_master[col].dtype == "object"
]

for col in categorical_cols:
    train_master[col] = train_master[col].fillna("missing").astype(str)
    test_master[col] = test_master[col].fillna("missing").astype(str)

low_card_cols = []
target_encode_cols = []
for col in categorical_cols:
    nunique = pd.concat([train_master[col], test_master[col]], axis=0).nunique(dropna=False)
    if nunique <= LOW_CARD_THRESHOLD:
        low_card_cols.append(col)
    elif nunique <= TARGET_ENCODE_MAX_CARD:
        target_encode_cols.append(col)

for col in categorical_cols:
    freq = pd.concat([train_master[col], test_master[col]], axis=0).value_counts(normalize=True)
    train_master[f"{col}_freq"] = train_master[col].map(freq).fillna(0).astype(np.float32)
    test_master[f"{col}_freq"] = test_master[col].map(freq).fillna(0).astype(np.float32)

for col in target_encode_cols:
    tr_enc, te_enc = cross_fit_target_encode(train_master, test_master, col, "next_3m_txn_count")
    train_master[f"{col}_te"] = tr_enc
    test_master[f"{col}_te"] = te_enc

if low_card_cols:
    stacked = pd.concat(
        [
            train_master[low_card_cols].assign(_source="train"),
            test_master[low_card_cols].assign(_source="test"),
        ],
        axis=0,
    )
    dummies = pd.get_dummies(stacked, columns=low_card_cols, dtype=np.float32)
    train_dummies = dummies[dummies["_source_train"] == 1].drop(columns=["_source_train", "_source_test"])
    test_dummies = dummies[dummies["_source_test"] == 1].drop(columns=["_source_train", "_source_test"])
    train_dummies.index = train_master.index
    test_dummies.index = test_master.index
else:
    train_dummies = pd.DataFrame(index=train_master.index)
    test_dummies = pd.DataFrame(index=test_master.index)

drop_cols = ["UniqueID", "next_3m_txn_count"] + categorical_cols
base_feature_cols = [col for col in train_master.columns if col not in drop_cols]

X_num = train_master[base_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_num = test_master[base_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

X = pd.concat([X_num, train_dummies], axis=1)
X_test = pd.concat([X_test_num, test_dummies], axis=1)
X, X_test = X.align(X_test, join="outer", axis=1, fill_value=0)

constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

for frame in [X, X_test]:
    for col in frame.columns:
        if frame[col].dtype == bool:
            frame[col] = frame[col].astype(np.int8)

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)
y = train_master["next_3m_txn_count"].astype(np.float32).values
y_log = np.log1p(y)
folds = build_folds(X, y_log, n_splits=N_FOLDS, seed=SEED)

print("train matrix:", X.shape)
print("test matrix :", X_test.shape)
print("low-card columns:", len(low_card_cols))
print("target-encoded columns:", len(target_encode_cols))
print("constant columns removed:", len(constant_cols))


# ## Train Models
# 
# The ensemble combines three tree models trained on the log-target:
# 
# - CatBoost
# - LightGBM
# - XGBoost
# 
# A small Ridge model on the same design matrix is included as a stabiliser.
# Final blend weights are derived from inverse CV error.

# In[ ]:


CAT_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 2200,
    "learning_rate": 0.025,
    "depth": 6,
    "l2_leaf_reg": 12.0,
    "subsample": 0.80,
    "random_strength": 2.0,
    "bootstrap_type": "Bernoulli",
    "verbose": 0,
    "allow_writing_files": False,
    "random_seed": SEED,
}

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2600,
    "learning_rate": 0.02,
    "num_leaves": 48,
    "max_depth": 6,
    "min_child_samples": 80,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "reg_alpha": 1.0,
    "reg_lambda": 8.0,
    "random_state": SEED,
    "verbosity": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_estimators": 2400,
    "learning_rate": 0.02,
    "max_depth": 5,
    "min_child_weight": 18,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "reg_alpha": 1.0,
    "reg_lambda": 8.0,
    "gamma": 0.2,
    "seed": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

RIDGE_ALPHA = 5.0

oof_cat = np.zeros(len(X), dtype=np.float32)
oof_lgb = np.zeros(len(X), dtype=np.float32)
oof_xgb = np.zeros(len(X), dtype=np.float32)
oof_ridge = np.zeros(len(X), dtype=np.float32)

pred_cat = np.zeros(len(X_test), dtype=np.float32)
pred_lgb = np.zeros(len(X_test), dtype=np.float32)
pred_xgb = np.zeros(len(X_test), dtype=np.float32)
pred_ridge = np.zeros(len(X_test), dtype=np.float32)

scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    X_tr = X.iloc[tr_idx]
    X_val = X.iloc[val_idx]
    y_tr = y_log[tr_idx]
    y_val = y_log[val_idx]

    cat_model = CatBoostRegressor(**CAT_PARAMS)
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=False,
    )
    oof_cat[val_idx] = np.clip(cat_model.predict(X_val), 0, None)
    pred_cat += np.clip(cat_model.predict(X_test), 0, None) / N_FOLDS

    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof_lgb[val_idx] = np.clip(lgb_model.predict(X_val), 0, None)
    pred_lgb += np.clip(lgb_model.predict(X_test), 0, None) / N_FOLDS

    xgb_model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=200)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = np.clip(xgb_model.predict(X_val), 0, None)
    pred_xgb += np.clip(xgb_model.predict(X_test), 0, None) / N_FOLDS

    ridge_model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge_model.fit(X_scaled[tr_idx], y_tr)
    oof_ridge[val_idx] = np.clip(ridge_model.predict(X_scaled[val_idx]), 0, None)
    pred_ridge += np.clip(ridge_model.predict(X_test_scaled), 0, None) / N_FOLDS

    fold_scores.append(
        {
            "fold": fold,
            "cat": rmse_on_logs(y_val, oof_cat[val_idx]),
            "lgb": rmse_on_logs(y_val, oof_lgb[val_idx]),
            "xgb": rmse_on_logs(y_val, oof_xgb[val_idx]),
            "ridge": rmse_on_logs(y_val, oof_ridge[val_idx]),
        }
    )
    print(f"fold {fold} complete")

    gc.collect()

cv_scores = {
    "cat": rmse_on_logs(y_log, oof_cat),
    "lgb": rmse_on_logs(y_log, oof_lgb),
    "xgb": rmse_on_logs(y_log, oof_xgb),
    "ridge": rmse_on_logs(y_log, oof_ridge),
}

inverse = {k: 1.0 / v for k, v in cv_scores.items()}
total_inv = sum(inverse.values())
blend_weights = {k: v / total_inv for k, v in inverse.items()}

oof_blend = (
    blend_weights["cat"] * oof_cat
    + blend_weights["lgb"] * oof_lgb
    + blend_weights["xgb"] * oof_xgb
    + blend_weights["ridge"] * oof_ridge
)
pred_blend_log = (
    blend_weights["cat"] * pred_cat
    + blend_weights["lgb"] * pred_lgb
    + blend_weights["xgb"] * pred_xgb
    + blend_weights["ridge"] * pred_ridge
)
pred_blend_raw = np.expm1(np.clip(pred_blend_log, 0, None))

print("Fold scores:")
display(pd.DataFrame(fold_scores))
print("CV model scores:", json.dumps(cv_scores, indent=2))
print("Blend weights  :", json.dumps(blend_weights, indent=2))
print("Blend CV       :", rmse_on_logs(y_log, oof_blend))


# ## Create Submission
# 
# Per the official brief, the submission file must contain:
# 
# - `UniqueID`
# - `next_3m_txn_count`
# 
# And the second column must be **`log1p` of the raw predicted transaction count**.

# In[ ]:


submission_path = OUTPUT_PATH / "submission_v26.csv"
submission = build_submission(test["UniqueID"], pred_blend_raw, submission_path)

print("Saved:", submission_path)
print("submission shape:", submission.shape)
display(submission.head())

assert submission.columns.tolist() == ["UniqueID", "next_3m_txn_count"]
assert submission.shape[0] == test.shape[0]
assert submission["UniqueID"].nunique() == test.shape[0]
assert submission["next_3m_txn_count"].notna().all()


# ## Optional Local Evaluation
# 
# If you have a local reference file such as `PublicReference.csv`, you can score the submission with:
# 
# ```bash
# python evaluate.py submission_v26.csv PublicReference.csv
# ```
# 
# This notebook does not assume the reference file is present.
