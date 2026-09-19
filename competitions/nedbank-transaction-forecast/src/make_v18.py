from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path("/mnt/c/Users/oluhl/Desktop/Nedbank competition")
SRC = ROOT / "nedbank_v17.ipynb"
DST = ROOT / "nedbank_v18.ipynb"


cell_updates = {
    0: dedent(
        """
        # Nedbank Transaction Forecasting - v18
        """
    ).strip(),
    2: dedent(
        """
        %pip install -q catboost optuna fastparquet lightgbm xgboost scikit-learn seaborn

        import gc
        import json
        import warnings
        from pathlib import Path

        import lightgbm as lgb
        import numpy as np
        import optuna
        import pandas as pd
        import seaborn as sns
        import xgboost as xgb
        from catboost import CatBoostRegressor
        from matplotlib import pyplot as plt
        from scipy.optimize import minimize
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_squared_log_error
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        warnings.filterwarnings("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        pd.set_option("display.max_columns", 200)
        np.random.seed(42)
        """
    ).strip(),
    4: dedent(
        """
        IN_COLAB = False
        try:
            from google.colab import drive
            IN_COLAB = True
        except Exception:
            drive = None

        LOCAL_PATH = Path.cwd()
        DRIVE_PATH = Path("/content/drive/MyDrive/Nedbank competition")

        if IN_COLAB and not DRIVE_PATH.exists():
            drive.mount("/content/drive")

        DATA_PATH = DRIVE_PATH if DRIVE_PATH.exists() else LOCAL_PATH
        OUTPUT_PATH = DATA_PATH
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        SEED = 42
        CUTOFF = pd.Timestamp("2015-10-31")
        N_FOLDS = 5
        TUNE_FOLDS = 3
        N_SEEDS = [42, 123, 456]
        N_TRIALS_CAT = 55
        N_TRIALS_LGB = 50
        N_TRIALS_XGB = 50
        LOW_CARD_OHE_THRESHOLD = 25
        HIGH_CARD_THRESHOLD = 80
        TARGET_ENCODE_SMOOTHING = 60
        RIDGE_ALPHAS = np.logspace(-3, 3, 25)

        print(f"DATA_PATH  : {DATA_PATH}")
        print(f"OUTPUT_PATH: {OUTPUT_PATH}")
        print(f"CUTOFF     : {CUTOFF.date()}")
        """
    ).strip(),
    6: dedent(
        """
        def rmsle(y_true, y_pred):
            y_pred = np.clip(np.asarray(y_pred), 0, None)
            return np.sqrt(mean_squared_log_error(y_true, y_pred))


        def rmsle_from_logs(y_log_true, y_log_pred):
            y_log_pred = np.clip(np.asarray(y_log_pred), 0, None)
            return float(np.sqrt(np.mean((np.asarray(y_log_true) - y_log_pred) ** 2)))


        def make_target_bins(y_log, n_bins=12):
            y_log = pd.Series(y_log)
            n_bins = min(n_bins, y_log.nunique())
            if n_bins < 2:
                return np.zeros(len(y_log), dtype=int)
            return pd.qcut(y_log, q=n_bins, labels=False, duplicates="drop")


        def build_stratified_folds(X, y_log, n_splits, seed):
            bins = make_target_bins(y_log, n_bins=max(n_splits * 2, 8))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(X, bins))


        def safe_ratio(num, den, offset=1.0):
            return num / (den + offset)


        def target_encode_cv(train_df, test_df, col, target, n_splits=5, smoothing=60, seed=SEED):
            global_mean = train_df[target].mean()
            train_encoded = np.zeros(len(train_df), dtype=np.float32)
            bins = make_target_bins(np.log1p(train_df[target].values), n_bins=min(10, n_splits * 2))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for tr_idx, val_idx in splitter.split(train_df, bins):
                stats = train_df.iloc[tr_idx].groupby(col)[target].agg(["mean", "count"])
                stats["encoded"] = (
                    stats["mean"] * stats["count"] + global_mean * smoothing
                ) / (stats["count"] + smoothing)
                train_encoded[val_idx] = (
                    train_df.iloc[val_idx][col].map(stats["encoded"]).fillna(global_mean).astype(np.float32).values
                )

            full_stats = train_df.groupby(col)[target].agg(["mean", "count"])
            full_stats["encoded"] = (
                full_stats["mean"] * full_stats["count"] + global_mean * smoothing
            ) / (full_stats["count"] + smoothing)
            test_encoded = (
                test_df[col].map(full_stats["encoded"]).fillna(global_mean).astype(np.float32).values
            )
            return train_encoded, test_encoded


        def build_submission(unique_ids, raw_preds, path):
            raw_preds = np.clip(np.asarray(raw_preds, dtype=np.float64), 0, None)
            submission = pd.DataFrame(
                {
                    "UniqueID": unique_ids,
                    "next_3m_txn_count": np.log1p(raw_preds),
                }
            )
            assert submission.columns.tolist() == ["UniqueID", "next_3m_txn_count"]
            assert submission["next_3m_txn_count"].notna().all()
            submission.to_csv(path, index=False)
            return submission


        def group_slope(df, value_col, lookback):
            tail = df.groupby("UniqueID").tail(lookback).copy()
            tail["t"] = tail.groupby("UniqueID").cumcount().astype(np.float32)

            def _fit(group):
                if len(group) < 2:
                    return 0.0
                return float(np.polyfit(group["t"], group[value_col], 1)[0])

            return tail.groupby("UniqueID").apply(_fit)


        def recent_group_stat(df, group_col, value_col, lookback, agg="mean"):
            tail = df.groupby(group_col).tail(lookback)
            return getattr(tail.groupby(group_col)[value_col], agg)()
        """
    ).strip(),
    10: dedent(
        """
        txn = txn.copy()
        txn["TransactionDate"] = pd.to_datetime(txn["TransactionDate"], errors="coerce")
        txn = txn[txn["TransactionDate"].notna() & (txn["TransactionDate"] <= CUTOFF)].copy()
        txn["TransactionAmount"] = pd.to_numeric(txn["TransactionAmount"], errors="coerce").fillna(0.0)
        txn["AbsAmount"] = txn["TransactionAmount"].abs()
        txn["IsDebit"] = (txn["TransactionAmount"] < 0).astype(np.int8)
        txn["IsCredit"] = (txn["TransactionAmount"] > 0).astype(np.int8)
        if "StatementBalance" in txn.columns:
            txn["StatementBalance"] = pd.to_numeric(txn["StatementBalance"], errors="coerce")
        txn = txn.sort_values(["UniqueID", "TransactionDate"]).reset_index(drop=True)

        demo = demo.copy()
        if "BirthDate" in demo.columns:
            demo["BirthDate"] = pd.to_datetime(demo["BirthDate"], errors="coerce")
            demo["Age"] = ((CUTOFF - demo["BirthDate"]).dt.days / 365.25).clip(18, 90)
        if "AnnualGrossIncome" in demo.columns:
            demo["income_missing"] = demo["AnnualGrossIncome"].isna().astype(np.int8)
            demo["AnnualGrossIncome"] = demo["AnnualGrossIncome"].fillna(demo["AnnualGrossIncome"].median())
            demo["AnnualGrossIncome_log"] = np.log1p(demo["AnnualGrossIncome"].clip(lower=0))


        def build_transaction_features(txn, cutoff):
            txn = txn.copy()
            txn["year_month"] = txn["TransactionDate"].dt.to_period("M")
            txn["month_num"] = txn["TransactionDate"].dt.month
            txn["year"] = txn["TransactionDate"].dt.year

            grp = txn.groupby("UniqueID")
            feats = pd.DataFrame(index=grp.size().index)
            feats.index.name = "UniqueID"

            first_txn = grp["TransactionDate"].min()
            last_txn = grp["TransactionDate"].max()
            tenure_months = ((cutoff.to_period("M") - first_txn.dt.to_period("M")).apply(lambda x: x.n) + 1).clip(lower=1)

            feats["txn_count_all"] = grp.size()
            feats["txn_active_days_all"] = grp["TransactionDate"].nunique()
            feats["days_since_first_txn"] = (cutoff - first_txn).dt.days.clip(lower=0)
            feats["days_since_last_txn"] = (cutoff - last_txn).dt.days.clip(lower=0)
            feats["tenure_months"] = tenure_months.astype(np.float32)
            feats["last_txn_in_oct_2015"] = (last_txn.dt.to_period("M") == pd.Period("2015-10")).astype(np.int8)

            amount_aggs = grp["AbsAmount"].agg(["mean", "median", "std", "max", "sum"])
            amount_aggs.columns = [f"abs_amount_{c}_all" for c in amount_aggs.columns]
            feats = feats.join(amount_aggs)
            feats["ticket_cv_all"] = feats["abs_amount_std_all"] / (feats["abs_amount_mean_all"] + 1.0)

            if "StatementBalance" in txn.columns:
                balance_aggs = grp["StatementBalance"].agg(["last", "mean", "std"])
                balance_aggs.columns = [f"statement_balance_{c}" for c in balance_aggs.columns]
                feats = feats.join(balance_aggs)

            if "TransactionTypeDescription" in txn.columns:
                feats["txn_type_nunique_all"] = grp["TransactionTypeDescription"].nunique()
            if "TransactionBatchDescription" in txn.columns:
                feats["txn_batch_nunique_all"] = grp["TransactionBatchDescription"].nunique()

            txn["prev_txn_date"] = grp["TransactionDate"].shift(1)
            txn["gap_days"] = (txn["TransactionDate"] - txn["prev_txn_date"]).dt.days
            gap_stats = grp["gap_days"].agg(["mean", "std", "max"])
            gap_stats.columns = [f"gap_days_{c}" for c in gap_stats.columns]
            feats = feats.join(gap_stats)
            feats["gap_days_recent5_mean"] = recent_group_stat(txn, "UniqueID", "gap_days", 5, "mean")
            feats["is_dormant_30d"] = (feats["days_since_last_txn"] > 30).astype(np.int8)
            feats["is_dormant_60d"] = (feats["days_since_last_txn"] > 60).astype(np.int8)
            feats["is_dormant_90d"] = (feats["days_since_last_txn"] > 90).astype(np.int8)

            for days, label in [(30, "1m"), (60, "2m"), (90, "3m"), (180, "6m"), (365, "12m")]:
                mask = txn["TransactionDate"] > cutoff - pd.Timedelta(days=days)
                w = txn.loc[mask]
                g = w.groupby("UniqueID")
                feats[f"txn_count_{label}"] = g.size()
                feats[f"txn_active_days_{label}"] = g["TransactionDate"].nunique()
                feats[f"abs_amount_sum_{label}"] = g["AbsAmount"].sum()
                feats[f"abs_amount_mean_{label}"] = g["AbsAmount"].mean()
                feats[f"abs_amount_std_{label}"] = g["AbsAmount"].std()
                feats[f"debit_share_{label}"] = g["IsDebit"].mean()
                feats[f"credit_share_{label}"] = g["IsCredit"].mean()
                if "TransactionTypeDescription" in w.columns:
                    feats[f"txn_type_nunique_{label}"] = g["TransactionTypeDescription"].nunique()
                if "TransactionBatchDescription" in w.columns:
                    feats[f"txn_batch_nunique_{label}"] = g["TransactionBatchDescription"].nunique()

            monthly = (
                txn.groupby(["UniqueID", "year_month"])
                .agg(
                    monthly_txn_count=("TransactionAmount", "size"),
                    monthly_abs_sum=("AbsAmount", "sum"),
                    monthly_abs_mean=("AbsAmount", "mean"),
                    monthly_debit_share=("IsDebit", "mean"),
                    monthly_active_days=("TransactionDate", "nunique"),
                    monthly_credit_count=("IsCredit", "sum"),
                    monthly_debit_count=("IsDebit", "sum"),
                )
                .reset_index()
                .sort_values(["UniqueID", "year_month"])
            )
            monthly["month_num"] = monthly["year_month"].dt.month
            monthly["year"] = monthly["year_month"].dt.year

            monthly_count_stats = monthly.groupby("UniqueID")["monthly_txn_count"].agg(["mean", "std", "max", "min", "last"])
            monthly_count_stats.columns = [f"monthly_txn_count_{c}" for c in monthly_count_stats.columns]
            feats = feats.join(monthly_count_stats)

            monthly_amount_stats = monthly.groupby("UniqueID")["monthly_abs_sum"].agg(["mean", "std", "max", "last"])
            monthly_amount_stats.columns = [f"monthly_abs_sum_{c}" for c in monthly_amount_stats.columns]
            feats = feats.join(monthly_amount_stats)

            monthly_active_stats = monthly.groupby("UniqueID")["monthly_active_days"].agg(["mean", "std", "max", "last"])
            monthly_active_stats.columns = [f"monthly_active_days_{c}" for c in monthly_active_stats.columns]
            feats = feats.join(monthly_active_stats)

            active_months = monthly.groupby("UniqueID").size()
            feats["active_months"] = active_months
            feats["active_month_ratio"] = feats["active_months"] / feats["tenure_months"]
            feats["inactive_month_ratio"] = 1.0 - feats["active_month_ratio"].clip(upper=1.0)
            feats["monthly_txn_count_cv"] = feats["monthly_txn_count_std"] / (feats["monthly_txn_count_mean"] + 1.0)
            feats["monthly_abs_sum_cv"] = feats["monthly_abs_sum_std"] / (feats["monthly_abs_sum_mean"] + 1.0)
            feats["monthly_active_days_cv"] = feats["monthly_active_days_std"] / (feats["monthly_active_days_mean"] + 1.0)

            feats["monthly_txn_count_last2_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_txn_count", 2, "mean")
            feats["monthly_txn_count_last3_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_txn_count", 3, "mean")
            feats["monthly_txn_count_last6_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_txn_count", 6, "mean")
            feats["monthly_abs_sum_last2_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_abs_sum", 2, "mean")
            feats["monthly_abs_sum_last3_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_abs_sum", 3, "mean")
            feats["monthly_abs_sum_last6_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_abs_sum", 6, "mean")
            feats["monthly_active_days_last3_mean"] = recent_group_stat(monthly, "UniqueID", "monthly_active_days", 3, "mean")
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
                mask = (txn["TransactionDate"] >= start) & (txn["TransactionDate"] <= end)
                g = txn.loc[mask].groupby("UniqueID")
                feats[f"{prefix}_count"] = g.size()
                feats[f"{prefix}_abs_sum"] = g["AbsAmount"].sum()
                feats[f"{prefix}_avg_ticket"] = g["AbsAmount"].mean()
                feats[f"{prefix}_debit_share"] = g["IsDebit"].mean()
                feats[f"{prefix}_active_days"] = g["TransactionDate"].nunique()

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

            oct_2015 = feats["txn_count_2015_10"]
            sep_2015 = feats["txn_count_2015_09"]
            aug_2015 = feats["txn_count_2015_08"]
            oct_2014 = feats["txn_count_2014_10"]
            sep_2014 = feats["txn_count_2014_09"]
            aug_2014 = feats["txn_count_2014_08"]

            feats["preholiday_2015_count"] = sep_2015 + oct_2015
            feats["preholiday_2014_count"] = sep_2014 + oct_2014
            feats["preholiday_2015_abs_sum"] = feats["abs_sum_2015_09"] + feats["abs_sum_2015_10"]
            feats["preholiday_2014_abs_sum"] = feats["abs_sum_2014_09"] + feats["abs_sum_2014_10"]
            feats["preholiday_2015_active_days"] = feats["active_days_2015_09"] + feats["active_days_2015_10"]
            feats["preholiday_2014_active_days"] = feats["active_days_2014_09"] + feats["active_days_2014_10"]

            feats["holiday_count_yoy_delta"] = feats["holiday_2014_2015_count"] - feats["holiday_2013_2014_count"]
            feats["holiday_count_yoy_ratio"] = safe_ratio(feats["holiday_2014_2015_count"], feats["holiday_2013_2014_count"])
            feats["holiday_abs_sum_yoy_ratio"] = safe_ratio(feats["holiday_2014_2015_abs_sum"], feats["holiday_2013_2014_abs_sum"])
            feats["holiday_dependency_ratio"] = safe_ratio(feats["holiday_2014_2015_count"], feats["txn_count_12m"])
            feats["nov_2014_to_baseline"] = safe_ratio(feats["txn_count_2014_11"], feats["monthly_txn_count_mean"])
            feats["dec_2014_to_baseline"] = safe_ratio(feats["txn_count_2014_12"], feats["monthly_txn_count_mean"])
            feats["jan_2015_to_baseline"] = safe_ratio(feats["txn_count_2015_01"], feats["monthly_txn_count_mean"])

            feats["oct_2015_to_oct_2014_ratio"] = safe_ratio(oct_2015, oct_2014)
            feats["sep_2015_to_sep_2014_ratio"] = safe_ratio(sep_2015, sep_2014)
            feats["aug_2015_to_aug_2014_ratio"] = safe_ratio(aug_2015, aug_2014)
            feats["preholiday_count_yoy_delta"] = feats["preholiday_2015_count"] - feats["preholiday_2014_count"]
            feats["preholiday_count_yoy_ratio"] = safe_ratio(feats["preholiday_2015_count"], feats["preholiday_2014_count"])
            feats["preholiday_abs_sum_yoy_ratio"] = safe_ratio(feats["preholiday_2015_abs_sum"], feats["preholiday_2014_abs_sum"])
            feats["oct_sep_count_delta"] = oct_2015 - sep_2015
            feats["oct_sep_count_ratio"] = safe_ratio(oct_2015, sep_2015)
            feats["oct_aug_count_ratio"] = safe_ratio(oct_2015, aug_2015)
            feats["oct_vs_last3_mean"] = safe_ratio(oct_2015, feats["monthly_txn_count_last3_mean"])
            feats["sep_oct_vs_last6_rate"] = safe_ratio(feats["preholiday_2015_count"], feats["txn_count_6m"] / 3.0)
            feats["weighted_recent_count"] = 0.2 * aug_2015 + 0.3 * sep_2015 + 0.5 * oct_2015
            feats["weighted_recent_amount"] = (
                0.2 * feats["abs_sum_2015_08"] + 0.3 * feats["abs_sum_2015_09"] + 0.5 * feats["abs_sum_2015_10"]
            )
            feats["oct_share_12m"] = safe_ratio(oct_2015, feats["txn_count_12m"])
            feats["sep_oct_share_12m"] = safe_ratio(feats["preholiday_2015_count"], feats["txn_count_12m"])
            feats["active_in_both_sep_oct_2015"] = ((sep_2015 > 0) & (oct_2015 > 0)).astype(np.int8)
            feats["reactivated_in_oct_2015"] = ((sep_2015 == 0) & (oct_2015 > 0)).astype(np.int8)
            feats["slowing_before_nov"] = ((sep_2015 > 0) & (oct_2015 < sep_2015)).astype(np.int8)
            feats["accelerating_before_nov"] = (oct_2015 > sep_2015).astype(np.int8)

            feats["recent_count_vs_12m_rate"] = feats["txn_count_3m"] / (feats["txn_count_12m"] / 4.0 + 1.0)
            feats["recent_amount_vs_12m_rate"] = feats["abs_amount_sum_3m"] / (feats["abs_amount_sum_12m"] / 4.0 + 1.0)
            feats["volatility_x_volume"] = feats["monthly_txn_count_cv"] * feats["monthly_txn_count_mean"]
            feats["recency_x_recent_rate"] = feats["txn_count_3m"] / (np.log1p(feats["days_since_last_txn"]) + 1.0)
            feats["oct_recency_interaction"] = oct_2015 / (np.log1p(feats["days_since_last_txn"]) + 1.0)

            return feats.replace([np.inf, -np.inf], np.nan).fillna(0)


        def build_financial_features(fin, cutoff):
            fin = fin.copy()
            date_col = None
            for candidate in ["SnapshotDate", "RunDate"]:
                if candidate in fin.columns:
                    date_col = candidate
                    break

            if date_col is not None:
                fin[date_col] = pd.to_datetime(fin[date_col], errors="coerce")
                fin = fin[fin[date_col].notna() & (fin[date_col] <= cutoff)].copy()

            numeric_cols = [
                c
                for c in fin.columns
                if c not in ["UniqueID", "AccountID", "Product", date_col]
                and pd.api.types.is_numeric_dtype(fin[c])
            ]

            sort_cols = ["UniqueID"] + ([date_col] if date_col is not None else [])
            if "Product" in fin.columns:
                sort_cols.append("Product")
            fin = fin.sort_values(sort_cols)

            latest = fin.groupby("UniqueID").tail(1).set_index("UniqueID")
            feats = latest[numeric_cols].add_prefix("fin_latest_")

            if date_col is not None and len(numeric_cols) > 0:
                first_run = fin.groupby("UniqueID")[date_col].min()
                last_run = fin.groupby("UniqueID")[date_col].max()
                feats["fin_days_since_last_run"] = (cutoff - last_run).dt.days
                feats["fin_history_days"] = (last_run - first_run).dt.days.clip(lower=0)

                last3 = fin.groupby("UniqueID").tail(3).groupby("UniqueID")[numeric_cols].mean().add_prefix("fin_mean_3m_")
                last6 = fin.groupby("UniqueID").tail(6).groupby("UniqueID")[numeric_cols].mean().add_prefix("fin_mean_6m_")
                std3 = fin.groupby("UniqueID").tail(3).groupby("UniqueID")[numeric_cols].std().add_prefix("fin_std_3m_")
                feats = feats.join(last3).join(last6).join(std3)

                def delta_frame(source, lag, prefix):
                    pieces = []
                    for uid, group in source.groupby("UniqueID"):
                        group = group.reset_index(drop=True)
                        row = {"UniqueID": uid}
                        for col in numeric_cols:
                            row[f"{prefix}{col}"] = (
                                group.loc[len(group) - 1, col] - group.loc[len(group) - 1 - lag, col]
                                if len(group) > lag else np.nan
                            )
                        pieces.append(row)
                    return pd.DataFrame(pieces).set_index("UniqueID")

                feats = feats.join(delta_frame(fin, 1, "fin_delta_1m_"))
                feats = feats.join(delta_frame(fin, 3, "fin_delta_3m_"))

            if date_col is not None and "Product" in fin.columns and len(numeric_cols) > 0:
                fin["year_month"] = fin[date_col].dt.to_period("M")
                latest_product = fin.groupby(["UniqueID", "Product"]).tail(1)
                for col in numeric_cols:
                    pivot = latest_product.pivot_table(index="UniqueID", columns="Product", values=col, aggfunc="last")
                    pivot.columns = [f"fin_prod_latest_{col}_{str(product).lower()}" for product in pivot.columns]
                    feats = feats.join(pivot)

                last3_fin = fin.groupby("UniqueID").tail(3).copy()
                if not last3_fin.empty:
                    prod_mix = (
                        last3_fin.pivot_table(
                            index="UniqueID",
                            columns="Product",
                            values=numeric_cols[0],
                            aggfunc="size",
                            fill_value=0,
                        )
                    )
                    prod_mix.columns = [f"fin_prod_obs_3m_{str(product).lower()}" for product in prod_mix.columns]
                    feats = feats.join(prod_mix)

            for col in list(feats.columns):
                feats[col] = feats[col].astype(np.float32)
                if feats[col].min() >= 0:
                    feats[f"{col}_log"] = np.log1p(feats[col])

            return feats.replace([np.inf, -np.inf], np.nan).fillna(0)


        txn_feats = build_transaction_features(txn, CUTOFF)
        fin_feats = build_financial_features(fin, CUTOFF)

        print("Transaction features:", txn_feats.shape)
        print("Financial features  :", fin_feats.shape)
        """
    ).strip(),
    12: dedent(
        """
        master = (
            train
            .merge(txn_feats.reset_index(), on="UniqueID", how="left")
            .merge(fin_feats.reset_index(), on="UniqueID", how="left")
            .merge(demo, on="UniqueID", how="left")
        )
        test_master = (
            test
            .merge(txn_feats.reset_index(), on="UniqueID", how="left")
            .merge(fin_feats.reset_index(), on="UniqueID", how="left")
            .merge(demo, on="UniqueID", how="left")
        )

        for df in [master, test_master]:
            datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
            for col in datetime_cols:
                df[col] = (CUTOFF - df[col]).dt.days.astype(np.float32)

        categorical_cols = [
            c for c in master.columns
            if c not in ["UniqueID", "next_3m_txn_count"] and master[c].dtype == "object"
        ]

        for col in categorical_cols:
            master[col] = master[col].astype(str).fillna("missing")
            test_master[col] = test_master[col].astype(str).fillna("missing")

        low_card_cols = []
        high_card_cols = []
        te_cols = []
        freq_cols = []

        for col in categorical_cols:
            combined_nunique = pd.concat([master[col], test_master[col]], axis=0).nunique(dropna=False)
            if combined_nunique <= LOW_CARD_OHE_THRESHOLD:
                low_card_cols.append(col)
            else:
                high_card_cols.append(col)

        for col in high_card_cols:
            counts = pd.concat([master[col], test_master[col]], axis=0).value_counts(normalize=True)
            master[f"{col}_freq"] = master[col].map(counts).fillna(0).astype(np.float32)
            test_master[f"{col}_freq"] = test_master[col].map(counts).fillna(0).astype(np.float32)
            freq_cols.append(col)

            if pd.concat([master[col], test_master[col]], axis=0).nunique(dropna=False) <= HIGH_CARD_THRESHOLD:
                tr_enc, te_enc = target_encode_cv(
                    master,
                    test_master,
                    col,
                    "next_3m_txn_count",
                    n_splits=N_FOLDS,
                    smoothing=TARGET_ENCODE_SMOOTHING,
                )
                master[f"{col}_te"] = tr_enc
                test_master[f"{col}_te"] = te_enc
                te_cols.append(col)

        if low_card_cols:
            combined_low = pd.concat(
                [master[low_card_cols].assign(_split="train"), test_master[low_card_cols].assign(_split="test")],
                axis=0,
            )
            low_dummies = pd.get_dummies(combined_low, columns=low_card_cols, dtype=np.float32)
            train_dummies = low_dummies[low_dummies["_split_train"] == 1].drop(columns=["_split_train", "_split_test"])
            test_dummies = low_dummies[low_dummies["_split_test"] == 1].drop(columns=["_split_train", "_split_test"])
            train_dummies.index = master.index
            test_dummies.index = test_master.index
        else:
            train_dummies = pd.DataFrame(index=master.index)
            test_dummies = pd.DataFrame(index=test_master.index)

        drop_cols = ["UniqueID", "next_3m_txn_count"] + categorical_cols
        numeric_feature_cols = [c for c in master.columns if c not in drop_cols]

        X_num = master[numeric_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        X_test_num = test_master[numeric_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        X = pd.concat([X_num, train_dummies], axis=1)
        X_test = pd.concat([X_test_num, test_dummies], axis=1)
        X, X_test = X.align(X_test, join="outer", axis=1, fill_value=0)

        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        X = X.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

        y = master["next_3m_txn_count"].astype(np.float32).values
        y_log = np.log1p(y)
        y_bins = make_target_bins(y_log, n_bins=12)

        print("One-hot columns               :", len(low_card_cols))
        print("Frequency-encoded columns     :", freq_cols)
        print("Target-encoded columns        :", te_cols)
        print("Dropped constant columns      :", len(constant_cols))
        print("Final train matrix            :", X.shape)
        print("Final test matrix             :", X_test.shape)
        """
    ).strip(),
    14: dedent(
        """
        tune_folds = build_stratified_folds(X, y_log, n_splits=TUNE_FOLDS, seed=SEED)
        print("Tune folds:", len(tune_folds), "folds")
        """
    ).strip(),
    16: dedent(
        """
        def cat_objective(trial):
            params = {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "iterations": trial.suggest_int("iterations", 900, 2800),
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.05, log=True),
                "depth": trial.suggest_int("depth", 4, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 80.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 180),
                "subsample": trial.suggest_float("subsample", 0.65, 0.9),
                "random_strength": trial.suggest_float("random_strength", 0.5, 25.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 128),
                "bootstrap_type": "Bernoulli",
                "verbose": 0,
                "random_seed": SEED,
                "allow_writing_files": False,
            }
            scores = []
            for tr_idx, val_idx in tune_folds:
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y_log[tr_idx], y_log[val_idx]
                model = CatBoostRegressor(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=150,
                    verbose=False,
                )
                pred_log = np.clip(model.predict(X_val), 0, None)
                scores.append(rmsle_from_logs(y_val, pred_log))
            return float(np.mean(scores))


        cat_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        )
        cat_study.optimize(cat_objective, n_trials=N_TRIALS_CAT, show_progress_bar=True)
        BEST_CAT = cat_study.best_params.copy()
        BEST_CAT.update(
            {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "bootstrap_type": "Bernoulli",
                "verbose": 0,
                "random_seed": SEED,
                "allow_writing_files": False,
            }
        )
        print("BEST_CAT:", BEST_CAT)
        """
    ).strip(),
    17: dedent(
        """
        def lgb_objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 1000, 3200),
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 24, 96),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 40, 220),
                "subsample": trial.suggest_float("subsample", 0.65, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 80.0, log=True),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.01, 1.0, log=True),
                "random_state": SEED,
            }
            scores = []
            for tr_idx, val_idx in tune_folds:
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y_log[tr_idx], y_log[val_idx]
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(200, verbose=False)],
                )
                pred_log = np.clip(model.predict(X_val), 0, None)
                scores.append(rmsle_from_logs(y_val, pred_log))
            return float(np.mean(scores))


        lgb_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        )
        lgb_study.optimize(lgb_objective, n_trials=N_TRIALS_LGB, show_progress_bar=True)
        BEST_LGB = lgb_study.best_params.copy()
        BEST_LGB.update({"objective": "regression", "metric": "rmse", "verbosity": -1, "n_jobs": -1, "random_state": SEED})
        print("BEST_LGB:", BEST_LGB)
        """
    ).strip(),
    18: dedent(
        """
        def xgb_objective(trial):
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "verbosity": 0,
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 1000, 3200),
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.05, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.65, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 80),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 80.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.01, 8.0, log=True),
                "seed": SEED,
            }
            scores = []
            for tr_idx, val_idx in tune_folds:
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y_log[tr_idx], y_log[val_idx]
                model = xgb.XGBRegressor(**params, early_stopping_rounds=200)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                pred_log = np.clip(model.predict(X_val), 0, None)
                scores.append(rmsle_from_logs(y_val, pred_log))
            return float(np.mean(scores))


        xgb_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        )
        xgb_study.optimize(xgb_objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)
        BEST_XGB = xgb_study.best_params.copy()
        BEST_XGB.update(
            {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist", "verbosity": 0, "n_jobs": -1, "seed": SEED}
        )
        print("BEST_XGB:", BEST_XGB)
        """
    ).strip(),
    20: dedent(
        """
        def stratified_folds(X, y_log, seed):
            return build_stratified_folds(X, y_log, n_splits=N_FOLDS, seed=seed)


        all_cat_oof, all_lgb_oof, all_xgb_oof, all_ridge_oof = [], [], [], []
        cat_best_iters, lgb_best_iters, xgb_best_iters, ridge_alphas = [], [], [], []

        for seed in N_SEEDS:
            folds = stratified_folds(X, y_log, seed)

            oof_cat = np.zeros(len(X), dtype=np.float32)
            oof_lgb = np.zeros(len(X), dtype=np.float32)
            oof_xgb = np.zeros(len(X), dtype=np.float32)
            oof_ridge = np.zeros(len(X), dtype=np.float32)

            for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y_log[tr_idx], y_log[val_idx]

                cat_model = CatBoostRegressor(**{**BEST_CAT, "random_seed": seed, "allow_writing_files": False})
                cat_model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=150,
                    verbose=False,
                )
                oof_cat[val_idx] = np.clip(cat_model.predict(X_val), 0, None)
                cat_best_iter = cat_model.get_best_iteration()
                cat_best_iters.append(int(cat_best_iter if cat_best_iter not in [None, -1] else BEST_CAT["iterations"]))

                lgb_model = lgb.LGBMRegressor(**{**BEST_LGB, "random_state": seed})
                lgb_model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(200, verbose=False)],
                )
                oof_lgb[val_idx] = np.clip(lgb_model.predict(X_val), 0, None)
                lgb_best_iters.append(int(getattr(lgb_model, "best_iteration_", BEST_LGB["n_estimators"]) or BEST_LGB["n_estimators"]))

                xgb_model = xgb.XGBRegressor(**{**BEST_XGB, "seed": seed}, early_stopping_rounds=200)
                xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                oof_xgb[val_idx] = np.clip(xgb_model.predict(X_val), 0, None)
                xgb_best_iter = getattr(xgb_model, "best_iteration", None)
                xgb_best_iters.append(int(xgb_best_iter + 1 if xgb_best_iter is not None else BEST_XGB["n_estimators"]))

                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
                ridge_model = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge_model.fit(X_tr_scaled, y_tr)
                oof_ridge[val_idx] = np.clip(ridge_model.predict(X_val_scaled), 0, None)
                ridge_alphas.append(float(ridge_model.alpha_))

                gc.collect()
                print(f"Seed {seed} Fold {fold} complete")

            all_cat_oof.append(oof_cat)
            all_lgb_oof.append(oof_lgb)
            all_xgb_oof.append(oof_xgb)
            all_ridge_oof.append(oof_ridge)

        oof_cat_log = np.mean(all_cat_oof, axis=0)
        oof_lgb_log = np.mean(all_lgb_oof, axis=0)
        oof_xgb_log = np.mean(all_xgb_oof, axis=0)
        oof_ridge_log = np.mean(all_ridge_oof, axis=0)

        print("CatBoost OOF RMSLE :", rmsle_from_logs(y_log, oof_cat_log))
        print("LightGBM OOF RMSLE :", rmsle_from_logs(y_log, oof_lgb_log))
        print("XGBoost OOF RMSLE  :", rmsle_from_logs(y_log, oof_xgb_log))
        print("Ridge OOF RMSLE    :", rmsle_from_logs(y_log, oof_ridge_log))
        """
    ).strip(),
    22: dedent(
        """
        oof_stack = np.column_stack([oof_cat_log, oof_lgb_log, oof_xgb_log, oof_ridge_log])
        model_names = ["CatBoost", "LightGBM", "XGBoost", "Ridge"]


        def blend_loss(weights):
            weights = np.asarray(weights, dtype=np.float64)
            blend_log = np.clip(oof_stack @ weights, 0, None)
            return rmsle_from_logs(y_log, blend_log)


        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * oof_stack.shape[1]
        starts = [
            np.array([0.30, 0.30, 0.30, 0.10]),
            np.array([0.40, 0.25, 0.25, 0.10]),
            np.array([0.45, 0.20, 0.20, 0.15]),
        ]

        best_result = None
        for start in starts:
            result = minimize(blend_loss, x0=start, method="SLSQP", bounds=bounds, constraints=constraints)
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        blend_weights = best_result.x
        blend_weights = blend_weights / blend_weights.sum()

        W_CAT, W_LGB, W_XGB, W_RIDGE = blend_weights
        blend_oof_log = np.clip(oof_stack @ blend_weights, 0, None)
        pred_corr = pd.DataFrame(oof_stack, columns=model_names).corr()

        print("Blend weights:", dict(zip(model_names, np.round(blend_weights, 6))))
        print("OOF prediction correlation:")
        display(pred_corr)
        print("Blend OOF RMSLE:", rmsle_from_logs(y_log, blend_oof_log))
        """
    ).strip(),
    24: dedent(
        """
        final_cat_params = BEST_CAT.copy()
        final_cat_params["iterations"] = int(np.median(cat_best_iters)) if cat_best_iters else BEST_CAT["iterations"]

        final_lgb_params = BEST_LGB.copy()
        final_lgb_params["n_estimators"] = int(np.median(lgb_best_iters)) if lgb_best_iters else BEST_LGB["n_estimators"]

        final_xgb_params = BEST_XGB.copy()
        final_xgb_params["n_estimators"] = int(np.median(xgb_best_iters)) if xgb_best_iters else BEST_XGB["n_estimators"]

        final_cat = CatBoostRegressor(**final_cat_params)
        final_lgb = lgb.LGBMRegressor(**final_lgb_params)
        final_xgb = xgb.XGBRegressor(**final_xgb_params)
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        X_test_scaled = final_scaler.transform(X_test)
        final_ridge = RidgeCV(alphas=RIDGE_ALPHAS)

        final_cat.fit(X, y_log, verbose=False)
        final_lgb.fit(X, y_log)
        final_xgb.fit(X, y_log)
        final_ridge.fit(X_scaled, y_log)

        test_cat_log = np.clip(final_cat.predict(X_test), 0, None)
        test_lgb_log = np.clip(final_lgb.predict(X_test), 0, None)
        test_xgb_log = np.clip(final_xgb.predict(X_test), 0, None)
        test_ridge_log = np.clip(final_ridge.predict(X_test_scaled), 0, None)

        final_test_log = (
            W_CAT * test_cat_log
            + W_LGB * test_lgb_log
            + W_XGB * test_xgb_log
            + W_RIDGE * test_ridge_log
        )
        final_test_log = np.clip(final_test_log, 0, None)
        final_test_raw = np.expm1(final_test_log)

        submission_path = OUTPUT_PATH / "submission_v18.csv"
        submission = build_submission(test["UniqueID"], final_test_raw, submission_path)

        assert submission.shape[0] == sample_submission.shape[0] == len(test)
        assert submission["UniqueID"].nunique() == len(test)
        assert submission.columns.tolist() == ["UniqueID", "next_3m_txn_count"]

        print("Saved submission:", submission_path)
        display(submission.head())
        """
    ).strip(),
    26: dedent(
        """
        importance_frames = []
        for model_name, model in [
            ("CatBoost", final_cat),
            ("LightGBM", final_lgb),
            ("XGBoost", final_xgb),
        ]:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                importance_frames.append(
                    pd.DataFrame(
                        {
                            "feature": X.columns,
                            "importance": importance / (np.sum(importance) + 1e-9),
                            "model": model_name,
                        }
                    )
                )

        ridge_importance = np.abs(final_ridge.coef_)
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": ridge_importance / (np.sum(ridge_importance) + 1e-9),
                    "model": "Ridge",
                }
            )
        )

        feat_imp = pd.concat(importance_frames, axis=0)
        avg_feat_imp = feat_imp.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)

        plt.figure(figsize=(12, 10))
        sns.barplot(data=avg_feat_imp.head(30), x="importance", y="feature", palette="viridis")
        plt.title("Top 30 Ensemble Features")
        plt.tight_layout()
        plt.show()

        key_features = [
            "txn_count_2015_10",
            "txn_count_2015_09",
            "oct_2015_to_oct_2014_ratio",
            "preholiday_count_yoy_ratio",
            "monthly_txn_count_slope_3m",
            "weighted_recent_count",
            "oct_recency_interaction",
        ]

        print("Key engineered feature ranks:")
        for feature in key_features:
            if feature in avg_feat_imp["feature"].values:
                rank = avg_feat_imp.reset_index(drop=True).index[avg_feat_imp["feature"].values == feature][0] + 1
                score = avg_feat_imp.loc[avg_feat_imp["feature"] == feature, "importance"].iloc[0]
                print(f"{rank:>3} | {feature:<30} | {score:.6f}")

        metrics = {
            "version": "v18",
            "oof_rmsle": {
                "catboost": float(rmsle_from_logs(y_log, oof_cat_log)),
                "lightgbm": float(rmsle_from_logs(y_log, oof_lgb_log)),
                "xgboost": float(rmsle_from_logs(y_log, oof_xgb_log)),
                "ridge": float(rmsle_from_logs(y_log, oof_ridge_log)),
                "blend": float(rmsle_from_logs(y_log, blend_oof_log)),
            },
            "blend_weights": {
                "catboost": float(W_CAT),
                "lightgbm": float(W_LGB),
                "xgboost": float(W_XGB),
                "ridge": float(W_RIDGE),
            },
            "median_best_iterations": {
                "catboost": int(np.median(cat_best_iters)) if cat_best_iters else int(BEST_CAT["iterations"]),
                "lightgbm": int(np.median(lgb_best_iters)) if lgb_best_iters else int(BEST_LGB["n_estimators"]),
                "xgboost": int(np.median(xgb_best_iters)) if xgb_best_iters else int(BEST_XGB["n_estimators"]),
                "ridge_alpha": float(np.median(ridge_alphas)) if ridge_alphas else float(RIDGE_ALPHAS[len(RIDGE_ALPHAS) // 2]),
            },
            "submission_file": str(submission_path),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }

        metrics_path = OUTPUT_PATH / "metrics_v18.json"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        print("Metrics saved:", metrics_path)
        print("Blend RMSLE   :", metrics["oof_rmsle"]["blend"])
        """
    ).strip(),
}


nb = nbformat.read(SRC, as_version=4)
for idx, source in cell_updates.items():
    nb.cells[idx].source = source

nbformat.write(nb, DST)
print(f"Wrote {DST}")
