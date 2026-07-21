# -*- coding: utf-8 -*-
"""
================================================================================
Experiment F1: Signal Layer Evaluation (WFO ROC-AUC, PR-AUC, Calibration, SHAP)
================================================================================
Reviewer Feedback Response (#8 Signal Layer Evaluation + ESWA SHAP Analysis):
1. Instruments rolling Walk-Forward Optimization (WFO) to collect out-of-sample
   predictions (y_prob) vs ground truth targets (y_true).
2. Computes ROC-AUC, PR-AUC, Brier Score, and Calibration Curve data.
3. Performs SHAP TreeExplainer analysis on Random Forest models across windows
   to compute feature importance (mean |SHAP|) across the 16 technical features.

Outputs:
- exp_F1_signal_evaluation.csv
- exp_F1_calibration_curve.csv
- exp_F1_shap_importance.csv
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
import shap

warnings.filterwarnings("ignore")

# Resolve local path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0

CACHE_FILE = "all_assets_rf_2019_2023.pkl"

FEATURE_NAMES = [
    '收益率_lag1', '收益率_lag2', '收益率_lag3',
    'RSI_14', 'KDJ_J',
    'MA5_偏离', 'MA10_偏离', 'MA60_偏离', 'EMA13_偏离',
    'MA均线交叉', '多空线偏离', 'BBI偏离', '趋势线偏离',
    '成交量变化', '振幅', 'Volatility'
]

def walk_forward_predict_with_shap(df: pd.DataFrame, config: strategy.StrategyConfig) -> Tuple[np.ndarray, np.ndarray, int, List[np.ndarray]]:
    """Runs WFO and extracts out-of-sample probabilities, targets, and SHAP values per window."""
    train_window = config.train_window
    retrain_every = config.retrain_every
    
    # Feature matrix & target vector
    X = df[strategy.DEFAULT_FEATURE_COLS].values
    y = df['Target_方向'].values
    n = len(df)

    probabilities = np.full(n, np.nan)
    targets = np.full(n, np.nan)
    shap_window_importance = []

    first_valid = train_window

    for train_end in range(train_window, n, retrain_every):
        test_end = min(train_end + retrain_every, n)
        X_train_raw = X[train_end - train_window : train_end - config.future_return_days]
        y_train_raw = y[train_end - train_window : train_end - config.future_return_days]

        valid_idx = ~np.isnan(y_train_raw)
        X_train = X_train_raw[valid_idx]
        y_train = y_train_raw[valid_idx].astype(int)

        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        if len(X_test) == 0 or len(X_train) == 0 or len(np.unique(y_train)) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        probabilities[train_end:test_end] = probs
        targets[train_end:test_end] = y_test

        # Compute SHAP values for current model on test window
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                # Class 1 SHAP values
                vals = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                vals = shap_values[:, :, 1]
            else:
                vals = shap_values

            # Mean absolute SHAP value for this window
            mean_abs_shap = np.mean(np.abs(vals), axis=0)
            shap_window_importance.append(mean_abs_shap)
        except Exception:
            pass

    return probabilities, targets, first_valid, shap_window_importance

def main():
    output_dir = os.path.abspath(os.path.dirname(__file__))
    file_eval = os.path.join(output_dir, "exp_F1_signal_evaluation.csv")
    file_calib = os.path.join(output_dir, "exp_F1_calibration_curve.csv")
    file_shap = os.path.join(output_dir, "exp_F1_shap_importance.csv")

    print("=== Starting Experiment F1: Signal Layer Evaluation & SHAP Analysis ===")

    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache file {CACHE_FILE} not found!")

    with open(CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)

    all_probs = []
    all_targets = []
    all_dates = []
    all_shap_list = []

    print(f"Processing WFO prediction signals and SHAP for {len(cache_data)} assets from {CACHE_FILE}...")

    for i, (ticker, item) in enumerate(cache_data.items()):
        df = pd.DataFrame({
            '日期': pd.to_datetime(item['dates']),
            '开盘': item['open'],
            '高': item['high'],
            '低': item['low'],
            '收盘': item['close'],
            '交易量': np.ones(len(item['close']))
        })

        try:
            df_feat = strategy.prepare_features(df, config)
        except Exception:
            continue

        if len(df_feat) < config.min_data_length:
            continue

        probs, targets, first_valid, shap_list = walk_forward_predict_with_shap(df_feat, config)

        sliced_df = df_feat.iloc[first_valid:].reset_index(drop=True)
        valid_probs = probs[first_valid:]
        valid_targets = targets[first_valid:]

        mask = ~np.isnan(valid_probs) & ~np.isnan(valid_targets)

        all_probs.extend(valid_probs[mask])
        all_targets.extend(valid_targets[mask])
        all_dates.extend(sliced_df['日期'].values[mask])
        all_shap_list.extend(shap_list)

        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(cache_data)} assets...")

    y_prob = np.array(all_probs)
    y_true = np.array(all_targets).astype(int)
    dates = pd.to_datetime(np.array(all_dates))

    print(f"\nCollected total out-of-sample prediction samples: {len(y_true)}")

    # 1. Compute Overall & Annual Metrics
    roc_auc_overall = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_overall = auc(rec, prec)
    brier_overall = brier_score_loss(y_true, y_prob)

    print(f"\nOverall Signal Layer Metrics (2019-2023):")
    print(f"   ROC-AUC:     {roc_auc_overall:.4f}")
    print(f"   PR-AUC:      {pr_auc_overall:.4f}")
    print(f"   Brier Score: {brier_overall:.4f}")

    annual_metrics = []
    annual_metrics.append({
        'period': 'Overall (2019-2023)',
        'sample_count': len(y_true),
        'roc_auc': roc_auc_overall,
        'pr_auc': pr_auc_overall,
        'brier_score': brier_overall
    })

    years = sorted(list(set(dates.year)))
    for yr in years:
        yr_mask = (dates.year == yr)
        if np.sum(yr_mask) > 0:
            y_p_yr = y_prob[yr_mask]
            y_t_yr = y_true[yr_mask]
            if len(np.unique(y_t_yr)) > 1:
                auc_yr = roc_auc_score(y_t_yr, y_p_yr)
                p_yr, r_yr, _ = precision_recall_curve(y_t_yr, y_p_yr)
                pr_auc_yr = auc(r_yr, p_yr)
                brier_yr = brier_score_loss(y_t_yr, y_p_yr)

                annual_metrics.append({
                    'period': f'Year {yr}',
                    'sample_count': len(y_t_yr),
                    'roc_auc': auc_yr,
                    'pr_auc': pr_auc_yr,
                    'brier_score': brier_yr
                })
                print(f"   Year {yr} -> ROC-AUC: {auc_yr:.4f} | PR-AUC: {pr_auc_yr:.4f} | Brier: {brier_yr:.4f}")

    df_eval = pd.DataFrame(annual_metrics)
    df_eval.to_csv(file_eval, index=False, encoding='utf-8-sig')

    # 2. Compute Calibration Curve (10 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    df_calib = pd.DataFrame({
        'bin_index': range(1, len(prob_true) + 1),
        'mean_predicted_prob': prob_pred,
        'fraction_of_positives': prob_true
    })
    df_calib.to_csv(file_calib, index=False, encoding='utf-8-sig')

    # 3. Compute Aggregated SHAP Feature Importance
    if len(all_shap_list) > 0:
        avg_shap = np.mean(all_shap_list, axis=0)
        df_shap = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'mean_abs_shap': avg_shap
        }).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
        df_shap['rank'] = range(1, len(df_shap) + 1)
        df_shap = df_shap[['rank', 'feature', 'mean_abs_shap']]
        df_shap.to_csv(file_shap, index=False, encoding='utf-8-sig')

        print("\n--- SHAP Feature Importance (Top 16) ---")
        for idx, row in df_shap.iterrows():
            print(f"   Rank {row['rank']:2d}: {row['feature']:<15} | mean(|SHAP|): {row['mean_abs_shap']:.6f}")

    print(f"\nSuccessfully saved all F1 signal layer results:")
    print(f" - {file_eval}")
    print(f" - {file_calib}")
    print(f" - {file_shap}")

if __name__ == "__main__":
    main()
