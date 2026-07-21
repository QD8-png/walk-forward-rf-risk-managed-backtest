# -*- coding: utf-8 -*-
"""
================================================================================
Experiment A1: 30-Seed Resampling (Broad-Based Pool, 2020-06-01 Start)
================================================================================
Reviewer Feedback Response (#1 Diagram/Text Consistency + #6 Narrative):
Runs 30 random seed samplings (seeds 1..30, plus seed 100 as Group 0 reference)
on a broad-based candidate pool starting from 2020-06-01 to 2023-12-31.
Compares System 5 vs MA Crossover (5/20) and computes paired delta Calmar metrics.

Outputs: exp_A1_seed_sampling_results.csv
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats

warnings.filterwarnings("ignore")

# Resolve local path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0

CACHE_FILES = ["sci_all_assets_2019_2023.pkl", "sci_baostock_assets_200_robust.pkl"]

def load_combined_asset_pool() -> Dict[str, Dict[str, Any]]:
    combined_pool = {}
    for cfile in CACHE_FILES:
        if os.path.exists(cfile):
            with open(cfile, 'rb') as f:
                cdata = pickle.load(f)
                for name, item in cdata.items():
                    if 'ma5' not in item:
                        close_s = pd.Series(item['close'])
                        item['ma5'] = close_s.rolling(5).mean().values
                        item['ma20'] = close_s.rolling(20).mean().values
                        item['momentum'] = (close_s.shift(21) / close_s.shift(252) - 1).values
                    combined_pool[name] = item

    if len(combined_pool) == 0:
        raise FileNotFoundError("No valid cache files found for asset pool!")
    return combined_pool

def run_single_seed_experiment(seed: int, pool: Dict[str, Dict[str, Any]], sample_size: int = 80) -> Tuple[Dict[str, float], Dict[str, float]]:
    all_keys = sorted(list(pool.keys()))
    rng = np.random.default_rng(seed=seed)
    
    # Randomly sample assets for this seed
    n_sample = min(sample_size, len(all_keys))
    sampled_keys = rng.choice(all_keys, size=n_sample, replace=False)
    sub_pool = {k: pool[k] for k in sampled_keys}

    # Run System 5 (from 2020-06-01 to 2023-12-31)
    res_sys5 = strategy.run_portfolio_backtest(
        sub_pool, config,
        enable_ml=True, enable_atr=True, enable_bbl=True, enable_toce=True, enable_full_exits=True,
        start_date=pd.Timestamp('2020-06-01'), end_date=pd.Timestamp('2023-12-31')
    )

    # Run MA Crossover Baseline (from 2020-06-01 to 2023-12-31)
    res_ma = strategy.run_portfolio_backtest(
        sub_pool, config,
        enable_ml=False, enable_atr=False, enable_bbl=False, enable_toce=False, enable_full_exits=False,
        use_ma_crossover=True,
        start_date=pd.Timestamp('2020-06-01'), end_date=pd.Timestamp('2023-12-31')
    )

    sys5_metrics = {
        'return': res_sys5['cum_return'],
        'mdd': res_sys5['max_drawdown'],
        'sharpe': res_sys5['sharpe_ratio'],
        'calmar': res_sys5['calmar_ratio']
    }
    ma_metrics = {
        'return': res_ma['cum_return'],
        'mdd': res_ma['max_drawdown'],
        'sharpe': res_ma['sharpe_ratio'],
        'calmar': res_ma['calmar_ratio']
    }
    return sys5_metrics, ma_metrics

def main():
    output_dir = os.path.abspath(os.path.dirname(__file__))
    output_csv = os.path.join(output_dir, "exp_A1_seed_sampling_results.csv")

    print("=== Starting Experiment A1: 30-Seed Resampling (2020-06-01 Start) ===")
    pool = load_combined_asset_pool()
    print(f"Loaded {len(pool)} total unique broad-based assets in pool.")

    # Seeds list: Group 0 (seed=100 reference) + Seeds 1 to 30
    seeds_to_run = [100] + list(range(1, 31))
    results = []

    for i, seed in enumerate(seeds_to_run):
        seed_label = f"Seed_{seed}" if seed != 100 else "Group_0 (Seed_100)"
        print(f"[{i+1}/{len(seeds_to_run)}] Running {seed_label}...")
        try:
            s5, ma = run_single_seed_experiment(seed, pool, sample_size=80)
            results.append({
                'seed_group': seed_label,
                'seed': seed,
                'sys5_return': s5['return'],
                'sys5_mdd': s5['mdd'],
                'sys5_sharpe': s5['sharpe'],
                'sys5_calmar': s5['calmar'],
                'ma_return': ma['return'],
                'ma_mdd': ma['mdd'],
                'ma_sharpe': ma['sharpe'],
                'ma_calmar': ma['calmar'],
                'delta_calmar': s5['calmar'] - ma['calmar']
            })
            print(f"   -> Sys5 Calmar: {s5['calmar']:.4f} | MA Calmar: {ma['calmar']:.4f} | Delta Calmar: {s5['calmar'] - ma['calmar']:.4f}")
        except Exception as e:
            print(f"   -> Error running seed {seed}: {e}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved {len(df_res)} seed runs to {output_csv}")

    # Statistical analysis of Delta Calmar across seeds 1..30 (excluding seed 100)
    df_eval = df_res[df_res['seed'] != 100]
    deltas = df_eval['delta_calmar'].values
    median_delta = np.median(deltas)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    pos_count = np.sum(deltas > 0)
    n_total = len(deltas)
    sign_test_p = stats.binomtest(pos_count, n_total, p=0.5).pvalue if hasattr(stats, 'binomtest') else 0.5

    print("\n--- Summary Statistics across 30 Seeds (Seeds 1..30) ---")
    print(f"Paired Delta Calmar Median: {median_delta:.4f}")
    print(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Sign Test: {pos_count}/{n_total} positive | p-value: {sign_test_p:.4f}")

if __name__ == "__main__":
    main()
