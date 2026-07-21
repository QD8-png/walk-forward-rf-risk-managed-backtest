# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy
from run_full_academic_pipeline import custom_ablation_backtest

def main():
    print("=" * 70)
    print("★ Running Multi-Seed Asset Sub-Sampling Validation (30 Runs) ★")
    print("=" * 70)
    
    # 1. Load and merge predictions
    files = [
        'sci_baostock_assets_2016_2023.pkl',
        'sci_baostock_assets_200_robust.pkl',
        'sci_all_assets_2019_2023.pkl'
    ]
    
    all_stocks = {}
    for f in files:
        if os.path.exists(f):
            print(f"Loading predictions from {f}...")
            with open(f, 'rb') as pf:
                data = pickle.load(pf)
                for k, v in data.items():
                    if k not in all_stocks:
                        all_stocks[k] = v
                        
    print(f"Total unique stocks loaded: {len(all_stocks)}")
    
    # 2. Align stock timelines
    # We require the stock data to start on or before 2020-06-01 and end on or after 2023-12-25
    aligned_stocks = {}
    start_cutoff = pd.Timestamp('2020-06-01')
    end_cutoff = pd.Timestamp('2023-12-25')
    
    for k, v in all_stocks.items():
        dates = pd.to_datetime(v['dates'])
        if len(dates) == 0:
            continue
        if dates.min() <= start_cutoff and dates.max() >= end_cutoff:
            aligned_stocks[k] = v
            
    print(f"Number of aligned stocks (Jan 2019/2020 to Dec 2023): {len(aligned_stocks)}")
    if len(aligned_stocks) < 117:
        print(f"Warning: Aligned stocks ({len(aligned_stocks)}) is less than target sample size (117)!")
        print("Loosening start date cutoff to 2021-01-01...")
        aligned_stocks = {}
        start_cutoff = pd.Timestamp('2021-01-01')
        for k, v in all_stocks.items():
            dates = pd.to_datetime(v['dates'])
            if len(dates) == 0:
                continue
            if dates.min() <= start_cutoff and dates.max() >= end_cutoff:
                aligned_stocks[k] = v
        print(f"New number of aligned stocks: {len(aligned_stocks)}")
        
    stock_names = list(aligned_stocks.keys())
    
    config = strategy.StrategyConfig()
    config.portfolio_capital = 1_000_000.0
    
    results = []
    
    # 3. Run 30 random sub-sampling trials
    n_trials = 30
    sample_size = 117
    
    # Use seed 42 to generate reproducible seeds for the 30 trials
    rng_master = np.random.default_rng(seed=42)
    seeds = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(n_trials)]
    
    print(f"\nStarting {n_trials} trials of sampling {sample_size} stocks...")
    for trial_idx in range(n_trials):
        t0 = time.time()
        trial_seed = seeds[trial_idx]
        rng_trial = np.random.default_rng(seed=trial_seed)
        
        # Randomly sample 117 stocks
        sampled_keys = rng_trial.choice(stock_names, size=sample_size, replace=False)
        sampled_assets = {k: aligned_stocks[k] for k in sampled_keys}
        
        # Run MA Crossover
        ma_ret, ma_mdd, ma_sharpe, ma_calmar, _, _, _ = custom_ablation_backtest(
            sampled_assets, config, strategy_type='ma_crossover',
            enable_atr=False, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False
        )
        
        # Run System 5 (Full ARMS)
        s5_ret, s5_mdd, s5_sharpe, s5_calmar, _, _, _ = custom_ablation_backtest(
            sampled_assets, config, strategy_type='ml_rules',
            enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True
        )
        
        trial_res = {
            'Trial': trial_idx + 1,
            'Seed': trial_seed,
            'MA_Return': ma_ret,
            'MA_MDD': ma_mdd,
            'MA_Sharpe': ma_sharpe,
            'MA_Calmar': ma_calmar,
            'S5_Return': s5_ret,
            'S5_MDD': s5_mdd,
            'S5_Sharpe': s5_sharpe,
            'S5_Calmar': s5_calmar,
            'Diff_Return': ma_ret - s5_ret,
            'Diff_Calmar': ma_calmar - s5_calmar
        }
        results.append(trial_res)
        
        t_elapsed = time.time() - t0
        print(f"Trial {trial_idx+1:02d}/{n_trials:02d} | Seed: {trial_seed:<10d} | "
              f"MA Calmar: {ma_calmar:.4f} (Ret: {ma_ret*100:.2f}%) | "
              f"S5 Calmar: {s5_calmar:.4f} (Ret: {s5_ret*100:.2f}%, MDD: {s5_mdd*100:.2f}%) | "
              f"Time: {t_elapsed:.1f}s")
              
    df_results = pd.DataFrame(results)
    df_results.to_csv("multiseed_sampling_results.csv", index=False)
    print("\nSaved trial results to multiseed_sampling_results.csv")
    
    # 4. Compute statistics
    def format_stat(series, is_pct=False):
        mean = series.mean()
        std = series.std()
        # 95% CI of the distribution
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        
        factor = 100.0 if is_pct else 1.0
        unit = "%" if is_pct else ""
        return f"{mean*factor:.2f}{unit} ± {std*factor:.2f}{unit} (95% CI: [{ci_lower*factor:.2f}{unit}, {ci_upper*factor:.2f}{unit}])"

    print("\n" + "=" * 70)
    print("★ Multi-Seed Sub-Sampling Summary Statistics (N = 30) ★")
    print("=" * 70)
    
    summary = {
        "Metric": ["Annualized Return", "Maximum Drawdown", "Sharpe Ratio", "Calmar Ratio"],
        "MA Crossover (Benchmark)": [
            format_stat(df_results["MA_Return"], is_pct=True),
            format_stat(df_results["MA_MDD"], is_pct=True),
            format_stat(df_results["MA_Sharpe"]),
            format_stat(df_results["MA_Calmar"])
        ],
        "System 5 (Full ARMS)": [
            format_stat(df_results["S5_Return"], is_pct=True),
            format_stat(df_results["S5_MDD"], is_pct=True),
            format_stat(df_results["S5_Sharpe"]),
            format_stat(df_results["S5_Calmar"])
        ],
        "Difference (MA - S5)": [
            format_stat(df_results["Diff_Return"], is_pct=True),
            "-",
            "-",
            format_stat(df_results["Diff_Calmar"])
        ]
    }
    
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Check if MA Crossover consistently outperforms System 5
    ma_better_ret = np.mean(df_results["MA_Return"] > df_results["S5_Return"])
    ma_better_calmar = np.mean(df_results["MA_Calmar"] > df_results["S5_Calmar"])
    print("\n" + "=" * 70)
    print(f"Probability that MA Crossover outperforms System 5 in Return: {ma_better_ret*100:.1f}%")
    print(f"Probability that MA Crossover outperforms System 5 in Calmar: {ma_better_calmar*100:.1f}%")
    
    # Check回撤压缩的跨资产池不变性 (System 5 MDD stability)
    s5_mdd_mean = df_results["S5_MDD"].mean()
    s5_mdd_std = df_results["S5_MDD"].std()
    print(f"System 5 Maximum Drawdown: {s5_mdd_mean*100:.2f}% ± {s5_mdd_std*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
