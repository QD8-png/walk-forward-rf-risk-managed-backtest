# -*- coding: utf-8 -*-
import os
import sys
import copy
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy
from run_full_academic_pipeline import custom_ablation_backtest

CACHE_FILE = "sci_baostock_assets_2016_2023.pkl"

def _permutation_worker(args):
    all_assets, config, seed = args
    rng = np.random.default_rng(seed=seed)
    
    # Deepcopy to avoid modifying the shared reference
    shuffled_assets = copy.deepcopy(all_assets)
    
    for stock_name, asset in shuffled_assets.items():
        n_len = len(asset['y_pred'])
        idx = rng.permutation(n_len)
        asset['y_pred'] = asset['y_pred'][idx]
        asset['y_prob'] = asset['y_prob'][idx]
        
    # Run System 5 backtest
    # enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True
    r5_ret, r5_mdd, r5_sharpe, r5_calmar, _, _, _ = custom_ablation_backtest(
        shuffled_assets, config, strategy_type='ml_rules', 
        enable_atr=True, enable_bbl=True, enable_toce=True, 
        enable_trailing=True, enable_bbi_tp=True
    )
    
    return {
        'seed': seed,
        'return': r5_ret,
        'mdd': r5_mdd,
        'sharpe': r5_sharpe
    }

def main():
    print("[OK] Loading predictions cache...")
    with open(CACHE_FILE, 'rb') as f:
        all_assets = pickle.load(f)
        
    config = strategy.StrategyConfig()
    config.portfolio_capital = 1_000_000.0
    
    # First, get the actual return of System 5 without shuffling
    print("[OK] Running baseline System 5...")
    actual_ret, actual_mdd, actual_sharpe, actual_calmar, _, _, _ = custom_ablation_backtest(
        all_assets, config, strategy_type='ml_rules', 
        enable_atr=True, enable_bbl=True, enable_toce=True, 
        enable_trailing=True, enable_bbi_tp=True
    )
    print(f"System 5 Actual Return: {actual_ret*100:.2f}%, MDD: {actual_mdd*100:.2f}%")
    
    # Now run permutations
    n_permutations = 50
    print(f"\n[OK] Starting Monte Carlo Permutation Test ({n_permutations} runs)...")
    
    rng = np.random.default_rng(seed=42)
    seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(n_permutations)]
    
    tasks = [(all_assets, config, s) for s in seeds]
    results = []
    
    max_workers = min(multiprocessing.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_permutation_worker, args): i for i, args in enumerate(tasks)}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if i % 10 == 0 or i == n_permutations:
                print(f"  [OK] Completed {i} / {n_permutations} permutations")
                
    # Calculate statistics
    returns = np.array([r['return'] for r in results])
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    p_value = np.mean(returns >= actual_ret)
    
    print("\n" + "="*60)
    print("★ Monte Carlo Permutation Test Results ★")
    print("="*60)
    print(f"Actual System 5 Return:   {actual_ret*100:.2f}%")
    print(f"Randomized Mean Return:   {mean_ret*100:.2f}%")
    print(f"Randomized Std Dev:       ±{std_ret*100:.2f}%")
    print(f"Randomized Max Return:    {returns.max()*100:.2f}%")
    print(f"Randomized Min Return:    {returns.min()*100:.2f}%")
    print(f"Empirical p-value:        {p_value:.4f}")
    
    if p_value < 0.05:
        print("\nConclusion: Alpha is statistically significant (p < 0.05).")
    else:
        print("\nConclusion: NO SIGNIFICANT ALPHA (p >= 0.05).")
        print("The ML predictions provide no predictive power beyond random guessing.")
        print("This strongly supports the Complexity Trap hypothesis.")
    print("="*60)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("permutation_test_results.csv", index=False)
    print("\n[OK] Results saved to permutation_test_results.csv")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
