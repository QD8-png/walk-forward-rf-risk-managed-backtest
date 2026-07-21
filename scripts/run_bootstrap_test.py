# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def circular_block_bootstrap_indices(n_days, block_size, rng):
    """
    Generates bootstrapped indices of length n_days using Circular Block Bootstrap.
    This preserves the autocorrelation and path-dependency of time-series data.
    """
    indices = []
    # Calculate how many blocks we need
    num_blocks = int(np.ceil(n_days / block_size))
    
    for _ in range(num_blocks):
        # Pick a random start index
        start_idx = rng.integers(0, n_days)
        # Create block indices with circular wrap-around
        block = [(start_idx + i) % n_days for i in range(block_size)]
        indices.extend(block)
        
    # Truncate to exact length
    return np.array(indices[:n_days])

def calculate_metrics(returns):
    """Reconstructs NAV from daily returns and calculates performance metrics."""
    nav = np.cumprod(1.0 + returns)
    total_return = nav[-1] - 1.0
    
    # Calculate Maximum Drawdown (MDD)
    peak = np.maximum.accumulate(nav)
    # Avoid division by zero
    peak = np.maximum(peak, 1e-8)
    drawdowns = (nav - peak) / peak
    mdd = drawdowns.min()
    
    # Annualized Return
    n_days = len(returns)
    ann_factor = 252.0
    ann_return = (1.0 + total_return) ** (ann_factor / n_days) - 1.0 if n_days > 0 else 0.0
    
    # Calmar Ratio
    calmar = ann_return / abs(mdd) if mdd != 0.0 else 0.0
    
    return ann_return, mdd, calmar

def main():
    print("=" * 70)
    print("★ Running Paired Circular Block Bootstrap Significance Test (5000 Runs) ★")
    print("=" * 70)
    
    csv_path = "equity_curves_117_pool.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run run_full_academic_pipeline.py first.")
        return
        
    print(f"Loading daily NAV data from {csv_path}...")
    df_nav = pd.read_csv(csv_path)
    
    # Calculate daily returns
    df_ret = pd.DataFrame({'Date': df_nav['Date']})
    df_ret['MA_Ret'] = df_nav['MA Crossover'].pct_change()
    df_ret['S5_Ret'] = df_nav['System 5'].pct_change()
    df_ret = df_ret.dropna().reset_index(drop=True)
    
    ma_returns = df_ret['MA_Ret'].values
    s5_returns = df_ret['S5_Ret'].values
    n_days = len(df_ret)
    
    print(f"Total trading days: {n_days}")
    
    # Point estimates on the actual original data
    ma_ann_ret_orig, ma_mdd_orig, ma_calmar_orig = calculate_metrics(ma_returns)
    s5_ann_ret_orig, s5_mdd_orig, s5_calmar_orig = calculate_metrics(s5_returns)
    
    print("\n--- Point Estimates on Original Data ---")
    print(f"MA Crossover | Ann. Return: {ma_ann_ret_orig*100:.2f}% | MDD: {ma_mdd_orig*100:.2f}% | Calmar: {ma_calmar_orig:.4f}")
    print(f"System 5     | Ann. Return: {s5_ann_ret_orig*100:.2f}% | MDD: {s5_mdd_orig*100:.2f}% | Calmar: {s5_calmar_orig:.4f}")
    print(f"Difference   | Ann. Return Diff: {(ma_ann_ret_orig - s5_ann_ret_orig)*100:.2f}% | Calmar Diff: {ma_calmar_orig - s5_calmar_orig:.4f}")
    
    # Run Bootstrap
    n_bootstraps = 5000
    block_size = 20 # 20 trading days (~1 month)
    
    results = []
    
    rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
    
    print(f"\nRunning {n_bootstraps} Circular Block Bootstrap resamples (block size = {block_size} days)...")
    for boot_idx in range(n_bootstraps):
        boot_indices = circular_block_bootstrap_indices(n_days, block_size, rng)
        
        # Draw pairwise returns to maintain chronological alignment of market regimes
        ma_boot_ret = ma_returns[boot_indices]
        s5_boot_ret = s5_returns[boot_indices]
        
        ma_ann_ret, ma_mdd, ma_calmar = calculate_metrics(ma_boot_ret)
        s5_ann_ret, s5_mdd, s5_calmar = calculate_metrics(s5_boot_ret)
        
        results.append({
            'MA_Return': ma_ann_ret,
            'MA_MDD': ma_mdd,
            'MA_Calmar': ma_calmar,
            'S5_Return': s5_ann_ret,
            'S5_MDD': s5_mdd,
            'S5_Calmar': s5_calmar,
            'Diff_Return': ma_ann_ret - s5_ann_ret,
            'Diff_Calmar': ma_calmar - s5_calmar
        })
        
        if (boot_idx + 1) % 1000 == 0 or (boot_idx + 1) == n_bootstraps:
            print(f"  Completed {boot_idx + 1} / {n_bootstraps} resamples")
            
    df_boot = pd.DataFrame(results)
    df_boot.to_csv("bootstrap_test_results.csv", index=False)
    print("\nSaved bootstrap results to bootstrap_test_results.csv")
    
    # Calculate statistics and confidence intervals
    def get_stats(series, orig_val, is_pct=False):
        mean_val = series.mean()
        std_val = series.std()
        
        # Percentile bootstrap confidence interval
        ci_lower = np.percentile(series, 2.5)
        ci_upper = np.percentile(series, 97.5)
        
        factor = 100.0 if is_pct else 1.0
        unit = "%" if is_pct else ""
        
        return f"{orig_val*factor:.2f}{unit} | Mean: {mean_val*factor:.2f}{unit} ± {std_val*factor:.2f}{unit} (95% CI: [{ci_lower*factor:.2f}{unit}, {ci_upper*factor:.2f}{unit}])"
        
    print("\n" + "=" * 80)
    print("★ Circular Block Bootstrap Summary Statistics (N = 5000) ★")
    print("=" * 80)
    
    # Empirical p-values for difference (H0: MA <= S5 vs H1: MA > S5)
    # One-sided p-value: proportion of bootstrap differences <= 0
    p_val_calmar = np.mean(df_boot['Diff_Calmar'] <= 0)
    p_val_return = np.mean(df_boot['Diff_Return'] <= 0)
    
    # Two-sided p-value:
    # 2 * min(P(diff <= 0), P(diff > 0))
    p_val_calmar_2sided = 2 * min(np.mean(df_boot['Diff_Calmar'] <= 0), np.mean(df_boot['Diff_Calmar'] > 0))
    p_val_return_2sided = 2 * min(np.mean(df_boot['Diff_Return'] <= 0), np.mean(df_boot['Diff_Return'] > 0))
    
    print(f"MA Crossover Calmar Ratio  : {get_stats(df_boot['MA_Calmar'], ma_calmar_orig)}")
    print(f"System 5 Calmar Ratio      : {get_stats(df_boot['S5_Calmar'], s5_calmar_orig)}")
    print(f"Calmar Ratio Diff (MA - S5) : {get_stats(df_boot['Diff_Calmar'], ma_calmar_orig - s5_calmar_orig)}")
    print(f"  --> Empirical p-value (one-sided MA > S5): {p_val_calmar:.6f}")
    print(f"  --> Empirical p-value (two-sided):        {p_val_calmar_2sided:.6f}")
    
    print("-" * 80)
    print(f"MA Crossover Ann. Return   : {get_stats(df_boot['MA_Return'], ma_ann_ret_orig, is_pct=True)}")
    print(f"System 5 Ann. Return       : {get_stats(df_boot['S5_Return'], s5_ann_ret_orig, is_pct=True)}")
    print(f"Ann. Return Diff (MA - S5)  : {get_stats(df_boot['Diff_Return'], ma_ann_ret_orig - s5_ann_ret_orig, is_pct=True)}")
    print(f"  --> Empirical p-value (one-sided MA > S5): {p_val_return:.6f}")
    print(f"  --> Empirical p-value (two-sided):        {p_val_return_2sided:.6f}")
    
    print("-" * 80)
    print(f"MA Crossover Max Drawdown  : {get_stats(df_boot['MA_MDD'], ma_mdd_orig, is_pct=True)}")
    print(f"System 5 Max Drawdown      : {get_stats(df_boot['S5_MDD'], s5_mdd_orig, is_pct=True)}")
    print("=" * 80)
    
    if p_val_calmar < 0.05:
        print("\nConclusion: The Calmar ratio of MA Crossover is significantly higher than System 5 (p < 0.05).")
        print("This provides strong empirical support for the 'Complexity Trap' hypothesis.")
    else:
        print("\nConclusion: The difference in Calmar ratio is not statistically significant.")
        
if __name__ == '__main__':
    main()
