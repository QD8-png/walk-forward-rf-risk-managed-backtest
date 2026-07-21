# -*- coding: utf-8 -*-
"""Robustness test on a non-overlapping pool of non-leader assets."""
import os
import sys
import pickle
import multiprocessing
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy
from run_full_academic_pipeline import process_single_asset_custom, custom_ablation_backtest

config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0
config.n_shuffles = 20
CACHE_FILE = "sci_baostock_assets_200_robust.pkl"

# The 49 leader/ETF tickers to EXCLUDE
LEADER_TICKERS = set([
    "sh.510300", "sh.510500", "sh.510050", "sz.159915",
    "sh.512000", "sh.512480", "sh.512170", "sh.512660",
    "sh.515030", "sh.512690", "sh.512880",
    "sh.600519", "sz.000858", "sh.600887", "sz.002714", 
    "sh.601933", "sz.002508", "sh.603288", "sh.600009",
    "sz.300750", "sz.002594", "sh.601012", "sz.002460",
    "sh.601899", "sh.600019", "sh.603993", "sh.600547",
    "sz.000977", "sh.603019", "sz.002230", "sz.002415",
    "sh.600584", "sz.000063", "sh.600745", "sz.300059",
    "sh.601138", "sz.002027", "sh.600036", "sh.601318",
    "sh.600030", "sh.601398", "sh.601688", "sz.000001",
    "sh.600900", "sh.601857", "sh.600028", "sh.600150",
    "sh.600276", "sz.300015"
])

def generate_robust_tickers(count: int = 800) -> List[str]:
    """
    Generate exactly the same way as the 117 pool, but EXPLICITLY filter out
    the 49 leader tickers. We use a different seed.
    """
    tickers = set()
    rng = np.random.default_rng(seed=42)  # Different seed from original
    sz_prefixes = ["000", "001", "002", "300"]
    ss_prefixes = ["600", "601", "603", "605"]
    
    while len(tickers) < count:
        is_sz = rng.choice([True, False])
        prefix = rng.choice(sz_prefixes) if is_sz else rng.choice(ss_prefixes)
        suffix = f"{rng.integers(1, 1000):03d}"
        ticker = f"{'sz' if is_sz else 'sh'}.{prefix}{suffix}"
        
        # KEY ROBUSTNESS LOGIC: Explicitly exclude any overlap with original leaders
        if ticker not in LEADER_TICKERS:
            tickers.add(ticker)
            
    return list(tickers)

def load_or_build_robust_predictions(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(CACHE_FILE):
        print(f"\n[CACHE] Loading from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
            
    print(f"\n[PREDICT] No cache found, downloading data for {len(tickers)} tickers from Baostock...")
    import baostock as bs
    bs.login()
    
    valid_dfs = {}
    for i, t in enumerate(tickers):
        if i % 100 == 0:
            print(f"   Downloading {i}/{len(tickers)}...")
        rs = bs.query_history_k_data_plus(t, "date,open,high,low,close,volume", start_date='2016-01-01', end_date='2023-12-31', frequency="d", adjustflag="3")
        if rs.error_code == '0':
            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())
            if len(data_list) > config.min_data_length:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                if len(df) > config.min_data_length:
                    valid_dfs[t] = df
    bs.logout()

    print(f"   Valid assets from Baostock: {len(valid_dfs)}. Starting WFO modeling...")
    all_assets = {}
    max_workers = min(4, multiprocessing.cpu_count(), len(valid_dfs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_asset_custom, t, df, config, 'rf'): t for t, df in valid_dfs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                all_assets[res['name']] = res
            if i % 20 == 0 or i == len(valid_dfs):
                print(f"   Progress: {i} / {len(valid_dfs)}")
                
    # If we have more than 200, strictly truncate to 200 for the test
    final_assets = dict(list(all_assets.items())[:200])
    
    print(f"   Modeling done. Final Universe size: {len(final_assets)}. Caching...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(final_assets, f)
    return final_assets

def main():
    print("=" * 70)
    print("★ ROBUSTNESS TEST (Non-Leader 200 Pool) ★")
    print("=" * 70)
    
    tickers = generate_robust_tickers(800) # Request 800 to ensure we get >200 valid after baostock filter
    all_assets = load_or_build_robust_predictions(tickers)
    
    print(f"\nRunning tests on {len(all_assets)} completely independent, non-leader stocks...")
    
    results = []
    
    print("   [0-A] Pure Rules Equivalent to S4...")
    r0a_ret, r0a_mdd, r0a_sharpe, r0a_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='pure_rules', enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False)
    results.append({"System": "System 0-A: Pure Rules Baseline", "Return": r0a_ret, "MDD": r0a_mdd})
    
    print("   [5] Full ARMS...")
    r5_ret, r5_mdd, r5_sharpe, r5_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='ml_rules', enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True)
    results.append({"System": "System 5: Full ARMS Framework", "Return": r5_ret, "MDD": r5_mdd})

    df = pd.DataFrame(results)
    for col in ["Return", "MDD"]:
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
        
    df.to_csv("robustness_200_results.csv", index=False)
    print("\n[SAVE] Robustness results saved to robustness_200_results.csv")
    print(df.to_string())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
