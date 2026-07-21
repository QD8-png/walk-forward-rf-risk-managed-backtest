# -*- coding: utf-8 -*-
"""
================================================================================
Experiment B1: ETF Stratification (49 Sector Leaders -> 38 Pure Equities + 11 ETFs)
================================================================================
Reviewer Feedback Response (#3 ETF Stratification):
Evaluates System 4 and System 5 performance across:
1. Full 49 Sector Leaders Pool
2. 11 ETF Sub-pool
3. 38 Pure Stock Equities Sub-pool

Outputs: exp_B1_etf_stratification.csv
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

# Resolve local path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0

CACHE_FILE = "all_assets_rf_2019_2023.pkl"

ETF_TICKERS = [
    '510500.SS', '510300.SS', '510050.SS', '159915.SZ', '512480.SS',
    '512170.SS', '512000.SS', '515030.SS', '512690.SS', '512660.SS', '512880.SS'
]

def load_processed_49_assets() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache file {CACHE_FILE} not found!")

    with open(CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)

    # Validate elements contain required fields
    for name, item in cache_data.items():
        if 'ma5' not in item:
            close_s = pd.Series(item['close'])
            item['ma5'] = close_s.rolling(5).mean().values
            item['ma20'] = close_s.rolling(20).mean().values
            item['momentum'] = (close_s.shift(21) / close_s.shift(252) - 1).values

    return cache_data

def run_system_evaluation(assets: Dict[str, Dict[str, Any]], system_name: str, enable_full_exits: bool) -> Dict[str, float]:
    res = strategy.run_portfolio_backtest(
        assets, config,
        enable_ml=True, enable_atr=True, enable_bbl=True, enable_toce=True,
        enable_full_exits=enable_full_exits,
        start_date=pd.Timestamp('2019-01-01'), end_date=pd.Timestamp('2023-12-31')
    )
    return {
        'system': system_name,
        'cum_return': res['cum_return'],
        'max_drawdown': res['max_drawdown'],
        'sharpe_ratio': res['sharpe_ratio'],
        'calmar_ratio': res['calmar_ratio'],
        'total_trades': res['total_trades']
    }

def main():
    output_dir = os.path.abspath(os.path.dirname(__file__))
    output_csv = os.path.join(output_dir, "exp_B1_etf_stratification.csv")

    print("=== Starting Experiment B1: ETF Stratification Analysis ===")
    all_49_assets = load_processed_49_assets()
    print(f"Successfully loaded {len(all_49_assets)} total sector leader assets from {CACHE_FILE}.")

    # Split into ETFs and Stocks
    etf_assets = {k: v for k, v in all_49_assets.items() if k in ETF_TICKERS}
    stock_assets = {k: v for k, v in all_49_assets.items() if k not in ETF_TICKERS}

    print(f" -> Full Pool: {len(all_49_assets)} assets")
    print(f" -> ETF Pool: {len(etf_assets)} assets ({list(etf_assets.keys())[:3]}...)")
    print(f" -> Pure Stock Pool: {len(stock_assets)} assets")

    results = []

    pools = [
        ("Full 49 Sector Leaders", all_49_assets),
        ("11 ETFs Sub-pool", etf_assets),
        ("38 Pure Equities Sub-pool", stock_assets)
    ]

    for pool_label, pool_dict in pools:
        print(f"\n--- Testing Pool: {pool_label} ({len(pool_dict)} assets) ---")
        # System 4 (ML + ATR + BBL + TOCE, no full trailing exits)
        s4 = run_system_evaluation(pool_dict, "System 4 (ML+ATR+BBL+TOCE)", enable_full_exits=False)
        s4['pool_name'] = pool_label
        s4['asset_count'] = len(pool_dict)
        results.append(s4)
        print(f"   System 4 -> Return: {s4['cum_return']*100:.2f}% | MDD: {s4['max_drawdown']*100:.2f}% | Sharpe: {s4['sharpe_ratio']:.4f} | Calmar: {s4['calmar_ratio']:.4f}")

        # System 5 (Full ARMS Framework)
        s5 = run_system_evaluation(pool_dict, "System 5 (Full ARMS)", enable_full_exits=True)
        s5['pool_name'] = pool_label
        s5['asset_count'] = len(pool_dict)
        results.append(s5)
        print(f"   System 5 -> Return: {s5['cum_return']*100:.2f}% | MDD: {s5['max_drawdown']*100:.2f}% | Sharpe: {s5['sharpe_ratio']:.4f} | Calmar: {s5['calmar_ratio']:.4f}")

    df_out = pd.DataFrame(results)[['pool_name', 'asset_count', 'system', 'cum_return', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio', 'total_trades']]
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved ETF stratification results to {output_csv}")

if __name__ == "__main__":
    main()
