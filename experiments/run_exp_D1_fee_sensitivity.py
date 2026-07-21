# -*- coding: utf-8 -*-
"""
================================================================================
Experiment D1: Transaction Fee Sensitivity Analysis (13 bps, 20 bps, 30 bps)
================================================================================
Reviewer Feedback Response (#5 Fee Sensitivity):
Evaluates System 5 and MA Crossover (5/20) under three round-trip fee regimes:
1. 13 bps (0.0013) - Baseline
2. 20 bps (0.0020) - Intermediate friction
3. 30 bps (0.0030) - High friction / retail scale

Tested on the unbiased random A-share pool (2019-2023).
Outputs: exp_D1_fee_sensitivity.csv
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

CACHE_FILE = "sci_all_assets_2019_2023.pkl"

def load_random_pool_assets() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache file {CACHE_FILE} not found!")

    with open(CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)

    for name, item in cache_data.items():
        if 'ma5' not in item:
            close_s = pd.Series(item['close'])
            item['ma5'] = close_s.rolling(5).mean().values
            item['ma20'] = close_s.rolling(20).mean().values
            item['momentum'] = (close_s.shift(21) / close_s.shift(252) - 1).values

    return cache_data

def main():
    output_dir = os.path.abspath(os.path.dirname(__file__))
    output_csv = os.path.join(output_dir, "exp_D1_fee_sensitivity.csv")

    print("=== Starting Experiment D1: Fee Sensitivity Analysis ===")
    assets_112 = load_random_pool_assets()
    print(f"Loaded {len(assets_112)} random broad-based A-share assets from {CACHE_FILE}.")

    fee_rates = [
        ("13 bps (0.13%)", 0.0013),
        ("20 bps (0.20%)", 0.0020),
        ("30 bps (0.30%)", 0.0030)
    ]

    results = []

    for fee_name, fee_val in fee_rates:
        print(f"\n--- Testing Fee Tier: {fee_name} ---")
        cfg_custom = strategy.StrategyConfig()
        cfg_custom.portfolio_capital = 1_000_000.0
        cfg_custom.fee_rate = fee_val

        # 1. System 5 (Full ARMS)
        res_s5 = strategy.run_portfolio_backtest(
            assets_112, cfg_custom,
            enable_ml=True, enable_atr=True, enable_bbl=True, enable_toce=True, enable_full_exits=True,
            start_date=pd.Timestamp('2019-01-01'), end_date=pd.Timestamp('2023-12-31')
        )
        results.append({
            'fee_tier': fee_name,
            'fee_rate': fee_val,
            'strategy': 'System 5 (Full ARMS)',
            'cum_return': res_s5['cum_return'],
            'max_drawdown': res_s5['max_drawdown'],
            'sharpe_ratio': res_s5['sharpe_ratio'],
            'calmar_ratio': res_s5['calmar_ratio'],
            'total_trades': res_s5['total_trades']
        })
        print(f"   System 5 -> Return: {res_s5['cum_return']*100:.2f}% | MDD: {res_s5['max_drawdown']*100:.2f}% | Sharpe: {res_s5['sharpe_ratio']:.4f} | Calmar: {res_s5['calmar_ratio']:.4f}")

        # 2. MA Crossover Baseline
        res_ma = strategy.run_portfolio_backtest(
            assets_112, cfg_custom,
            enable_ml=False, enable_atr=False, enable_bbl=False, enable_toce=False, enable_full_exits=False,
            use_ma_crossover=True,
            start_date=pd.Timestamp('2019-01-01'), end_date=pd.Timestamp('2023-12-31')
        )
        results.append({
            'fee_tier': fee_name,
            'fee_rate': fee_val,
            'strategy': 'MA Crossover (5/20)',
            'cum_return': res_ma['cum_return'],
            'max_drawdown': res_ma['max_drawdown'],
            'sharpe_ratio': res_ma['sharpe_ratio'],
            'calmar_ratio': res_ma['calmar_ratio'],
            'total_trades': res_ma['total_trades']
        })
        print(f"   MA Cross -> Return: {res_ma['cum_return']*100:.2f}% | MDD: {res_ma['max_drawdown']*100:.2f}% | Sharpe: {res_ma['sharpe_ratio']:.4f} | Calmar: {res_ma['calmar_ratio']:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved fee sensitivity matrix to {output_csv}")

if __name__ == "__main__":
    main()
