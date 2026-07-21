# -*- coding: utf-8 -*-
"""
================================================================================
Experiment E1: ATR Stop-Loss Multiplier Sensitivity Grid (1.5x, 2.0x, 2.5x)
================================================================================
Reviewer Feedback Response (#7 ATR Grid):
Tests ATR trailing stop-loss multipliers (1.5x, 2.0x, 2.5x) on Pure Rules + ATR setup
to evaluate MDD compression and scale invariance across volatility parameters.

Tested on unbiased random A-share assets (2019-2023).
Outputs: exp_E1_atr_grid.csv
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
    output_csv = os.path.join(output_dir, "exp_E1_atr_grid.csv")

    print("=== Starting Experiment E1: ATR Multiplier Grid Analysis ===")
    assets_112 = load_random_pool_assets()
    print(f"Loaded {len(assets_112)} random broad-based A-share assets from {CACHE_FILE}.")

    atr_multipliers = [1.5, 2.0, 2.5]
    results = []

    for mult in atr_multipliers:
        print(f"\n--- Testing ATR Multiplier: {mult}x ---")
        cfg_custom = strategy.StrategyConfig()
        cfg_custom.portfolio_capital = 1_000_000.0
        cfg_custom.atr_multiplier = mult

        # Pure Rules + ATR Setup
        res = strategy.run_portfolio_backtest(
            assets_112, cfg_custom,
            enable_ml=False, enable_atr=True, enable_bbl=False, enable_toce=False, enable_full_exits=False,
            start_date=pd.Timestamp('2019-01-01'), end_date=pd.Timestamp('2023-12-31')
        )

        results.append({
            'atr_multiplier': f"{mult}x",
            'multiplier_val': mult,
            'cum_return': res['cum_return'],
            'max_drawdown': res['max_drawdown'],
            'sharpe_ratio': res['sharpe_ratio'],
            'calmar_ratio': res['calmar_ratio'],
            'total_trades': res['total_trades']
        })
        print(f"   ATR {mult}x -> Return: {res['cum_return']*100:.2f}% | MDD: {res['max_drawdown']*100:.2f}% | Sharpe: {res['sharpe_ratio']:.4f} | Calmar: {res['calmar_ratio']:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved ATR multiplier grid results to {output_csv}")

if __name__ == "__main__":
    main()
