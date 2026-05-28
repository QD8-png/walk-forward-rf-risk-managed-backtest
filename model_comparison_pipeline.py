# -*- coding: utf-8 -*-
"""
Model Comparison Pipeline for Route B: Model Upgrade
===================================================
Compares: RF vs LightGBM vs LSTM vs Ensemble
Uses:
- Centrally managed data downloads.
- Clean walk-forward predictions caching.
- Shared-capital multi-asset backtester (2019-2023).
- High-fidelity TradingView dark comparison visualizer.
"""

import os
import sys
import copy
import pickle
import multiprocessing
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

# Matplotlib styling for dark TradeView-style charts
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# Import strategy
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0

def process_single_asset_custom(ticker: str, df: pd.DataFrame, config: strategy.StrategyConfig, model_type: str) -> Optional[Dict[str, Any]]:
    """Runs purged walk-forward ML predictions bypassing standard dates limits."""
    try:
        df = strategy.prepare_features(df, config)
    except Exception as e:
        return None

    if len(df) < config.min_data_length:
        return None

    if df['日期'].max() < pd.Timestamp('2023-12-01'):
        return None

    predictions, probabilities, first_valid = strategy.walk_forward_predict(df, config, model_type=model_type)
    sliced = df.iloc[first_valid:].reset_index(drop=True)
    
    return {
        'name': ticker,
        'dates': sliced['日期'].values,
        'close': sliced['收盘'].values,
        'open': sliced['开盘'].values,
        'low': sliced['低'].values,
        'high': sliced['高'].values,
        'bb_line': sliced['多空线'].values,
        'bbi': sliced['BBI'].values,
        'kdj_j': sliced['KDJ_J'].values,
        'ma120_slope': sliced['MA120_slope'].values,
        'atr': sliced['ATR'].values,
        'y_pred': predictions[first_valid:],
        'y_prob': probabilities[first_valid:],
    }

def get_expanded_50_tickers() -> List[str]:
    """Returns the same stock tickers used in expanded test cycle."""
    return [
        "510300.SS", "510500.SS", "510050.SS", "159915.SZ",
        "512000.SS", "512480.SS", "512170.SS", "512660.SS",
        "515030.SS", "512690.SS", "512880.SS",
        "600519.SS", "000858.SZ", "600887.SS", "002714.SZ", 
        "601933.SS", "002508.SZ", "603288.SS", "600009.SS",
        "300750.SZ", "002594.SZ", "601012.SS", "002460.SZ",
        "601899.SS", "600019.SS", "603993.SS", "600547.SS",
        "000977.SZ", "603019.SS", "002230.SZ", "002415.SZ",
        "600584.SS", "000063.SZ", "600745.SS", "300059.SZ",
        "601138.SS", "002027.SZ", "600036.SS", "601318.SS",
        "600030.SS", "601398.SS", "601688.SS", "000001.SZ",
        "600900.SS", "601857.SS", "600028.SS", "600150.SS",
        "600276.SS", "300015.SZ"
    ]

def load_or_build_predictions(tickers: List[str], model_type: str) -> Dict[str, Dict[str, Any]]:
    """Loads prediction results from local cache or executes the full WFO process and caches it."""
    cache_file = f"all_assets_{model_type}_2019_2023.pkl"
    
    # Optimization: Copy Route A's Random Forest prediction cache if it exists and rf is requested
    if model_type == 'rf' and not os.path.exists(cache_file) and os.path.exists("all_assets_2019_2023.pkl"):
        import shutil
        print(f"[OPTIMIZATION] 发现已存在的随机森林缓存 all_assets_2019_2023.pkl，正在复制至 {cache_file}...")
        try:
            shutil.copy("all_assets_2019_2023.pkl", cache_file)
        except Exception as e:
            print(f"复制缓存失败: {e}")

    if os.path.exists(cache_file):
        print(f"[CACHE] 找到模型 {model_type.upper()} 的缓存文件 {cache_file}，正在加载...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
            
    print(f"\n[PREDICT] 未找到 {model_type.upper()} 缓存，启动历史数据下载 (2016-01-01 至 2023-12-31)...")
    data = yf.download(tickers, start="2016-01-01", end="2023-12-31", group_by='ticker', threads=True, progress=False)
    
    valid_dfs = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].copy()
            else:
                df = data.copy() if len(tickers) == 1 else None
                if df is None: continue
            df = df.dropna(how='all')
            if len(df) > config.min_data_length:
                df = df.reset_index()
                valid_dfs[t] = df
        except Exception as e:
            print(f"   下载标的 {t} 失败: {e}")

    print(f"   足额有效标的数: {len(valid_dfs)}。启动 WFO {model_type.upper()} 并发计算...")
    all_assets = {}
    
    # LSTM can use a bit more memory or PyTorch multiprocessing, keep workers low to avoid CPU thrashing
    max_workers = min(2, multiprocessing.cpu_count(), len(valid_dfs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_asset_custom, t, df, config, model_type): t for t, df in valid_dfs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                all_assets[res['name']] = res
            if i % 10 == 0 or i == len(valid_dfs):
                print(f"   {model_type.upper()} 建模进度: {i} / {len(valid_dfs)}")
                
    print(f"   {model_type.upper()} 建模完毕，可用资产池大小: {len(all_assets)}。正在保存至缓存...")
    with open(cache_file, 'wb') as f:
        pickle.dump(all_assets, f)
    print(f"   [CACHE] 缓存成功: {cache_file}")
    return all_assets

def custom_backtest(
    all_assets: Dict[str, Dict[str, Any]], 
    config: strategy.StrategyConfig, 
) -> Tuple[float, float, float, int, List[float], List[pd.Timestamp]]:
    """
    Standard full strategy backtester.
    """
    capital = config.portfolio_capital
    holdings: Dict[str, strategy.Holding] = {}
    cooldowns: Dict[str, int] = {}
    
    # Gather dates
    all_dates = set()
    for asset in all_assets.values():
        all_dates.update(asset['dates'])
    all_dates = sorted(list(all_dates))
    
    asset_date_map = {name: {d: i for i, d in enumerate(asset['dates'])} for name, asset in all_assets.items()}
    
    portfolio_values = []
    portfolio_dates = []
    total_trades = 0
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2023-12-31')
    
    for day_idx, today in enumerate(all_dates):
        today_ts = pd.Timestamp(today)
        if today_ts < start_date or today_ts > end_date:
            continue
            
        # 1. Daily mark-to-market
        daily_portfolio_return = 0.0
        for stock_name in list(holdings.keys()):
            h = holdings[stock_name]
            idx_map = asset_date_map[stock_name]
            if today not in idx_map:
                continue
            local_idx = idx_map[today]
            asset = all_assets[stock_name]
            current_close = asset['close'][local_idx]
            prev_close = asset['close'][local_idx - 1] if local_idx > 0 else current_close
            
            if prev_close > 0:
                daily_return = current_close / prev_close - 1
                daily_portfolio_return += daily_return * h.position_weight
            
            if current_close > h.max_price:
                h.max_price = current_close
                
        capital = capital * (1 + daily_portfolio_return)
        
        # 2. Risk check & exits
        stocks_to_sell = []
        for stock_name in list(holdings.keys()):
            h = holdings[stock_name]
            idx_map = asset_date_map[stock_name]
            if today not in idx_map:
                continue
            local_idx = idx_map[today]
            asset = all_assets[stock_name]
            current_close = asset['close'][local_idx]
            
            should_exit = False
            unrealized = current_close / h.entry_price - 1
            holding_days = day_idx - h.entry_day_idx
            
            # Layer A: Bull-Bear Line Trend Interceptor
            if current_close < asset['bb_line'][local_idx]:
                should_exit = True
            # Layer B: ATR Volatility Stop Loss
            elif current_close < h.entry_stop_price:
                should_exit = True
            # Layer C: TOCE (Time-based Opportunity Cost Exit)
            elif holding_days >= config.patience_days and unrealized < config.patience_return:
                should_exit = True
            else:
                # Layer D: Trailing Profit Lock
                if unrealized < config.trailing_activate_pct:
                    if asset['y_pred'][local_idx] == 0 and unrealized > 0:
                        should_exit = True
                else:
                    if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                        should_exit = True
                            
            if should_exit:
                stocks_to_sell.append(stock_name)
            else:
                # BBI Deviation Ladder profit take
                bbi_dev = current_close / asset['bbi'][local_idx] - 1
                is_bull = (current_close / asset['open'][local_idx] - 1) >= config.big_bull_threshold
                if bbi_dev >= config.bbi_dev_threshold and is_bull and h.position_weight > config.min_remaining_position * config.max_weight_per_stock:
                    capital *= (1 - config.fee_rate * (h.position_weight / 2))
                    h.position_weight /= 2
                        
        for stock_name in stocks_to_sell:
            capital *= (1 - config.fee_rate * holdings[stock_name].position_weight)
            cooldowns[stock_name] = day_idx + config.cooldown_days
            del holdings[stock_name]
            total_trades += 1
            
        # 3. Entries screening
        candidates = []
        if len(holdings) < config.max_holdings:
            for stock_name, asset in all_assets.items():
                if stock_name in holdings:
                    continue
                if stock_name in cooldowns and day_idx < cooldowns[stock_name]:
                    continue
                    
                idx_map = asset_date_map[stock_name]
                if today not in idx_map:
                    continue
                local_idx = idx_map[today]
                
                # Enforce WFO signals
                if asset['y_pred'][local_idx] != 1:
                    continue
                if asset['ma120_slope'][local_idx] <= 0:
                    continue
                    
                is_strong_trend = asset['ma120_slope'][local_idx] > 0.01
                high_confidence = asset['y_prob'][local_idx] > 0.65
                if not (is_strong_trend or high_confidence):
                    if asset['kdj_j'][local_idx] >= config.kdj_panic_threshold:
                        continue
                        
                if asset['close'][local_idx] < asset['bb_line'][local_idx]:
                    continue
                    
                candidates.append((stock_name, asset['y_prob'][local_idx]))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for stock_name, y_prob_i in candidates:
            if len(holdings) >= config.max_holdings:
                break
            rem_cap = 1.0 - sum(h.position_weight for h in holdings.values())
            if rem_cap <= 0.01:
                break
                
            raw_pos = min(config.max_position_size, max(config.min_position_size, (y_prob_i - 0.5) * config.position_scale_factor))
            target_weight = min(raw_pos * config.max_weight_per_stock, rem_cap)
            
            if target_weight < 0.01:
                continue
                
            local_idx = asset_date_map[stock_name][today]
            asset = all_assets[stock_name]
            
            capital *= (1 - config.fee_rate * target_weight)
            total_trades += 1
            holdings[stock_name] = strategy.Holding(
                name=stock_name,
                entry_price=asset['close'][local_idx],
                entry_stop_price=asset['low'][local_idx] - config.atr_multiplier * asset['atr'][local_idx],
                max_price=asset['close'][local_idx],
                position_weight=target_weight,
                entry_day_idx=day_idx,
            )
            
        portfolio_values.append(capital)
        portfolio_dates.append(today)
        
    total_return = capital / config.portfolio_capital - 1
    curve = pd.Series(portfolio_values)
    mdd = ((curve - curve.cummax()) / curve.cummax()).min() if len(curve) > 0 else 0.0
    
    # Sharpe ratio
    daily_returns = curve.pct_change().dropna()
    ann_factor = 252
    n_days = len(portfolio_values)
    ann_ret = (1 + total_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    ann_vol = daily_returns.std() * np.sqrt(ann_factor) if len(daily_returns) > 0 else 0
    sharpe = (ann_ret - config.risk_free_rate) / ann_vol if ann_vol != 0 else 0.0
    
    return total_return, mdd, sharpe, total_trades, portfolio_values, portfolio_dates

def main():
    print("=" * 75)
    print("★ 启动 ARMS 量化框架 - 路线B 模型升级与多算法实证对照 (2019-2023) ★")
    print("=" * 75)
    
    tickers = get_expanded_50_tickers()
    models = ['rf', 'lgbm', 'lstm', 'ensemble']
    
    results = {}
    
    for model_type in models:
        print(f"\n>>> 正在运行模型: {model_type.upper()} ...")
        all_assets = load_or_build_predictions(tickers, model_type)
        print(f"    数据集大小: {len(all_assets)} 标的。启动回测...")
        ret, mdd, sharpe, trades, vals, dates = custom_backtest(all_assets, config)
        print(f"    [RESULTS] {model_type.upper()} | Return: {ret*100:.2f}%, MDD: {mdd*100:.2f}%, Sharpe: {sharpe:.4f}, Trades: {trades}")
        
        results[model_type] = {
            'return': ret,
            'mdd': mdd,
            'sharpe': sharpe,
            'trades': trades,
            'vals': vals,
            'dates': dates
        }
        
    # Save results to CSV
    summary_data = []
    for model_type in models:
        res = results[model_type]
        summary_data.append({
            "Model": model_type.upper(),
            "Total_Return": f"{res['return']*100:.2f}%",
            "Max_Drawdown": f"{res['mdd']*100:.2f}%",
            "Sharpe_Ratio": f"{res['sharpe']:.4f}",
            "Total_Trades": res['trades']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("model_comparison_results.csv", index=False, encoding='utf-8-sig')
    print("\n[SAVE] 模型对比数据报表已保存: model_comparison_results.csv")
    
    # Plot comparative equity curves
    fig, ax = plt.subplots(figsize=(12.5, 6.5), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    dates_clean = pd.to_datetime(results['rf']['dates'])
    
    ax.plot(dates_clean, np.array(results['rf']['vals'])/1e6, color='#FF9800', linewidth=1.5, label='RF (Baseline Classifier)', alpha=0.7)
    ax.plot(dates_clean, np.array(results['lgbm']['vals'])/1e6, color='#4CAF50', linewidth=1.7, label='LightGBM (Gradient Boosting)', alpha=0.8)
    ax.plot(dates_clean, np.array(results['lstm']['vals'])/1e6, color='#00F2FE', linewidth=2.0, label='LSTM (Temporal Neural Network)', alpha=0.9)
    ax.plot(dates_clean, np.array(results['ensemble']['vals'])/1e6, color='#E22D7D', linewidth=2.3, label='Ensemble (RF + LGBM + LSTM)', alpha=0.95)
    
    ax.axhline(y=1.0, color='#8B949E', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('ARMS 量化系统底座算法升级对照实验累计净值曲线 (2019-2023)', fontsize=13, color='#F0F6FC', pad=18, weight='bold')
    ax.set_ylabel('资产权益净值 (百万 CNY)', color='#F0F6FC', fontsize=11)
    ax.set_xlabel('年份', color='#F0F6FC', fontsize=11)
    ax.tick_params(colors='#8B949E', labelsize=10)
    ax.grid(True, color='#21262D', linestyle='-', linewidth=0.7, alpha=0.5)
    
    leg = ax.legend(loc='upper left', facecolor='#161B22', edgecolor='#30363D', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#F0F6FC')
        
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/academic_model_comparison.png", facecolor='#0D1117', edgecolor='none', dpi=200)
    print("[SAVE] 对比图表已保存: plots/academic_model_comparison.png")
    
    print("\n" + "=" * 75)
    print("★ 模型升级实证对照实验全量完成！ ★")
    print("=" * 75)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
