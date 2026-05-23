# -*- coding: utf-8 -*-
"""
Full cycle backtest (2019-2023) on an Expanded Stock Universe of 50 sector leaders
==============================================================================
Uses the exact machine learning rolling forecasting, technical indicators, and multi-stage 
risk controls from our git_review/strategy.py codebase.
"""

import os
import sys
import copy
import multiprocessing
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up matplotlib dark style
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

# Resolve path to import our strategy code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "git_review")))
import strategy

# We will inherit StrategyConfig but adjust n_shuffles for speed if needed
config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0
config.n_shuffles = 20  # Reduced for fast validation on 5-year period

def custom_portfolio_backtest(all_assets: Dict[str, Dict[str, Any]], config: strategy.StrategyConfig) -> Tuple[float, float, List[float], List[pd.Timestamp]]:
    """
    Custom portfolio backtest running exactly from 2019-01-01 to 2023-12-31.
    Uses the exact same transaction cost, rotation, entry gate, and exits as strategy.py.
    """
    capital = config.portfolio_capital
    holdings: Dict[str, strategy.Holding] = {}
    cooldowns: Dict[str, int] = {}
    
    # Consolidate all timeline dates
    all_dates = set()
    for asset in all_assets.values():
        all_dates.update(asset['dates'])
    all_dates = sorted(list(all_dates))
    
    asset_date_map: Dict[str, Dict] = {}
    for name, asset in all_assets.items():
        asset_date_map[name] = {d: i for i, d in enumerate(asset['dates'])}

    portfolio_values = []
    portfolio_dates = []
    total_trades = 0
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2023-12-31')

    print("\n--- 启动 2019-2023 周期性滚动回测 ---")
    print(f"回测区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"初始资产: CNY {capital:,.2f}")
    
    for day_idx, today in enumerate(all_dates):
        today_ts = pd.Timestamp(today)
        if today_ts < start_date or today_ts > end_date:
            continue
            
        # 1. Update mark-to-market daily returns and peak prices
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
        
        # Parallel-sum weighted update of capital pool (fixes compounding return bug)
        capital = capital * (1 + daily_portfolio_return)
        
        # 2. Risk checks and Exits execution
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
            
            # Layer A: Price falls below core Bull-Bear Line
            if current_close < asset['bb_line'][local_idx]:
                should_exit = True
            # Layer B: Hard Adaptive Stop-loss (2.0 * ATR below low of entry day)
            elif current_close < h.entry_stop_price:
                should_exit = True
            else:
                unrealized = current_close / h.entry_price - 1
                # Layer C: Trailing Stop profit lock or minor take-profit exit
                if unrealized < config.trailing_activate_pct:
                    # ML turns bearish with small/no profit: Exit (catch small fish)
                    if asset['y_pred'][local_idx] == 0 and unrealized > 0:
                        should_exit = True
                else:
                    # Trailing Stop triggered: Drop 5% from historical peak
                    if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                        should_exit = True
            
            if should_exit:
                stocks_to_sell.append(stock_name)
            else:
                # Layer D: BBI deviation ladder take-profit (halve position size)
                bbi_dev = current_close / asset['bbi'][local_idx] - 1
                is_bull = (current_close / asset['open'][local_idx] - 1) >= config.big_bull_threshold
                if bbi_dev >= config.bbi_dev_threshold and is_bull and h.position_weight > config.min_remaining_position * config.max_weight_per_stock:
                    capital *= (1 - config.fee_rate * (h.position_weight / 2))
                    h.position_weight /= 2
        
        # Liquidate exited stocks and trigger cool-down locks
        for stock_name in stocks_to_sell:
            capital *= (1 - config.fee_rate * holdings[stock_name].position_weight)
            cooldowns[stock_name] = day_idx + config.cooldown_days
            del holdings[stock_name]
            total_trades += 1
        
        # 3. Entry Signals and Allocation
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
                
                # Five independent layers of screening
                if asset['y_pred'][local_idx] != 1:
                    continue
                if asset['ma120_slope'][local_idx] <= 0:
                    continue
                
                # Relax KDJ_J panic oversold filter if trend is strong or model confidence is high
                is_strong_trend = asset['ma120_slope'][local_idx] > 0.01
                high_confidence = asset['y_prob'][local_idx] > 0.65
                if not (is_strong_trend or high_confidence):
                    if asset['kdj_j'][local_idx] >= config.kdj_panic_threshold:
                        continue
                
                if asset['close'][local_idx] < asset['bb_line'][local_idx]:
                    continue
                
                candidates.append((stock_name, asset['y_prob'][local_idx]))
        
        # Priority order by model prediction probability (confidence)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate available capital
        for stock_name, y_prob_i in candidates:
            if len(holdings) >= config.max_holdings:
                break
            
            rem_cap = 1.0 - sum(h.position_weight for h in holdings.values())
            if rem_cap <= 0.01:
                break
            
            # Dynamic leverage scaling
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
            )
        
        portfolio_values.append(capital)
        portfolio_dates.append(today)

    total_return = capital / config.portfolio_capital - 1
    ann_factor = 252
    n_days = len(portfolio_values)
    curve = pd.Series(portfolio_values)
    
    if len(curve) > 0:
        daily_returns = curve.pct_change().dropna()
        annualized_return = (1 + total_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
        annualized_vol = daily_returns.std() * np.sqrt(ann_factor) if len(daily_returns) > 0 else 0
        sharpe = (annualized_return - config.risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
        max_drawdown = ((curve - curve.cummax()) / curve.cummax()).min() if n_days > 0 else 0
    else:
        annualized_return, annualized_vol, sharpe, max_drawdown = 0, 0, 0, 0

    print("\n" + "=" * 50)
    print("2019-2023 跨越牛熊周期 - 50只行业龙头海选轮动 - 绩效报告")
    print("=" * 50)
    print(f"  初始资金:     CNY {config.portfolio_capital:,.2f}")
    print(f"  终末资金:     CNY {capital:,.2f}")
    print(f"  总收益率:     {total_return * 100:.2f}%")
    print(f"  年化收益:     {annualized_return * 100:.2f}%")
    print(f"  年化波动:     {annualized_vol * 100:.2f}%")
    print(f"  夏普比率:     {sharpe:.4f}")
    print(f"  最大回撤:     {max_drawdown * 100:.2f}%")
    print(f"  总交易次数:   {total_trades}")
    print(f"  回测天数:     {n_days}")
    print("=" * 50)
        
    return total_return, max_drawdown, portfolio_values, portfolio_dates

def get_expanded_50_tickers() -> List[str]:
    """Returns a highly diversified A-shares expanded universe of 50 sector leaders and ETFs."""
    return [
        # ETFs
        "510300.SS", "510500.SS", "510050.SS", "159915.SZ", # CSI 300, CSI 500, SSE 50, ChiNext
        "512000.SS", "512480.SS", "512170.SS", "512660.SS", # Brokerage, Semiconductor, Healthcare, Defense
        "515030.SS", "512690.SS", "512880.SS", # Bank removed because of delisted error, NEV, Liquor, Infrastructure
        
        # Consumption leaders
        "600519.SS", "000858.SZ", "600887.SS", "002714.SZ", 
        "601933.SS", "002508.SZ", "603288.SS", "600009.SS",
        
        # Energy & Materials
        "300750.SZ", "002594.SZ", "601012.SS", "002460.SZ",
        "601899.SS", "600019.SS", "603993.SS", "600547.SS",
        
        # Technology / Hardware / AI
        "000977.SZ", "603019.SS", "002230.SZ", "002415.SZ",
        "600584.SS", "000063.SZ", "600745.SS", "300059.SZ",
        "601138.SS", "002027.SZ",
        
        # Financials
        "600036.SS", "601318.SS", "600030.SS", "601398.SS",
        "601688.SS", "000001.SZ",
        
        # Utilities, Oil, Industry
        "600900.SS", "601857.SS", "600028.SS", "600150.SS",
        "600276.SS", "300015.SZ"
    ]

def process_single_asset_custom(ticker: str, df: pd.DataFrame, config: strategy.StrategyConfig, model_type: str = 'rf') -> Optional[Dict[str, Any]]:
    """Custom version of process_single_asset that bypasses the 2024 date limit."""
    try:
        df = strategy.prepare_features(df, config)
    except Exception as e:
        return None

    if len(df) < config.min_data_length:
        return None

    # Adapt date check to our 2019-2023 range
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

def main():
    tickers = get_expanded_50_tickers()
    
    print("=" * 60)
    print("★ 启动学术级实证扩展：2019-2023 牛熊完整周期大样本回测 ★")
    print("=" * 60)
    
    # We download from 2016-01-01 to give walk-forward modeling a 3-year warm-up
    # so out-of-sample backtesting starts exactly on 2019-01-01.
    print(f"1. 批量下载及构建 {len(tickers)} 只标的自 2016-01-01 至 2023-12-31 的技术特征...")
    data = yf.download(tickers, start="2016-01-01", end="2023-12-31", group_by='ticker', threads=True, progress=False)
    
    valid_dfs = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].copy()
            else:
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    continue
            df = df.dropna(how='all')
            if len(df) > config.min_data_length:
                df = df.reset_index()
                valid_dfs[t] = df
        except Exception as e:
            print(f"   标的 {t} 报错: {e}")
            
    print(f"   下载完毕。包含足额历史(>600天)的标的数: {len(valid_dfs)}")
    if len(valid_dfs) == 0:
        print("   错误: 没有可用标的数据，请检查网络！")
        return

    # Use Random Forest modeling in walk-forward
    model_type = 'rf'
    print(f"\n2. 启动多进程并发加速：前向滚动 RF 建模预测...")
    all_assets = {}
    max_workers = min(multiprocessing.cpu_count(), len(valid_dfs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_asset_custom, t, df, config, model_type): t for t, df in valid_dfs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                all_assets[res['name']] = res
            if i % 10 == 0 or i == len(valid_dfs):
                print(f"   已完成模组计算: {i} / {len(valid_dfs)}")
                
    print(f"\n   训练全部结束。共有 {len(all_assets)} 只标的具备完整信号，进入海选轮动池。")
    if len(all_assets) == 0:
        print("   没有符合预测时间线标准的标的，回测结束。")
        return

    # 3. Custom portfolio rotation backtest
    actual_return, max_drawdown, values, dates = custom_portfolio_backtest(all_assets, config)
    
    # 4. Premium graph saving
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "cycle_portfolio_equity.png")
    strategy.plot_portfolio_equity(dates, values, len(all_assets), save_path=plot_path)

    # Save summary indicators to CSV for reference
    ann_factor = 252
    n_days = len(values)
    curve = pd.Series(values)
    daily_returns = curve.pct_change().dropna()
    annualized_return = (1 + actual_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    annualized_vol = daily_returns.std() * np.sqrt(ann_factor) if len(daily_returns) > 0 else 0
    sharpe = (annualized_return - config.risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    
    summary_df = pd.DataFrame([{
        "Universe_Size": len(all_assets),
        "Start_Date": "2019-01-01",
        "End_Date": "2023-12-31",
        "Total_Return": f"{actual_return * 100:.2f}%",
        "Annualized_Return": f"{annualized_return * 100:.2f}%",
        "Annualized_Vol": f"{annualized_vol * 100:.2f}%",
        "Sharpe_Ratio": f"{sharpe:.4f}",
        "Max_Drawdown": f"{max_drawdown * 100:.2f}%"
    }])
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cycle_results_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  [SAVE] 周期回测指标数据已保存: {summary_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
