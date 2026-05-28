# -*- coding: utf-8 -*-
"""
Academic Empirical Pipeline for the ARMS Framework & TOCE Mechanism
===================================================================
Performs:
1. Cache-supported Walk-Forward modeling on 49 sector leaders (2019-2023).
2. Five-stage Ablation Study on risk management layers.
3. Parametric Sensitivity Analysis grid-search on TOCE parameters.
4. Buy & Hold equal-weighted benchmark comparison.
5. Publication-grade dark-themed data visualizations.
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

# Path alignment to import strategy.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

# Initialize base strategy configuration
config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0
config.n_shuffles = 20  # Fast verification

CACHE_FILE = "all_assets_2019_2023.pkl"

def process_single_asset_custom(ticker: str, df: pd.DataFrame, config: strategy.StrategyConfig, model_type: str = 'rf') -> Optional[Dict[str, Any]]:
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
    """Returns the same 50 stock tickers used in expanded test cycle."""
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

def load_or_build_predictions(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Loads prediction results from local cache or executes the full WFO process and caches it."""
    if os.path.exists(CACHE_FILE):
        print(f"\n[CACHE] 找到缓存文件 {CACHE_FILE}，正在加载...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
            
    print(f"\n[PREDICT] 未找到缓存，启动历史数据下载 (2016-01-01 至 2023-12-31)...")
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

    print(f"   足额有效标的数: {len(valid_dfs)}。启动去偏 WFO 随机森林计算...")
    all_assets = {}
    max_workers = min(2, multiprocessing.cpu_count(), len(valid_dfs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_asset_custom, t, df, config, 'rf'): t for t, df in valid_dfs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                all_assets[res['name']] = res
            if i % 10 == 0 or i == len(valid_dfs):
                print(f"   建模进度: {i} / {len(valid_dfs)}")
                
    print(f"   建模完毕，可用资产池大小: {len(all_assets)}。正在保存至缓存...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(all_assets, f)
    print(f"   [CACHE] 缓存成功: {CACHE_FILE}")
    return all_assets

def custom_ablation_backtest(
    all_assets: Dict[str, Dict[str, Any]], 
    config: strategy.StrategyConfig, 
    enable_atr: bool = True, 
    enable_bbl: bool = True, 
    enable_toce: bool = True, 
    enable_trailing: bool = True, 
    enable_bbi_tp: bool = True, 
    non_compounding: bool = False
) -> Tuple[float, float, float, List[float], List[pd.Timestamp]]:
    """
    Highly customized backtester that runs dynamic risk/exit layers to compile an Ablation Study.
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
            if enable_bbl and current_close < asset['bb_line'][local_idx]:
                should_exit = True
            # Layer B: ATR Volatility Stop Loss
            elif enable_atr and current_close < h.entry_stop_price:
                should_exit = True
            # Layer C: TOCE (Time-based Opportunity Cost Exit)
            elif enable_toce and holding_days >= config.patience_days and unrealized < config.patience_return:
                should_exit = True
            else:
                # Layer D: Trailing Profit Lock
                if enable_trailing:
                    if unrealized < config.trailing_activate_pct:
                        if asset['y_pred'][local_idx] == 0 and unrealized > 0:
                            should_exit = True
                    else:
                        if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                            should_exit = True
                            
            # Baseline Pure ML Exit (no risk management enabled at all)
            if not should_exit and not enable_bbl and not enable_atr and not enable_toce and not enable_trailing:
                if asset['y_pred'][local_idx] == 0:
                    should_exit = True
                    
            if should_exit:
                stocks_to_sell.append(stock_name)
            else:
                # BBI Deviation Ladder profit take
                if enable_bbi_tp:
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
    
    return total_return, mdd, sharpe, portfolio_values, portfolio_dates

def compute_buy_and_hold(all_assets: Dict[str, Dict[str, Any]], initial_capital: float) -> Tuple[float, float, List[float], List[pd.Timestamp]]:
    """Computes equal-weighted buy-and-hold index returns across all active assets."""
    all_dates = set()
    for asset in all_assets.values():
        all_dates.update(asset['dates'])
    all_dates = sorted(list(all_dates))
    
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2023-12-31')
    active_dates = [d for d in all_dates if start_date <= pd.Timestamp(d) <= end_date]
    
    asset_date_map = {name: {d: i for i, d in enumerate(asset['dates'])} for name, asset in all_assets.items()}
    
    capital = initial_capital
    portfolio_values = []
    portfolio_dates = []
    
    for today in active_dates:
        daily_portfolio_return = 0.0
        active_count = 0
        
        for name, asset in all_assets.items():
            idx_map = asset_date_map[name]
            if today not in idx_map:
                continue
            local_idx = idx_map[today]
            current_close = asset['close'][local_idx]
            prev_close = asset['close'][local_idx - 1] if local_idx > 0 else current_close
            
            if prev_close > 0:
                daily_return = current_close / prev_close - 1
                daily_portfolio_return += daily_return
                active_count += 1
                
        if active_count > 0:
            avg_return = daily_portfolio_return / active_count
        else:
            avg_return = 0.0
            
        capital = capital * (1 + avg_return)
        portfolio_values.append(capital)
        portfolio_dates.append(today)
        
    total_return = capital / initial_capital - 1
    curve = pd.Series(portfolio_values)
    mdd = ((curve - curve.cummax()) / curve.cummax()).min() if len(curve) > 0 else 0.0
    return total_return, mdd, portfolio_values, portfolio_dates

def main():
    print("=" * 70)
    print("★ 启动 ARMS 量化框架学术实证与 TOCE 机制验证流水线 ★")
    print("=" * 70)
    
    tickers = get_expanded_50_tickers()
    all_assets = load_or_build_predictions(tickers)
    
    # Make sure plot folder exists
    os.makedirs("plots", exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. BUY & HOLD BENCHMARK
    # --------------------------------------------------------------------------
    print("\n--- 正在计算等权买入持有(Buy & Hold)基准... ---")
    bh_ret, bh_mdd, bh_vals, bh_dates = compute_buy_and_hold(all_assets, config.portfolio_capital)
    print(f"   Buy & Hold 收益率: {bh_ret*100:.2f}%, 最大回撤: {bh_mdd*100:.2f}%")
    
    # --------------------------------------------------------------------------
    # 2. ABLATION STUDY
    # --------------------------------------------------------------------------
    print("\n--- 启动五阶段风控层级消融对照实验 (2019-2023) ---")
    
    # System 1: Pure ML
    print("   [Ablation 1] 运行纯机器学习信号 (System 1)...")
    s1_ret, s1_mdd, s1_sharpe, s1_vals, s1_dates = custom_ablation_backtest(
        all_assets, config, enable_atr=False, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False
    )
    
    # System 2: ML + ATR
    print("   [Ablation 2] 运行 ML + ATR 波动止损 (System 2)...")
    s2_ret, s2_mdd, s2_sharpe, s2_vals, _ = custom_ablation_backtest(
        all_assets, config, enable_atr=True, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False
    )
    
    # System 3: ML + ATR + BBL
    print("   [Ablation 3] 运行 ML + ATR + BBL 多空线拦截 (System 3)...")
    s3_ret, s3_mdd, s3_sharpe, s3_vals, _ = custom_ablation_backtest(
        all_assets, config, enable_atr=True, enable_bbl=True, enable_toce=False, enable_trailing=False, enable_bbi_tp=False
    )
    
    # System 4: ML + ATR + BBL + TOCE (Proposed Framework Core)
    print("   [Ablation 4] 运行 ML + ATR + BBL + TOCE 耐心时间退出 (System 4)...")
    s4_ret, s4_mdd, s4_sharpe, s4_vals, _ = custom_ablation_backtest(
        all_assets, config, enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False
    )
    
    # System 5: Full ARMS Strategy (With Trailing Stop and BBI TP)
    print("   [Ablation 5] 运行完整自适应风控策略 (Full ARMS System 5)...")
    s5_ret, s5_mdd, s5_sharpe, s5_vals, _ = custom_ablation_backtest(
        all_assets, config, enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True
    )
    
    # Save Ablation Results
    ablation_df = pd.DataFrame([
        {"System": "System 1: Pure ML Baseline", "Total_Return": f"{s1_ret*100:.2f}%", "Max_Drawdown": f"{s1_mdd*100:.2f}%", "Sharpe_Ratio": f"{s1_sharpe:.4f}"},
        {"System": "System 2: ML + ATR Stop", "Total_Return": f"{s2_ret*100:.2f}%", "Max_Drawdown": f"{s2_mdd*100:.2f}%", "Sharpe_Ratio": f"{s2_sharpe:.4f}"},
        {"System": "System 3: ML + ATR + BBL", "Total_Return": f"{s3_ret*100:.2f}%", "Max_Drawdown": f"{s3_mdd*100:.2f}%", "Sharpe_Ratio": f"{s3_sharpe:.4f}"},
        {"System": "System 4: ML + ATR + BBL + TOCE", "Total_Return": f"{s4_ret*100:.2f}%", "Max_Drawdown": f"{s4_mdd*100:.2f}%", "Sharpe_Ratio": f"{s4_sharpe:.4f}"},
        {"System": "System 5: Full ARMS Framework", "Total_Return": f"{s5_ret*100:.2f}%", "Max_Drawdown": f"{s5_mdd*100:.2f}%", "Sharpe_Ratio": f"{s5_sharpe:.4f}"},
        {"System": "Benchmark: Buy & Hold (B&H)", "Total_Return": f"{bh_ret*100:.2f}%", "Max_Drawdown": f"{bh_mdd*100:.2f}%", "Sharpe_Ratio": "N/A"}
    ])
    ablation_df.to_csv("ablation_study_results.csv", index=False, encoding='utf-8-sig')
    print("\n   [SAVE] 消融实验表格已保存: ablation_study_results.csv")
    
    # Plot Ablation Curves
    fig, ax = plt.subplots(figsize=(12.5, 6.5), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    dates_clean = pd.to_datetime(bh_dates)
    ax.plot(dates_clean, np.array(s1_vals)/1e6, color='#8B949E', linestyle=':', label='System 1: Pure ML Baseline', alpha=0.6)
    ax.plot(dates_clean, np.array(s2_vals)/1e6, color='#FFC107', linestyle='--', label='System 2: ML + ATR Stop', alpha=0.7)
    ax.plot(dates_clean, np.array(s3_vals)/1e6, color='#4CAF50', linestyle='-.', label='System 3: ML + ATR + BBL', alpha=0.8)
    ax.plot(dates_clean, np.array(s4_vals)/1e6, color='#00F2FE', linewidth=2.0, label='System 4: ML + ATR + BBL + TOCE (Proposed)', alpha=0.9)
    ax.plot(dates_clean, np.array(s5_vals)/1e6, color='#E22D7D', linewidth=2.2, label='System 5: Full ARMS Framework', alpha=0.9)
    ax.plot(dates_clean, np.array(bh_vals)/1e6, color='#F0F6FC', linewidth=1.5, label='Benchmark: Buy & Hold (B&H)', alpha=0.5)
    
    ax.axhline(y=1.0, color='#8B949E', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('ARMS 量化系统规则层级消融实验累计净值曲线 (2019-2023)', fontsize=13, color='#F0F6FC', pad=18, weight='bold')
    ax.set_ylabel('资产权益净值 (百万 CNY)', color='#F0F6FC', fontsize=11)
    ax.set_xlabel('年份', color='#F0F6FC', fontsize=11)
    ax.tick_params(colors='#8B949E', labelsize=10)
    ax.grid(True, color='#21262D', linestyle='-', linewidth=0.7, alpha=0.5)
    
    leg = ax.legend(loc='upper left', facecolor='#161B22', edgecolor='#30363D', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#F0F6FC')
        
    plt.tight_layout()
    plt.savefig("plots/academic_ablation_equity_comparison.png", facecolor='#0D1117', edgecolor='none', dpi=200)
    print("   [SAVE] 消融对比图表已保存: plots/academic_ablation_equity_comparison.png")
    
    # --------------------------------------------------------------------------
    # 3. TOCE PARAMETER SENSITIVITY GRID SEARCH
    # --------------------------------------------------------------------------
    print("\n--- 启动 TOCE 耐心天数及收益率敏感性网格搜索 ---")
    patience_days_list = [2, 3, 5, 7, 10, 15]
    patience_return_list = [0.0, 0.002, 0.005, 0.010, 0.015]
    
    sensitivity_results = []
    
    for p_days in patience_days_list:
        for p_ret in patience_return_list:
            sens_config = copy.deepcopy(config)
            sens_config.patience_days = p_days
            sens_config.patience_return = p_ret
            
            # Run s4 config (ML+ATR+BBL+TOCE) to measure the raw impact of TOCE parameter shifting
            ret_i, mdd_i, sharpe_i, _, _ = custom_ablation_backtest(
                all_assets, sens_config, enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False
            )
            print(f"   [Grid Search] patience_days={p_days:2d}, patience_return={p_ret:.3f} | Return: {ret_i*100:6.2f}%, MDD: {mdd_i*100:6.2f}%")
            
            sensitivity_results.append({
                "patience_days": p_days,
                "patience_return": p_ret,
                "Total_Return": ret_i,
                "Max_Drawdown": mdd_i,
                "Sharpe_Ratio": sharpe_i
            })
            
    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_csv("sensitivity_patience_results.csv", index=False, encoding='utf-8-sig')
    print("\n   [SAVE] 耐心敏感度网格数据已保存: sensitivity_patience_results.csv")
    
    # Plot Parameter Sensitivity Heatmap
    # Reshape total return to grid
    grid_return = sens_df.pivot(index="patience_days", columns="patience_return", values="Total_Return").values * 100
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    im = ax.imshow(grid_return, cmap='magma', origin='lower')
    
    # Show tickers and labels
    ax.set_xticks(np.arange(len(patience_return_list)))
    ax.set_yticks(np.arange(len(patience_days_list)))
    ax.set_xticklabels([f"{r*100:.1f}%" for r in patience_return_list], color='#F0F6FC')
    ax.set_yticklabels(patience_days_list, color='#F0F6FC')
    
    ax.set_xlabel('TOCE 滞涨阈值 (patience_return)', color='#F0F6FC', fontsize=11, labelpad=10)
    ax.set_ylabel('TOCE 耐心周期 (patience_days)', color='#F0F6FC', fontsize=11, labelpad=10)
    ax.set_title('TOCE 机制参数网格敏感性分析 (2019-2023 牛熊周期总收益率 %)', fontsize=12, color='#F0F6FC', pad=15, weight='bold')
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(patience_days_list)):
        for j in range(len(patience_return_list)):
            text = ax.text(j, i, f"{grid_return[i, j]:.2f}%",
                           ha="center", va="center", color="#F0F6FC" if grid_return[i, j] < 20 else "#000000",
                           fontsize=10, weight='bold')
            
    cbar = fig.colorbar(im, ax=ax, pad=0.08)
    cbar.ax.yaxis.set_tick_params(color='#8B949E')
    cbar.ax.set_ylabel('组合总收益率 (%)', color='#F0F6FC', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=9.5, colors='#8B949E')
    
    plt.tight_layout()
    plt.savefig("plots/academic_sensitivity_heatmap.png", facecolor='#0D1117', edgecolor='none', dpi=200)
    print("   [SAVE] 敏感性热力图已保存: plots/academic_sensitivity_heatmap.png")
    
    # --------------------------------------------------------------------------
    # 4. COOLDOWN DAYS SENSITIVITY STUDY
    # --------------------------------------------------------------------------
    print("\n--- 启动冷却锁定期(cooldown_days)敏感度验证 (patience_days=5) ---")
    cooldown_list = [5, 10, 15, 30, 60, 120]
    cooldown_results = []
    
    for cd in cooldown_list:
        cd_config = copy.deepcopy(config)
        cd_config.cooldown_days = cd
        ret_i, mdd_i, sharpe_i, _, _ = custom_ablation_backtest(
            all_assets, cd_config, enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False
        )
        print(f"   [Cooldown Search] cooldown_days={cd:3d} | Return: {ret_i*100:6.2f}%, MDD: {mdd_i*100:6.2f}%, Sharpe: {sharpe_i:.4f}")
        cooldown_results.append({
            "cooldown_days": cd,
            "Total_Return": ret_i,
            "Max_Drawdown": mdd_i,
            "Sharpe_Ratio": sharpe_i
        })
        
    cd_df = pd.DataFrame(cooldown_results)
    cd_df.to_csv("sensitivity_cooldown_results.csv", index=False, encoding='utf-8-sig')
    print("   [SAVE] 冷却敏感度数据已保存: sensitivity_cooldown_results.csv")
    
    # Print clear wrap-up
    print("\n" + "=" * 70)
    print("★ 实证流水线全量执行完毕！全部数据表格与出图均已高标准生成 ★")
    print("=" * 70)
    print(f"1. 消融实验结果: ablation_study_results.csv")
    print(f"2. 敏感性网格数据: sensitivity_patience_results.csv")
    print(f"3. 冷却敏感度数据: sensitivity_cooldown_results.csv")
    print(f"4. 累计净值对比图: plots/academic_ablation_equity_comparison.png")
    print(f"5. 参数敏感度热力图: plots/academic_sensitivity_heatmap.png")
    print("=" * 70)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
