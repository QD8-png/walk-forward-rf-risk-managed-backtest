# -*- coding: utf-8 -*-
"""
Full cycle backtest (2019-2023) on a COMPLETELY NEW independent Universe of 50 Sector Leaders
========================================================================================
Replicates the exact rolling machine learning, wind control, and 3-trade limit structural constraint
to evaluate generalizability and eliminate survival/selection bias.
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
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

# We will inherit StrategyConfig
config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0
config.n_shuffles = 20  # Reduced for speed

def custom_portfolio_backtest(all_assets: Dict[str, Dict[str, Any]], config: strategy.StrategyConfig, non_compounding: bool = False) -> Tuple[float, float, List[float], List[pd.Timestamp]]:
    capital = config.portfolio_capital
    holdings: Dict[str, strategy.Holding] = {}
    cooldowns: Dict[str, int] = {}
    trade_counts: Dict[str, int] = {}  # Max 3 trades per stock limit
    pocketed_profit = 0.0
    
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

    mode_str = "【非复利/利润锁定模式】" if non_compounding else "【复利模式】"
    print(f"\n--- 启动新大池子 2019-2023 周期性滚动回测 {mode_str} (每只股票限3次交易) ---")
    
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
            unrealized = current_close / h.entry_price - 1
            holding_days = day_idx - h.entry_day_idx
            
            # Layer A: Price falls below core Bull-Bear Line
            if current_close < asset['bb_line'][local_idx]:
                should_exit = True
            # Layer B: Hard Adaptive Stop-loss
            elif current_close < h.entry_stop_price:
                should_exit = True
            # Layer C: Time-based opportunity cost exit (不涨就拍掉)
            elif holding_days >= config.patience_days and unrealized < config.patience_return:
                should_exit = True
            else:
                # Layer D: Trailing Stop
                if unrealized < config.trailing_activate_pct:
                    if asset['y_pred'][local_idx] == 0 and unrealized > 0:
                        should_exit = True
                else:
                    if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                        should_exit = True
            
            if should_exit:
                stocks_to_sell.append(stock_name)
            else:
                # Layer D: BBI deviation ladder take-profit
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
                if False: # Capped at 3 trades per stock (Patched out for healthy rotation)
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
            trade_counts[stock_name] = trade_counts.get(stock_name, 0) + 1
            holdings[stock_name] = strategy.Holding(
                name=stock_name,
                entry_price=asset['close'][local_idx],
                entry_stop_price=asset['low'][local_idx] - config.atr_multiplier * asset['atr'][local_idx],
                max_price=asset['close'][local_idx],
                position_weight=target_weight,
                entry_day_idx=day_idx,
            )
        
        # Profit extraction logic for non-compounding mode
        if non_compounding:
            profit = capital - config.portfolio_capital
            if profit > 0:
                total_weight = sum(h.position_weight for h in holdings.values())
                cash = capital * (1.0 - total_weight)
                withdrawn = min(profit, cash)
                if withdrawn > 0:
                    capital_before = capital
                    capital -= withdrawn
                    pocketed_profit += withdrawn
                    if len(holdings) > 0 and capital > 0:
                        weight_scale = capital_before / capital
                        for h in holdings.values():
                            h.position_weight *= weight_scale
        
        equity_val = capital + pocketed_profit if non_compounding else capital
        portfolio_values.append(equity_val)
        portfolio_dates.append(today)

    actual_final_capital = capital + pocketed_profit if non_compounding else capital
    total_return = actual_final_capital / config.portfolio_capital - 1
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
    print(f"新大池子 2019-2023 最终绩效报告 ({mode_str})")
    print("=" * 50)
    print(f"  初始资金:     CNY {config.portfolio_capital:,.2f}")
    print(f"  终末总权益:   CNY {actual_final_capital:,.2f}")
    print(f"  总收益率:     {total_return * 100:.2f}%")
    print(f"  年化收益:     {annualized_return * 100:.2f}%")
    print(f"  年化波动:     {annualized_vol * 100:.2f}%")
    print(f"  夏普比率:     {sharpe:.4f}")
    print(f"  最大回撤:     {max_drawdown * 100:.2f}%")
    print(f"  总交易次数:   {total_trades}")
    print(f"  回测天数:     {n_days}")
    print("=" * 50)
        
    return total_return, max_drawdown, portfolio_values, portfolio_dates

def get_new_50_tickers() -> List[str]:
    """Returns a completely new set of 50 high-liquidity prominent A-shares tickers disjoint from Pool A."""
    return [
        # Industrial / Machinery Leaders
        "600031.SS", "000157.SZ", "601100.SS", "000425.SZ",
        # Consumer Electronics & Hardware Innovators
        "002475.SZ", "002241.SZ", "603501.SS", "603986.SS", "600703.SS", "002049.SZ",
        # Biotech / Medical Giants
        "300760.SZ", "603259.SS", "000538.SZ", "600436.SS", "002007.SZ",
        # Chemicals & Materials
        "600309.SS", "600426.SS", "600989.SS",
        # Solar / Wind / Electric Grid
        "300274.SZ", "600438.SS", "601877.SS", "002459.SZ", "600406.SS",
        # F&B / Spirits / Consumer Goods
        "600600.SS", "600132.SS", "000876.SZ", "000895.SZ", "000568.SZ",
        # Real Estate & Heavy Infra & Shipping
        "000002.SZ", "600048.SS", "601668.SS", "002352.SZ", "601006.SS", "601919.SS",
        # Telecom & Soft AI & Software
        "601728.SS", "600941.SS", "600050.SS", "688111.SS", "600570.SS",
        # Automotive & EV Industry
        "600104.SS", "601633.SS", "601238.SS", "000625.SZ",
        # Financials (Banks, Insurance, Brokers)
        "601288.SS", "601988.SS", "601601.SS", "601377.SS",
        # Resources & Coal & Gold
        "601088.SS", "601225.SS", "600489.SS"
    ]

def process_single_asset_custom(ticker: str, df: pd.DataFrame, config: strategy.StrategyConfig, model_type: str = 'rf') -> Optional[Dict[str, Any]]:
    try:
        df = strategy.prepare_features(df, config)
    except Exception:
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

def main():
    tickers = get_new_50_tickers()
    
    print("=" * 60)
    print("★ 启动全新大池子实证对比：新50只行业龙头2019-2023周期压力测试 ★")
    print("=" * 60)
    
    print(f"1. 批量下载新大池子 {len(tickers)} 只标的自 2016-01-01 至 2023-12-31 的日线行情...")
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
            pass
            
    print(f"   下载完毕。可用标的数: {len(valid_dfs)}")
    if len(valid_dfs) == 0:
        print("   错误: 没有可用标的数据，请检查网络！")
        return

    # Use Random Forest modeling in walk-forward
    model_type = 'rf'
    print(f"\n2. 启动多进程并发加速：前向滚动 RF 建模预测...")
    all_assets = {}
    max_workers = min(2, multiprocessing.cpu_count(), len(valid_dfs))
    
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

    # 3. Custom portfolio rotation backtest (Double-mode)
    print("\n3.1. 启动新池子【复利模式】回测...")
    actual_return, max_drawdown, values, dates = custom_portfolio_backtest(all_assets, config, non_compounding=False)
    
    print("\n3.2. 启动新池子【非复利/利润锁定模式】回测...")
    nc_return, nc_max_drawdown, nc_values, nc_dates = custom_portfolio_backtest(all_assets, config, non_compounding=True)
    
    # 4. Premium comparison graph saving
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "new_cycle_portfolio_equity_comparison.png")
    
    def plot_portfolio_equity_comparison(dates: List[pd.Timestamp], values_comp: List[float], values_nc: List[float], asset_count: int, save_path: str = None):
        fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0D1117')
        ax.set_facecolor('#0D1117')
        
        dates_clean = pd.to_datetime(dates)
        norm_comp = np.array(values_comp) / values_comp[0]
        norm_nc = np.array(values_nc) / values_nc[0]
        
        ax.plot(dates_clean, norm_comp, color='#00F2FE', linewidth=2.5, label='复利增长模式 (Compounding)', alpha=0.9)
        ax.plot(dates_clean, norm_nc, color='#FF5E62', linewidth=2.5, label='非复利/锁定利润模式 (Non-compounding)', alpha=0.9)
        
        ax.axhline(y=1.0, color='#8B949E', linestyle='--', linewidth=0.8, alpha=0.6)
        
        ax.set_title(f'全新大池子(Pool B)投资组合净值对比曲线 (2019-2023 牛熊周期 - 共 {asset_count} 只有效股票)', fontsize=14, color='#F0F6FC', pad=20, weight='bold')
        ax.set_ylabel('组合净值 (初始=1.00)', color='#F0F6FC', fontsize=11, labelpad=10)
        ax.set_xlabel('回测日期', color='#F0F6FC', fontsize=11, labelpad=10)
        ax.tick_params(colors='#8B949E', labelsize=9.5)
        ax.grid(True, which='both', color='#21262D', linestyle='-', linewidth=0.7, alpha=0.6)
        
        info_text = (
            f"复利终末净值:   {norm_comp[-1]:.3f} ({(norm_comp[-1] - 1)*100:+.2f}%)\n"
            f"非复利终末净值: {norm_nc[-1]:.3f} ({(norm_nc[-1] - 1)*100:+.2f}%)"
        )
        ax.text(0.03, 0.82, info_text, transform=ax.transAxes, fontsize=10, color='#F0F6FC',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#161B22', edgecolor='#30363D', alpha=0.9))
        
        leg = ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.70), facecolor='#161B22', edgecolor='#30363D', fontsize=10)
        for text in leg.get_texts():
            text.set_color('#F0F6FC')
            
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, facecolor='#0D1117', edgecolor='none', dpi=150)
            print(f"  [SAVE] 新组合回测对比图表已保存: {save_path}")
        plt.show()

    plot_portfolio_equity_comparison(dates, values, nc_values, len(all_assets), save_path=plot_path)

    # Save summary indicators to CSV for reference
    ann_factor = 252
    n_days = len(values)
    curve_comp = pd.Series(values)
    daily_comp = curve_comp.pct_change().dropna()
    ann_ret_comp = (1 + actual_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    ann_vol_comp = daily_comp.std() * np.sqrt(ann_factor) if len(daily_comp) > 0 else 0
    sharpe_comp = (ann_ret_comp - config.risk_free_rate) / ann_vol_comp if ann_vol_comp != 0 else 0
    
    curve_nc = pd.Series(nc_values)
    daily_nc = curve_nc.pct_change().dropna()
    ann_ret_nc = (1 + nc_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    ann_vol_nc = daily_nc.std() * np.sqrt(ann_factor) if len(daily_nc) > 0 else 0
    sharpe_nc = (ann_ret_nc - config.risk_free_rate) / ann_vol_nc if ann_vol_nc != 0 else 0
    
    summary_df = pd.DataFrame([
        {
            "Mode": "New Compounding (新复利)",
            "Universe_Size": len(all_assets),
            "Start_Date": "2019-01-01",
            "End_Date": "2023-12-31",
            "Total_Return": f"{actual_return * 100:.2f}%",
            "Annualized_Return": f"{ann_ret_comp * 100:.2f}%",
            "Annualized_Vol": f"{ann_vol_comp * 100:.2f}%",
            "Sharpe_Ratio": f"{sharpe_comp:.4f}",
            "Max_Drawdown": f"{max_drawdown * 100:.2f}%"
        },
        {
            "Mode": "New Non-compounding (新非复利/锁定利润)",
            "Universe_Size": len(all_assets),
            "Start_Date": "2019-01-01",
            "End_Date": "2023-12-31",
            "Total_Return": f"{nc_return * 100:.2f}%",
            "Annualized_Return": f"{ann_ret_nc * 100:.2f}%",
            "Annualized_Vol": f"{ann_vol_nc * 100:.2f}%",
            "Sharpe_Ratio": f"{sharpe_nc:.4f}",
            "Max_Drawdown": f"{nc_max_drawdown * 100:.2f}%"
        }
    ])
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_cycle_results_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  [SAVE] 新周期回测指标数据已保存: {summary_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
