# -*- coding: utf-8 -*-
"""
Single Ticker Walk-Forward Random Forest Backtest Runner
=========================================================
Fetches historical market data online, builds advanced quant features,
executes purged walk-forward model training, and runs the multi-stage
risk managed backtesting engine on a single asset.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Setup path to import our base strategy
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

def fetch_and_test(ticker: str, start_date: str = "2020-01-01"):
    print("=" * 60)
    print(f"★ 正在抓取个股数据并执行滚动预测与多层风控实测 ★")
    print(f"  个股代码: {ticker}")
    print(f"  数据起点: {start_date}")
    print("=" * 60)
    
    # 1. Fetch data
    print(f"1. 正在通过 yfinance 下载 {ticker} 的历史日线行情...")
    df_raw = yf.download(ticker, start=start_date, progress=False)
    if df_raw.empty:
        print(f"   错误: 无法下载 {ticker} 的行情数据，请检查网络或股票代码是否正确。")
        return
        
    print(f"   下载成功！共获取到 {len(df_raw)} 条日线交易数据。")
    
    # Flatten MultiIndex columns if necessary
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.reset_index()
    
    # 2. Build Technical Features
    print(f"\n2. 正在构建 16 维动量、趋势偏离、摆动与波动率因子特征工程...")
    config = strategy.StrategyConfig()
    try:
        df_feat = strategy.prepare_features(df_raw, config)
    except Exception as e:
        print(f"   特征构建失败: {e}")
        return
        
    print(f"   有效数据过滤完毕。用于前向学习与回测的样本天数: {len(df_feat)}")
    if len(df_feat) < config.min_data_length:
        print(f"   警告: 数据长度为 {len(df_feat)}，低于推荐的最低数据量 {config.min_data_length}。将尝试继续运行。")
    
    # 3. Walk-Forward Rolling Predictive Engine
    print(f"\n3. 启动无泄漏滚动 WFO 决策树集成引擎进行前向学习预测...")
    predictions, probabilities, first_valid = strategy.walk_forward_predict(df_feat, config, model_type='rf')
    
    sliced = df_feat.iloc[first_valid:].reset_index(drop=True)
    y_pred_sliced = predictions[first_valid:]
    y_prob_sliced = probabilities[first_valid:]
    
    print(f"   样本外预测生成完毕。滚动检验区间起点: {sliced['日期'].min().strftime('%Y-%m-%d')}，终点: {sliced['日期'].max().strftime('%Y-%m-%d')}")
    
    # 4. Single Stock Backtest Engine
    # Run the backtest using a simplified single-asset capital allocation loop
    print(f"\n4. 启动单只股票混合风控与智能仓位量化回测...")
    
    capital = config.portfolio_capital
    initial_cap = capital
    position = 0
    position_weight = 0.0
    entry_price = 0.0
    entry_stop_price = 0.0
    max_price = 0.0
    cooldown_until = 0
    
    capital_history = []
    dates_history = []
    trades = 0
    
    for i in range(len(sliced)):
        today_date = sliced['日期'].iloc[i]
        current_close = sliced['收盘'].iloc[i]
        current_open = sliced['开盘'].iloc[i]
        current_low = sliced['低'].iloc[i]
        current_bb = sliced['多空线'].iloc[i]
        current_bbi = sliced['BBI'].iloc[i]
        current_kdj_j = sliced['KDJ_J'].iloc[i]
        current_slope = sliced['MA120_slope'].iloc[i]
        current_atr = sliced['ATR'].iloc[i]
        
        y_p = y_pred_sliced[i]
        y_prob_i = y_prob_sliced[i]
        
        # Daily Mark-to-Market Profit/Loss
        if position == 1 and i > 0:
            prev_close = sliced['收盘'].iloc[i - 1]
            daily_ret = current_close / prev_close - 1
            capital = capital * (1 + daily_ret * position_weight)
            if current_close > max_price:
                max_price = current_close
                
        capital_history.append(capital)
        dates_history.append(today_date)
        
        # Decision logic
        if position == 0:
            # Entry Conditions check (ML prediction + 120-day trend + KDJ Panic filter + Price position)
            should_buy = (y_p == 1) and (current_slope > 0) and (current_close >= current_bb)
            
            # KDJ panic oversold filter (relaxed under strong momentum or high confidence)
            is_strong = current_slope > 0.01
            high_conf = y_prob_i > 0.65
            if not (is_strong or high_conf):
                if current_kdj_j >= config.kdj_panic_threshold:
                    should_buy = False
                    
            if should_buy and i >= cooldown_until:
                position = 1
                entry_price = current_close
                entry_stop_price = current_low - config.atr_multiplier * current_atr
                max_price = current_close
                
                # Dynamic weight leverage sizing based on prediction confidence
                raw_pos = min(config.max_position_size, max(config.min_position_size, (y_prob_i - 0.5) * config.position_scale_factor))
                position_weight = raw_pos
                
                # Apply transaction fee
                capital *= (1 - config.fee_rate * position_weight)
                trades += 1
                
        else:
            # Exit check
            should_exit = False
            
            # Layer A: Price crosses below main bull-bear line
            if current_close < current_bb:
                should_exit = True
            # Layer B: Hard adaptive Stop Loss
            elif current_close < entry_stop_price:
                should_exit = True
            else:
                unrealized = current_close / entry_price - 1
                # Layer C: Trailing stop profit locks or minor exit
                if unrealized < config.trailing_activate_pct:
                    # ML turns bearish with small/no profit
                    if y_p == 0 and unrealized > 0:
                        should_exit = True
                else:
                    # Trailing Stop triggered: Drop 5% from peak
                    if max_price > entry_price and current_close < max_price * (1 - config.trailing_stop_pct):
                        should_exit = True
                        
            if should_exit:
                position = 0
                capital *= (1 - config.fee_rate * position_weight)
                position_weight = 0.0
                cooldown_until = i + config.cooldown_days
                trades += 1
            else:
                # Layer D: BBI deviation ladder take-profit (halve position size)
                bbi_dev = current_close / current_bbi - 1
                is_bull = (current_close / current_open - 1) >= config.big_bull_threshold
                if bbi_dev >= config.bbi_dev_threshold and is_bull and position_weight > config.min_remaining_position:
                    capital *= (1 - config.fee_rate * (position_weight / 2))
                    position_weight /= 2
                    trades += 1

    total_return = capital / initial_cap - 1
    buy_hold_return = sliced['收盘'].iloc[-1] / sliced['收盘'].iloc[0] - 1
    
    curve = pd.Series(capital_history)
    daily_returns = curve.pct_change().dropna()
    ann_factor = 252
    n_days = len(sliced)
    
    annualized_return = (1 + total_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    annualized_vol = daily_returns.std() * np.sqrt(ann_factor) if len(daily_returns) > 0 else 0
    sharpe = (annualized_return - config.risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    max_drawdown = ((curve - curve.cummax()) / curve.cummax()).min() if n_days > 0 else 0

    print("\n" + "=" * 50)
    print(f"★ 个股实测绩效报告 ({ticker}) ★")
    print("=" * 50)
    print(f"  初始资金:     CNY {initial_cap:,.2f}")
    print(f"  终末净值:     CNY {capital:,.2f}")
    print(f"  策略累计收益: {total_return * 100:+.2f}%")
    print(f"  基准买入持有: {buy_hold_return * 100:+.2f}%")
    print(f"  年化收益率:   {annualized_return * 100:.2f}%")
    print(f"  年化波动率:   {annualized_vol * 100:.2f}%")
    print(f"  夏普比率:     {sharpe:.4f}")
    print(f"  最大资产回撤: {max_drawdown * 100:.2f}%")
    print(f"  总调仓操作数: {trades}")
    print(f"  回测天数:     {n_days} 天")
    print("=" * 50)

    # 5. Visualizer Plots
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    norm_values = np.array(capital_history) / initial_cap
    norm_benchmark = sliced['收盘'].values / sliced['收盘'].values[0]
    dates_clean = pd.to_datetime(dates_history)
    
    ax.plot(dates_clean, norm_values, color='#00F2FE', linewidth=2.5, label=f'本策略净值 (WFO RF+风控)', alpha=0.95)
    ax.fill_between(dates_clean, norm_values, 1.0, where=(norm_values >= 1.0), facecolor='#00F2FE', alpha=0.08)
    
    ax.plot(dates_clean, norm_benchmark, color='#FF5E62', linewidth=1.5, label=f'{ticker} 被动买入持有基准', alpha=0.6, linestyle='--')
    ax.axhline(y=1.0, color='#8B949E', linestyle='--', linewidth=0.8, alpha=0.6)
    
    ax.set_title(f'个股滚动预测与量化回测对照曲线 ({ticker})', fontsize=14, color='#F0F6FC', pad=20, weight='bold')
    ax.set_ylabel('净值 (初始=1.00)', color='#F0F6FC', fontsize=11, labelpad=10)
    ax.set_xlabel('回测日期', color='#F0F6FC', fontsize=11, labelpad=10)
    ax.tick_params(colors='#8B949E', labelsize=9.5)
    ax.grid(True, which='both', color='#21262D', linestyle='-', linewidth=0.7, alpha=0.6)
    
    info_text = (
        f"策略累计收益: {total_return * 100:+.2f}%\n"
        f"基准累计收益: {buy_hold_return * 100:+.2f}%\n"
        f"策略最大回撤: {max_drawdown * 100:.2f}%\n"
        f"基准最大回撤: {(((sliced['收盘'] - sliced['收盘'].cummax()) / sliced['收盘'].cummax()).min() * 100):.2f}%"
    )
    ax.text(0.03, 0.78, info_text, transform=ax.transAxes, fontsize=10, color='#F0F6FC',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#161B22', edgecolor='#30363D', alpha=0.9))
    
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.70), facecolor='#161B22', edgecolor='#30363D', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#F0F6FC')
        
    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"single_stock_{ticker.replace('.', '_')}.png")
    plt.savefig(save_path, facecolor='#0D1117', edgecolor='none', dpi=150)
    print(f"  [SAVE] 个股回测净值对比曲线图已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    ticker = "300750.SZ"  # 宁德时代 (CATL) as default
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    fetch_and_test(ticker)
