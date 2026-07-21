import os
import sys
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Matplotlib dark theme
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Configuration class matching strategy configs
class Config:
    portfolio_capital = 1_000_000.0
    max_weight_per_stock = 0.25
    max_holdings = 4
    cooldown_days = 15  # Optimized value
    patience_days = 5
    patience_return = 0.005
    atr_multiplier = 2.0
    trailing_activate_pct = 0.03
    trailing_stop_pct = 0.05
    min_remaining_position = 0.05
    big_bull_threshold = 0.02
    bbi_dev_threshold = 0.03
    fee_rate = 0.0013
    risk_free_rate = 0.02

config = Config()

@dataclass
class Holding:
    name: str
    entry_price: float
    entry_stop_price: float
    max_price: float
    position_weight: float
    entry_day_idx: int = 0

LEADER_BLACKLIST = {
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
}

def generate_random_tickers(count: int = 350) -> List[str]:
    """Generates unique, valid-format A-share tickers excluding blacklisted leaders."""
    tickers = set()
    rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
    
    sz_prefixes = ["000", "001", "002", "300"]
    ss_prefixes = ["600", "601", "603", "605"]
    
    # Let's generate a list of candidates
    while len(tickers) < count:
        is_sz = rng.choice([True, False])
        if is_sz:
            prefix = rng.choice(sz_prefixes)
            # Normal range of suffix numbers
            suffix = f"{rng.integers(1, 1000):03d}"
            ticker = f"{prefix}{suffix}.SZ"
        else:
            prefix = rng.choice(ss_prefixes)
            suffix = f"{rng.integers(1, 1000):03d}"
            ticker = f"{prefix}{suffix}.SS"
            
        if ticker not in LEADER_BLACKLIST:
            tickers.add(ticker)
            
    return list(tickers)

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all technical indicators required for the pure rules backtest."""
    df = df.copy()
    
    # Standardize columns
    col_map = {
        'Date': '日期', 'Open': '开盘', 'High': '高', 'Low': '低',
        'Close': '收盘', 'Volume': '交易量'
    }
    df.rename(columns=col_map, inplace=True)
    df = df.dropna(subset=['收盘']).reset_index(drop=True)
    
    # Indicators calculation
    df['收益率'] = df['收盘'].pct_change()
    
    # KDJ
    kdj_n = 9
    low_n = df['低'].rolling(window=kdj_n).min()
    high_n = df['高'].rolling(window=kdj_n).max()
    rsv = (df['收盘'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    df['KDJ_K'] = rsv.ewm(com=2, adjust=False).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2, adjust=False).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
    
    # MA120 slope
    df['MA120'] = df['收盘'].rolling(window=120).mean()
    df['MA120_slope'] = (df['MA120'] - df['MA120'].shift(20)) / df['MA120'].shift(20)
    
    # Bull Bear Line (14, 28, 57, 114)
    df['多空线'] = (df['收盘'].rolling(14).mean() + df['收盘'].rolling(28).mean()
                  + df['收盘'].rolling(57).mean() + df['收盘'].rolling(114).mean()) / 4
                  
    # BBI (3, 6, 12, 24)
    df['BBI'] = (df['收盘'].rolling(3).mean() + df['收盘'].rolling(6).mean()
                + df['收盘'].rolling(12).mean() + df['收盘'].rolling(24).mean()) / 4
                
    # ATR 14
    prev_close = df['收盘'].shift(1)
    tr1 = df['高'] - df['低']
    tr2 = (df['高'] - prev_close).abs()
    tr3 = (df['低'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()
    
    required = ['多空线', 'BBI', '收盘', 'KDJ_J', 'MA120_slope', 'ATR', '日期']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=required).reset_index(drop=True)
    return df

def run_pure_rules_backtest(all_assets: Dict[str, pd.DataFrame], config: Config, enable_atr: bool = True, enable_bbl: bool = True, enable_toce: bool = True, enable_trailing: bool = True, enable_bbi_tp: bool = True) -> Tuple[float, float, float, List[float], List[pd.Timestamp], int]:
    capital = config.portfolio_capital
    holdings: Dict[str, Holding] = {}
    cooldowns: Dict[str, int] = {}
    
    # Consolidate all timeline dates in 2020-2025
    all_dates = set()
    for df in all_assets.values():
        all_dates.update(df['日期'].values)
    all_dates = sorted(list(all_dates))
    
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2025-12-31')
    active_dates = [d for d in all_dates if start_date <= pd.Timestamp(d) <= end_date]
    
    asset_date_map = {name: {df['日期'].iloc[i]: i for i in range(len(df))} for name, df in all_assets.items()}
    
    portfolio_values = []
    portfolio_dates = []
    total_trades = 0
    
    for day_idx, today in enumerate(active_dates):
        # 1. Update mark-to-market
        daily_portfolio_return = 0.0
        for stock_name in list(holdings.keys()):
            h = holdings[stock_name]
            df = all_assets[stock_name]
            idx_map = asset_date_map[stock_name]
            if today not in idx_map:
                continue
            local_idx = idx_map[today]
            current_close = df['收盘'].iloc[local_idx]
            prev_close = df['收盘'].iloc[local_idx - 1] if local_idx > 0 else current_close
            
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
            df = all_assets[stock_name]
            idx_map = asset_date_map[stock_name]
            if today not in idx_map:
                continue
            local_idx = idx_map[today]
            current_close = df['收盘'].iloc[local_idx]
            
            should_exit = False
            unrealized = current_close / h.entry_price - 1
            holding_days = day_idx - h.entry_day_idx
            
            # Layer A: BBL filter
            if enable_bbl and current_close < df['多空线'].iloc[local_idx]:
                should_exit = True
            # Layer B: ATR Stop
            elif enable_atr and current_close < h.entry_stop_price:
                should_exit = True
            # Layer C: TOCE
            elif enable_toce and holding_days >= config.patience_days and unrealized < config.patience_return:
                should_exit = True
            else:
                # Layer D: Trailing Profit Lock
                if enable_trailing:
                    if unrealized >= config.trailing_activate_pct:
                        if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                            should_exit = True
            
            if should_exit:
                stocks_to_sell.append(stock_name)
            else:
                # BBI Deviation Ladder profit take
                if enable_bbi_tp:
                    bbi_dev = current_close / df['BBI'].iloc[local_idx] - 1
                    is_bull = (current_close / df['开盘'].iloc[local_idx] - 1) >= config.big_bull_threshold
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
            for stock_name, df in all_assets.items():
                if stock_name in holdings:
                    continue
                if stock_name in cooldowns and day_idx < cooldowns[stock_name]:
                    continue
                    
                idx_map = asset_date_map[stock_name]
                if today not in idx_map:
                    continue
                local_idx = idx_map[today]
                
                # Rule-based entry criteria (No ML check)
                if df['MA120_slope'].iloc[local_idx] <= 0:
                    continue
                    
                is_strong_trend = df['MA120_slope'].iloc[local_idx] > 0.01
                if not is_strong_trend:
                    if df['KDJ_J'].iloc[local_idx] >= 20: # Panic threshold
                        continue
                        
                if df['收盘'].iloc[local_idx] < df['多空线'].iloc[local_idx]:
                    continue
                    
                # Score based on trend slope strength (stronger trend has higher priority)
                score = df['MA120_slope'].iloc[local_idx]
                candidates.append((stock_name, score))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for stock_name, score in candidates:
            if len(holdings) >= config.max_holdings:
                break
            rem_cap = 1.0 - sum(h.position_weight for h in holdings.values())
            if rem_cap <= 0.01:
                break
                
            raw_pos = 1.0 # Pure Rules: fixed leverage factor 1.0
            target_weight = min(raw_pos * config.max_weight_per_stock, rem_cap)
            
            if target_weight < 0.01:
                continue
                
            local_idx = asset_date_map[stock_name][today]
            df = all_assets[stock_name]
            
            capital *= (1 - config.fee_rate * target_weight)
            total_trades += 1
            holdings[stock_name] = Holding(
                name=stock_name,
                entry_price=df['收盘'].iloc[local_idx],
                entry_stop_price=df['低'].iloc[local_idx] - config.atr_multiplier * df['ATR'].iloc[local_idx],
                max_price=df['收盘'].iloc[local_idx],
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
    
    return total_return, mdd, sharpe, portfolio_values, portfolio_dates, total_trades

def run_buy_and_hold(all_assets: Dict[str, pd.DataFrame], config: Config) -> Tuple[float, float, List[float], List[pd.Timestamp]]:
    """Runs equal-weighted buy-and-hold across all available active assets in 2020-2025."""
    all_dates = set()
    for df in all_assets.values():
        all_dates.update(df['日期'].values)
    all_dates = sorted(list(all_dates))
    
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2025-12-31')
    active_dates = [d for d in all_dates if start_date <= pd.Timestamp(d) <= end_date]
    
    asset_date_map = {name: {df['日期'].iloc[i]: i for i in range(len(df))} for name, df in all_assets.items()}
    
    capital = config.portfolio_capital
    portfolio_values = []
    portfolio_dates = []
    
    for today in active_dates:
        daily_portfolio_return = 0.0
        active_count = 0
        
        for name, df in all_assets.items():
            idx_map = asset_date_map[name]
            if today not in idx_map:
                continue
            local_idx = idx_map[today]
            current_close = df['收盘'].iloc[local_idx]
            prev_close = df['收盘'].iloc[local_idx - 1] if local_idx > 0 else current_close
            
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
        
    total_return = capital / config.portfolio_capital - 1
    curve = pd.Series(portfolio_values)
    mdd = ((curve - curve.cummax()) / curve.cummax()).min() if len(curve) > 0 else 0.0
    return total_return, mdd, portfolio_values, portfolio_dates

def main():
    print("=" * 60)
    print("★ 纯规则策略稳健性验证 (200只非龙头普通股, 时段: 2020-2025) ★")
    print("=" * 60)
    
    # 1. Generate 380 random tickers
    print("正在生成 380 只候选非龙头 A 股代码...")
    candidates = generate_random_tickers(380)
    
    # 2. Download data in parallel
    print("开始并行下载历史 OHLCV 行情数据 (2017-01-01 至 2025-12-31)...")
    raw_data = yf.download(candidates, start="2017-01-01", end="2025-12-31", group_by='ticker', threads=True, progress=False)
    
    # 3. Clean and filter to keep 200 valid assets
    print("正在过滤数据完整且流动性合规的非龙头股票...")
    valid_assets = {}
    
    # If single asset returned (unlikely for list, but for safety)
    if not isinstance(raw_data.columns, pd.MultiIndex):
        print("Error downloading dataset - please check network.")
        return
        
    for ticker in candidates:
        if ticker not in raw_data.columns.levels[0]:
            continue
        df_ticker = raw_data[ticker].dropna(how='all')
        
        # Check minimum data length (approx 6 years, 1500 days)
        if len(df_ticker) >= 1500:
            df_ticker = df_ticker.reset_index()
            required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df_ticker.columns for col in required):
                try:
                    df_ind = prepare_indicators(df_ticker)
                    if len(df_ind) >= 1200: # Ensure enough points after indicators calculations
                        valid_assets[ticker] = df_ind
                except Exception:
                    continue
                    
        # Stop once we hit exactly 200 stocks
        if len(valid_assets) == 200:
            break
            
    print(f"成功筛选出 {len(valid_assets)} 只符合历史数据要求的非龙头股票资产池。")
    if len(valid_assets) < 100:
        print("警告: 符合要求的股票数量太少，回测代表性不足。")
        return
        
    # 4. Run Backtests
    print("\n--- 正在运行 2020-2025 组合回测 ---")
    
    # Benchmark: Buy & Hold
    print("运行等权买入持有 (Buy & Hold) 基准...")
    bh_ret, bh_mdd, bh_vals, bh_dates = run_buy_and_hold(valid_assets, config)
    print(f"  B&H 累计收益: {bh_ret*100:.2f}%, 最大回撤: {bh_mdd*100:.2f}%")
    
    # System 4 Pure Rules
    print("运行纯规则 System 4 (ATR+BBL+TOCE, 无 ML)...")
    s4_ret, s4_mdd, s4_sharpe, s4_vals, s4_dates, s4_trades = run_pure_rules_backtest(
        valid_assets, config, enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False
    )
    print(f"  S4 累计收益: {s4_ret*100:.2f}%, 最大回撤: {s4_mdd*100:.2f}%, Sharpe: {s4_sharpe:.4f}, 交易数: {s4_trades}")
    
    # System 5 Pure Rules
    print("运行纯规则 System 5 (Full wind-control, 无 ML)...")
    s5_ret, s5_mdd, s5_sharpe, s5_vals, _, s5_trades = run_pure_rules_backtest(
        valid_assets, config, enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True
    )
    print(f"  S5 累计收益: {s5_ret*100:.2f}%, 最大回撤: {s5_mdd*100:.2f}%, Sharpe: {s5_sharpe:.4f}, 交易数: {s5_trades}")
    
    # 5. Output comparison results
    results = [
        {"Strategy": "Benchmark: Buy & Hold (B&H)", "Total Return": f"{bh_ret*100:.2f}%", "Max Drawdown": f"{bh_mdd*100:.2f}%", "Sharpe Ratio": "N/A", "Total Trades": "N/A"},
        {"Strategy": "Pure Rules (S4 Equivalent)", "Total Return": f"{s4_ret*100:.2f}%", "Max Drawdown": f"{s4_mdd*100:.2f}%", "Sharpe Ratio": f"{s4_sharpe:.4f}", "Total Trades": s4_trades},
        {"Strategy": "Pure Rules (S5 Equivalent)", "Total Return": f"{s5_ret*100:.2f}%", "Max Drawdown": f"{s5_mdd*100:.2f}%", "Sharpe Ratio": f"{s5_sharpe:.4f}", "Total Trades": s5_trades}
    ]
    df_res = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("稳健性验证结果对比表 (2020-2025, 200只非龙头普通股)")
    print("=" * 60)
    print(df_res.to_string(index=False))
    print("=" * 60)
    
    # Save CSV
    output_dir = r"C:\Users\qwe\.gemini\antigravity\scratch\walk-forward-rf-risk-managed-backtest"
    output_csv = os.path.join(output_dir, "pure_rules_robustness_results.csv")
    df_res.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"对比数据已保存至: {output_csv}")
    
    # Plot curves
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    dates_clean = pd.to_datetime(s4_dates)
    ax.plot(dates_clean, np.array(bh_vals)/1e6, color='#8B949E', linewidth=1.5, label='Benchmark: Buy & Hold (B&H)', alpha=0.6)
    ax.plot(dates_clean, np.array(s4_vals)/1e6, color='#FF9800', linewidth=2.2, label='Pure Rules (S4 Equivalent: ATR+BBL+TOCE)', alpha=0.9)
    ax.plot(dates_clean, np.array(s5_vals)/1e6, color='#E91E63', linewidth=2.2, label='Pure Rules (S5 Equivalent: Full Wind-Control)', alpha=0.9)
    
    ax.axhline(y=1.0, color='#8B949E', linestyle=':', alpha=0.5)
    ax.set_title(f'Pure Rules Robustness Verification (2020-2025, {len(valid_assets)} Random Non-Leader Stocks)', fontsize=13, color='#F0F6FC', pad=18, weight='bold')
    ax.set_ylabel('Portfolio Equity (Million CNY)', color='#F0F6FC')
    ax.set_xlabel('Year', color='#F0F6FC')
    ax.tick_params(colors='#8B949E', labelsize=10)
    ax.grid(True, color='#21262D', linestyle='-', linewidth=0.7, alpha=0.5)
    
    leg = ax.legend(loc='upper left', facecolor='#161B22', edgecolor='#30363D', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#F0F6FC')
        
    plt.tight_layout()
    output_png = os.path.join(output_dir, "plots", "pure_rules_robustness_comparison.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, facecolor='#0D1117', edgecolor='none', dpi=200)
    print(f"净值对比图表已保存至: {output_png}")
    
if __name__ == "__main__":
    main()
