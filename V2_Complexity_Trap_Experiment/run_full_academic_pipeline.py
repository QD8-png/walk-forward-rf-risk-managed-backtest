# -*- coding: utf-8 -*-
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

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

config = strategy.StrategyConfig()
config.portfolio_capital = 1_000_000.0
config.n_shuffles = 20  # Fast verification for now
CACHE_FILE = "sci_baostock_assets_2016_2023.pkl"

def generate_sci_tickers(count: int = 400) -> List[str]:
    tickers = set([
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
    rng = np.random.default_rng(seed=100)
    sz_prefixes = ["000", "001", "002", "300"]
    ss_prefixes = ["600", "601", "603", "605"]
    
    while len(tickers) < count:
        is_sz = rng.choice([True, False])
        prefix = rng.choice(sz_prefixes) if is_sz else rng.choice(ss_prefixes)
        suffix = f"{rng.integers(1, 1000):03d}"
        ticker = f"{'sz' if is_sz else 'sh'}.{prefix}{suffix}"
        tickers.add(ticker)
    return list(tickers)

def process_single_asset_custom(ticker: str, df: pd.DataFrame, config: strategy.StrategyConfig, model_type: str = 'rf') -> Optional[Dict[str, Any]]:
    try:
        df = strategy.prepare_features(df, config)
    except Exception:
        return None

    if len(df) < config.min_data_length:
        return None
    if df['日期'].max() < pd.Timestamp('2023-12-01'):
        return None

    # Calculate additional baselines indicators
    df['ma5'] = df['收盘'].rolling(5).mean()
    df['ma20'] = df['收盘'].rolling(20).mean()
    # 12M - 1M momentum (approx 252 - 21 trading days)
    df['momentum'] = df['收盘'].shift(21) / df['收盘'].shift(252) - 1

    df = df.dropna(subset=['ma5', 'ma20', 'momentum']).reset_index(drop=True)
    if len(df) < config.min_data_length:
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
        'ma5': sliced['ma5'].values,
        'ma20': sliced['ma20'].values,
        'momentum': sliced['momentum'].values
    }

def load_or_build_predictions(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(CACHE_FILE):
        print(f"\n[CACHE] Loading from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
            
    print(f"\n[PREDICT] No cache found, downloading data for {len(tickers)} tickers from Baostock...")
    import baostock as bs
    bs.login()
    
    valid_dfs = {}
    for i, t in enumerate(tickers):
        print(f"   Downloading {i}/{len(tickers)}: {t}")
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

    print(f"   Valid assets: {len(valid_dfs)}. Starting WFO modeling...")
    all_assets = {}
    max_workers = min(4, multiprocessing.cpu_count(), len(valid_dfs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_asset_custom, t, df, config, 'rf'): t for t, df in valid_dfs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                all_assets[res['name']] = res
            if i % 10 == 0 or i == len(valid_dfs):
                print(f"   Progress: {i} / {len(valid_dfs)}")
                
    print(f"   Modeling done. Universe size: {len(all_assets)}. Caching...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(all_assets, f)
    return all_assets

def custom_ablation_backtest(
    all_assets: Dict[str, Dict[str, Any]], 
    config: strategy.StrategyConfig, 
    strategy_type: str = 'ml_rules',  # ml_rules, pure_rules, ma_crossover, momentum
    enable_atr: bool = True, 
    enable_bbl: bool = True, 
    enable_toce: bool = True, 
    enable_trailing: bool = True, 
    enable_bbi_tp: bool = True
) -> Tuple[float, float, float, float, List[float], List[pd.Timestamp], List[Dict[str, Any]]]:
    
    capital = config.portfolio_capital
    holdings: Dict[str, strategy.Holding] = {}
    cooldowns: Dict[str, int] = {}
    trade_log: List[Dict[str, Any]] = []
    
    all_dates = set()
    for asset in all_assets.values():
        all_dates.update(asset['dates'])
    all_dates = sorted(list(all_dates))
    asset_date_map = {name: {d: i for i, d in enumerate(asset['dates'])} for name, asset in all_assets.items()}
    
    portfolio_values = []
    portfolio_dates = []
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2023-12-31')
    
    for day_idx, today in enumerate(all_dates):
        today_ts = pd.Timestamp(today)
        if today_ts < start_date or today_ts > end_date:
            continue
            
        daily_portfolio_return = 0.0
        for stock_name in list(holdings.keys()):
            h = holdings[stock_name]
            if today not in asset_date_map[stock_name]:
                continue
            local_idx = asset_date_map[stock_name][today]
            asset = all_assets[stock_name]
            current_close = asset['close'][local_idx]
            prev_close = asset['close'][local_idx - 1] if local_idx > 0 else current_close
            
            if prev_close > 0:
                daily_portfolio_return += (current_close / prev_close - 1) * h.position_weight
            if current_close > h.max_price:
                h.max_price = current_close
                
        capital = capital * (1 + daily_portfolio_return)
        
        # Risk check & exits
        stocks_to_sell = []
        for stock_name in list(holdings.keys()):
            h = holdings[stock_name]
            if today not in asset_date_map[stock_name]:
                continue
            local_idx = asset_date_map[stock_name][today]
            asset = all_assets[stock_name]
            current_close = asset['close'][local_idx]
            
            should_exit = False
            unrealized = current_close / h.entry_price - 1
            holding_days = day_idx - h.entry_day_idx
            
            if strategy_type == 'ma_crossover':
                if asset['ma5'][local_idx] < asset['ma20'][local_idx]:
                    should_exit = True
            
            if enable_bbl and current_close < asset['bb_line'][local_idx]:
                should_exit = True
            elif enable_atr and current_close < h.entry_stop_price:
                should_exit = True
            elif enable_toce and holding_days >= config.patience_days and unrealized < config.patience_return:
                should_exit = True
            else:
                if enable_trailing:
                    if unrealized >= config.trailing_activate_pct:
                        if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                            should_exit = True
                            
            if strategy_type == 'ml_rules' and not should_exit and not enable_bbl and not enable_atr and not enable_toce and not enable_trailing:
                if asset['y_pred'][local_idx] == 0:
                    should_exit = True
                    
            if should_exit:
                stocks_to_sell.append(stock_name)
            else:
                if enable_bbi_tp:
                    bbi_dev = current_close / asset['bbi'][local_idx] - 1
                    is_bull = (current_close / asset['open'][local_idx] - 1) >= config.big_bull_threshold
                    if bbi_dev >= config.bbi_dev_threshold and is_bull and h.position_weight > config.min_remaining_position * config.max_weight_per_stock:
                        capital *= (1 - config.fee_rate * (h.position_weight / 2))
                        h.position_weight /= 2
                        
        for stock_name in stocks_to_sell:
            h = holdings[stock_name]
            asset = all_assets[stock_name]
            local_idx = asset_date_map[stock_name][today]
            exit_price = asset['close'][local_idx]
            
            raw_return = exit_price / h.entry_price - 1
            net_return = raw_return - 2 * config.fee_rate
            
            trade_log.append({
                'stock': stock_name,
                'entry_date': all_dates[h.entry_day_idx],
                'exit_date': today,
                'raw_return': raw_return,
                'net_return': net_return
            })
            
            capital *= (1 - config.fee_rate * holdings[stock_name].position_weight)
            cooldowns[stock_name] = day_idx + config.cooldown_days
            del holdings[stock_name]
            
        # Entries screening
        candidates = []
        if len(holdings) < config.max_holdings:
            for stock_name, asset in all_assets.items():
                if stock_name in holdings or (stock_name in cooldowns and day_idx < cooldowns[stock_name]):
                    continue
                if today not in asset_date_map[stock_name]:
                    continue
                local_idx = asset_date_map[stock_name][today]
                
                if strategy_type == 'ml_rules':
                    if asset['y_pred'][local_idx] != 1 or asset['ma120_slope'][local_idx] <= 0:
                        continue
                    if not (asset['ma120_slope'][local_idx] > 0.01 or asset['y_prob'][local_idx] > 0.65) and asset['kdj_j'][local_idx] >= config.kdj_panic_threshold:
                        continue
                    if asset['close'][local_idx] < asset['bb_line'][local_idx]:
                        continue
                    candidates.append((stock_name, asset['y_prob'][local_idx]))
                
                elif strategy_type == 'pure_rules':
                    if asset['ma120_slope'][local_idx] <= 0:
                        continue
                    if asset['ma120_slope'][local_idx] <= 0.01 and asset['kdj_j'][local_idx] >= config.kdj_panic_threshold:
                        continue
                    if asset['close'][local_idx] < asset['bb_line'][local_idx]:
                        continue
                    candidates.append((stock_name, asset['ma120_slope'][local_idx]))
                    
                elif strategy_type == 'ma_crossover':
                    if asset['ma5'][local_idx] > asset['ma20'][local_idx]:
                        candidates.append((stock_name, asset['ma5'][local_idx] / asset['ma20'][local_idx]))
                        
                elif strategy_type == 'momentum':
                    if asset['momentum'][local_idx] > 0:
                        candidates.append((stock_name, asset['momentum'][local_idx]))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for stock_name, score in candidates:
            if len(holdings) >= config.max_holdings:
                break
            rem_cap = 1.0 - sum(h.position_weight for h in holdings.values())
            if rem_cap <= 0.01:
                break
                
            if strategy_type == 'ml_rules':
                raw_pos = min(config.max_position_size, max(config.min_position_size, (score - 0.5) * config.position_scale_factor))
            else:
                raw_pos = 1.0
                
            target_weight = min(raw_pos * config.max_weight_per_stock, rem_cap)
            if target_weight < 0.01:
                continue
                
            local_idx = asset_date_map[stock_name][today]
            asset = all_assets[stock_name]
            capital *= (1 - config.fee_rate * target_weight)
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
    
    daily_returns = curve.pct_change().dropna()
    ann_factor = 252
    n_days = len(portfolio_values)
    ann_ret = (1 + total_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    ann_vol = daily_returns.std() * np.sqrt(ann_factor) if len(daily_returns) > 0 else 0
    sharpe = (ann_ret - config.risk_free_rate) / ann_vol if ann_vol != 0 else 0.0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0
    
    return total_return, mdd, sharpe, calmar, portfolio_values, portfolio_dates, trade_log

def compute_buy_and_hold(all_assets: Dict[str, Dict[str, Any]], initial_capital: float) -> Tuple[float, float, float, List[float], List[pd.Timestamp]]:
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
            if today not in asset_date_map[name]: continue
            local_idx = asset_date_map[name][today]
            current_close = asset['close'][local_idx]
            prev_close = asset['close'][local_idx - 1] if local_idx > 0 else current_close
            if prev_close > 0:
                daily_portfolio_return += current_close / prev_close - 1
                active_count += 1
                
        avg_return = daily_portfolio_return / active_count if active_count > 0 else 0.0
        capital = capital * (1 + avg_return)
        portfolio_values.append(capital)
        portfolio_dates.append(today)
        
    total_return = capital / initial_capital - 1
    curve = pd.Series(portfolio_values)
    mdd = ((curve - curve.cummax()) / curve.cummax()).min() if len(curve) > 0 else 0.0
    
    n_days = len(portfolio_values)
    ann_factor = 252
    ann_ret = (1 + total_return) ** (ann_factor / n_days) - 1 if n_days > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0
    
    return total_return, mdd, calmar, portfolio_values, portfolio_dates

def main():
    print("=" * 70)
    print("★ SCI Academic Empirical Pipeline (Route A: Synergy & Emergence) ★")
    print("=" * 70)
    
    tickers = generate_sci_tickers(200) # Fast testing
    all_assets = load_or_build_predictions(tickers)
    os.makedirs("plots", exist_ok=True)
    
    print("\n--- 1. Passive Benchmark ---")
    bh_ret, bh_mdd, bh_calmar, bh_vals, bh_dates = compute_buy_and_hold(all_assets, config.portfolio_capital)
    print(f"   Buy & Hold Return: {bh_ret*100:.2f}%, MDD: {bh_mdd*100:.2f}%, Calmar: {bh_calmar:.4f}")
    
    print("\n--- 2. Factor Baselines ---")
    ma_ret, ma_mdd, ma_sharpe, ma_calmar, ma_vals, ma_dates, _ = custom_ablation_backtest(all_assets, config, strategy_type='ma_crossover', enable_atr=False, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False)
    print(f"   MA Crossover (5/20) Return: {ma_ret*100:.2f}%, MDD: {ma_mdd*100:.2f}%, Sharpe: {ma_sharpe:.4f}, Calmar: {ma_calmar:.4f}")
    
    ma_atr_ret, ma_atr_mdd, ma_atr_sharpe, ma_atr_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='ma_crossover', enable_atr=True, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False)
    print(f"   MA + ATR Crossover Return: {ma_atr_ret*100:.2f}%, MDD: {ma_atr_mdd*100:.2f}%, Sharpe: {ma_atr_sharpe:.4f}, Calmar: {ma_atr_calmar:.4f}")
    
    mom_ret, mom_mdd, mom_sharpe, mom_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='momentum', enable_atr=False, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False)
    print(f"   Momentum (12M-1M) Return: {mom_ret*100:.2f}%, MDD: {mom_mdd*100:.2f}%, Sharpe: {mom_sharpe:.4f}, Calmar: {mom_calmar:.4f}")
    
    print("\n--- 3. Ablation Study (Pure Rules vs ML+Rules) ---")
    results = []
    
    print("   [0-A] Pure Rules Equivalent to S4...")
    r0a_ret, r0a_mdd, r0a_sharpe, r0a_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='pure_rules', enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False)
    results.append({"System": "System 0-A: Pure Rules (S4 Equivalent)", "Return": r0a_ret, "MDD": r0a_mdd, "Sharpe": r0a_sharpe, "Calmar": r0a_calmar})
    
    print("   [0-B] Pure Rules Equivalent to S5...")
    r0b_ret, r0b_mdd, r0b_sharpe, r0b_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='pure_rules', enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True)
    results.append({"System": "System 0-B: Pure Rules (S5 Equivalent)", "Return": r0b_ret, "MDD": r0b_mdd, "Sharpe": r0b_sharpe, "Calmar": r0b_calmar})

    print("   [1] ML Only...")
    r1_ret, r1_mdd, r1_sharpe, r1_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='ml_rules', enable_atr=False, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False)
    results.append({"System": "System 1: Pure ML Baseline", "Return": r1_ret, "MDD": r1_mdd, "Sharpe": r1_sharpe, "Calmar": r1_calmar})

    print("   [2] ML + ATR...")
    r2_ret, r2_mdd, r2_sharpe, r2_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='ml_rules', enable_atr=True, enable_bbl=False, enable_toce=False, enable_trailing=False, enable_bbi_tp=False)
    results.append({"System": "System 2: ML + ATR Stop", "Return": r2_ret, "MDD": r2_mdd, "Sharpe": r2_sharpe, "Calmar": r2_calmar})

    print("   [3] ML + ATR + BBL...")
    r3_ret, r3_mdd, r3_sharpe, r3_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='ml_rules', enable_atr=True, enable_bbl=True, enable_toce=False, enable_trailing=False, enable_bbi_tp=False)
    results.append({"System": "System 3: ML + ATR + BBL", "Return": r3_ret, "MDD": r3_mdd, "Sharpe": r3_sharpe, "Calmar": r3_calmar})

    print("   [4] ML + ATR + BBL + TOCE (Proposed)...")
    r4_ret, r4_mdd, r4_sharpe, r4_calmar, _, _, _ = custom_ablation_backtest(all_assets, config, strategy_type='ml_rules', enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False)
    results.append({"System": "System 4: ML + ATR + BBL + TOCE", "Return": r4_ret, "MDD": r4_mdd, "Sharpe": r4_sharpe, "Calmar": r4_calmar})

    print("   [5] Full ARMS...")
    r5_ret, r5_mdd, r5_sharpe, r5_calmar, s5_vals, s5_dates, s5_trade_log = custom_ablation_backtest(all_assets, config, strategy_type='ml_rules', enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True)
    results.append({"System": "System 5: Full ARMS Framework", "Return": r5_ret, "MDD": r5_mdd, "Sharpe": r5_sharpe, "Calmar": r5_calmar})

    results.append({"System": "Benchmark: Buy & Hold", "Return": bh_ret, "MDD": bh_mdd, "Sharpe": 0.0, "Calmar": bh_calmar})
    results.append({"System": "Benchmark: MA Crossover", "Return": ma_ret, "MDD": ma_mdd, "Sharpe": ma_sharpe, "Calmar": ma_calmar})
    results.append({"System": "Benchmark: MA + ATR Crossover", "Return": ma_atr_ret, "MDD": ma_atr_mdd, "Sharpe": ma_atr_sharpe, "Calmar": ma_atr_calmar})
    results.append({"System": "Benchmark: Momentum", "Return": mom_ret, "MDD": mom_mdd, "Sharpe": mom_sharpe, "Calmar": mom_calmar})
    
    df = pd.DataFrame(results)
    for col in ["Return", "MDD"]:
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    df["Sharpe"] = df["Sharpe"].apply(lambda x: f"{x:.4f}" if x != 0.0 else "N/A")
    df["Calmar"] = df["Calmar"].apply(lambda x: f"{x:.4f}")
    df.to_csv("sci_ablation_results_v2.csv", index=False, encoding='utf-8-sig')
    print("\n[SAVE] Ablation & Baselines saved to sci_ablation_results_v2.csv")
    print(df.to_string())
    
    print("\n--- 4. Deep Analysis Outputs ---")
    
    print("\n[Yearly Breakdown] System 5 vs MA Crossover")
    s5_series = pd.Series(s5_vals, index=pd.to_datetime(s5_dates))
    ma_series = pd.Series(ma_vals, index=pd.to_datetime(ma_dates))
    s5_yearly = s5_series.resample('YE').apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    ma_yearly = ma_series.resample('YE').apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    yearly_df = pd.DataFrame({'System 5': s5_yearly, 'MA Crossover': ma_yearly})
    yearly_df.index = yearly_df.index.year
    print(yearly_df.applymap(lambda x: f"{x*100:.2f}%"))
    yearly_df.to_csv("yearly_breakdown.csv")
    
    print("\n[Stock Profitability] System 5")
    trades_df = pd.DataFrame(s5_trade_log)
    if not trades_df.empty:
        stock_pnl = trades_df.groupby('stock')['net_return'].sum()
        profitable_stocks = (stock_pnl > 0).sum()
        total_traded_stocks = len(stock_pnl)
        total_universe = len(all_assets)
        print(f"Total Universe: {total_universe} stocks")
        print(f"Traded Stocks: {total_traded_stocks} stocks")
        print(f"Profitable Stocks: {profitable_stocks} stocks")
        print(f"Win Rate (Profitable / Traded): {profitable_stocks / total_traded_stocks * 100:.2f}%")
        print(f"Win Rate (Profitable / Universe): {profitable_stocks / total_universe * 100:.2f}%")
    else:
        print("No trades executed by System 5.")
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
