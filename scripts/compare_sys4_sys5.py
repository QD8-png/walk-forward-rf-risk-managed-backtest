# -*- coding: utf-8 -*-
import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd

# Path alignment to import strategy.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import strategy

CACHE_FILE = "all_assets_2019_2023.pkl"

def run_logged_backtest(
    all_assets,
    config,
    enable_atr=True,
    enable_bbl=True,
    enable_toce=True,
    enable_trailing=True,
    enable_bbi_tp=True,
    system_name="System"
):
    capital = config.portfolio_capital
    holdings = {} # stock_name -> Holding
    cooldowns = {}
    
    # Gather dates
    all_dates = set()
    for asset in all_assets.values():
        all_dates.update(asset['dates'])
    all_dates = sorted(list(all_dates))
    
    asset_date_map = {name: {d: i for i, d in enumerate(asset['dates'])} for name, asset in all_assets.items()}
    
    portfolio_values = []
    portfolio_dates = []
    
    # Trade logging
    trades = [] # List of closed trades: {stock_name, entry_date, exit_date, entry_day_idx, exit_day_idx, exit_reason, unrealized_return}
    exit_reasons_count = {"ATR止损": 0, "TOCE清退": 0, "BBL清退": 0, "移动止盈": 0, "BBI减仓": 0}
    cooldown_blocked_count = 0
    
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2023-12-31')
    
    print(f"\n==========================================")
    print(f" 运行 {system_name} 详细交易日志...")
    print(f"==========================================")
    
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
            exit_reason = None
            unrealized = current_close / h.entry_price - 1
            holding_days = day_idx - h.entry_day_idx
            
            # Layer A: Bull-Bear Line Trend Interceptor
            if enable_bbl and current_close < asset['bb_line'][local_idx]:
                should_exit = True
                exit_reason = "BBL清退"
            # Layer B: ATR Volatility Stop Loss
            elif enable_atr and current_close < h.entry_stop_price:
                should_exit = True
                exit_reason = "ATR止损"
            # Layer C: TOCE (Time-based Opportunity Cost Exit)
            elif enable_toce and holding_days >= config.patience_days and unrealized < config.patience_return:
                should_exit = True
                exit_reason = "TOCE清退"
            else:
                # Layer D: Trailing Profit Lock
                if enable_trailing:
                    if unrealized < config.trailing_activate_pct:
                        if asset['y_pred'][local_idx] == 0 and unrealized > 0:
                            should_exit = True
                            exit_reason = "移动止盈"
                    else:
                        if h.max_price > h.entry_price and current_close < h.max_price * (1 - config.trailing_stop_pct):
                            should_exit = True
                            exit_reason = "移动止盈"
                            
            # Baseline Pure ML Exit (no risk management enabled at all)
            if not should_exit and not enable_bbl and not enable_atr and not enable_toce and not enable_trailing:
                if asset['y_pred'][local_idx] == 0:
                    should_exit = True
                    exit_reason = "ML退出"
                    
            if should_exit:
                stocks_to_sell.append((stock_name, exit_reason))
            else:
                # BBI Deviation Ladder profit take (partial exit)
                if enable_bbi_tp:
                    bbi_dev = current_close / asset['bbi'][local_idx] - 1
                    is_bull = (current_close / asset['open'][local_idx] - 1) >= config.big_bull_threshold
                    if bbi_dev >= config.bbi_dev_threshold and is_bull and h.position_weight > config.min_remaining_position * config.max_weight_per_stock:
                        print(f"[{today_ts.strftime('%Y-%m-%d')}] {stock_name} 触及 BBI偏离 减仓减半 (持仓权重 {h.position_weight:.4f} -> {h.position_weight/2:.4f})")
                        capital *= (1 - config.fee_rate * (h.position_weight / 2))
                        h.position_weight /= 2
                        exit_reasons_count["BBI减仓"] += 1
                        
        for stock_name, reason in stocks_to_sell:
            h = holdings[stock_name]
            idx_map = asset_date_map[stock_name]
            local_idx = idx_map[today]
            current_close = all_assets[stock_name]['close'][local_idx]
            unrealized = current_close / h.entry_price - 1
            holding_days = day_idx - h.entry_day_idx
            
            # Print exit details
            entry_date_str = all_dates[h.entry_day_idx]
            print(f"[{today_ts.strftime('%Y-%m-%d')}] 卖出 {stock_name} | 原因: {reason} | 买入日: {entry_date_str} | 持仓天数: {holding_days} | 收益率: {unrealized*100:.2f}%")
            
            trades.append({
                "stock_name": stock_name,
                "entry_date": entry_date_str,
                "exit_date": today,
                "holding_days": holding_days,
                "exit_reason": reason,
                "return": unrealized
            })
            if reason in exit_reasons_count:
                exit_reasons_count[reason] += 1
            else:
                exit_reasons_count[reason] = exit_reasons_count.get(reason, 0) + 1
                
            capital *= (1 - config.fee_rate * h.position_weight)
            cooldowns[stock_name] = day_idx + config.cooldown_days
            del holdings[stock_name]
            
        # 3. Entries screening
        candidates = []
        if len(holdings) < config.max_holdings:
            for stock_name, asset in all_assets.items():
                if stock_name in holdings:
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
                
                # This is a valid candidate signal! Let's check if blocked by cooldown lock
                if stock_name in cooldowns and day_idx < cooldowns[stock_name]:
                    cooldown_blocked_count += 1
                    # print(f"  [冷却阻挡] {stock_name} 在冷却期内被锁 (距离解锁还剩 {cooldowns[stock_name] - day_idx} 天)")
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
        
    # Stats compilation
    avg_holding_days = np.mean([t['holding_days'] for t in trades]) if len(trades) > 0 else 0
    total_exit_count = sum(exit_reasons_count[k] for k in ["ATR止损", "TOCE清退", "BBL清退", "移动止盈"])
    
    print(f"\n==========================================")
    print(f" {system_name} 统计报告:")
    print(f"==========================================")
    print(f"  总交易笔数 (完全退出): {len(trades)}")
    print(f"  平均持仓天数:         {avg_holding_days:.2f} 天")
    print(f"  各类退出原因统计 (总完全退出: {total_exit_count} 次):")
    for k, v in exit_reasons_count.items():
        if k == "BBI减仓":
            print(f"    - {k}: {v} 次 (部分平仓，不计入完全退出占比)")
        else:
            pct = (v / total_exit_count * 100) if total_exit_count > 0 else 0
            print(f"    - {k}: {v} 次 ({pct:.2f}%)")
    print(f"  被冷却期阻挡的入场信号次数: {cooldown_blocked_count} 次")
    print(f"  终末净值:             {capital:.2f}")
    print(f"==========================================\n")
    
    return {
        "trades": trades,
        "avg_holding_days": avg_holding_days,
        "exit_reasons_count": exit_reasons_count,
        "cooldown_blocked_count": cooldown_blocked_count,
        "final_capital": capital
    }

def main():
    if not os.path.exists(CACHE_FILE):
        print(f"未找到缓存文件 {CACHE_FILE}，请先运行 academic_empirical_pipeline.py 生成缓存！")
        return
        
    with open(CACHE_FILE, 'rb') as f:
        all_assets = pickle.load(f)
        
    config = strategy.StrategyConfig()
    config.portfolio_capital = 1_000_000.0
    
    # System 4: ML + ATR + BBL + TOCE (Proposed Framework Core)
    sys4_results = run_logged_backtest(
        all_assets, config,
        enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=False, enable_bbi_tp=False,
        system_name="System 4 (ML + ATR + BBL + TOCE)"
    )
    
    # System 5: Full ARMS Strategy (With Trailing Stop and BBI TP)
    sys5_results = run_logged_backtest(
        all_assets, config,
        enable_atr=True, enable_bbl=True, enable_toce=True, enable_trailing=True, enable_bbi_tp=True,
        system_name="System 5 (Full ARMS Framework)"
    )

if __name__ == "__main__":
    main()
