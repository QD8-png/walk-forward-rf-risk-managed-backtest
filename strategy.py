# -*- coding: utf-8 -*-
"""
================================================================================
Walk-Forward Random Forest Trading Strategy with Multi-Layer Risk Regimes
================================================================================
An institutional-grade quantitative backtesting framework combining rolling
non-linear machine learning forecasts with multi-frequency structural risk controls,
portfolio-level shared-capital rotation, and Monte Carlo statistical validation.

Core Philosophy: "Predict with Machine Learning, Control with Structural Rules."
"""

import os
import copy
import warnings
import multiprocessing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# Ignore harmless warnings
warnings.filterwarnings("ignore")

# Matplotlib dark UI settings
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 0. Global Parameters & Config Configuration
# ==============================================================================
@dataclass
class StrategyConfig:
    """Strategy hyperparameters and risk limits, centralized to avoid hardcoding."""
    # Walk-Forward Modeling
    train_window: int = 500             # Rolling history training window (days)
    retrain_every: int = 60             # Model retraining frequency (days)

    # Random Forest Configuration
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 10
    random_state: int = 42

    # Transaction Costs
    fee_rate: float = 0.0013            # Round-trip commission + stamp + slippage (13 bps)

    # Risk Management & Exit Regimes
    ma120_slope_lookback: int = 20      # Days to compute the slope of MA120
    bbi_dev_threshold: float = 0.03     # Profit ladder BBI deviation (3%)
    big_bull_threshold: float = 0.02    # Daily return required for BBI profit ladder (2%)
    cooldown_days: int = 120            # Asset cooling period post-liquidation (days)
    
    # ATR Volatility-Based Stop-Loss
    atr_period: int = 14                # ATR calculation lookback window
    atr_multiplier: float = 2.0         # Stop = Low_entry - atr_multiplier × ATR_entry
    
    # Trailing Profit-Take (TP)
    trailing_activate_pct: float = 0.03 # floating profit trigger for trailing stop (3%)
    trailing_stop_pct: float = 0.05      # Max drawdown from peak before liquidating (5%)
    min_remaining_position: float = 0.05# BBI profit ladder minimum residual weight (5%)

    # Position Sizing
    min_position_size: float = 0.3      # Minimum conviction leverage multiplier
    max_position_size: float = 1.0      # Maximum conviction leverage multiplier
    position_scale_factor: float = 3.33 # Sizer scaling slope

    # Indicators Parameter
    kdj_period: int = 9
    kdj_panic_threshold: float = 20     # KDJ_J threshold representing panic oversold
    bbi_periods: Tuple[int, ...] = (3, 6, 12, 24)
    bb_periods: Tuple[int, ...] = (14, 28, 57, 114) # Bollinger-like MA bands
    min_data_length: int = 600

    # ML Labeling Target
    future_return_days: int = 5
    future_return_threshold: float = 0.01

    # Portfolio Sizing
    portfolio_capital: float = 1_000_000.0
    max_weight_per_stock: float = 0.25  # Limit single asset allocation to 25%
    max_holdings: int = 4               # Absolute max parallel holdings

    # Statistical Validation
    n_shuffles: int = 50                # Monte Carlo shuffles
    risk_free_rate: float = 0.02


DEFAULT_FEATURE_COLS: List[str] = [
    '收益率_lag1', '收益率_lag2', '收益率_lag3',
    'RSI_14', 'KDJ_J',
    'MA5_偏离', 'MA10_偏离', 'MA均线交叉',
    '成交量变化', '振幅', 'Volatility',
    '趋势线偏离', 'MA60_偏离', 'EMA13_偏离', '多空线偏离', 'BBI偏离'
]


# ==============================================================================
# 1. Feature Engineering
# ==============================================================================
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilders EWM Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def prepare_features(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Builds technical indicator features and target labels."""
    df = df.copy()
    col_map = {
        'Date': '日期', 'Open': '开盘', 'High': '高', 'Low': '低',
        'Close': '收盘', 'Volume': '交易量'
    }
    df.rename(columns=col_map, inplace=True)
    df = df.dropna(subset=['收盘']).reset_index(drop=True)

    # 1. Standard quant elements
    df['收益率'] = df['收盘'].pct_change()
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA10'] = df['收盘'].rolling(window=10).mean()
    df['Volatility'] = df['收益率'].rolling(window=20).std()

    # Label target setting (future N-day returns > threshold)
    df['未来5日收益'] = df['收盘'].shift(-config.future_return_days) / df['收盘'] - 1
    df['Target_方向'] = (df['未来5日收益'] > config.future_return_threshold).astype(float)
    df.loc[df['未来5日收益'].isna(), 'Target_方向'] = np.nan

    # 2. Advanced technical overlays
    df['RSI_14'] = _compute_rsi(df['收盘'], 14)
    df['MA5_偏离'] = df['收盘'] / df['MA5'] - 1
    df['MA10_偏离'] = df['收盘'] / df['MA10'] - 1
    df['MA均线交叉'] = df['MA5'] / df['MA10'] - 1
    df['成交量变化'] = df['交易量'].pct_change()
    df['振幅'] = (df['高'] - df['低']) / df['收盘']

    # Lags
    df['收益率_lag1'] = df['收益率'].shift(1)
    df['收益率_lag2'] = df['收益率'].shift(2)
    df['收益率_lag3'] = df['收益率'].shift(3)

    # KDJ Oscillator
    kdj_n = config.kdj_period
    low_n = df['低'].rolling(window=kdj_n).min()
    high_n = df['高'].rolling(window=kdj_n).max()
    rsv = (df['收盘'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    df['KDJ_K'] = rsv.ewm(com=2, adjust=False).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2, adjust=False).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    # Trend Overlays
    ema10 = df['收盘'].ewm(span=10, adjust=False).mean()
    df['短期趋势线'] = ema10.ewm(span=10, adjust=False).mean()
    df['趋势线偏离'] = df['收盘'] / df['短期趋势线'] - 1

    df['MA60'] = df['收盘'].rolling(window=60).mean()
    df['EMA13'] = df['收盘'].ewm(span=13, adjust=False).mean()
    df['MA60_偏离'] = df['收盘'] / df['MA60'] - 1
    df['EMA13_偏离'] = df['收盘'] / df['EMA13'] - 1

    df['MA120'] = df['收盘'].rolling(window=120).mean()
    df['MA120_slope'] = (df['MA120'] - df['MA120'].shift(config.ma120_slope_lookback)) / df['MA120'].shift(config.ma120_slope_lookback)

    # Bull Bear Line & BBI
    M1, M2, M3, M4 = config.bb_periods
    df['多空线'] = (df['收盘'].rolling(M1).mean() + df['收盘'].rolling(M2).mean()
                  + df['收盘'].rolling(M3).mean() + df['收盘'].rolling(M4).mean()) / 4
    df['多空线偏离'] = df['收盘'] / df['多空线'] - 1

    b1, b2, b3, b4 = config.bbi_periods
    df['BBI'] = (df['收盘'].rolling(b1).mean() + df['收盘'].rolling(b2).mean()
                + df['收盘'].rolling(b3).mean() + df['收盘'].rolling(b4).mean()) / 4
    df['BBI偏离'] = df['收盘'] / df['BBI'] - 1

    # ATR (Average True Range) for dynamic volatility-based stop-loss
    prev_close = df['收盘'].shift(1)
    tr1 = df['高'] - df['低']
    tr2 = (df['高'] - prev_close).abs()
    tr3 = (df['低'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(span=config.atr_period, adjust=False).mean()

    # Cleanup invalid / missing data rows
    required_cols = DEFAULT_FEATURE_COLS + ['多空线', 'BBI', '收盘', 'KDJ_J', 'MA120_slope', 'ATR']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


# ==============================================================================
# 2. Walk-Forward Predictive Engine
# ==============================================================================
def walk_forward_predict(df: pd.DataFrame, config: StrategyConfig, model_type: str = 'rf') -> Tuple[np.ndarray, np.ndarray, int]:
    """Generates machine learning predictions using rolling walk-forward validation with baseline model support."""
    train_window = config.train_window
    retrain_every = config.retrain_every
    X = df[DEFAULT_FEATURE_COLS].values
    y = df['Target_方向'].values
    n = len(df)
    predictions = np.full(n, -1, dtype=int)
    probabilities = np.full(n, 0.5)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    for train_end in range(train_window, n, retrain_every):
        test_end = min(train_end + retrain_every, n)
        # Prevent look-ahead bias: subtract future_return_days from training window
        X_train_raw = X[train_end - train_window : train_end - config.future_return_days]
        y_train_raw = y[train_end - train_window : train_end - config.future_return_days]
        
        # Only keep rows in training set where target is not NaN
        valid_idx = ~np.isnan(y_train_raw)
        X_train = X_train_raw[valid_idx]
        y_train = y_train_raw[valid_idx].astype(int)
        
        X_test = X[train_end:test_end]

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        # Feature scaling for Logistic Regression
        if model_type == 'lr':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Model instantiation
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=config.n_estimators, max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf, random_state=config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_type == 'lgbm':
            model = lgb.LGBMClassifier(
                n_estimators=config.n_estimators, max_depth=config.max_depth,
                random_state=config.random_state, class_weight='balanced',
                n_jobs=-1, verbosity=-1
            )
        elif model_type == 'lr':
            model = LogisticRegression(
                penalty='l2', C=1.0, random_state=config.random_state,
                class_weight='balanced', max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
        prob_col = probs[:, 1] if probs.shape[1] > 1 else np.full(len(X_test), 0.5)

        predictions[train_end:test_end] = preds
        probabilities[train_end:test_end] = prob_col

    return predictions, probabilities, train_window


# Parallelized feature generation and machine learning worker
def process_single_asset(ticker: str, df: pd.DataFrame, config: StrategyConfig, model_type: str = 'rf') -> Optional[Dict[str, Any]]:
    """Helper method to run technical analysis and rolling training on single asset."""
    try:
        df = prepare_features(df, config)
    except Exception:
        return None

    if len(df) < config.min_data_length:
        return None

    if df['日期'].max() < pd.Timestamp('2024-01-01'):
        return None

    predictions, probabilities, first_valid = walk_forward_predict(df, config, model_type=model_type)
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


# ==============================================================================
# 3. Portfolio Multi-Asset shared Capital backtester
# ==============================================================================
@dataclass
class Holding:
    name: str
    entry_price: float
    entry_stop_price: float
    max_price: float
    position_weight: float


def portfolio_backtest_2024(all_assets: Dict[str, Dict[str, Any]], config: StrategyConfig, verbose=True) -> Tuple[float, float, List[float], List[pd.Timestamp]]:
    """Backtests a shared capital pool rotation portfolio over multi-asset universe."""
    capital = config.portfolio_capital
    holdings: Dict[str, Holding] = {}
    cooldowns: Dict[str, int] = {}
    trade_counts: Dict[str, int] = {}  # Force broader rotation: max 3 trades per stock
    
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
    start_date_2024 = pd.Timestamp('2024-01-01')

    for day_idx, today in enumerate(all_dates):
        if today < start_date_2024:
            continue
            
        # 1. Update mark-to-market daily returns and update peak prices
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
            # Layer B: Hard Adaptive Stop-loss (2% below buy-day LOW)
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
                if trade_counts.get(stock_name, 0) >= 3:
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
            trade_counts[stock_name] = trade_counts.get(stock_name, 0) + 1
            holdings[stock_name] = Holding(
                name=stock_name,
                entry_price=asset['close'][local_idx],
                # ATR volatility-based dynamic stop-loss: Stop = Low_entry - λ × ATR_entry
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

    if verbose:
        print("\n" + "=" * 50)
        print("2024至今 - 多资产海选轮动 - 组合回测报告")
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


# ==============================================================================
# 4. Statistical Validation via Monte Carlo
# ==============================================================================
def portfolio_monte_carlo(all_assets: Dict[str, Dict[str, Any]], config: StrategyConfig, actual_return: float):
    """Executes a Monte Carlo permutation test by shuffling model predictions to evaluate Alpha validity."""
    print("\n" + "=" * 50)
    print("正在启动组合级别蒙特卡洛置换检验...")
    print(f"正在进行 {config.n_shuffles} 次随机信号置换，模拟纯抛硬币的表现...")
    print("=" * 50)
    
    random_returns = []
    rng = np.random.default_rng(seed=config.random_state)
    
    for i in range(config.n_shuffles):
        # Deepcopy to avoid mutating source signals
        shuffled_assets = copy.deepcopy(all_assets)
        
        # Shuffle temporal mapping of AI predictions
        for stock_name, asset in shuffled_assets.items():
            n_len = len(asset['y_pred'])
            idx = rng.permutation(n_len)
            asset['y_pred'] = asset['y_pred'][idx]
            asset['y_prob'] = asset['y_prob'][idx]
            
        ret, _, _, _ = portfolio_backtest_2024(shuffled_assets, config, verbose=False)
        random_returns.append(ret)
        
        if (i + 1) % 10 == 0:
            print(f"  [Progress] 已完成随机模拟 {i + 1} / {config.n_shuffles} 次")
            
    random_returns_arr = np.array(random_returns)
    p_value = np.mean(random_returns_arr >= actual_return)
    
    print("\n" + "=" * 50)
    print("★ 蒙特卡洛统计学检验报告 ★")
    print("=" * 50)
    print(f"  策略真实实际收益: {actual_return * 100:.2f}%")
    print(f"  随机置换平均收益: {random_returns_arr.mean() * 100:.2f}%")
    print(f"  随机置换最大收益: {random_returns_arr.max() * 100:.2f}%")
    print(f"  随机置换最小收益: {random_returns_arr.min() * 100:.2f}%")
    print(f"  经验显著性 p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  结论: [ 极其显著 p < 0.05 ] 本策略超额收益以 >95% 的概率优于猴子丢飞镖，Alpha真实有效！")
    else:
        print("  结论: [ 统计学不显著 p >= 0.05 ] 无法拒绝随机假说，策略表现可能受市场红利或巧合主导。")
    print("=" * 50)


# ==============================================================================
# 5. Visualizer
# ==============================================================================
def plot_portfolio_equity(dates: List[pd.Timestamp], values: List[float], asset_count: int, save_path: str = None):
    """Draws premium TradingView-themed portfolio equity curves."""
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    dates_clean = pd.to_datetime(dates)
    norm_values = np.array(values) / values[0]
    
    # Dynamic styles
    bg_glow_color = '#00F2FE'
    ax.plot(dates_clean, norm_values, color=bg_glow_color, linewidth=2.5, label='策略资产组合净值 (V4自适应版)', alpha=0.95)
    ax.fill_between(dates_clean, norm_values, 1.0, where=(norm_values >= 1.0), facecolor=bg_glow_color, alpha=0.08)
    
    # Index Benchmark comparison estimation
    ax.axhline(y=1.0, color='#8B949E', linestyle='--', linewidth=0.8, alpha=0.6)
    
    ax.set_title(f'多资产海选轮动组合净值曲线 (2024至今 - 共 {asset_count} 只有效股票)', fontsize=14, color='#F0F6FC', pad=20, weight='bold')
    ax.set_ylabel('组合净值 (初始=1.00)', color='#F0F6FC', fontsize=11, labelpad=10)
    ax.set_xlabel('回测日期', color='#F0F6FC', fontsize=11, labelpad=10)
    ax.tick_params(colors='#8B949E', labelsize=9.5)
    ax.grid(True, which='both', color='#21262D', linestyle='-', linewidth=0.7, alpha=0.6)
    
    # Card details window placement
    info_text = (
        f"组合最终统计绩:\n"
        f"  - 最终净值: {norm_values[-1]:.3f}\n"
        f"  - 累计收益: {(norm_values[-1] - 1)*100:+.2f}%"
    )
    ax.text(0.03, 0.82, info_text, transform=ax.transAxes, fontsize=10, color='#F0F6FC',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#161B22', edgecolor='#30363D', alpha=0.9))
    
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.75), facecolor='#161B22', edgecolor='#30363D', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#F0F6FC')
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, facecolor='#0D1117', edgecolor='none', dpi=150)
        print(f"  [SAVE] 组合回测图表已成功保存: {save_path}")
    plt.show()


# ==============================================================================
# 6. Unified Execution Entry Point
# ==============================================================================
def get_default_50_tickers() -> List[str]:
    """Generates standard stock tickers for testing (mixture of high-beta and defensive)."""
    # Expanded stock selection of representative market sector giants
    return [
        "600519.SS", "300750.SZ", "600900.SS", "600036.SS", "002594.SZ",
        "000977.SZ", "603019.SS", "601127.SS", "002230.SZ", "002050.SZ",
        "510300.SS", "512000.SS", "512480.SS", "512170.SS"
    ]


def main() -> None:
    config = StrategyConfig()
    tickers = get_default_50_tickers()
    
    print("=" * 60)
    print("★ 启动 V4 最新合并版本：多资产共享资金池海选轮动系统 ★")
    print("=" * 60)
    print(f"1. 批量下载及构建 {len(tickers)} 只代表性标的技术特征...")
    
    # Single-thread safe fallback or multithreaded download
    data = yf.download(tickers, start="2020-01-01", group_by='ticker', threads=True, progress=False)
    
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
        except Exception:
            pass
            
    print(f"   下载完毕。包含可用足额历史的标的数: {len(valid_dfs)}")
    if len(valid_dfs) == 0:
        print("   错误: 没有可用标的数据，请检查网络！")
        return

    # Multiprocessing accelerated Walk-Forward RF engineering
    # 可在此处修改 model_type 来选用不同的机器学习底座: 'rf' (随机森林), 'lgbm' (LightGBM), 'lr' (逻辑回归)
    model_type = 'rf'
    print(f"\n2. 启动多进程并发加速：特征工程建模与 Walk-Forward 滚动训练 (模型类型: {model_type.upper()})...")
    all_assets = {}
    max_workers = min(multiprocessing.cpu_count(), len(valid_dfs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_asset, t, df, config, model_type): t for t, df in valid_dfs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                all_assets[res['name']] = res
            if i % 5 == 0 or i == len(valid_dfs):
                print(f"   已完成模组计算: {i} / {len(valid_dfs)}")
                
    print(f"\n   训练全部结束。共有 {len(all_assets)} 只标的在 2024-01-01 后具备完整预测信号，进入海选轮动池。")
    if len(all_assets) == 0:
        print("   没有符合预测时间线标准的标的，回测结束。")
        return

    # 3. Shared capital portfolio rotation backtest
    print("\n3. 启动 2024 至今多资产共享资金池组合回测 (包含 ATR 波动率自适应止损)...")
    actual_return, max_drawdown, values, dates = portfolio_backtest_2024(all_assets, config, verbose=True)
    
    # 4. Premium graph saving
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "v4_portfolio_equity.png")
    plot_portfolio_equity(dates, values, len(all_assets), save_path=plot_path)

    # 5. Monte Carlo validation
    portfolio_monte_carlo(all_assets, config, actual_return)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
