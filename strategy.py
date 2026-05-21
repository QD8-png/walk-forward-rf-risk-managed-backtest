# -*- coding: utf-8 -*-
"""
低买高卖量化回测系统
策略: 随机森林(Walk-Forward滚动训练) + 多频均线风控 + KDJ情绪 + 智能仓位 + 蒙特卡洛验证
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 策略超参数配置（集中管理，便于调参）
# ==========================================
@dataclass
class StrategyConfig:
    """策略全局超参数，集中管理避免硬编码散落各处"""
    # Walk-Forward 参数
    train_window: int = 500           # 滚动训练窗口大小（交易日）
    retrain_every: int = 60           # 重新训练间隔（交易日）

    # 随机森林参数
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 10
    random_state: int = 42

    # 回测参数
    initial_capital: float = 100000
    fee_rate: float = 0.0013          # 单边交易手续费率
    bbi_dev_threshold: float = 0.03   # BBI止盈偏离阈值
    entry_stop_ticks: float = 0.05         # 入场止损：买入当天最低价下方5个价位(0.05元)
    big_bull_threshold: float = 0.02  # 大阳线判定阈值（日涨幅>2%）
    cooldown_days: int = 120           # 冷却期：离场后120个交易日内不再入场
    trailing_stop_pct: float = 0.05    # 移动止盈：从最高价回撤5%触发清仓
    trailing_activate_pct: float = 0.03   # 移动止盈激活门槛：浮盈≥3%后启用
    min_remaining_position: float = 0.05  # 阶梯减仓最低仓位：低于此不再减仓

    # 仓位管理参数
    min_position_size: float = 0.3    # 最低仓位比例
    max_position_size: float = 1.0    # 最高仓位比例
    position_scale_factor: float = 3.33  # 仓位缩放因子

    # KDJ 参数
    kdj_period: int = 9
    kdj_panic_threshold: float = 20   # KDJ J值恐慌阈值（低于此值才允许入场）

    # BBI 均线周期
    bbi_periods: Tuple[int, ...] = (3, 6, 12, 24)

    # 多空线均线周期
    bb_periods: Tuple[int, ...] = (14, 28, 57, 114)

    # 蒙特卡洛参数
    n_shuffles: int = 200

    # 数据最低条数要求
    min_data_length: int = 600

    # 目标标签参数
    future_return_days: int = 5       # 未来N日收益
    future_return_threshold: float = 0.01  # 标签阈值


@dataclass
class BacktestData:
    """回测所需的行情与指标数据，打包传递避免散装参数"""
    dates: pd.Series
    close: pd.Series
    open_price: pd.Series   # 开盘价（用于检测大阳线）
    low: pd.Series          # 最低价（用于入场止损参考）
    bb_line: pd.Series      # 多空线
    bbi_line: pd.Series     # BBI线
    kdj_j: pd.Series        # KDJ J值
    ma120_slope: pd.Series  # MA120斜率

    @staticmethod
    def from_dataframe(df: pd.DataFrame, start_idx: int) -> 'BacktestData':
        """从完整 DataFrame 中按 start_idx 切片，一行构造"""
        sliced = df.iloc[start_idx:].reset_index(drop=True)
        return BacktestData(
            dates=sliced['日期'],
            close=sliced['收盘'],
            open_price=sliced['开盘'],
            low=sliced['低'],
            bb_line=sliced['多空线'],
            bbi_line=sliced['BBI'],
            kdj_j=sliced['KDJ_J'],
            ma120_slope=sliced['MA120_slope'],
        )


# 默认特征列
DEFAULT_FEATURE_COLS: List[str] = [
    '收益率_lag1', '收益率_lag2', '收益率_lag3',
    'RSI_14', 'KDJ_J',
    'MA5_偏离', 'MA10_偏离', 'MA均线交叉',
    '成交量变化', '振幅', 'Volatility',
    '趋势线偏离', 'MA60_偏离', 'EMA13_偏离', '多空线偏离', 'BBI偏离'
]


# ==========================================
# 1. 数据准备与特征工程
# ==========================================
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder EWM RSI

    Args:
        series: 价格序列
        period: RSI周期

    Returns:
        RSI值序列
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def prepare_features(source: str, feature_cols: List[str],
                     config: StrategyConfig) -> pd.DataFrame:
    """读取CSV或通过yfinance API在线获取数据，构建全部技术特征

    Args:
        source: CSV文件路径或股票代码（如 "512000.SS"）
        feature_cols: 特征列名列表
        config: 策略配置参数

    Returns:
        包含特征和标签的DataFrame

    Raises:
        ValueError: 数据缺失必要列或文件编码无法解析
        ConnectionError: yfinance API调用失败
    """
    if os.path.isfile(source):
        # 本地CSV模式 — 尝试多种编码，全部失败时抛出明确异常
        df = None
        encodings = ['utf-8', 'gbk', 'utf-8-sig', 'gb18030']
        for enc in encodings:
            try:
                df = pd.read_csv(source, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(
                f"无法解码文件 '{source}'，已尝试编码: {encodings}")
    else:
        # API在线模式：source为股票代码如 "512000.SS"
        try:
            ticker = yf.Ticker(source)
            df = ticker.history(period='max').reset_index()
        except Exception as e:
            raise ConnectionError(
                f"yfinance API获取 '{source}' 数据失败: {e}") from e

        if df.empty:
            raise ValueError(
                f"yfinance返回空数据，请检查代码 '{source}' 是否正确")
        df.rename(columns={'Date': '日期'}, inplace=True)

    # 统一列名
    col_map = {
        'Date': '日期', 'Open': '开盘', 'High': '高', 'Low': '低',
        'Close': '收盘', 'Volume': '交易量',
        'date': '日期', 'open': '开盘', 'high': '高', 'low': '低',
        'close': '收盘', 'volume': '交易量',
    }
    df.rename(columns=col_map, inplace=True)

    for col in ['开盘', '收盘', '高', '低', '交易量']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', ''), errors='coerce')

    df = df.dropna(subset=['收盘']).reset_index(drop=True)

    # --- 基础指标 ---
    df['收益率'] = df['收盘'].pct_change()

    if 'MA5' not in df.columns:
        df['MA5'] = df['收盘'].rolling(window=5).mean()
    if 'MA10' not in df.columns:
        df['MA10'] = df['收盘'].rolling(window=10).mean()
    if '交易量' not in df.columns:
        df['交易量'] = 0

    if ('Volatility' not in df.columns) or (df['Volatility'].isna().all()):
        df['Volatility'] = df['收益率'].rolling(window=20).std()

    # --- 低买高卖目标 ---
    # 如果未来N个交易日的收盘价比今天高出阈值以上，说明今天是"低位买点"
    df['未来5日收益'] = (
        df['收盘'].shift(-config.future_return_days) / df['收盘'] - 1
    )
    df['Target_方向'] = (
        df['未来5日收益'] > config.future_return_threshold
    ).astype(float)
    df.loc[df['未来5日收益'].isna(), 'Target_方向'] = np.nan

    df['RSI_14'] = _compute_rsi(df['收盘'], 14)

    df['MA5_偏离'] = df['收盘'] / df['MA5'] - 1
    df['MA10_偏离'] = df['收盘'] / df['MA10'] - 1
    df['MA均线交叉'] = df['MA5'] / df['MA10'] - 1

    df['成交量变化'] = df['交易量'].pct_change()
    df['振幅'] = (df['高'] - df['低']) / df['收盘']

    # 动量特征（全部使用历史数据，避免数据泄露）
    df['收益率_lag1'] = df['收益率'].shift(1)
    df['收益率_lag2'] = df['收益率'].shift(2)
    df['收益率_lag3'] = df['收益率'].shift(3)

    # KDJ指标（情绪极端值探测器）
    kdj_n = config.kdj_period
    low_n = df['低'].rolling(window=kdj_n).min()
    high_n = df['高'].rolling(window=kdj_n).max()
    rsv = (df['收盘'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    df['KDJ_K'] = rsv.ewm(com=2, adjust=False).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2, adjust=False).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    # 引入趋势指标
    ema10 = df['收盘'].ewm(span=10, adjust=False).mean()
    df['短期趋势线'] = ema10.ewm(span=10, adjust=False).mean()
    df['趋势线偏离'] = df['收盘'] / df['短期趋势线'] - 1

    df['MA60'] = df['收盘'].rolling(window=60).mean()
    df['EMA13'] = df['收盘'].ewm(span=13, adjust=False).mean()
    df['MA60_偏离'] = df['收盘'] / df['MA60'] - 1
    df['EMA13_偏离'] = df['收盘'] / df['EMA13'] - 1

    df['MA120'] = df['收盘'].rolling(window=120).mean()
    df['MA120_slope'] = (
        (df['MA120'] - df['MA120'].shift(20)) / df['MA120'].shift(20)
    )

    M1, M2, M3, M4 = config.bb_periods
    df['多空线'] = (
        df['收盘'].rolling(M1).mean() + df['收盘'].rolling(M2).mean()
        + df['收盘'].rolling(M3).mean() + df['收盘'].rolling(M4).mean()
    ) / 4
    df['多空线偏离'] = df['收盘'] / df['多空线'] - 1

    # BBI多空指标（Bull Bear Index）：中频均线系统，用于动态止盈"放飞"基准
    b1, b2, b3, b4 = config.bbi_periods
    df['BBI'] = (
        df['收盘'].rolling(b1).mean() + df['收盘'].rolling(b2).mean()
        + df['收盘'].rolling(b3).mean() + df['收盘'].rolling(b4).mean()
    ) / 4
    df['BBI偏离'] = df['收盘'] / df['BBI'] - 1


    # 只需要保证我们用到的特征和关键列不为空
    required_cols = (
        feature_cols
        + ['Target_方向', '多空线', 'BBI', '收盘', 'KDJ_J', 'MA120_slope']
    )

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失必要的数据列: {missing_cols}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


# ==========================================
# 2. Walk-Forward 滚动训练模块
# ==========================================
def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: StrategyConfig,
) -> Tuple[np.ndarray, np.ndarray, int, pd.Series]:
    """
    Walk-Forward滚动训练：每隔 retrain_every 天用最近 train_window 天数据重新训练
    避免"一次训练吃一辈子"的模型过时风险

    Args:
        df: 含特征和标签的DataFrame
        feature_cols: 特征列名列表
        config: 策略配置参数

    Returns:
        (predictions, probabilities, first_valid_index, feature_importances)

    Raises:
        ValueError: 训练数据不足以完成任何一轮训练
    """
    train_window = config.train_window
    retrain_every = config.retrain_every

    X = df[feature_cols].values
    y = df['Target_方向'].values.astype(int)
    n = len(df)

    predictions = np.full(n, -1, dtype=int)
    probabilities = np.full(n, 0.5)
    all_y_test: List[int] = []
    all_y_pred: List[int] = []
    all_importances: List[np.ndarray] = []  # 累积每轮特征重要性
    rf = None

    for train_end in range(train_window, n, retrain_every):
        test_end = min(train_end + retrain_every, n)
        X_train = X[train_end - train_window:train_end]
        y_train = y[train_end - train_window:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        if len(X_test) == 0:
            continue

        rf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)
        prob_col = (
            probs[:, 1] if probs.shape[1] > 1
            else np.full(len(X_test), 0.5)
        )

        predictions[train_end:test_end] = preds
        probabilities[train_end:test_end] = prob_col
        all_y_test.extend(y_test)
        all_y_pred.extend(preds)

        all_importances.append(rf.feature_importances_)

    first_valid = train_window

    if not all_importances:
        raise ValueError(
            "训练数据不足，无法完成任何一轮Walk-Forward训练"
        )

    # 各轮特征重要性取均值，而非只用最后一轮，更能代表整体
    mean_importances = np.mean(all_importances, axis=0)
    importances = pd.Series(
        mean_importances, index=feature_cols
    ).sort_values(ascending=True)

    acc = accuracy_score(all_y_test, all_y_pred) if all_y_test else 0
    n_rounds = len(range(train_window, n, retrain_every))
    print(f"  Walk-Forward 滚动训练: {n_rounds} 轮 "
          f"(窗口{train_window}天, 步长{retrain_every}天)")
    print(f"  滚动样本外准确率: {acc * 100:.2f}%")

    return predictions[first_valid:], probabilities[first_valid:], first_valid, importances


# ==========================================
# 3. 回测决策辅助函数（从主循环中提取，职责单一）
# ==========================================
def _should_buy(
    close: float, bb_line: float, kdj_j: float, slope: float,
    y_pred_i: int, day_idx: int, cooldown_until: int,
    config: StrategyConfig,
) -> bool:
    """入场条件检查：五重过滤，只做最漂亮的图形

    过滤逻辑:
        1. 模型预测看多 (y_pred == 1)
        2. 大趋势向上 (MA120斜率 > 0)
        3. 情绪恐慌 (KDJ J < 阈值)
        4. 强势区域 (收盘价 >= 多空线)
        5. 冷却期已过 (距上次离场 >= 120天)
    """
    if y_pred_i != 1:
        return False
    if slope <= 0:
        return False
    if kdj_j >= config.kdj_panic_threshold:
        return False
    if close < bb_line:
        return False
    if day_idx < cooldown_until:
        return False
    return True


def _check_exit(
    close: float, entry_price: float, entry_stop_price: float,
    bb_line: float, max_price: float, y_pred_i: int,
    config: StrategyConfig,
) -> Tuple[bool, str]:
    """离场条件检查 — 分阶段混合止盈（不含BBI阶梯减仓）

    设计理念（钓鱼策略）:
        小鱼直接抓 — 浮盈 < trailing_activate_pct 时，模型转空即落袋为安
        大鱼放线溜 — 浮盈 ≥ trailing_activate_pct 后，切换为移动止盈放飞利润

    优先级从高到低:
        1. 跌破多空线 → 强制止损 ('bb_stop')
        2. 跌破入场止损价 → 入场止损 ('entry_stop')
        3. 浮盈 < 激活门槛 且模型转空 → 小鱼落袋 ('model_tp')
        4. 浮盈 ≥ 激活门槛 且从最高价回撤超限 → 大鱼放飞 ('trailing_tp')

    Returns:
        (should_exit, reason): reason 为 'bb_stop' / 'entry_stop' / 'model_tp' / 'trailing_tp' / ''
    """
    # --- 防守层：无条件止损 ---
    if close < bb_line:
        return True, 'bb_stop'
    if close < entry_stop_price:
        return True, 'entry_stop'

    unrealized = close / entry_price - 1

    # --- 进攻层：分阶段止盈 ---
    if unrealized < config.trailing_activate_pct:
        # 小鱼阶段：模型转空且有浮盈 → 立即落袋为安
        if y_pred_i == 0 and unrealized > 0:
            return True, 'model_tp'
    else:
        # 大鱼阶段：从持仓最高价回撤超过阈值 → 移动止盈清仓
        if max_price > entry_price and close < max_price * (1 - config.trailing_stop_pct):
            return True, 'trailing_tp'

    return False, ''


# ==========================================
# 4. 回测模块 (智能仓位 + 五重风控)
# ==========================================
def backtest_with_stoploss(
    data: BacktestData,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    config: StrategyConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    低买高卖回测引擎（多空线止损 + 入场止损 + 分阶段止盈 + BBI阶梯减仓 + 智能仓位）

    仓位管理：根据模型置信度(predict_proba)动态调整仓位
      - 高置信度(>70%) → 重仓; 低置信度(~50%) → 轻仓(30%)

    资金曲线说明：capital_history 记录的是每日盘后结算后、调仓执行前的资金净值。
    交易手续费在调仓时即时扣除，体现在下一日的资金记录中。

    Args:
        data: 打包的行情与指标数据
        y_pred: 模型预测信号（0/1）
        y_prob: 模型预测概率
        config: 策略配置参数
        verbose: 是否打印回测结果

    Returns:
        包含 capital_history, total_return, sharpe, max_drawdown, trades 的字典
    """
    capital = config.initial_capital
    position = 0
    capital_history: List[float] = []
    entry_price = 0.0
    entry_stop_price = 0.0          # 入场止损价位（买入当天最低价 - 5个价位）
    max_price_since_entry = 0.0     # 持仓期间最高收盘价（用于移动止盈）
    current_position_size = 0.0
    cooldown_until = 0                  # 冷却期截止日索引

    trades = 0
    stoploss_count = 0
    entry_stop_count = 0
    model_tp_count = 0            # 模型止盈次数（小鱼落袋）
    trailing_tp_count = 0         # 移动止盈次数（大鱼放飞）
    takeprofit_count = 0          # BBI阶梯减仓次数

    close_vals = data.close.values
    open_vals = data.open_price.values
    low_vals = data.low.values
    bb_vals = data.bb_line.values
    bbi_vals = data.bbi_line.values
    kdj_vals = data.kdj_j.values
    slope_vals = data.ma120_slope.values

    for i in range(len(close_vals)):
        current_close = close_vals[i]
        current_bb_line = bb_vals[i]
        current_kdj_j = kdj_vals[i]
        current_slope = slope_vals[i]

        # === 步骤1: 结算持仓的每日盈亏（按仓位比例） ===
        if position == 1 and i > 0:
            daily_return = current_close / close_vals[i - 1] - 1
            capital = capital * (1 + daily_return * current_position_size)
            # 更新持仓最高价（用于移动止盈）
            if current_close > max_price_since_entry:
                max_price_since_entry = current_close
        capital_history.append(capital)

        # === 步骤2: 低买高卖决策引擎 ===
        target_position = position  # 默认保持当前仓位，防御性初始化

        if position == 0:
            if _should_buy(current_close, current_bb_line, current_kdj_j,
                           current_slope, y_pred[i], i, cooldown_until, config):
                target_position = 1
            else:
                target_position = 0
        else:
            should_exit, reason = _check_exit(
                current_close, entry_price, entry_stop_price,
                current_bb_line, max_price_since_entry, y_pred[i],
                config)

            if should_exit:
                target_position = 0
                if reason == 'bb_stop':
                    stoploss_count += 1
                elif reason == 'entry_stop':
                    entry_stop_count += 1
                elif reason == 'model_tp':
                    model_tp_count += 1
                elif reason == 'trailing_tp':
                    trailing_tp_count += 1

            # --- 持有区：检查是否触发BBI阶梯减仓（可多次触发） ---
            else:
                target_position = 1
                # BBI上方 + 大阳线 + 仓位仍高于最低门槛 → 减仓一半
                bbi_deviation = current_close / bbi_vals[i] - 1
                is_big_bull = (
                    (current_close / open_vals[i] - 1) >= config.big_bull_threshold
                )
                if (bbi_deviation >= config.bbi_dev_threshold
                        and is_big_bull
                        and current_position_size > config.min_remaining_position):
                    # 阶梯减仓：卖出当前仓位的一半
                    sell_size = current_position_size / 2
                    capital *= (1 - config.fee_rate * sell_size)
                    current_position_size /= 2
                    takeprofit_count += 1

        # === 步骤3: 执行调仓（含智能仓位） ===
        if target_position != position:
            if target_position == 1 and position == 0:
                position = 1
                entry_price = current_close
                entry_stop_price = low_vals[i] - config.entry_stop_ticks
                max_price_since_entry = current_close  # 初始化最高价
                # 仓位 = clamp(min_pos, (prob - 0.5) * scale, max_pos)
                # prob=0.5 → 0.3(轻仓); prob=0.8 → 1.0(满仓)
                current_position_size = min(
                    config.max_position_size,
                    max(config.min_position_size,
                        (y_prob[i] - 0.5) * config.position_scale_factor),
                )
                capital *= (1 - config.fee_rate * current_position_size)
                trades += 1
            elif target_position == 0 and position == 1:
                position = 0
                capital *= (1 - config.fee_rate * current_position_size)
                entry_price = 0.0
                max_price_since_entry = 0.0
                current_position_size = 0.0
                cooldown_until = i + config.cooldown_days
                trades += 1

    total_return = capital / config.initial_capital - 1
    buy_hold_return = close_vals[-1] / close_vals[0] - 1

    curve_series = pd.Series(capital_history)
    daily_returns = curve_series.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if len(daily_returns) > 0 and daily_returns.std() != 0
        else 0
    )

    cum_max = curve_series.cummax()
    drawdown = (curve_series - cum_max) / cum_max
    max_drawdown = drawdown.min()

    if verbose:
        print(f"  策略总收益:   {total_return * 100:.2f}%")
        print(f"  买入持有收益: {buy_hold_return * 100:.2f}%")
        print(f"  夏普比率:     {sharpe:.4f}")
        print(f"  最大回撤:     {max_drawdown * 100:.2f}%")
        print(f"  总交易次数:   {trades}")
        print(f"  模型止盈(小鱼): {model_tp_count}")
        print(f"  移动止盈(大鱼): {trailing_tp_count}")
        print(f"  BBI阶梯减仓:  {takeprofit_count}")
        print(f"  入场止损次数:  {entry_stop_count}")
        print(f"  多空线止损:    {stoploss_count}")
        print(f"  冷却期:       {config.cooldown_days}天")

    return {
        'capital_history': capital_history,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': trades,
    }


# ==========================================
# 4. 蒙特卡洛统计验证
# ==========================================
def statistical_validation(
    data: BacktestData,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    config: StrategyConfig,
) -> None:
    """蒙特卡洛置换检验：随机打乱预测信号N次，验证策略收益是否显著优于随机

    零假设 (H0):
        模型预测信号对策略收益无增量贡献，即策略收益完全来自风控规则
        （KDJ过滤、多空线止损、BBI止盈、MA120趋势过滤）而非ML预测信号。

    检验方法:
        仅打乱 y_pred 和 y_prob 的时序对应关系，保留风控逻辑原序
        （KDJ、多空线、MA120斜率等不打乱），从而验证的是
        "ML预测信号相对于随机信号的增量价值"。

    p值解读:
        在 n_shuffles 次随机排列中，有多少比例的随机收益 >= 策略真实收益。
        p < 0.05 表示策略预测信号显著优于随机。

    注意:
        使用固定种子 (seed=42) 的 numpy.random.Generator 以保证结果可复现。
    """
    actual = backtest_with_stoploss(data, y_pred, y_prob, config=config, verbose=False)
    actual_return = actual['total_return']

    random_returns: List[float] = []
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)
    rng = np.random.default_rng(seed=config.random_state)

    for _ in range(config.n_shuffles):
        idx = rng.permutation(len(y_pred_arr))
        result = backtest_with_stoploss(
            data, y_pred_arr[idx], y_prob_arr[idx],
            config=config, verbose=False,
        )
        random_returns.append(result['total_return'])

    random_returns_arr = np.array(random_returns)
    p_value = np.mean(random_returns_arr >= actual_return)

    print(f"\n--- 统计验证 (蒙特卡洛置换检验, n={config.n_shuffles}) ---")
    print(f"  策略真实收益: {actual_return * 100:.2f}%")
    print(f"  随机信号平均: {random_returns_arr.mean() * 100:.2f}%")
    print(f"  p值: {p_value:.4f} "
          f"{'[显著 p<0.05]' if p_value < 0.05 else '[不显著]'}")


# ==========================================
# 5. 可视化模块（从main中提取，职责分离）
# ==========================================
def plot_results(
    asset_name: str,
    data: BacktestData,
    capital_curve: List[float],
    importances: pd.Series,
    save_path: str = None,
) -> None:
    """绘制策略净值曲线（双轴）与特征重要性图 (高颜值 TradingView 暗黑风格)

    Args:
        asset_name: 标的名称
        data: 打包的行情数据
        capital_curve: 策略资金曲线
        importances: 特征重要性 Series
        save_path: 图片保存路径，若不为 None 则保存到指定文件
    """
    # matplotlib.pyplot 和 numpy 已在文件头部 import，此处无需重复

    # 设置暗黑风格的全局属性
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 调色盘定义 (TradingView 风格)
    bg_color = '#0C1017'       # 深蓝黑底色
    card_color = '#161B22'     # 卡片深灰底色
    grid_color = '#21262D'     # 网格色
    text_color = '#C9D1D9'     # 主体文字淡灰色
    muted_text = '#8B949E'     # 辅助文字灰色
    
    strategy_color = '#00F2FE'  # 策略线颜色：极光青
    benchmark_color = '#FF4D6D' # 基准线颜色：极光红
    
    dates_test = data.dates
    close_test = data.close
    strategy_norm = np.array(capital_curve) / capital_curve[0]
    buy_hold_norm = close_test.values / close_test.values[0]

    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=bg_color)
    
    # ==================== 1. 净值曲线图 ====================
    ax1 = axes[0]
    ax1.set_facecolor(bg_color)
    
    # 隐藏四周无用的边框
    for spine in ax1.spines.values():
        spine.set_visible(False)
        
    # 双轴设置
    ax2 = ax1.twinx()
    for spine in ax2.spines.values():
        spine.set_visible(False)
        
    # 绘制基准收益线 (右轴，极光红，虚线或半透明细线)
    line2 = ax2.plot(
        dates_test.values, buy_hold_norm,
        label='买入持有基准 (右轴)', linewidth=1.5, color=benchmark_color, alpha=0.55, linestyle='--'
    )
    # 基准下方填充半透明淡红色
    ax2.fill_between(
        dates_test.values, buy_hold_norm, 1.0,
        where=(buy_hold_norm >= 1.0), facecolor=benchmark_color, alpha=0.04
    )
    ax2.fill_between(
        dates_test.values, buy_hold_norm, 1.0,
        where=(buy_hold_norm < 1.0), facecolor=benchmark_color, alpha=0.04
    )
    
    # 绘制策略净值线 (左轴，极光青，粗实线)
    line1 = ax1.plot(
        dates_test.values[:len(strategy_norm)], strategy_norm,
        label='策略净值 (左轴)', linewidth=2.5, color=strategy_color
    )
    # 策略下方填充半透明青色渐变效果
    ax1.fill_between(
        dates_test.values[:len(strategy_norm)], strategy_norm, 1.0,
        where=(strategy_norm >= 1.0), facecolor=strategy_color, alpha=0.12
    )
    ax1.fill_between(
        dates_test.values[:len(strategy_norm)], strategy_norm, 1.0,
        where=(strategy_norm < 1.0), facecolor=strategy_color, alpha=0.06
    )

    # 刻度及标签样式优化
    ax1.set_title(f'{asset_name} 资金曲线对比 (TradingView风格)', fontsize=14, color='#F0F6FC', pad=20, weight='bold')
    ax1.set_ylabel('策略净值 (初始=1)', color=strategy_color, fontsize=12, labelpad=10)
    ax1.tick_params(colors=muted_text, labelsize=10)
    ax1.set_xlabel('日期', color=text_color, fontsize=11, labelpad=10)
    
    ax2.set_ylabel('基准净值 (倍数)', color=benchmark_color, fontsize=12, labelpad=10)
    ax2.tick_params(colors=muted_text, labelsize=10)
    
    # 网格线
    ax1.grid(True, which='both', color=grid_color, linestyle='-', linewidth=0.8, alpha=0.7)
    
    # 限制X轴展示刻度数以防拥挤
    x_len = len(dates_test)
    step = max(1, x_len // 6)
    ax1.set_xticks(dates_test.values[::step])
    ax1.set_xticklabels(dates_test.values[::step], rotation=15, ha='right', color=muted_text)
 
    # 合并图例，并移到更开阔的区域
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    leg = ax1.legend(lines, labels, loc='upper left', facecolor=card_color, edgecolor=grid_color, fontsize=10)
    for text in leg.get_texts():
        text.set_color(text_color)
        
    # 计算一些关键统计数据放在图表中展示
    strat_ret = (strategy_norm[-1] - 1) * 100
    bh_ret = (buy_hold_norm[-1] - 1) * 100
    
    # 添加好看的信息浮窗
    info_text = (
        f"策略核心指标:\n"
        f"  - 策略最终净值: {strategy_norm[-1]:.3f} ({strat_ret:+.2f}%)\n"
        f"  - 基准最终净值: {buy_hold_norm[-1]:.3f} ({bh_ret:+.2f}%)\n"
        f"  - 策略相对超额: {strat_ret - bh_ret:+.2f}%"
    )
    ax1.text(0.03, 0.72, info_text, transform=ax1.transAxes, fontsize=9.5, color=text_color,
             bbox=dict(boxstyle='round,pad=0.8', facecolor=card_color, edgecolor=grid_color, alpha=0.9))
 
    # ==================== 2. 特征重要性图 ====================
    ax3 = axes[1]
    ax3.set_facecolor(bg_color)
    for spine in ax3.spines.values():
        spine.set_visible(False)
        
    # 绘制水平条形图，并用渐变色彩
    y_pos = np.arange(len(importances))
    norm_importances = importances.values
    
    # 使用渐变颜色：从暗青色到亮青色
    colors = [plt.cm.cool(x) for x in np.linspace(0.3, 0.85, len(importances))]
    
    bars = ax3.barh(y_pos, norm_importances, align='center', color=colors, height=0.6, alpha=0.95)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(importances.index, fontsize=10, color=text_color)
    ax3.set_xlabel('相对重要性权重', color=text_color, fontsize=11, labelpad=10)
    ax3.set_title(f'随机森林多期滚动特征重要性', fontsize=14, color='#F0F6FC', pad=20, weight='bold')
    ax3.tick_params(colors=muted_text, labelsize=10)
    ax3.grid(True, axis='x', color=grid_color, linestyle='--', alpha=0.5)

    # 往条形图内部/外部加百分比文本标注
    max_val = max(norm_importances) if len(norm_importances) > 0 else 1.0
    for bar in bars:
        width = bar.get_width()
        ax3.text(
            width + max_val * 0.01, bar.get_y() + bar.get_height()/2,
            f'{width*100:.1f}%',
            va='center', ha='left', fontsize=9, color=strategy_color, weight='bold'
        )

    plt.tight_layout()
    
    # 是否保存
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, facecolor=bg_color, edgecolor='none', dpi=150)
        print(f"  [SAVE] 净值图已保存至: {save_path}")
        
    plt.show()


# ==========================================
# 6. 主程序：批量测试框架
# ==========================================
def main() -> None:
    config = StrategyConfig()

    test_assets: Dict[str, str] = {
        # --- AI 算力 ---
        "浪潮信息 (000977)": "000977.SZ",
        "中科曙光 (603019)": "603019.SS",
        # --- 人形机器人 ---
        "拓普集团 (601689)": "601689.SS",
        "三花智控 (002050)": "002050.SZ",
        # --- 低空经济 ---
        "万丰奥威 (002085)": "002085.SZ",
        # --- 固态电池 / 新能源 ---
        "宁德时代 (300750)": "300750.SZ",
        # --- 华为产业链 ---
        "赛力斯 (601127)": "601127.SS",
        # --- 军工 / 卫星互联网 ---
        "中国卫星 (600118)": "600118.SS",
    }

    feature_cols = DEFAULT_FEATURE_COLS

    print("=" * 60)
    print("低买高卖量化回测 (Walk-Forward + 智能仓位 + 统计验证)")
    print("=" * 60)

    for asset_name, filepath in test_assets.items():
        print(f"\n>>> 正在测试标的: {asset_name}")

        try:
            df = prepare_features(filepath, feature_cols, config)
        except Exception as e:
            print(f"  数据处理失败: {e}")
            continue

        print(f"  有效数据量: {len(df)} 条")
        if len(df) < config.min_data_length:
            print(f"  数据量不足{config.min_data_length}条，跳过Walk-Forward")
            continue

        # 1. Walk-Forward 滚动训练
        print("\n--- 模型评估 (Walk-Forward) ---")
        y_pred, y_prob, start_idx, importances = walk_forward_predict(
            df, feature_cols, config)

        # 一行构造回测数据包，替代原来6行重复切片
        bt_data = BacktestData.from_dataframe(df, start_idx)

        # 2. 回测（智能仓位 + 四重风控）
        print("\n--- 回测表现 ---")
        result = backtest_with_stoploss(bt_data, y_pred, y_prob, config=config)

        # 3. 蒙特卡洛统计验证
        statistical_validation(bt_data, y_pred, y_prob, config=config)

        # 4. 绘图
        plot_results(asset_name, bt_data, result['capital_history'], importances)


if __name__ == "__main__":
    main()
