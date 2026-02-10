"""
Realistic Backtesting Module - Train Once, Test Separately

All functions here use the REALISTIC methodology:
1. Train model ONCE on training period (e.g., 2020-2022)
2. FREEZE the model
3. Test on all points in test period (e.g., 2023-2024)
4. NO retraining during test

This matches real-world deployment where you build a model,
deploy it, and it has to work without constant updates.
"""

from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from model import compute_technical_features, add_fundamental_features
from config import MIN_DATA_POINTS


# Default date ranges
DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_TEST_START = "2023-01-01"
DEFAULT_TEST_END = "2024-12-31"


def fetch_historical_data(ticker: str, start_date: str = "2019-01-01") -> Optional[pd.DataFrame]:
    """Fetch historical data for backtesting."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=datetime.now().strftime("%Y-%m-%d"), interval="1d")
        if df.empty or len(df) < 200:
            return None
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df[["open", "high", "low", "close", "volume"]]
    except:
        return None


def run_screener_backtest_realistic(
    stocks: List[str],
    days: int = 5,
    min_confidence: float = 0.75,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> Optional[Dict]:
    """
    REALISTIC screener backtest - train ONCE, test separately.

    Methodology:
    1. For each stock, train model on train_start to train_end
    2. Freeze the model
    3. Test on all non-overlapping periods from test_start to test_end
    4. Aggregate results across all stocks

    Returns accuracy, win rate, and risk metrics.
    """
    all_picks = []
    all_predictions = []
    stocks_tested = 0
    stocks_failed = 0

    # Fetch SPY for benchmark
    spy_df = fetch_historical_data("SPY", start_date="2019-01-01")
    if spy_df is not None:
        spy_df["forward_return"] = spy_df["close"].shift(-days) / spy_df["close"] - 1

    for ticker in stocks:
        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            stocks_failed += 1
            continue

        # Compute features
        features = compute_technical_features(df.reset_index())
        feature_cols = features.columns.tolist()

        # Target
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["close"] = df["close"]
        data = data.ffill().bfill()

        # Split by date
        train_mask = (data.index >= train_start) & (data.index <= train_end)
        test_mask = (data.index >= test_start) & (data.index <= test_end)

        train_data = data[train_mask].dropna()
        test_data = data[test_mask].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            stocks_failed += 1
            continue

        stocks_tested += 1

        # Train model ONCE
        X_train = train_data[feature_cols].replace([np.inf, -np.inf], 0)
        y_train = train_data["target"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Test on non-overlapping windows
        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            test_row = test_data.iloc[[idx]]
            X_test = test_row[feature_cols].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            pred = model.predict(X_test_scaled)[0]
            proba = model.predict_proba(X_test_scaled)[0]
            confidence = float(max(proba))

            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / test_data.iloc[idx]["close"] - 1)
            actual = 1 if actual_return > 0 else 0

            # Get SPY return for same period
            spy_return = None
            try:
                test_date = test_data.index[idx]
                spy_idx = spy_df.index.get_indexer([test_date], method='nearest')[0]
                if spy_idx + days < len(spy_df):
                    spy_return = float(spy_df.iloc[spy_idx]["forward_return"])
            except:
                pass

            prediction = {
                "ticker": ticker,
                "date": str(test_data.index[idx].date()),
                "pred": pred,
                "actual": actual,
                "correct": pred == actual,
                "confidence": confidence,
                "actual_return": actual_return,
                "spy_return": spy_return,
            }
            all_predictions.append(prediction)

            # Screener pick: predicted UP with high confidence
            if pred == 1 and confidence >= min_confidence:
                all_picks.append({
                    **prediction,
                    "direction_correct": actual == 1,
                    "beat_spy": actual_return > spy_return if spy_return is not None else None,
                })

    if len(all_picks) == 0:
        return None

    # Compute metrics
    df_picks = pd.DataFrame(all_picks)
    df_all = pd.DataFrame(all_predictions)

    # Basic metrics
    total_picks = len(df_picks)
    correct = df_picks["direction_correct"].sum()
    accuracy = correct / total_picks
    avg_return = df_picks["actual_return"].mean()
    win_rate = (df_picks["actual_return"] > 0).mean()
    winning_picks = int((df_picks["actual_return"] > 0).sum())

    # Baseline (all predictions and SPY returns)
    baseline_accuracy = df_all["correct"].mean()
    picks_with_spy = df_picks[df_picks["spy_return"].notna()]
    avg_baseline_return = float(picks_with_spy["spy_return"].mean()) if len(picks_with_spy) > 0 else 0
    alpha_vs_baseline = avg_return - avg_baseline_return

    # Confidence buckets
    confidence_buckets = []
    for name, low, high in [("70-75%", 0.70, 0.75), ("75-80%", 0.75, 0.80), ("80%+", 0.80, 1.01)]:
        bucket = df_picks[(df_picks["confidence"] >= low) & (df_picks["confidence"] < high)]
        if len(bucket) > 0:
            bucket_worst = float(bucket["actual_return"].min()) * 100
            confidence_buckets.append({
                "bucket": name,
                "trades": int(len(bucket)),
                "win_rate": round(float((bucket["actual_return"] > 0).mean()) * 100, 1),
                "avg_return": round(float(bucket["actual_return"].mean()) * 100, 2),
                "worst_trade": round(bucket_worst, 2),
            })

    # Risk metrics
    returns = df_picks["actual_return"].tolist()
    worst_trade = min(returns)

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for ret in returns:
        cumulative += ret
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Worst loss streak
    worst_streak_length = 0
    worst_streak_loss = 0
    current_streak = 0
    current_loss = 0
    for ret in returns:
        if ret < 0:
            current_streak += 1
            current_loss += ret
            if current_streak > worst_streak_length:
                worst_streak_length = current_streak
                worst_streak_loss = current_loss
        else:
            current_streak = 0
            current_loss = 0

    # Beat SPY
    beat_spy_count = int(picks_with_spy["beat_spy"].sum()) if len(picks_with_spy) > 0 else 0
    beat_spy_pct = float(picks_with_spy["beat_spy"].mean()) if len(picks_with_spy) > 0 else 0

    return {
        "backtest_type": "realistic",
        "methodology": f"Train on {train_start} to {train_end}, test on {test_start} to {test_end} (no retraining)",
        "summary": {
            "stocks_tested": int(stocks_tested),
            "stocks_failed": int(stocks_failed),
            "total_picks": int(total_picks),
            "winning_picks": winning_picks,
            "accuracy": round(float(accuracy) * 100, 2),
            "win_rate": round(float(win_rate) * 100, 2),
            "avg_return": round(float(avg_return) * 100, 2),
            "avg_pick_return": round(float(avg_return) * 100, 2),
            "avg_baseline_return": round(float(avg_baseline_return) * 100, 2),
            "alpha_vs_baseline": round(float(alpha_vs_baseline) * 100, 2),
            "baseline_accuracy": round(float(baseline_accuracy) * 100, 2),
            "improvement_vs_baseline": round(float(accuracy - baseline_accuracy) * 100, 2),
        },
        "confidence_buckets": confidence_buckets,
        "risk_metrics": {
            "worst_trade": round(float(worst_trade) * 100, 2),
            "worst_single_trade": round(float(worst_trade) * 100, 2),
            "max_drawdown": round(float(max_dd) * 100, 2),
            "worst_streak_length": int(worst_streak_length),
            "worst_streak_loss": round(float(worst_streak_loss) * 100, 2),
            "beat_spy_pct": round(float(beat_spy_pct) * 100, 1),
        },
        "market_comparison": {
            "picks_with_benchmark": int(len(picks_with_spy)),
            "beat_market_count": beat_spy_count,
            "beat_market_pct": round(float(beat_spy_pct) * 100, 1),
            "avg_alpha": round(float(alpha_vs_baseline) * 100, 2),
        },
        "train_period": f"{train_start} to {train_end}",
        "test_period": f"{test_start} to {test_end}",
    }


def run_regime_backtest_realistic(
    stocks: List[str],
    days: int = 5,
    min_confidence: float = 0.75,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> Optional[Dict]:
    """
    REALISTIC regime-gated backtest - train ONCE, test separately.

    Compares three modes:
    1. No Gate: Standard screener (75% confidence)
    2. Bear Suppressed: No trades when SPY trailing < -1%
    3. Bear High Confidence: Require 80% when SPY trailing < -1%
    """
    BEAR_THRESHOLD = -0.01
    NORMAL_CONF = min_confidence
    BEAR_CONF = 0.80

    # Fetch SPY for regime detection and benchmark
    spy_df = fetch_historical_data("SPY", start_date="2019-01-01")
    if spy_df is None:
        return None

    # Trailing 5-day return for regime detection (no lookahead)
    spy_df["trailing_return"] = spy_df["close"] / spy_df["close"].shift(5) - 1
    spy_df["forward_return"] = spy_df["close"].shift(-days) / spy_df["close"] - 1

    picks_no_gate = []
    picks_suppressed = []
    picks_high_conf = []

    stocks_tested = 0
    bear_periods = 0
    total_periods = 0

    for ticker in stocks:
        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        features = compute_technical_features(df.reset_index())
        feature_cols = features.columns.tolist()

        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["close"] = df["close"]
        data = data.ffill().bfill()

        train_data = data[(data.index >= train_start) & (data.index <= train_end)].dropna()
        test_data = data[(data.index >= test_start) & (data.index <= test_end)].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            continue

        stocks_tested += 1

        # Train ONCE
        X_train = train_data[feature_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train_scaled, train_data["target"])

        # Test
        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            test_date = test_data.index[idx]

            # Get SPY trailing return for regime detection and forward return for benchmark
            try:
                spy_idx = spy_df.index.get_indexer([test_date], method='nearest')[0]
                spy_trailing = float(spy_df.iloc[spy_idx]["trailing_return"])
                spy_forward = float(spy_df.iloc[spy_idx]["forward_return"]) if spy_idx + days < len(spy_df) else None
                is_bear = spy_trailing < BEAR_THRESHOLD
            except:
                spy_trailing = 0
                spy_forward = None
                is_bear = False

            if ticker == stocks[0]:  # Count once per period
                total_periods += 1
                if is_bear:
                    bear_periods += 1

            # Predict
            test_row = test_data.iloc[[idx]]
            X_test = test_row[feature_cols].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            pred = model.predict(X_test_scaled)[0]
            proba = model.predict_proba(X_test_scaled)[0]
            conf = float(max(proba))

            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / test_data.iloc[idx]["close"] - 1)

            pick = {
                "ticker": ticker,
                "date": str(test_date.date()),
                "confidence": conf,
                "actual_return": actual_return,
                "correct": actual_return > 0,
                "is_bear": is_bear,
                "spy_return": spy_forward,
                "beat_spy": actual_return > spy_forward if spy_forward is not None else None,
            }

            # Mode 1: No gate
            if pred == 1 and conf >= NORMAL_CONF:
                picks_no_gate.append(pick)

            # Mode 2: Suppressed (no trades in bear)
            if not is_bear:
                if pred == 1 and conf >= NORMAL_CONF:
                    picks_suppressed.append(pick)

            # Mode 3: High confidence in bear
            if is_bear:
                if pred == 1 and conf >= BEAR_CONF:
                    picks_high_conf.append(pick)
            else:
                if pred == 1 and conf >= NORMAL_CONF:
                    picks_high_conf.append(pick)

    def compute_metrics(picks, baseline_trades=None):
        if len(picks) == 0:
            return {
                "total_trades": 0,
                "trades": 0,
                "accuracy": 0,
                "win_rate": 0,
                "avg_return": 0,
                "worst_trade": 0,
                "worst_streak_length": 0,
                "worst_streak_loss": 0,
                "max_drawdown": 0,
                "beat_spy_pct": 0,
                "trade_reduction": 0,
                "drawdown_improvement": 0,
            }
        df = pd.DataFrame(picks)
        acc = df["correct"].mean()
        avg_ret = df["actual_return"].mean()
        win_rate = (df["actual_return"] > 0).mean()
        worst_trade = df["actual_return"].min()

        # Beat SPY calculation
        picks_with_spy = df[df["beat_spy"].notna()]
        beat_spy_pct = float(picks_with_spy["beat_spy"].mean()) * 100 if len(picks_with_spy) > 0 else 0

        # Max drawdown and worst streak
        cumulative = 0
        peak = 0
        max_dd = 0
        worst_streak_length = 0
        worst_streak_loss = 0
        current_streak = 0
        current_loss = 0

        for ret in df["actual_return"].tolist():
            cumulative += ret
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

            # Track losing streaks
            if ret < 0:
                current_streak += 1
                current_loss += ret
                if current_streak > worst_streak_length:
                    worst_streak_length = current_streak
                    worst_streak_loss = current_loss
            else:
                current_streak = 0
                current_loss = 0

        # Trade reduction vs baseline
        trade_reduction = 0
        if baseline_trades and baseline_trades > 0:
            trade_reduction = round((1 - len(df) / baseline_trades) * 100, 1)

        return {
            "total_trades": int(len(df)),
            "trades": int(len(df)),
            "accuracy": round(float(acc) * 100, 2),
            "win_rate": round(float(win_rate) * 100, 2),
            "avg_return": round(float(avg_ret) * 100, 2),
            "worst_trade": round(float(worst_trade) * 100, 2),
            "worst_streak_length": int(worst_streak_length),
            "worst_streak_loss": round(float(worst_streak_loss) * 100, 2),
            "max_drawdown": round(float(max_dd) * 100, 2),
            "beat_spy_pct": round(float(beat_spy_pct), 1),
            "trade_reduction": 0,  # Will be set after
            "drawdown_improvement": 0,  # Will be set after
        }

    no_gate = compute_metrics(picks_no_gate)
    suppressed = compute_metrics(picks_suppressed, no_gate["total_trades"])
    high_conf = compute_metrics(picks_high_conf, no_gate["total_trades"])

    # Calculate trade reduction and drawdown improvement
    if no_gate["total_trades"] > 0:
        suppressed["trade_reduction"] = round((1 - suppressed["total_trades"] / no_gate["total_trades"]) * 100, 1)
        high_conf["trade_reduction"] = round((1 - high_conf["total_trades"] / no_gate["total_trades"]) * 100, 1)

    if no_gate["max_drawdown"] > 0:
        suppressed["drawdown_improvement"] = round(no_gate["max_drawdown"] - suppressed["max_drawdown"], 2)
        high_conf["drawdown_improvement"] = round(no_gate["max_drawdown"] - high_conf["max_drawdown"], 2)

    # Calculate bear periods percentage
    bear_periods_pct = round(bear_periods / total_periods * 100, 1) if total_periods > 0 else 0

    return {
        "backtest_type": "realistic",
        "methodology": f"Train on {train_start} to {train_end}, test on {test_start} to {test_end} (no retraining)",
        "days": days,
        "summary": {
            "stocks_tested": int(stocks_tested),
            "total_periods": int(total_periods),
            "bear_periods": int(bear_periods),
            "normal_periods": int(total_periods - bear_periods),
            "bear_threshold": f"SPY trailing < {BEAR_THRESHOLD*100:.0f}%",
            "normal_confidence": f"≥{NORMAL_CONF*100:.0f}%",
            "bear_confidence": f"≥{BEAR_CONF*100:.0f}%",
        },
        "period_breakdown": {
            "bear_periods_pct": bear_periods_pct,
            "normal_periods_pct": round(100 - bear_periods_pct, 1),
        },
        "comparison": {
            "no_gate": {"mode": "No Gate (Baseline)", **no_gate},
            "suppressed": {"mode": "Bear Suppressed", **suppressed},
            "high_confidence": {"mode": "Bear 80% Confidence", **high_conf},
        },
        "interpretation": _interpret_results(no_gate, suppressed, high_conf),
        "train_period": f"{train_start} to {train_end}",
        "test_period": f"{test_start} to {test_end}",
    }


def _interpret_results(no_gate, suppressed, high_conf):
    """Generate interpretation of regime results."""
    interp = []

    if suppressed["accuracy"] > no_gate["accuracy"]:
        interp.append(f"Suppressed mode improves accuracy: {no_gate['accuracy']}% → {suppressed['accuracy']}%")
    if high_conf["accuracy"] > no_gate["accuracy"]:
        interp.append(f"High-conf mode improves accuracy: {no_gate['accuracy']}% → {high_conf['accuracy']}%")
    if suppressed["max_drawdown"] < no_gate["max_drawdown"]:
        interp.append(f"Suppressed mode reduces drawdown: {no_gate['max_drawdown']}% → {suppressed['max_drawdown']}%")
    if suppressed["trades"] < no_gate["trades"]:
        reduction = round((1 - suppressed["trades"] / no_gate["trades"]) * 100, 1) if no_gate["trades"] > 0 else 0
        interp.append(f"Trade-off: Suppressed mode reduces trades by {reduction}%")

    return interp
