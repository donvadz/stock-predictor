"""
Alternative Modeling Approaches for Stock Prediction

This module implements and compares several alternative strategies to the
baseline RandomForest model (52% out-of-sample accuracy).

STRATEGIES:
1. Simple Momentum Strategy - Rules-based (no ML)
2. Mean Reversion Strategy - RSI-based rules
3. Volatility-Adjusted Model - Filter low-volatility stocks, use pruned RF
4. Trend Quality Filter - Only trade clear trends (ADX > 25)

Each strategy is backtested on 2023-2024 data for fair comparison.
"""

import argparse
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from model import compute_technical_features, add_fundamental_features
from config import MIN_DATA_POINTS
from data import fetch_fundamental_data


# =============================================================================
# CONFIGURATION
# =============================================================================

# Test period (same as validation.py)
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# Training period (for ML-based approaches)
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"

# Stocks to test (mix of large caps and volatile stocks)
TEST_STOCKS = [
    # Large caps (stable, lots of data)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Finance
    "JPM", "V", "MA", "BAC",
    # Healthcare
    "UNH", "JNJ", "PFE", "LLY",
    # Consumer
    "WMT", "KO", "NKE", "MCD", "HD",
    # ETFs
    "SPY", "QQQ",
    # Volatile stocks
    "AMD", "COIN", "RIVN",
]

# Top 15 features from validation.py analysis (typical results)
TOP_15_FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "volatility_5d", "volatility_20d",
    "rsi", "macd", "macd_histogram",
    "price_to_sma_20", "sma_ratio_5_20", "sma_ratio_20_50",
    "volume_ratio", "bb_width", "atr", "close_position",
]


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_historical_data(
    ticker: str,
    start_date: str = "2019-01-01",
    end_date: str = None,
) -> Optional[pd.DataFrame]:
    """Fetch historical data for backtesting."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")

        if df.empty or len(df) < 200:
            return None

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) for trend strength.

    ADX > 25: Strong trend
    ADX < 20: Weak/no trend (choppy market)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1

    # Set DM to 0 when conditions not met
    plus_dm = np.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), minus_dm.abs(), 0)

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smooth with EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    # Calculate ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx


# =============================================================================
# STRATEGY 1: SIMPLE MOMENTUM
# =============================================================================

def run_momentum_strategy(
    stocks: List[str] = None,
    days: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Simple Momentum Strategy (Rules-Based, No ML)

    RULES:
    - If price > SMA_20 AND SMA_20 > SMA_50 -> Predict UP
    - Otherwise -> Predict DOWN (or HOLD)

    This captures stocks in uptrends.
    """
    if stocks is None:
        stocks = TEST_STOCKS

    if verbose:
        print("\n" + "=" * 70)
        print("STRATEGY 1: SIMPLE MOMENTUM (No ML)")
        print("=" * 70)
        print("Rule: Price > SMA_20 AND SMA_20 > SMA_50 -> Predict UP")

    all_predictions = []
    stocks_processed = 0

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        # Calculate moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Filter to test period
        test_data = df[(df.index >= TEST_START) & (df.index <= TEST_END)].dropna()

        if len(test_data) < 10:
            continue

        stocks_processed += 1

        # Generate predictions at non-overlapping intervals
        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            row = test_data.iloc[idx]

            # Momentum rule
            price_above_sma20 = row["close"] > row["sma_20"]
            sma20_above_sma50 = row["sma_20"] > row["sma_50"]

            predicted = 1 if (price_above_sma20 and sma20_above_sma50) else 0

            # Get actual outcome
            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / row["close"] - 1)
            actual_direction = 1 if actual_return > 0 else 0

            all_predictions.append({
                "ticker": ticker,
                "date": str(test_data.index[idx].date()),
                "predicted": predicted,
                "actual": actual_direction,
                "correct": predicted == actual_direction,
                "actual_return": actual_return,
                "strategy_signal": "LONG" if predicted == 1 else "FLAT",
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            if stock_preds:
                stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds)
                print(f"  {len(stock_preds)} predictions, accuracy: {stock_acc*100:.1f}%")

    return _compute_strategy_metrics(all_predictions, stocks_processed, "Momentum")


# =============================================================================
# STRATEGY 2: MEAN REVERSION (RSI-BASED)
# =============================================================================

def run_mean_reversion_strategy(
    stocks: List[str] = None,
    days: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Mean Reversion Strategy (RSI-Based, No ML)

    RULES:
    - If RSI < 30 -> Predict UP (oversold bounce expected)
    - If RSI > 70 -> Predict DOWN (overbought pullback expected)
    - Otherwise -> No prediction (skip)

    This captures reversal opportunities at extremes.
    """
    if stocks is None:
        stocks = TEST_STOCKS

    if verbose:
        print("\n" + "=" * 70)
        print("STRATEGY 2: MEAN REVERSION (RSI-Based)")
        print("=" * 70)
        print("Rule: RSI < 30 -> UP (oversold), RSI > 70 -> DOWN (overbought)")

    all_predictions = []
    stocks_processed = 0
    signals_generated = 0

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Filter to test period
        test_data = df[(df.index >= TEST_START) & (df.index <= TEST_END)].dropna()

        if len(test_data) < 10:
            continue

        stocks_processed += 1

        # Generate predictions at non-overlapping intervals
        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            row = test_data.iloc[idx]
            rsi = row["rsi"]

            # Mean reversion rule - only trade at extremes
            if rsi < 30:
                predicted = 1  # Oversold -> expect bounce UP
                signal = "OVERSOLD_LONG"
            elif rsi > 70:
                predicted = 0  # Overbought -> expect pullback DOWN
                signal = "OVERBOUGHT_SHORT"
            else:
                continue  # No signal in neutral zone

            signals_generated += 1

            # Get actual outcome
            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / row["close"] - 1)
            actual_direction = 1 if actual_return > 0 else 0

            all_predictions.append({
                "ticker": ticker,
                "date": str(test_data.index[idx].date()),
                "predicted": predicted,
                "actual": actual_direction,
                "correct": predicted == actual_direction,
                "actual_return": actual_return,
                "rsi": rsi,
                "strategy_signal": signal,
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            if stock_preds:
                stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds)
                print(f"  {len(stock_preds)} signals, accuracy: {stock_acc*100:.1f}%")
            else:
                print(f"  No extreme RSI signals")

    result = _compute_strategy_metrics(all_predictions, stocks_processed, "Mean Reversion")
    result["summary"]["signals_generated"] = signals_generated
    return result


# =============================================================================
# STRATEGY 3: VOLATILITY-ADJUSTED MODEL
# =============================================================================

def run_volatility_adjusted_strategy(
    stocks: List[str] = None,
    days: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Volatility-Adjusted Model

    HYPOTHESIS:
    Low-volatility stocks are more predictable. High-volatility stocks add noise.

    APPROACH:
    1. Calculate 20-day volatility for each stock
    2. Only trade stocks with volatility < median
    3. Use pruned RandomForest (top 15 features) for predictions
    """
    if stocks is None:
        stocks = TEST_STOCKS

    if verbose:
        print("\n" + "=" * 70)
        print("STRATEGY 3: VOLATILITY-ADJUSTED MODEL")
        print("=" * 70)
        print("Filter: Only trade low-volatility stocks (< median)")
        print("Model: Pruned RandomForest (top 15 features)")

    # First pass: calculate volatility for all stocks
    stock_volatility = {}
    stock_data = {}

    for ticker in stocks:
        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        # Calculate volatility
        returns = df["close"].pct_change()
        vol_20d = returns.rolling(20).std()

        # Get median volatility in test period
        test_vol = vol_20d[(df.index >= TEST_START) & (df.index <= TEST_END)]
        if len(test_vol) > 0:
            median_vol = test_vol.median()
            stock_volatility[ticker] = median_vol
            stock_data[ticker] = df

    if len(stock_volatility) == 0:
        return {"error": "No stocks with valid data"}

    # Calculate median volatility threshold
    vol_threshold = np.median(list(stock_volatility.values()))

    # Filter to low-volatility stocks
    low_vol_stocks = [t for t, v in stock_volatility.items() if v <= vol_threshold]
    high_vol_stocks = [t for t, v in stock_volatility.items() if v > vol_threshold]

    if verbose:
        print(f"\nVolatility threshold (median): {vol_threshold*100:.2f}%")
        print(f"Low-vol stocks ({len(low_vol_stocks)}): {', '.join(low_vol_stocks[:5])}...")
        print(f"High-vol stocks ({len(high_vol_stocks)}): {', '.join(high_vol_stocks[:5])}...")

    all_predictions = []
    stocks_processed = 0

    # Use pruned features
    feature_cols = TOP_15_FEATURES

    for ticker in low_vol_stocks:
        if verbose:
            print(f"\nProcessing {ticker} (low-vol)...")

        df = stock_data[ticker]

        # Compute technical features
        features = compute_technical_features(df.reset_index())

        # Get fundamentals
        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        # Keep only top features that exist
        available_features = [f for f in feature_cols if f in features.columns]
        if len(available_features) < 10:
            continue

        # Construct target
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features[available_features].copy()
        data["target"] = target
        data["future_return"] = future_return
        data["close"] = df["close"]
        data = data.ffill().bfill()

        # Split by date
        train_data = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
        test_data = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            continue

        stocks_processed += 1

        # Train model
        X_train = train_data[available_features].replace([np.inf, -np.inf], 0)
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

        # Test
        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            test_row = test_data.iloc[[idx]]
            X_test = test_row[available_features].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            pred = model.predict(X_test_scaled)[0]
            proba = model.predict_proba(X_test_scaled)[0]
            confidence = float(max(proba))

            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / test_data.iloc[idx]["close"] - 1)
            actual_direction = 1 if actual_return > 0 else 0

            all_predictions.append({
                "ticker": ticker,
                "date": str(test_data.index[idx].date()),
                "predicted": pred,
                "actual": actual_direction,
                "correct": pred == actual_direction,
                "confidence": confidence,
                "actual_return": actual_return,
                "volatility": stock_volatility[ticker],
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            if stock_preds:
                stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds)
                print(f"  {len(stock_preds)} predictions, accuracy: {stock_acc*100:.1f}%")

    result = _compute_strategy_metrics(all_predictions, stocks_processed, "Volatility-Adjusted")
    result["summary"]["low_vol_stocks"] = len(low_vol_stocks)
    result["summary"]["high_vol_stocks"] = len(high_vol_stocks)
    result["summary"]["vol_threshold_pct"] = round(vol_threshold * 100, 2)
    return result


# =============================================================================
# STRATEGY 4: TREND QUALITY FILTER
# =============================================================================

def run_trend_quality_strategy(
    stocks: List[str] = None,
    days: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Trend Quality Filter Strategy

    HYPOTHESIS:
    Predictions are more accurate in trending markets. Skip choppy/sideways markets.

    APPROACH:
    1. Calculate ADX (Average Directional Index)
    2. Only trade when ADX > 25 (strong trend)
    3. Use pruned RandomForest for direction prediction

    ADX interpretation:
    - ADX < 20: Weak/no trend (choppy)
    - ADX 20-25: Developing trend
    - ADX > 25: Strong trend
    - ADX > 40: Very strong trend
    """
    if stocks is None:
        stocks = TEST_STOCKS

    if verbose:
        print("\n" + "=" * 70)
        print("STRATEGY 4: TREND QUALITY FILTER (ADX)")
        print("=" * 70)
        print("Filter: Only trade when ADX > 25 (strong trend)")
        print("Model: Pruned RandomForest (top 15 features)")

    all_predictions = []
    skipped_signals = 0
    stocks_processed = 0

    feature_cols = TOP_15_FEATURES

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        # Calculate ADX
        df["adx"] = compute_adx(df)

        # Compute technical features
        features = compute_technical_features(df.reset_index())

        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        available_features = [f for f in feature_cols if f in features.columns]
        if len(available_features) < 10:
            continue

        # Construct target
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features[available_features].copy()
        data["target"] = target
        data["future_return"] = future_return
        data["close"] = df["close"]
        data["adx"] = df["adx"]
        data = data.ffill().bfill()

        # Split by date
        train_data = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
        test_data = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            continue

        stocks_processed += 1

        # Train model on ALL training data (not just trending)
        X_train = train_data[available_features].replace([np.inf, -np.inf], 0)
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

        # Test - but only predict when ADX > 25
        test_indices = list(range(0, len(test_data) - days, days))
        stock_skipped = 0

        for idx in test_indices:
            test_row = test_data.iloc[[idx]]
            adx_value = test_row["adx"].values[0]

            # Skip if not in clear trend
            if adx_value < 25:
                stock_skipped += 1
                skipped_signals += 1
                continue

            X_test = test_row[available_features].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            pred = model.predict(X_test_scaled)[0]
            proba = model.predict_proba(X_test_scaled)[0]
            confidence = float(max(proba))

            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / test_data.iloc[idx]["close"] - 1)
            actual_direction = 1 if actual_return > 0 else 0

            all_predictions.append({
                "ticker": ticker,
                "date": str(test_data.index[idx].date()),
                "predicted": pred,
                "actual": actual_direction,
                "correct": pred == actual_direction,
                "confidence": confidence,
                "actual_return": actual_return,
                "adx": adx_value,
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            if stock_preds:
                stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds)
                print(f"  {len(stock_preds)} predictions (skipped {stock_skipped} low-ADX), accuracy: {stock_acc*100:.1f}%")
            else:
                print(f"  All signals skipped (low ADX)")

    result = _compute_strategy_metrics(all_predictions, stocks_processed, "Trend Quality")
    if result and "summary" in result:
        result["summary"]["skipped_low_adx"] = skipped_signals
    else:
        result = {
            "summary": {
                "strategy": "Trend Quality",
                "total_predictions": 0,
                "accuracy": 0,
                "skipped_low_adx": skipped_signals,
            }
        }
    return result


# =============================================================================
# BASELINE: RANDOM FOREST (FULL FEATURES)
# =============================================================================

def run_baseline_rf(
    stocks: List[str] = None,
    days: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Baseline RandomForest with all features (for comparison).
    This replicates the current model approach.
    """
    if stocks is None:
        stocks = TEST_STOCKS

    if verbose:
        print("\n" + "=" * 70)
        print("BASELINE: RANDOM FOREST (All Features)")
        print("=" * 70)

    all_predictions = []
    stocks_processed = 0
    feature_names = None

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        # Compute ALL features
        features = compute_technical_features(df.reset_index())
        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        if feature_names is None:
            feature_names = features.columns.tolist()

        # Construct target
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["future_return"] = future_return
        data["close"] = df["close"]
        data = data.ffill().bfill()

        train_data = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
        test_data = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            continue

        stocks_processed += 1

        X_train = train_data[feature_names].replace([np.inf, -np.inf], 0)
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

        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            test_row = test_data.iloc[[idx]]
            X_test = test_row[feature_names].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            pred = model.predict(X_test_scaled)[0]
            proba = model.predict_proba(X_test_scaled)[0]
            confidence = float(max(proba))

            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / test_data.iloc[idx]["close"] - 1)
            actual_direction = 1 if actual_return > 0 else 0

            all_predictions.append({
                "ticker": ticker,
                "predicted": pred,
                "actual": actual_direction,
                "correct": pred == actual_direction,
                "confidence": confidence,
                "actual_return": actual_return,
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            if stock_preds:
                stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds)
                print(f"  {len(stock_preds)} predictions, accuracy: {stock_acc*100:.1f}%")

    return _compute_strategy_metrics(all_predictions, stocks_processed, "Baseline RF")


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def _compute_strategy_metrics(
    predictions: List[Dict],
    stocks_processed: int,
    strategy_name: str,
) -> Dict:
    """Compute standard metrics for any strategy."""
    if len(predictions) == 0:
        return {
            "strategy": strategy_name,
            "error": "No predictions generated",
        }

    df_preds = pd.DataFrame(predictions)

    # Core metrics
    accuracy = df_preds["correct"].mean()

    # Win rate (predicted UP and actually went UP)
    up_predictions = df_preds[df_preds["predicted"] == 1]
    if len(up_predictions) > 0:
        win_rate = (up_predictions["actual"] == 1).mean()
        avg_return_when_long = up_predictions["actual_return"].mean()
    else:
        win_rate = 0
        avg_return_when_long = 0

    # Overall average return (if we followed the strategy)
    # Long when predicted UP, flat when predicted DOWN
    strategy_returns = df_preds.apply(
        lambda row: row["actual_return"] if row["predicted"] == 1 else 0,
        axis=1
    )
    avg_strategy_return = strategy_returns.mean()

    # High confidence analysis (if confidence exists)
    if "confidence" in df_preds.columns:
        high_conf = df_preds[df_preds["confidence"] >= 0.75]
        high_conf_accuracy = high_conf["correct"].mean() if len(high_conf) > 0 else 0
        high_conf_count = len(high_conf)
    else:
        high_conf_accuracy = 0
        high_conf_count = 0

    return {
        "strategy": strategy_name,
        "summary": {
            "stocks_processed": stocks_processed,
            "total_predictions": len(df_preds),
            "accuracy": round(accuracy * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "avg_return_when_long": round(avg_return_when_long * 100, 3),
            "avg_strategy_return": round(avg_strategy_return * 100, 3),
        },
        "high_confidence": {
            "count": high_conf_count,
            "accuracy": round(high_conf_accuracy * 100, 2),
        },
        "_raw_predictions": predictions,
    }


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def run_comparison(
    stocks: List[str] = None,
    days: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run all strategies and generate comparison report.
    """
    print("\n" + "=" * 70)
    print("ALTERNATIVE MODELS COMPARISON")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Prediction Horizon: {days} days")
    print(f"Stocks: {len(stocks or TEST_STOCKS)}")
    print("=" * 70)

    results = {}

    # Run baseline first
    print("\n[1/5] Running Baseline RandomForest...")
    results["baseline"] = run_baseline_rf(stocks, days, verbose)

    # Run alternative strategies
    print("\n[2/5] Running Momentum Strategy...")
    results["momentum"] = run_momentum_strategy(stocks, days, verbose)

    print("\n[3/5] Running Mean Reversion Strategy...")
    results["mean_reversion"] = run_mean_reversion_strategy(stocks, days, verbose)

    print("\n[4/5] Running Volatility-Adjusted Strategy...")
    results["volatility_adjusted"] = run_volatility_adjusted_strategy(stocks, days, verbose)

    print("\n[5/5] Running Trend Quality Strategy...")
    results["trend_quality"] = run_trend_quality_strategy(stocks, days, verbose)

    # Generate comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'Win Rate':>10} {'Avg Ret':>10} {'Signals':>10}")
    print("-" * 70)

    baseline_acc = 52.0  # Known baseline from validation

    strategy_order = ["baseline", "momentum", "mean_reversion", "volatility_adjusted", "trend_quality"]
    for key in strategy_order:
        res = results[key]
        if "error" in res:
            print(f"{res['strategy']:<25} {'ERROR':>10}")
            continue

        s = res["summary"]
        print(f"{res['strategy']:<25} {s['accuracy']:>9.1f}% {s['win_rate']:>9.1f}% {s['avg_strategy_return']:>9.2f}% {s['total_predictions']:>10}")

    # Determine winner
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    valid_results = [(k, r) for k, r in results.items() if "error" not in r]
    if valid_results:
        best_accuracy = max(valid_results, key=lambda x: x[1]["summary"]["accuracy"])
        best_return = max(valid_results, key=lambda x: x[1]["summary"]["avg_strategy_return"])

        print(f"\nBest Accuracy: {best_accuracy[1]['strategy']} at {best_accuracy[1]['summary']['accuracy']:.1f}%")
        print(f"Best Avg Return: {best_return[1]['strategy']} at {best_return[1]['summary']['avg_strategy_return']:.2f}%")

        # Compare to baseline
        for key, res in valid_results:
            if key == "baseline":
                continue
            s = res["summary"]
            baseline_s = results["baseline"]["summary"]
            acc_delta = s["accuracy"] - baseline_s["accuracy"]
            ret_delta = s["avg_strategy_return"] - baseline_s["avg_strategy_return"]

            status = "BETTER" if acc_delta > 1 else ("WORSE" if acc_delta < -1 else "SIMILAR")
            print(f"\n{res['strategy']} vs Baseline:")
            print(f"  Accuracy: {acc_delta:+.1f}% ({status})")
            print(f"  Return: {ret_delta:+.2f}%")

            # Strategy-specific insights
            if key == "mean_reversion" and "signals_generated" in s:
                print(f"  Note: Only {s['total_predictions']} signals at RSI extremes (selective)")
            if key == "volatility_adjusted" and "low_vol_stocks" in s:
                print(f"  Note: Traded only {s['low_vol_stocks']} low-vol stocks")
            if key == "trend_quality" and "skipped_low_adx" in s:
                print(f"  Note: Skipped {s['skipped_low_adx']} low-ADX (choppy) signals")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    recommendations = []

    # Check if any strategy significantly beats baseline
    if valid_results:
        best_acc = best_accuracy[1]["summary"]["accuracy"]
        if best_acc > baseline_acc + 3:
            recommendations.append(f"CONSIDER: {best_accuracy[1]['strategy']} shows {best_acc:.1f}% accuracy (vs 52% baseline)")
        else:
            recommendations.append("CAUTION: No strategy significantly outperforms baseline 52%")

        # Check return-based strategies
        if best_return[1]["summary"]["avg_strategy_return"] > 0.5:
            recommendations.append(f"RETURNS: {best_return[1]['strategy']} has best average return ({best_return[1]['summary']['avg_strategy_return']:.2f}%)")

        # Strategy-specific recommendations
        mr = results.get("mean_reversion", {})
        if "summary" in mr and mr["summary"]["accuracy"] > 55:
            recommendations.append("RSI EXTREMES: Mean reversion shows edge at oversold/overbought levels")

        tq = results.get("trend_quality", {})
        if "summary" in tq and tq["summary"]["accuracy"] > baseline_acc + 2:
            recommendations.append("TREND FILTER: ADX filtering improves accuracy - avoid choppy markets")

        va = results.get("volatility_adjusted", {})
        if "summary" in va and va["summary"]["accuracy"] > baseline_acc + 2:
            recommendations.append("VOLATILITY: Low-vol stocks are more predictable")

    if not recommendations:
        recommendations.append("No clear winning strategy - consider combining approaches or improving features")

    for rec in recommendations:
        print(f"  - {rec}")

    print("=" * 70)

    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare alternative stock prediction strategies")
    parser.add_argument("--days", type=int, default=5, help="Prediction horizon in days")
    parser.add_argument("--stocks", type=str, nargs="+", help="Specific stocks to test")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["all", "momentum", "mean_reversion", "volatility", "trend", "baseline"],
        default="all",
        help="Run specific strategy only"
    )

    args = parser.parse_args()

    stocks = args.stocks
    verbose = not args.quiet

    if args.strategy == "all":
        results = run_comparison(stocks, args.days, verbose)
    elif args.strategy == "momentum":
        results = run_momentum_strategy(stocks, args.days, verbose)
    elif args.strategy == "mean_reversion":
        results = run_mean_reversion_strategy(stocks, args.days, verbose)
    elif args.strategy == "volatility":
        results = run_volatility_adjusted_strategy(stocks, args.days, verbose)
    elif args.strategy == "trend":
        results = run_trend_quality_strategy(stocks, args.days, verbose)
    elif args.strategy == "baseline":
        results = run_baseline_rf(stocks, args.days, verbose)
