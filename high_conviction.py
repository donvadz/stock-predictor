"""
High Conviction Strategy - Targeting 60%+ Accuracy

APPROACH: Only trade when MULTIPLE independent signals agree.
The idea is that each signal alone is weak (52-56%), but consensus is stronger.

SIGNALS COMBINED:
1. ML Model confidence >= 80%
2. Trend alignment (price > SMA20 > SMA50)
3. Low volatility regime (vol < 25%)
4. Positive momentum (5-day return > 0)
5. Not overbought (RSI < 70)

TRADE ONLY WHEN: At least 4 of 5 signals agree on direction
"""

import argparse
from datetime import datetime
from typing import Optional, List, Dict

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

TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# Focus on stocks that showed predictability
SELECTED_STOCKS = [
    # Best performers from analysis (>55% accuracy, low volatility)
    "MCD", "HD", "QQQ", "MSFT", "AAPL", "MA", "SPY", "JNJ", "BAC", "KO", "V",
    # Add more stable large caps
    "PG", "WMT", "JPM", "VZ", "T", "PEP", "COST",
]

# Top 15 features from importance analysis
TOP_FEATURES = [
    "sma_ratio_50_200", "sma_ratio_20_50", "atr", "macd_signal", "volatility_20d",
    "obv_trend", "macd", "bb_width", "volatility_5d", "price_to_sma_50",
    "macd_histogram", "bb_upper", "sma_ratio_5_20", "bb_lower", "price_to_sma_20",
]


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
    except Exception as e:
        return None


def compute_signals(df: pd.DataFrame, ml_pred: int, ml_confidence: float) -> Dict:
    """
    Compute all trading signals for a given point in time.
    Returns dict with each signal and overall consensus.
    """
    idx = -1  # Latest data point

    # Price and moving averages
    close = df["close"].iloc[idx]
    sma_20 = df["close"].rolling(20).mean().iloc[idx]
    sma_50 = df["close"].rolling(50).mean().iloc[idx]

    # Volatility (20-day annualized)
    returns = df["close"].pct_change()
    volatility = returns.rolling(20).std().iloc[idx] * np.sqrt(252) * 100

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[idx]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[idx]
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # 5-day momentum
    momentum_5d = (close / df["close"].iloc[-6] - 1) * 100 if len(df) > 5 else 0

    # Individual signals (1 = bullish, 0 = neutral, -1 = bearish)
    signals = {}

    # Signal 1: ML Model (high confidence bullish)
    signals["ml_bullish"] = 1 if (ml_pred == 1 and ml_confidence >= 0.75) else 0
    signals["ml_high_conf"] = 1 if ml_confidence >= 0.80 else 0

    # Signal 2: Trend alignment
    signals["trend_aligned"] = 1 if (close > sma_20 > sma_50) else 0

    # Signal 3: Low volatility
    signals["low_volatility"] = 1 if volatility < 30 else 0

    # Signal 4: Positive momentum
    signals["positive_momentum"] = 1 if momentum_5d > 0 else 0

    # Signal 5: Not overbought
    signals["not_overbought"] = 1 if rsi < 70 else 0

    # Signal 6: Not oversold (contrarian - skip extremely oversold)
    signals["not_oversold"] = 1 if rsi > 25 else 0

    # Count bullish signals
    core_signals = ["ml_bullish", "trend_aligned", "low_volatility", "positive_momentum", "not_overbought"]
    bullish_count = sum(signals[s] for s in core_signals)

    signals["bullish_count"] = bullish_count
    signals["high_conviction"] = bullish_count >= 4  # At least 4 of 5 agree
    signals["ultra_conviction"] = bullish_count == 5  # All 5 agree

    # Additional metrics for analysis
    signals["volatility"] = volatility
    signals["rsi"] = rsi
    signals["momentum_5d"] = momentum_5d

    return signals


def run_high_conviction_backtest(
    stocks: List[str] = None,
    days: int = 5,
    min_signals: int = 4,
    verbose: bool = True,
) -> Dict:
    """
    Backtest high-conviction strategy requiring multiple signal agreement.
    """
    if stocks is None:
        stocks = SELECTED_STOCKS

    if verbose:
        print("=" * 70)
        print("HIGH CONVICTION BACKTEST")
        print(f"Required signals: {min_signals}/5")
        print(f"Train: {TRAIN_START} to {TRAIN_END}")
        print(f"Test:  {TEST_START} to {TEST_END}")
        print("=" * 70)

    all_predictions = []
    high_conviction_picks = []
    stocks_processed = 0

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        df = fetch_historical_data(ticker)
        if df is None:
            continue

        # Compute features
        features = compute_technical_features(df.reset_index())
        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        # Use only top features
        available_features = [f for f in TOP_FEATURES if f in features.columns]
        if len(available_features) < 10:
            continue

        # Target
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["close"] = df["close"]
        data = data.ffill().bfill()

        # Split
        train_data = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
        test_data = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            continue

        stocks_processed += 1

        # Train ML model once
        X_train = train_data[available_features].replace([np.inf, -np.inf], 0)
        y_train = train_data["target"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            min_samples_split=10, random_state=42, n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Test with non-overlapping periods
        test_indices = list(range(0, len(test_data) - days, days))
        stock_high_conviction = 0
        stock_correct = 0

        for idx in test_indices:
            test_row = test_data.iloc[[idx]]
            X_test = test_row[available_features].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            # ML prediction
            ml_pred = model.predict(X_test_scaled)[0]
            ml_proba = model.predict_proba(X_test_scaled)[0]
            ml_confidence = float(max(ml_proba))

            # Get slice of data up to test point for signal computation
            test_date = test_data.index[idx]
            df_slice = df[df.index <= test_date].tail(100)

            # Compute all signals
            signals = compute_signals(df_slice, ml_pred, ml_confidence)

            # Actual outcome
            actual_idx = idx + days
            if actual_idx >= len(test_data):
                continue

            actual_return = float(test_data.iloc[actual_idx]["close"] / test_data.iloc[idx]["close"] - 1)
            actual_direction = 1 if actual_return > 0 else 0

            # Record all predictions
            pred_record = {
                "ticker": ticker,
                "date": str(test_date.date()),
                "ml_pred": ml_pred,
                "ml_confidence": ml_confidence,
                "signals": signals,
                "bullish_count": signals["bullish_count"],
                "high_conviction": signals["bullish_count"] >= min_signals,
                "actual_direction": actual_direction,
                "actual_return": actual_return,
            }
            all_predictions.append(pred_record)

            # Track high conviction picks
            if signals["bullish_count"] >= min_signals and ml_pred == 1:
                stock_high_conviction += 1
                correct = actual_direction == 1
                if correct:
                    stock_correct += 1
                high_conviction_picks.append({
                    **pred_record,
                    "correct": correct,
                })

        if verbose and stock_high_conviction > 0:
            acc = stock_correct / stock_high_conviction * 100
            print(f"  {stock_high_conviction} high-conviction picks, accuracy: {acc:.1f}%")
        elif verbose:
            print(f"  No high-conviction signals")

    # Compute results
    if len(high_conviction_picks) == 0:
        return {"error": "No high-conviction picks found"}

    df_hc = pd.DataFrame(high_conviction_picks)
    hc_accuracy = df_hc["correct"].mean()
    hc_avg_return = df_hc["actual_return"].mean()
    hc_win_rate = (df_hc["actual_return"] > 0).mean()

    # Compare to baseline (all ML predictions)
    df_all = pd.DataFrame(all_predictions)
    baseline_accuracy = (df_all["ml_pred"] == df_all["actual_direction"]).mean()

    # By signal count
    signal_analysis = []
    for count in range(1, 6):
        subset = df_all[df_all["bullish_count"] == count]
        subset_bullish = subset[subset["ml_pred"] == 1]
        if len(subset_bullish) > 5:
            acc = (subset_bullish["actual_direction"] == 1).mean()
            signal_analysis.append({
                "signals": count,
                "picks": len(subset_bullish),
                "accuracy": round(acc * 100, 1),
            })

    # Ultra high conviction (5/5 signals)
    ultra_hc = df_all[(df_all["bullish_count"] == 5) & (df_all["ml_pred"] == 1)]
    ultra_accuracy = (ultra_hc["actual_direction"] == 1).mean() if len(ultra_hc) > 0 else 0

    return {
        "summary": {
            "stocks_tested": stocks_processed,
            "total_predictions": len(df_all),
            "baseline_accuracy": round(baseline_accuracy * 100, 2),
            "high_conviction_picks": len(df_hc),
            "high_conviction_accuracy": round(hc_accuracy * 100, 2),
            "high_conviction_win_rate": round(hc_win_rate * 100, 2),
            "high_conviction_avg_return": round(hc_avg_return * 100, 3),
            "ultra_conviction_picks": len(ultra_hc),
            "ultra_conviction_accuracy": round(ultra_accuracy * 100, 2),
        },
        "signal_analysis": signal_analysis,
        "improvement_vs_baseline": round((hc_accuracy - baseline_accuracy) * 100, 2),
    }


def run_time_horizon_test(
    stocks: List[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Test different prediction horizons to find optimal timeframe.
    """
    if stocks is None:
        stocks = SELECTED_STOCKS[:10]  # Subset for speed

    if verbose:
        print("\n" + "=" * 70)
        print("TIME HORIZON ANALYSIS")
        print("=" * 70)

    horizons = [3, 5, 7, 10, 15, 20, 30]
    results = []

    for days in horizons:
        if verbose:
            print(f"\nTesting {days}-day horizon...")

        horizon_preds = []

        for ticker in stocks:
            df = fetch_historical_data(ticker)
            if df is None:
                continue

            features = compute_technical_features(df.reset_index())
            fundamentals = fetch_fundamental_data(ticker)
            features = add_fundamental_features(features, fundamentals)

            available_features = [f for f in TOP_FEATURES if f in features.columns]
            if len(available_features) < 10:
                continue

            future_return = df["close"].shift(-days) / df["close"] - 1
            target = (future_return > 0).astype(int)

            features.index = df.index
            data = features.copy()
            data["target"] = target
            data["close"] = df["close"]
            data = data.ffill().bfill()

            train_data = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
            test_data = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

            if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
                continue

            X_train = train_data[available_features].replace([np.inf, -np.inf], 0)
            y_train = train_data["target"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)

            # Non-overlapping test
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
                actual = 1 if actual_return > 0 else 0

                horizon_preds.append({
                    "pred": pred,
                    "actual": actual,
                    "correct": pred == actual,
                    "confidence": confidence,
                    "return": actual_return,
                })

        if horizon_preds:
            df_h = pd.DataFrame(horizon_preds)
            accuracy = df_h["correct"].mean()

            # High confidence subset
            high_conf = df_h[df_h["confidence"] >= 0.75]
            high_conf_acc = high_conf["correct"].mean() if len(high_conf) > 0 else 0

            results.append({
                "days": days,
                "predictions": len(df_h),
                "accuracy": round(accuracy * 100, 1),
                "high_conf_accuracy": round(high_conf_acc * 100, 1),
                "high_conf_count": len(high_conf),
            })

            if verbose:
                print(f"  {days} days: {accuracy*100:.1f}% overall, {high_conf_acc*100:.1f}% high-conf ({len(high_conf)} picks)")

    return {"horizons": results}


def run_sector_rotation_test(
    verbose: bool = True,
) -> Dict:
    """
    Test sector rotation strategy - predict which sector will outperform.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SECTOR ROTATION ANALYSIS")
        print("=" * 70)

    # Sector ETFs
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
    }

    days = 10  # 2-week rotation
    all_rotations = []

    # Fetch all sector data
    sector_data = {}
    for etf, name in sectors.items():
        if verbose:
            print(f"Fetching {etf} ({name})...")
        df = fetch_historical_data(etf)
        if df is not None:
            sector_data[etf] = df

    if len(sector_data) < 5:
        return {"error": "Insufficient sector data"}

    # Get common date range
    common_dates = None
    for etf, df in sector_data.items():
        test_dates = df[(df.index >= TEST_START) & (df.index <= TEST_END)].index
        if common_dates is None:
            common_dates = set(test_dates)
        else:
            common_dates = common_dates.intersection(set(test_dates))

    common_dates = sorted(list(common_dates))

    if verbose:
        print(f"\nAnalyzing {len(common_dates)} common trading days...")

    # Test rotation every 'days' periods
    test_points = list(range(0, len(common_dates) - days, days))

    for i in test_points:
        start_date = common_dates[i]
        end_date = common_dates[i + days] if i + days < len(common_dates) else common_dates[-1]

        # Calculate forward returns for each sector
        forward_returns = {}
        trailing_momentum = {}

        for etf, df in sector_data.items():
            try:
                start_price = df.loc[start_date, "close"]
                end_price = df.loc[end_date, "close"]
                forward_returns[etf] = (end_price / start_price - 1) * 100

                # Trailing 20-day momentum as predictor
                lookback_date = common_dates[max(0, i - 20)]
                lookback_price = df.loc[lookback_date, "close"]
                trailing_momentum[etf] = (start_price / lookback_price - 1) * 100
            except:
                pass

        if len(forward_returns) >= 5:
            # Strategy: Pick top 3 sectors by trailing momentum
            sorted_by_momentum = sorted(trailing_momentum.items(), key=lambda x: x[1], reverse=True)
            top_picks = [etf for etf, _ in sorted_by_momentum[:3]]
            bottom_picks = [etf for etf, _ in sorted_by_momentum[-3:]]

            # Calculate returns
            top_return = np.mean([forward_returns.get(etf, 0) for etf in top_picks])
            bottom_return = np.mean([forward_returns.get(etf, 0) for etf in bottom_picks])
            spy_return = forward_returns.get("SPY", forward_returns.get(list(forward_returns.keys())[0], 0))

            all_rotations.append({
                "date": str(start_date.date()),
                "top_picks": top_picks,
                "top_return": top_return,
                "bottom_return": bottom_return,
                "spread": top_return - bottom_return,
                "beat_bottom": top_return > bottom_return,
            })

    if len(all_rotations) == 0:
        return {"error": "No rotation signals"}

    df_rot = pd.DataFrame(all_rotations)

    return {
        "summary": {
            "total_rotations": len(df_rot),
            "top_beat_bottom_pct": round(df_rot["beat_bottom"].mean() * 100, 1),
            "avg_spread": round(df_rot["spread"].mean(), 2),
            "avg_top_return": round(df_rot["top_return"].mean(), 2),
            "avg_bottom_return": round(df_rot["bottom_return"].mean(), 2),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High conviction strategy testing")
    parser.add_argument("--days", type=int, default=5, help="Prediction horizon")
    parser.add_argument("--min-signals", type=int, default=4, help="Minimum agreeing signals (1-5)")
    parser.add_argument("--test-horizons", action="store_true", help="Test different time horizons")
    parser.add_argument("--test-sectors", action="store_true", help="Test sector rotation")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("HIGH CONVICTION STRATEGY ANALYSIS")
    print("=" * 70)

    # Main high conviction test
    results = run_high_conviction_backtest(
        days=args.days,
        min_signals=args.min_signals,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if "error" not in results:
        s = results["summary"]
        print(f"\nBaseline (ML only):        {s['baseline_accuracy']}%")
        print(f"High Conviction (4+ sigs): {s['high_conviction_accuracy']}% ({s['high_conviction_picks']} picks)")
        print(f"Ultra Conviction (5 sigs): {s['ultra_conviction_accuracy']}% ({s['ultra_conviction_picks']} picks)")
        print(f"\nImprovement vs baseline:   {results['improvement_vs_baseline']:+.1f}%")

        print("\n" + "-" * 40)
        print("ACCURACY BY SIGNAL COUNT:")
        print("-" * 40)
        for row in results["signal_analysis"]:
            print(f"  {row['signals']} signals: {row['accuracy']}% ({row['picks']} picks)")

    # Optional: Test time horizons
    if args.test_horizons:
        horizon_results = run_time_horizon_test(verbose=True)
        print("\n" + "-" * 40)
        print("BEST TIME HORIZONS:")
        print("-" * 40)
        for h in sorted(horizon_results["horizons"], key=lambda x: x["high_conf_accuracy"], reverse=True)[:3]:
            print(f"  {h['days']} days: {h['high_conf_accuracy']}% high-conf accuracy")

    # Optional: Test sector rotation
    if args.test_sectors:
        sector_results = run_sector_rotation_test(verbose=True)
        if "error" not in sector_results:
            print("\n" + "-" * 40)
            print("SECTOR ROTATION:")
            print("-" * 40)
            s = sector_results["summary"]
            print(f"  Top beats Bottom: {s['top_beat_bottom_pct']}% of time")
            print(f"  Avg spread: {s['avg_spread']}%")
