"""
Stress Test - Test Strategy Against Historical Crisis Periods

Tests the optimal strategy against:
- 2008 Financial Crisis
- 2018 Q4 Selloff
- 2020 COVID Crash
- 2022 Bear Market

This answers: "Does the strategy survive when conditions turn hostile?"
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
# CRISIS PERIODS
# =============================================================================

CRISIS_PERIODS = {
    "2008_crisis": {
        "name": "2008 Financial Crisis",
        "train_start": "2005-01-01",
        "train_end": "2007-12-31",
        "test_start": "2008-01-01",
        "test_end": "2009-03-31",
        "description": "Lehman collapse, -50% drawdown",
    },
    "2018_selloff": {
        "name": "2018 Q4 Selloff",
        "train_start": "2015-01-01",
        "train_end": "2018-09-30",
        "test_start": "2018-10-01",
        "test_end": "2019-01-31",
        "description": "Fed tightening, -20% correction",
    },
    "2020_covid": {
        "name": "2020 COVID Crash",
        "train_start": "2017-01-01",
        "train_end": "2020-01-31",
        "test_start": "2020-02-01",
        "test_end": "2020-06-30",
        "description": "Fastest bear market in history",
    },
    "2022_bear": {
        "name": "2022 Bear Market",
        "train_start": "2019-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
        "description": "Fed hiking, tech collapse, -25%",
    },
}

# Stocks that existed during all periods (no RIVN, COIN, etc.)
STRESS_TEST_STOCKS = [
    "SPY", "QQQ", "AAPL", "MSFT", "JPM", "JNJ", "PG", "KO",
    "MCD", "WMT", "HD", "V", "MA", "XOM", "PFE",
]

TOP_FEATURES = [
    "sma_ratio_50_200", "sma_ratio_20_50", "atr", "macd_signal", "volatility_20d",
    "obv_trend", "macd", "bb_width", "volatility_5d", "price_to_sma_50",
    "macd_histogram", "bb_upper", "sma_ratio_5_20", "bb_lower", "price_to_sma_20",
]


def fetch_historical_data(ticker: str, start_date: str) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=datetime.now().strftime("%Y-%m-%d"), interval="1d")
        if df.empty or len(df) < 100:
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


def compute_signals(df: pd.DataFrame, ml_pred: int, ml_confidence: float) -> Dict:
    """Compute trading signals."""
    close = df["close"].iloc[-1]
    sma_20 = df["close"].rolling(20).mean().iloc[-1]
    sma_50 = df["close"].rolling(50).mean().iloc[-1]

    returns = df["close"].pct_change()
    volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    momentum_5d = (close / df["close"].iloc[-6] - 1) * 100 if len(df) > 5 else 0

    signals = {
        "ml_bullish": 1 if (ml_pred == 1 and ml_confidence >= 0.70) else 0,
        "trend_aligned": 1 if (close > sma_20 > sma_50) else 0,
        "low_volatility": 1 if volatility < 35 else 0,
        "positive_momentum": 1 if momentum_5d > 0 else 0,
        "not_overbought": 1 if rsi < 70 else 0,
    }

    return sum(signals.values())


def run_crisis_backtest(
    period_key: str,
    stocks: List[str] = None,
    days: int = 20,
    min_signals: int = 4,
    verbose: bool = True,
) -> Dict:
    """Run backtest during a specific crisis period."""

    if period_key not in CRISIS_PERIODS:
        return {"error": f"Unknown period: {period_key}"}

    period = CRISIS_PERIODS[period_key]

    if stocks is None:
        stocks = STRESS_TEST_STOCKS

    if verbose:
        print(f"\n{'='*70}")
        print(f"CRISIS TEST: {period['name']}")
        print(f"Description: {period['description']}")
        print(f"Train: {period['train_start']} to {period['train_end']}")
        print(f"Test:  {period['test_start']} to {period['test_end']}")
        print(f"{'='*70}")

    all_preds = []
    high_conviction = []
    stocks_tested = 0

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        # Fetch data starting well before train period
        fetch_start = str(int(period['train_start'][:4]) - 2) + period['train_start'][4:]
        df = fetch_historical_data(ticker, start_date=fetch_start)

        if df is None:
            continue

        # Compute features
        features = compute_technical_features(df.reset_index())

        # Skip fundamentals for historical (not available)
        available = [f for f in TOP_FEATURES if f in features.columns]
        if len(available) < 10:
            continue

        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["close"] = df["close"]
        data = data.ffill().bfill()

        # Split by crisis period dates
        train = data[(data.index >= period['train_start']) & (data.index <= period['train_end'])].dropna()
        test = data[(data.index >= period['test_start']) & (data.index <= period['test_end'])].dropna()

        if len(train) < MIN_DATA_POINTS or len(test) < 5:
            if verbose:
                print(f"  Skipped: insufficient data")
            continue

        stocks_tested += 1

        # Train
        X_train = train[available].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train_scaled, train["target"])

        # Test (non-overlapping)
        indices = list(range(0, len(test) - days, days))

        for idx in indices:
            row = test.iloc[[idx]]
            X = row[available].replace([np.inf, -np.inf], 0)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            conf = float(max(proba))

            # Compute signals
            df_slice = df[df.index <= test.index[idx]].tail(100)
            if len(df_slice) < 50:
                continue
            signal_count = compute_signals(df_slice, pred, conf)

            actual_idx = idx + days
            if actual_idx >= len(test):
                continue

            actual_return = float(test.iloc[actual_idx]["close"] / test.iloc[idx]["close"] - 1)
            actual = 1 if actual_return > 0 else 0

            record = {
                "ticker": ticker,
                "date": str(test.index[idx].date()),
                "pred": pred,
                "conf": conf,
                "signals": signal_count,
                "actual": actual,
                "correct": pred == actual,
                "return": actual_return,
            }
            all_preds.append(record)

            if signal_count >= min_signals and pred == 1:
                record["hc_correct"] = actual == 1
                high_conviction.append(record)

        if verbose:
            stock_hc = [p for p in high_conviction if p["ticker"] == ticker]
            if stock_hc:
                acc = sum(p["hc_correct"] for p in stock_hc) / len(stock_hc) * 100
                print(f"  {len(stock_hc)} high-conviction picks, accuracy: {acc:.1f}%")
            else:
                print(f"  No high-conviction signals")

    if not all_preds:
        return {"error": "No predictions generated"}

    df_all = pd.DataFrame(all_preds)
    baseline_acc = (df_all["pred"] == df_all["actual"]).mean()

    hc_acc = 0
    hc_return = 0
    if high_conviction:
        df_hc = pd.DataFrame(high_conviction)
        hc_acc = df_hc["hc_correct"].mean()
        hc_return = df_hc["return"].mean()

        # Calculate max drawdown for high conviction picks
        cumulative = 0
        peak = 0
        max_dd = 0
        for ret in df_hc["return"].tolist():
            cumulative += ret
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

    return {
        "period": period["name"],
        "description": period["description"],
        "stocks_tested": stocks_tested,
        "total_predictions": len(df_all),
        "baseline_accuracy": round(baseline_acc * 100, 1),
        "high_conviction_picks": len(high_conviction),
        "high_conviction_accuracy": round(hc_acc * 100, 1) if high_conviction else 0,
        "high_conviction_avg_return": round(hc_return * 100, 2) if high_conviction else 0,
        "max_drawdown": round(max_dd * 100, 2) if high_conviction else 0,
    }


def run_all_stress_tests(verbose: bool = True) -> Dict:
    """Run stress tests across all crisis periods."""

    if verbose:
        print("\n" + "=" * 70)
        print("STRESS TEST SUITE")
        print("Testing strategy against historical crisis periods")
        print("=" * 70)

    results = {}

    for period_key in CRISIS_PERIODS:
        result = run_crisis_backtest(period_key, verbose=verbose)
        results[period_key] = result

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("STRESS TEST SUMMARY")
        print("=" * 70)

        print(f"\n{'Period':<25} {'Baseline':>10} {'High Conv':>12} {'HC Picks':>10} {'Max DD':>10}")
        print("-" * 70)

        for key, r in results.items():
            if "error" not in r:
                print(f"{r['period'][:24]:<25} {r['baseline_accuracy']:>9.1f}% {r['high_conviction_accuracy']:>11.1f}% {r['high_conviction_picks']:>10} {r['max_drawdown']:>9.1f}%")

        # Overall assessment
        print("\n" + "-" * 70)
        print("ASSESSMENT:")
        print("-" * 70)

        surviving = 0
        failing = 0

        for key, r in results.items():
            if "error" in r:
                continue
            if r["high_conviction_accuracy"] >= 50:
                surviving += 1
                print(f"  {r['period']}: SURVIVED (accuracy didn't collapse)")
            else:
                failing += 1
                print(f"  {r['period']}: STRUGGLED (accuracy below 50%)")

        if surviving == len(results):
            print("\n  VERDICT: Strategy shows resilience across crisis periods")
        elif failing > surviving:
            print("\n  VERDICT: Strategy is regime-dependent - use with caution")
        else:
            print("\n  VERDICT: Mixed results - monitor carefully in volatile periods")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=str, help="Specific period to test")
    parser.add_argument("--days", type=int, default=20)

    args = parser.parse_args()

    if args.period:
        result = run_crisis_backtest(args.period, days=args.days)
        print(result)
    else:
        results = run_all_stress_tests()
