"""
Find Tier 1 Candidates - Stocks that behave like MCD, QQQ, MSFT, MA, COST, SPY

Criteria for Tier 1:
1. Low volatility (< 25% annualized)
2. Clear trend behavior (high SMA alignment)
3. High ML accuracy (> 60% on backtest)
4. Sufficient liquidity (large cap)

Tests each stock with the realistic backtest methodology.
"""

import argparse
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from model import compute_technical_features, add_fundamental_features
from config import MIN_DATA_POINTS
from data import fetch_fundamental_data


# Configuration
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# Current Tier 1 (for comparison)
CURRENT_TIER1 = ["MCD", "QQQ", "MSFT", "MA", "COST", "SPY"]

# Candidate universe - large cap, liquid stocks across sectors
CANDIDATE_STOCKS = [
    # Tech (beyond current)
    "AAPL", "GOOGL", "AMZN", "META", "NVDA", "ADBE", "CRM", "ORCL", "CSCO", "INTC",
    "IBM", "TXN", "AVGO", "QCOM", "ACN",

    # Finance
    "V", "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "USB",
    "PNC", "TFC", "COF", "CME", "ICE",

    # Healthcare
    "UNH", "JNJ", "PFE", "LLY", "MRK", "ABBV", "TMO", "DHR", "BMY", "AMGN",
    "MDT", "ISRG", "SYK", "ZTS", "GILD",

    # Consumer Staples (stable)
    "PG", "KO", "PEP", "WMT", "COST", "CL", "GIS", "K", "KHC", "MDLZ",
    "STZ", "MO", "PM", "EL", "KMB",

    # Consumer Discretionary
    "HD", "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST", "DG", "DLTR", "YUM",
    "CMG", "DHI", "LEN", "ORLY", "AZO",

    # Industrials
    "HON", "UPS", "CAT", "DE", "MMM", "GE", "RTX", "LMT", "BA", "UNP",
    "CSX", "NSC", "WM", "RSG", "EMR",

    # Utilities (very stable)
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED",
    "WEC", "ES", "AWK", "ATO", "NI",

    # REITs
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR",

    # ETFs
    "IVV", "VOO", "VTI", "DIA", "IWM", "XLF", "XLK", "XLV", "XLP", "XLU",
    "VIG", "SCHD", "VYM", "DGRO",
]

# Remove stocks already in Tier 1
CANDIDATES_TO_TEST = [s for s in CANDIDATE_STOCKS if s not in CURRENT_TIER1]


def fetch_historical_data(ticker: str, start_date: str = "2019-01-01") -> Optional[pd.DataFrame]:
    """Fetch historical data."""
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


def evaluate_stock(ticker: str, days: int = 20) -> Optional[Dict]:
    """
    Evaluate a single stock for Tier 1 candidacy.

    Returns metrics including accuracy, volatility, and signal quality.
    """
    df = fetch_historical_data(ticker)
    if df is None:
        return None

    # Compute features
    features = compute_technical_features(df.reset_index())
    fundamentals = fetch_fundamental_data(ticker)
    features = add_fundamental_features(features, fundamentals)

    feature_cols = features.columns.tolist()

    # Target
    future_return = df["close"].shift(-days) / df["close"] - 1
    target = (future_return > 0).astype(int)

    features.index = df.index
    data = features.copy()
    data["target"] = target
    data["close"] = df["close"]
    data["future_return"] = future_return
    data = data.ffill().bfill()

    # Calculate volatility
    returns = df["close"].pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252) * 100
    test_vol = volatility[(df.index >= TEST_START) & (df.index <= TEST_END)]
    avg_volatility = test_vol.mean() if len(test_vol) > 0 else 100

    # Calculate trend consistency (how often price > SMA50 > SMA200)
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    trend_aligned = ((df["close"] > sma50) & (sma50 > sma200)).astype(int)
    test_trend = trend_aligned[(df.index >= TEST_START) & (df.index <= TEST_END)]
    trend_pct = test_trend.mean() * 100 if len(test_trend) > 0 else 0

    # Split
    train = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
    test = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

    if len(train) < MIN_DATA_POINTS or len(test) < 10:
        return None

    # Train model ONCE
    X_train = train[feature_cols].replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        min_samples_split=10, random_state=42, n_jobs=-1,
    )
    model.fit(X_train_scaled, train["target"])

    # Test on non-overlapping windows
    test_indices = list(range(0, len(test) - days, days))

    predictions = []
    high_conf_predictions = []

    for idx in test_indices:
        row = test.iloc[[idx]]
        X_test = row[feature_cols].replace([np.inf, -np.inf], 0)
        X_test_scaled = scaler.transform(X_test)

        pred = model.predict(X_test_scaled)[0]
        proba = model.predict_proba(X_test_scaled)[0]
        conf = float(max(proba))

        actual_idx = idx + days
        if actual_idx >= len(test):
            continue

        actual_return = float(test.iloc[actual_idx]["close"] / test.iloc[idx]["close"] - 1)
        actual = 1 if actual_return > 0 else 0
        correct = pred == actual

        predictions.append({
            "pred": pred,
            "actual": actual,
            "correct": correct,
            "conf": conf,
            "return": actual_return,
        })

        # High confidence = predicted UP with conf >= 75%
        if pred == 1 and conf >= 0.75:
            high_conf_predictions.append({
                "correct": actual == 1,
                "return": actual_return,
            })

    if len(predictions) == 0:
        return None

    # Compute metrics
    accuracy = sum(p["correct"] for p in predictions) / len(predictions) * 100

    high_conf_accuracy = 0
    high_conf_count = len(high_conf_predictions)
    high_conf_return = 0

    if high_conf_count > 0:
        high_conf_accuracy = sum(p["correct"] for p in high_conf_predictions) / high_conf_count * 100
        high_conf_return = sum(p["return"] for p in high_conf_predictions) / high_conf_count * 100

    # Win rate when predicting UP
    up_preds = [p for p in predictions if p["pred"] == 1]
    win_rate = sum(p["actual"] == 1 for p in up_preds) / len(up_preds) * 100 if up_preds else 0

    return {
        "ticker": ticker,
        "accuracy": round(accuracy, 1),
        "high_conf_accuracy": round(high_conf_accuracy, 1),
        "high_conf_count": high_conf_count,
        "high_conf_return": round(high_conf_return, 2),
        "win_rate": round(win_rate, 1),
        "volatility": round(avg_volatility, 1),
        "trend_pct": round(trend_pct, 1),
        "total_signals": len(predictions),
    }


def find_tier1_candidates(
    candidates: List[str] = None,
    days: int = 20,
    min_accuracy: float = 60.0,
    max_volatility: float = 30.0,
    verbose: bool = True,
) -> List[Dict]:
    """
    Screen candidates for Tier 1 potential.
    """
    if candidates is None:
        candidates = CANDIDATES_TO_TEST

    print("=" * 70)
    print("TIER 1 CANDIDATE SCREENER")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Testing {len(candidates)} stocks")
    print(f"Criteria: accuracy > {min_accuracy}%, volatility < {max_volatility}%")
    print(f"Horizon: {days} days")
    print("=" * 70)

    # First, evaluate current Tier 1 for baseline
    print("\nEvaluating current Tier 1 stocks...")
    tier1_results = []
    for ticker in CURRENT_TIER1:
        result = evaluate_stock(ticker, days)
        if result:
            tier1_results.append(result)
            if verbose:
                print(f"  {ticker}: {result['accuracy']}% acc, {result['high_conf_accuracy']}% @ 75% conf, vol={result['volatility']}%")

    # Calculate Tier 1 benchmarks
    if tier1_results:
        avg_tier1_acc = np.mean([r["accuracy"] for r in tier1_results])
        avg_tier1_hc_acc = np.mean([r["high_conf_accuracy"] for r in tier1_results if r["high_conf_count"] > 0])
        avg_tier1_vol = np.mean([r["volatility"] for r in tier1_results])
        print(f"\nTier 1 benchmarks: {avg_tier1_acc:.1f}% acc, {avg_tier1_hc_acc:.1f}% HC acc, {avg_tier1_vol:.1f}% vol")

    # Evaluate candidates in parallel
    print(f"\nScreening {len(candidates)} candidates...")

    results = []

    def process_stock(ticker):
        try:
            return evaluate_stock(ticker, days)
        except Exception as e:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock, t): t for t in candidates}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            ticker = futures[future]
            result = future.result()
            if result:
                results.append(result)
                if verbose and result["accuracy"] >= min_accuracy:
                    print(f"  [{completed}/{len(candidates)}] {ticker}: {result['accuracy']}% acc, {result['high_conf_accuracy']}% HC, vol={result['volatility']}%")
            if completed % 20 == 0 and not verbose:
                print(f"  Progress: {completed}/{len(candidates)}")

    # Filter for Tier 1 candidates
    tier1_candidates = [
        r for r in results
        if r["accuracy"] >= min_accuracy and r["volatility"] <= max_volatility
    ]

    # Sort by high-confidence accuracy, then overall accuracy
    tier1_candidates.sort(key=lambda x: (x["high_conf_accuracy"], x["accuracy"]), reverse=True)

    # Print results
    print("\n" + "=" * 70)
    print("TIER 1 CANDIDATES FOUND")
    print("=" * 70)

    if not tier1_candidates:
        print("\nNo candidates met the criteria.")
        print("Try relaxing min_accuracy or max_volatility.")
    else:
        print(f"\n{'Ticker':<8} {'Accuracy':>10} {'HC Acc':>10} {'HC Cnt':>8} {'HC Ret':>10} {'Vol':>8} {'Trend%':>8}")
        print("-" * 70)

        for r in tier1_candidates[:25]:  # Show top 25
            print(f"{r['ticker']:<8} {r['accuracy']:>9.1f}% {r['high_conf_accuracy']:>9.1f}% {r['high_conf_count']:>8} {r['high_conf_return']:>9.2f}% {r['volatility']:>7.1f}% {r['trend_pct']:>7.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total candidates tested: {len(results)}")
    print(f"Met Tier 1 criteria: {len(tier1_candidates)}")

    if tier1_candidates:
        # Categorize
        excellent = [r for r in tier1_candidates if r["high_conf_accuracy"] >= 70]
        good = [r for r in tier1_candidates if 60 <= r["high_conf_accuracy"] < 70]

        if excellent:
            print(f"\nExcellent (HC >= 70%): {', '.join(r['ticker'] for r in excellent)}")
        if good:
            print(f"Good (60% <= HC < 70%): {', '.join(r['ticker'] for r in good)}")

        # Suggest additions to Tier 1
        top_candidates = [r for r in tier1_candidates if r["high_conf_accuracy"] >= 65][:6]
        if top_candidates:
            print(f"\nRECOMMENDED TIER 1 ADDITIONS:")
            for r in top_candidates:
                print(f"  - {r['ticker']}: {r['high_conf_accuracy']}% at 75% confidence, {r['volatility']}% volatility")

    print("=" * 70)

    return tier1_candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--min-accuracy", type=float, default=55.0)
    parser.add_argument("--max-volatility", type=float, default=35.0)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    candidates = find_tier1_candidates(
        days=args.days,
        min_accuracy=args.min_accuracy,
        max_volatility=args.max_volatility,
        verbose=not args.quiet,
    )
