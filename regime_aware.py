"""
Regime-Aware Strategy - Knows When to Trade and When to Sit Out

KEY INSIGHT FROM STRESS TESTS:
- The bullish multi-signal strategy works ONLY in favorable regimes
- During 2008, 2018, 2020 crashes: accuracy dropped to 28-33%
- The system kept generating bullish signals that failed

SOLUTION:
1. Detect market regime FIRST (VIX, SPY trend, drawdown)
2. In "Crisis Mode" → NO bullish signals, only defensive
3. In "Normal Mode" → Use the high-conviction bullish strategy
4. In "Recovery Mode" → Cautiously re-enter

This makes the system HONEST about when it should be silent.
"""

import argparse
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from model import compute_technical_features
from config import MIN_DATA_POINTS


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(spy_df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Detect current market regime based on SPY and VIX.

    Returns regime classification:
    - CRISIS: VIX > 30 or SPY down >10% from 50-day high
    - CAUTION: VIX > 20 or SPY below 50-day SMA
    - NORMAL: VIX < 20, SPY above SMAs, trending up
    - EUPHORIA: VIX < 15, SPY up >20% in 6 months (overextended)
    """
    close = spy_df["close"].iloc[-1]
    sma_20 = spy_df["close"].rolling(20).mean().iloc[-1]
    sma_50 = spy_df["close"].rolling(50).mean().iloc[-1]
    sma_200 = spy_df["close"].rolling(200).mean().iloc[-1]

    # Drawdown from recent high
    high_50d = spy_df["close"].rolling(50).max().iloc[-1]
    drawdown = (close - high_50d) / high_50d * 100

    # 6-month return
    if len(spy_df) > 126:
        return_6m = (close / spy_df["close"].iloc[-126] - 1) * 100
    else:
        return_6m = 0

    # Volatility
    volatility = spy_df["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    # VIX level (if available)
    vix_level = None
    if vix_df is not None and len(vix_df) > 0:
        vix_level = vix_df["close"].iloc[-1]

    # Classify regime
    regime = "NORMAL"
    confidence = 0.5
    action = "TRADE"

    # CRISIS detection
    if drawdown < -10 or volatility > 35 or (vix_level and vix_level > 30):
        regime = "CRISIS"
        confidence = 0.9
        action = "NO_TRADE"

    # CAUTION detection
    elif drawdown < -5 or close < sma_50 or volatility > 25 or (vix_level and vix_level > 20):
        regime = "CAUTION"
        confidence = 0.7
        action = "REDUCE"

    # EUPHORIA detection (overextended)
    elif return_6m > 25 and close > sma_20 * 1.1:
        regime = "EUPHORIA"
        confidence = 0.6
        action = "REDUCE"

    # NORMAL
    elif close > sma_20 > sma_50:
        regime = "NORMAL"
        confidence = 0.7
        action = "TRADE"

    # RECOVERY (coming out of crisis)
    elif close > sma_20 and close < sma_50:
        regime = "RECOVERY"
        confidence = 0.5
        action = "CAUTIOUS"

    return {
        "regime": regime,
        "confidence": confidence,
        "action": action,
        "metrics": {
            "spy_price": round(close, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2) if not np.isnan(sma_200) else None,
            "drawdown_from_50d_high": round(drawdown, 2),
            "return_6m": round(return_6m, 2),
            "volatility_annual": round(volatility, 1),
            "vix": round(vix_level, 1) if vix_level else None,
        },
        "explanation": _get_regime_explanation(regime, action),
    }


def _get_regime_explanation(regime: str, action: str) -> str:
    explanations = {
        "CRISIS": "Market is in crisis mode. High volatility, significant drawdown. Strategy historically fails in this regime. Recommendation: No new bullish positions.",
        "CAUTION": "Market showing stress signals. Elevated volatility or weakening trend. Recommendation: Reduce position sizes, require higher conviction.",
        "NORMAL": "Favorable conditions for the strategy. Trend intact, volatility contained. Recommendation: Execute high-conviction signals.",
        "RECOVERY": "Market recovering from stress. Trend improving but not confirmed. Recommendation: Small positions only, wait for confirmation.",
        "EUPHORIA": "Market may be overextended. Strong returns but potential for pullback. Recommendation: Tighten stops, be selective.",
    }
    return explanations.get(regime, "Unknown regime")


# =============================================================================
# REGIME-AWARE BACKTEST
# =============================================================================

def run_regime_aware_backtest(
    period: str = "2020-2024",
    verbose: bool = True,
) -> Dict:
    """
    Backtest showing how regime detection would have helped.

    Compares:
    - Strategy WITHOUT regime filter (always trade)
    - Strategy WITH regime filter (skip crisis periods)
    """

    periods = {
        "2008-2010": ("2005-01-01", "2007-12-31", "2008-01-01", "2010-12-31"),
        "2018-2020": ("2015-01-01", "2018-06-30", "2018-07-01", "2020-12-31"),
        "2020-2024": ("2017-01-01", "2019-12-31", "2020-01-01", "2024-12-31"),
    }

    if period not in periods:
        return {"error": f"Unknown period. Choose from: {list(periods.keys())}"}

    train_start, train_end, test_start, test_end = periods[period]

    if verbose:
        print("=" * 70)
        print(f"REGIME-AWARE BACKTEST: {period}")
        print(f"Test period: {test_start} to {test_end}")
        print("=" * 70)

    # Fetch SPY for regime detection
    spy = yf.Ticker("SPY")
    spy_df = spy.history(start=train_start, end=test_end, interval="1d")
    spy_df = spy_df.reset_index()
    spy_df = spy_df.rename(columns={"Date": "timestamp", "Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"})
    spy_df["timestamp"] = pd.to_datetime(spy_df["timestamp"])
    spy_df = spy_df.set_index("timestamp")

    # Try to get VIX
    try:
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(start=train_start, end=test_end, interval="1d")
        vix_df = vix_df.reset_index()
        vix_df = vix_df.rename(columns={"Date": "timestamp", "Close": "close"})
        vix_df["timestamp"] = pd.to_datetime(vix_df["timestamp"])
        vix_df = vix_df.set_index("timestamp")
    except:
        vix_df = None

    # Stocks to test
    stocks = ["SPY", "QQQ", "AAPL", "MSFT", "MCD", "JNJ", "KO", "PG", "HD", "MA"]

    TOP_FEATURES = [
        "sma_ratio_50_200", "sma_ratio_20_50", "atr", "macd_signal", "volatility_20d",
        "obv_trend", "macd", "bb_width", "volatility_5d", "price_to_sma_50",
    ]

    all_trades_unfiltered = []
    all_trades_filtered = []
    regime_history = []

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        stock = yf.Ticker(ticker)
        df = stock.history(start=train_start, end=test_end, interval="1d")

        if df.empty or len(df) < 200:
            continue

        df = df.reset_index()
        df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]]

        features = compute_technical_features(df.reset_index())
        available = [f for f in TOP_FEATURES if f in features.columns]

        if len(available) < 8:
            continue

        future_return = df["close"].shift(-20) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["close"] = df["close"]
        data = data.ffill().bfill()

        train = data[(data.index >= train_start) & (data.index <= train_end)].dropna()
        test = data[(data.index >= test_start) & (data.index <= test_end)].dropna()

        if len(train) < MIN_DATA_POINTS or len(test) < 10:
            continue

        # Train model
        X_train = train[available].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train_scaled, train["target"])

        # Test
        indices = list(range(0, len(test) - 20, 20))

        for idx in indices:
            test_date = test.index[idx]

            # Detect regime at this point
            spy_slice = spy_df[spy_df.index <= test_date].tail(250)
            vix_slice = vix_df[vix_df.index <= test_date].tail(1) if vix_df is not None else None

            if len(spy_slice) < 50:
                continue

            regime = detect_regime(spy_slice, vix_slice)

            # Record regime
            if ticker == "SPY":
                regime_history.append({
                    "date": str(test_date.date()),
                    "regime": regime["regime"],
                    "action": regime["action"],
                })

            # Get prediction
            row = test.iloc[[idx]]
            X = row[available].replace([np.inf, -np.inf], 0)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            conf = float(max(proba))

            actual_idx = idx + 20
            if actual_idx >= len(test):
                continue

            actual_return = float(test.iloc[actual_idx]["close"] / test.iloc[idx]["close"] - 1)
            actual = 1 if actual_return > 0 else 0
            correct = pred == actual

            # High conviction check
            is_hc = pred == 1 and conf >= 0.70

            trade = {
                "ticker": ticker,
                "date": str(test_date.date()),
                "regime": regime["regime"],
                "pred": pred,
                "conf": conf,
                "actual": actual,
                "correct": correct,
                "return": actual_return,
                "is_hc": is_hc,
            }

            # Unfiltered: take all high-conviction bullish
            if is_hc:
                all_trades_unfiltered.append(trade)

                # Filtered: only in NORMAL or RECOVERY regime
                if regime["action"] in ["TRADE", "CAUTIOUS"]:
                    all_trades_filtered.append(trade)

    # Compute results
    if not all_trades_unfiltered:
        return {"error": "No trades generated"}

    df_unfiltered = pd.DataFrame(all_trades_unfiltered)
    df_filtered = pd.DataFrame(all_trades_filtered) if all_trades_filtered else pd.DataFrame()

    unfiltered_acc = (df_unfiltered["actual"] == 1).mean()  # Bullish predictions that were correct
    unfiltered_return = df_unfiltered["return"].mean()

    if len(df_filtered) > 0:
        filtered_acc = (df_filtered["actual"] == 1).mean()
        filtered_return = df_filtered["return"].mean()
    else:
        filtered_acc = 0
        filtered_return = 0

    # Regime breakdown
    regime_breakdown = df_unfiltered.groupby("regime").agg({
        "actual": ["sum", "count"],
        "return": "mean"
    }).round(3)

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\n{'Strategy':<40} {'Trades':>10} {'Win Rate':>12} {'Avg Return':>12}")
        print("-" * 75)
        print(f"{'Without Regime Filter':<40} {len(df_unfiltered):>10} {unfiltered_acc*100:>11.1f}% {unfiltered_return*100:>11.2f}%")
        if len(df_filtered) > 0:
            print(f"{'With Regime Filter (skip CRISIS/CAUTION)':<40} {len(df_filtered):>10} {filtered_acc*100:>11.1f}% {filtered_return*100:>11.2f}%")

        print("\n" + "-" * 70)
        print("BREAKDOWN BY REGIME:")
        print("-" * 70)

        for regime in ["NORMAL", "RECOVERY", "CAUTION", "CRISIS", "EUPHORIA"]:
            regime_trades = df_unfiltered[df_unfiltered["regime"] == regime]
            if len(regime_trades) > 0:
                acc = (regime_trades["actual"] == 1).mean() * 100
                ret = regime_trades["return"].mean() * 100
                print(f"  {regime:<12} {len(regime_trades):>5} trades, {acc:>6.1f}% win rate, {ret:>+6.2f}% avg return")

        # Show trades we avoided
        avoided = df_unfiltered[df_unfiltered["regime"].isin(["CRISIS", "CAUTION"])]
        if len(avoided) > 0:
            avoided_return = avoided["return"].mean()
            print(f"\n  Trades AVOIDED by filter: {len(avoided)} (avg return would have been {avoided_return*100:+.2f}%)")

    return {
        "unfiltered": {
            "trades": len(df_unfiltered),
            "win_rate": round(unfiltered_acc * 100, 1),
            "avg_return": round(unfiltered_return * 100, 2),
        },
        "filtered": {
            "trades": len(df_filtered),
            "win_rate": round(filtered_acc * 100, 1) if len(df_filtered) > 0 else 0,
            "avg_return": round(filtered_return * 100, 2) if len(df_filtered) > 0 else 0,
        },
        "improvement": round((filtered_acc - unfiltered_acc) * 100, 1) if len(df_filtered) > 0 else 0,
    }


def get_current_regime(verbose: bool = True) -> Dict:
    """Get current market regime."""

    spy = yf.Ticker("SPY")
    spy_df = spy.history(period="1y", interval="1d")
    spy_df = spy_df.reset_index()
    spy_df = spy_df.rename(columns={"Date": "timestamp", "Close": "close"})
    spy_df["timestamp"] = pd.to_datetime(spy_df["timestamp"])
    spy_df = spy_df.set_index("timestamp")

    try:
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(period="5d", interval="1d")
        vix_df = vix_df.reset_index()
        vix_df = vix_df.rename(columns={"Date": "timestamp", "Close": "close"})
        vix_df["timestamp"] = pd.to_datetime(vix_df["timestamp"])
        vix_df = vix_df.set_index("timestamp")
    except:
        vix_df = None

    regime = detect_regime(spy_df, vix_df)

    if verbose:
        print("\n" + "=" * 70)
        print("CURRENT MARKET REGIME")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)

        print(f"\nREGIME: {regime['regime']}")
        print(f"ACTION: {regime['action']}")
        print(f"\n{regime['explanation']}")

        print("\nMETRICS:")
        for k, v in regime["metrics"].items():
            if v is not None:
                print(f"  {k}: {v}")

    return regime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", type=str, help="Run backtest for period (2008-2010, 2018-2020, 2020-2024)")
    parser.add_argument("--current", action="store_true", help="Show current regime")

    args = parser.parse_args()

    if args.current:
        get_current_regime()
    elif args.backtest:
        run_regime_aware_backtest(args.backtest)
    else:
        # Run all backtests
        for period in ["2008-2010", "2018-2020", "2020-2024"]:
            run_regime_aware_backtest(period)
