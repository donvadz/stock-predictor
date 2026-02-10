"""
Model Validation Module - Out-of-Sample Testing & Feature Analysis

PURPOSE:
This module performs rigorous validation to detect:
- Data leakage
- Overfitting
- Regime-locked performance
- Spurious confidence signals

KEY TESTS:
1. Year-Split Backtest: Train on 2020-2022, test on 2023-2024 (no retraining)
2. Feature Importance: Identify which features actually drive predictions
3. Pruned Model Test: Re-test with only top N features

INTERPRETATION:
- Year-split accuracy holds at 60%+ → real signal exists
- Year-split falls to 55-60% → smaller edge, still useful
- Year-split falls to ~50% → leakage or overfitting confirmed
- Pruned model matches/exceeds full model → original had noise features
"""

import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Year-split configuration
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# Stocks to validate (subset for faster testing)
VALIDATION_STOCKS = [
    # Large caps (stable, lots of data)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Finance
    "JPM", "V", "MA", "BAC",
    # Healthcare
    "UNH", "JNJ", "PFE", "LLY",
    # Consumer
    "WMT", "KO", "NKE", "MCD", "HD",
    # ETFs (for regime comparison)
    "SPY", "QQQ",
    # Volatile stocks (stress test)
    "AMD", "COIN", "RIVN",
]

# Feature pruning configuration
TOP_FEATURES_COUNT = 15


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_historical_data(
    ticker: str,
    start_date: str = "2019-01-01",  # Extra year for feature warm-up
    end_date: str = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for validation.

    Uses a longer lookback than production to cover full train+test period.
    """
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

        # Ensure timestamp is datetime for proper indexing
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


# =============================================================================
# YEAR-SPLIT BACKTEST
# =============================================================================

def run_year_split_backtest(
    stocks: List[str] = None,
    days: int = 5,
    min_confidence: float = 0.75,
    verbose: bool = True,
) -> Dict:
    """
    Run a strict year-split backtest.

    METHODOLOGY:
    1. Fetch data from 2019-2024 (2019 for feature warm-up)
    2. Train model ONCE on 2020-2022 data
    3. Test on 2023-2024 data WITHOUT retraining
    4. Track accuracy, confidence calibration, and regime performance

    This is the gold standard for detecting overfitting/leakage.
    """
    if stocks is None:
        stocks = VALIDATION_STOCKS

    if verbose:
        print("=" * 70)
        print("YEAR-SPLIT BACKTEST")
        print(f"Train: {TRAIN_START} to {TRAIN_END}")
        print(f"Test:  {TEST_START} to {TEST_END}")
        print(f"Prediction horizon: {days} days")
        print(f"Min confidence threshold: {min_confidence}")
        print("=" * 70)

    all_predictions = []
    stocks_processed = 0
    stocks_failed = 0

    # Track feature importances across all stocks
    all_feature_importances = []
    feature_names = None

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        # Fetch full historical data
        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            if verbose:
                print(f"  Skipped: No data")
            stocks_failed += 1
            continue

        # Compute features
        features = compute_technical_features(df.reset_index())

        # Get fundamentals (use current - this is a limitation but acceptable for validation)
        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        # Store feature names
        if feature_names is None:
            feature_names = features.columns.tolist()

        # Construct target (future return direction)
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        # Align indices
        features.index = df.index

        # Combine features and target
        data = features.copy()
        data["target"] = target
        data["future_return"] = future_return
        data["close"] = df["close"]

        # Fill NaN from rolling calculations
        data = data.ffill().bfill()

        # Split into train and test by date
        train_mask = (data.index >= TRAIN_START) & (data.index <= TRAIN_END)
        test_mask = (data.index >= TEST_START) & (data.index <= TEST_END)

        train_data = data[train_mask].dropna()
        test_data = data[test_mask].dropna()

        if len(train_data) < MIN_DATA_POINTS or len(test_data) < 10:
            if verbose:
                print(f"  Skipped: Insufficient data (train={len(train_data)}, test={len(test_data)})")
            stocks_failed += 1
            continue

        stocks_processed += 1

        # Prepare training data
        X_train = train_data[feature_names].replace([np.inf, -np.inf], 0)
        y_train = train_data["target"]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model ONCE on training period
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Store feature importances
        all_feature_importances.append(model.feature_importances_)

        # Test on out-of-sample period (non-overlapping windows)
        test_indices = list(range(0, len(test_data) - days, days))

        for idx in test_indices:
            test_row = test_data.iloc[[idx]]

            X_test = test_row[feature_names].replace([np.inf, -np.inf], 0)
            X_test_scaled = scaler.transform(X_test)

            # Predict
            pred = model.predict(X_test_scaled)[0]
            proba = model.predict_proba(X_test_scaled)[0]
            confidence = float(max(proba))

            # Get actual outcome
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
                "is_screener_pick": pred == 1 and confidence >= min_confidence,
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds) if stock_preds else 0
            print(f"  {len(stock_preds)} predictions, accuracy: {stock_acc*100:.1f}%")

    if len(all_predictions) == 0:
        return {"error": "No predictions generated"}

    # ==========================================================================
    # COMPUTE METRICS
    # ==========================================================================
    df_preds = pd.DataFrame(all_predictions)

    # Overall accuracy
    overall_accuracy = df_preds["correct"].mean()

    # Accuracy by confidence bucket
    confidence_buckets = []
    for name, low, high in [("50-60%", 0.5, 0.6), ("60-70%", 0.6, 0.7),
                            ("70-75%", 0.7, 0.75), ("75-80%", 0.75, 0.8), ("80%+", 0.8, 1.01)]:
        bucket = df_preds[(df_preds["confidence"] >= low) & (df_preds["confidence"] < high)]
        if len(bucket) > 0:
            confidence_buckets.append({
                "bucket": name,
                "count": len(bucket),
                "accuracy": round(bucket["correct"].mean() * 100, 1),
                "avg_return_when_correct": round(bucket[bucket["correct"]]["actual_return"].mean() * 100, 2) if bucket["correct"].sum() > 0 else 0,
            })

    # Screener picks performance (predicted UP with high confidence)
    screener_picks = df_preds[df_preds["is_screener_pick"]]
    if len(screener_picks) > 0:
        screener_accuracy = screener_picks["correct"].mean()
        screener_avg_return = screener_picks["actual_return"].mean()
        screener_win_rate = (screener_picks["actual_return"] > 0).mean()
    else:
        screener_accuracy = 0
        screener_avg_return = 0
        screener_win_rate = 0

    # Compute average feature importances
    avg_importances = np.mean(all_feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_importances,
    }).sort_values("importance", ascending=False)

    # ==========================================================================
    # INTERPRETATION
    # ==========================================================================
    interpretation = []

    if overall_accuracy >= 0.60:
        interpretation.append(f"PASS: Overall accuracy {overall_accuracy*100:.1f}% indicates real predictive signal.")
    elif overall_accuracy >= 0.55:
        interpretation.append(f"MARGINAL: Overall accuracy {overall_accuracy*100:.1f}% suggests a smaller edge exists.")
    else:
        interpretation.append(f"FAIL: Overall accuracy {overall_accuracy*100:.1f}% is near random - likely overfitting or leakage.")

    # Check if confidence correlates with accuracy
    high_conf = df_preds[df_preds["confidence"] >= 0.75]
    low_conf = df_preds[df_preds["confidence"] < 0.60]
    if len(high_conf) > 10 and len(low_conf) > 10:
        high_acc = high_conf["correct"].mean()
        low_acc = low_conf["correct"].mean()
        if high_acc > low_acc + 0.05:
            interpretation.append(f"GOOD: High-confidence predictions ({high_acc*100:.1f}%) outperform low-confidence ({low_acc*100:.1f}%).")
        else:
            interpretation.append(f"WARNING: Confidence not well-calibrated. High-conf: {high_acc*100:.1f}%, Low-conf: {low_acc*100:.1f}%.")

    if screener_accuracy >= 0.60:
        interpretation.append(f"SCREENER OK: Screener picks (conf >= {min_confidence}) have {screener_accuracy*100:.1f}% accuracy.")
    elif len(screener_picks) > 0:
        interpretation.append(f"SCREENER WEAK: Screener picks only {screener_accuracy*100:.1f}% accurate.")

    return {
        "summary": {
            "stocks_processed": stocks_processed,
            "stocks_failed": stocks_failed,
            "total_predictions": len(df_preds),
            "overall_accuracy": round(overall_accuracy * 100, 2),
            "train_period": f"{TRAIN_START} to {TRAIN_END}",
            "test_period": f"{TEST_START} to {TEST_END}",
        },
        "confidence_buckets": confidence_buckets,
        "screener_performance": {
            "total_picks": len(screener_picks),
            "accuracy": round(screener_accuracy * 100, 2),
            "win_rate": round(screener_win_rate * 100, 2),
            "avg_return": round(screener_avg_return * 100, 2),
        },
        "feature_importance": feature_importance_df.head(20).to_dict("records"),
        "interpretation": interpretation,
        "_raw_predictions": all_predictions,
        "_feature_names": feature_names,
        "_avg_importances": avg_importances.tolist(),
    }


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(
    year_split_results: Dict,
    top_n: int = TOP_FEATURES_COUNT,
    verbose: bool = True,
) -> Dict:
    """
    Analyze feature importance from the year-split backtest.

    Returns top N features ranked by importance.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)

    feature_names = year_split_results.get("_feature_names", [])
    avg_importances = year_split_results.get("_avg_importances", [])

    if not feature_names or not avg_importances:
        return {"error": "No feature importance data available"}

    # Create ranked list
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_importances,
    }).sort_values("importance", ascending=False)

    # Top N features
    top_features = importance_df.head(top_n)["feature"].tolist()
    top_importance_sum = importance_df.head(top_n)["importance"].sum()
    total_importance = importance_df["importance"].sum()

    if verbose:
        print(f"\nTop {top_n} features capture {top_importance_sum/total_importance*100:.1f}% of total importance:")
        print("-" * 50)
        for i, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']:30s} {row['importance']*100:5.2f}%")

        print(f"\nBottom 5 features (candidates for removal):")
        print("-" * 50)
        for i, row in importance_df.tail(5).iterrows():
            print(f"  {row['feature']:30s} {row['importance']*100:5.2f}%")

    return {
        "top_features": top_features,
        "top_features_importance_pct": round(top_importance_sum / total_importance * 100, 1),
        "all_features_ranked": importance_df.to_dict("records"),
        "recommendation": f"Retrain with top {top_n} features to reduce overfitting risk.",
    }


# =============================================================================
# PRUNED MODEL TEST
# =============================================================================

def run_pruned_model_test(
    top_features: List[str],
    stocks: List[str] = None,
    days: int = 5,
    min_confidence: float = 0.75,
    verbose: bool = True,
) -> Dict:
    """
    Re-run year-split backtest using ONLY the top N features.

    HYPOTHESIS:
    - If pruned model performs similarly or better → original model had noise
    - If pruned model performs much worse → removed features were important
    """
    if stocks is None:
        stocks = VALIDATION_STOCKS

    if verbose:
        print("\n" + "=" * 70)
        print(f"PRUNED MODEL TEST ({len(top_features)} features)")
        print("=" * 70)
        print(f"Features: {', '.join(top_features[:5])}...")

    all_predictions = []
    stocks_processed = 0

    for ticker in stocks:
        if verbose:
            print(f"\nProcessing {ticker}...")

        df = fetch_historical_data(ticker, start_date="2019-01-01")
        if df is None:
            continue

        # Compute ALL features first (needed for technical indicators)
        features = compute_technical_features(df.reset_index())
        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        # Keep only top features
        available_features = [f for f in top_features if f in features.columns]
        if len(available_features) < len(top_features) * 0.8:
            if verbose:
                print(f"  Skipped: Too many missing features")
            continue

        features = features[available_features]

        # Construct target
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
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

        # Train pruned model
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
                "predicted": pred,
                "actual": actual_direction,
                "correct": pred == actual_direction,
                "confidence": confidence,
                "actual_return": actual_return,
                "is_screener_pick": pred == 1 and confidence >= min_confidence,
            })

        if verbose:
            stock_preds = [p for p in all_predictions if p["ticker"] == ticker]
            stock_acc = sum(p["correct"] for p in stock_preds) / len(stock_preds) if stock_preds else 0
            print(f"  {len(stock_preds)} predictions, accuracy: {stock_acc*100:.1f}%")

    if len(all_predictions) == 0:
        return {"error": "No predictions generated"}

    df_preds = pd.DataFrame(all_predictions)
    overall_accuracy = df_preds["correct"].mean()

    # Screener picks
    screener_picks = df_preds[df_preds["is_screener_pick"]]
    screener_accuracy = screener_picks["correct"].mean() if len(screener_picks) > 0 else 0
    screener_avg_return = screener_picks["actual_return"].mean() if len(screener_picks) > 0 else 0

    return {
        "summary": {
            "features_used": len(top_features),
            "stocks_processed": stocks_processed,
            "total_predictions": len(df_preds),
            "overall_accuracy": round(overall_accuracy * 100, 2),
        },
        "screener_performance": {
            "total_picks": len(screener_picks),
            "accuracy": round(screener_accuracy * 100, 2),
            "avg_return": round(screener_avg_return * 100, 2),
        },
    }


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def generate_comparison_report(
    full_model_results: Dict,
    pruned_model_results: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Compare full model vs pruned model performance.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON: FULL MODEL vs PRUNED MODEL")
        print("=" * 70)

    full_acc = full_model_results["summary"]["overall_accuracy"]
    pruned_acc = pruned_model_results["summary"]["overall_accuracy"]

    full_screener = full_model_results["screener_performance"]["accuracy"]
    pruned_screener = pruned_model_results["screener_performance"]["accuracy"]

    comparison = {
        "full_model": {
            "features": "All (~50)",
            "accuracy": full_acc,
            "screener_accuracy": full_screener,
        },
        "pruned_model": {
            "features": pruned_model_results["summary"]["features_used"],
            "accuracy": pruned_acc,
            "screener_accuracy": pruned_screener,
        },
        "difference": {
            "accuracy_delta": round(pruned_acc - full_acc, 2),
            "screener_delta": round(pruned_screener - full_screener, 2),
        },
    }

    # Interpretation
    interpretation = []

    if pruned_acc >= full_acc - 1:
        interpretation.append("GOOD: Pruned model performs similarly. Extra features were likely noise.")
        interpretation.append("RECOMMENDATION: Use pruned model in production for more stable predictions.")
    elif pruned_acc >= full_acc - 3:
        interpretation.append("OK: Small accuracy drop with pruned model. Trade-off may be acceptable for stability.")
    else:
        interpretation.append("WARNING: Significant accuracy drop with pruning. Review which features were removed.")

    if pruned_screener >= full_screener:
        interpretation.append("SCREENER IMPROVED: Pruned model has better screener accuracy.")

    comparison["interpretation"] = interpretation

    if verbose:
        print(f"\n{'Metric':<25} {'Full Model':>15} {'Pruned Model':>15} {'Delta':>10}")
        print("-" * 70)
        print(f"{'Overall Accuracy':<25} {full_acc:>14.1f}% {pruned_acc:>14.1f}% {pruned_acc-full_acc:>+9.1f}%")
        print(f"{'Screener Accuracy':<25} {full_screener:>14.1f}% {pruned_screener:>14.1f}% {pruned_screener-full_screener:>+9.1f}%")
        print()
        for line in interpretation:
            print(f"  {line}")

    return comparison


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_full_validation(
    stocks: List[str] = None,
    days: int = 5,
    min_confidence: float = 0.75,
    top_features: int = TOP_FEATURES_COUNT,
) -> Dict:
    """
    Run complete validation suite:
    1. Year-split backtest with full model
    2. Feature importance analysis
    3. Pruned model test
    4. Comparison report
    """
    print("\n" + "=" * 70)
    print("STOCK PREDICTOR MODEL VALIDATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: {len(stocks or VALIDATION_STOCKS)}")
    print(f"Prediction horizon: {days} days")
    print(f"Confidence threshold: {min_confidence}")

    # Step 1: Year-split backtest
    print("\n" + "-" * 70)
    print("STEP 1/4: Running year-split backtest with full model...")
    print("-" * 70)
    full_results = run_year_split_backtest(
        stocks=stocks,
        days=days,
        min_confidence=min_confidence,
        verbose=True,
    )

    if "error" in full_results:
        print(f"ERROR: {full_results['error']}")
        return full_results

    # Step 2: Feature importance
    print("\n" + "-" * 70)
    print("STEP 2/4: Analyzing feature importance...")
    print("-" * 70)
    importance_results = analyze_feature_importance(
        full_results,
        top_n=top_features,
        verbose=True,
    )

    # Step 3: Pruned model test
    print("\n" + "-" * 70)
    print("STEP 3/4: Running year-split backtest with pruned model...")
    print("-" * 70)
    pruned_results = run_pruned_model_test(
        top_features=importance_results["top_features"],
        stocks=stocks,
        days=days,
        min_confidence=min_confidence,
        verbose=True,
    )

    # Step 4: Comparison
    print("\n" + "-" * 70)
    print("STEP 4/4: Generating comparison report...")
    print("-" * 70)
    comparison = generate_comparison_report(
        full_results,
        pruned_results,
        verbose=True,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE - FINAL SUMMARY")
    print("=" * 70)

    print("\nYear-Split Results (Full Model):")
    for line in full_results["interpretation"]:
        print(f"  {line}")

    print("\nFeature Analysis:")
    print(f"  Top {top_features} features capture {importance_results['top_features_importance_pct']}% of importance")

    print("\nModel Comparison:")
    for line in comparison["interpretation"]:
        print(f"  {line}")

    # Pass/Fail determination
    print("\n" + "-" * 70)
    overall_pass = full_results["summary"]["overall_accuracy"] >= 55
    if overall_pass:
        print("OVERALL: PASS - Model shows real predictive signal on out-of-sample data")
    else:
        print("OVERALL: FAIL - Model does not generalize well to out-of-sample data")
    print("-" * 70)

    return {
        "full_model_results": full_results,
        "feature_importance": importance_results,
        "pruned_model_results": pruned_results,
        "comparison": comparison,
        "overall_pass": overall_pass,
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate stock prediction model")
    parser.add_argument("--days", type=int, default=5, help="Prediction horizon in days")
    parser.add_argument("--confidence", type=float, default=0.75, help="Min confidence threshold")
    parser.add_argument("--top-features", type=int, default=15, help="Number of top features for pruning")
    parser.add_argument("--stocks", type=str, nargs="+", help="Specific stocks to test (default: VALIDATION_STOCKS)")

    args = parser.parse_args()

    results = run_full_validation(
        stocks=args.stocks,
        days=args.days,
        min_confidence=args.confidence,
        top_features=args.top_features,
    )
