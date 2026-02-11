"""
Compare Enhanced Models vs Baseline

This script tests different ML approaches on the same data to find
which model gives the best accuracy and confidence calibration.

Run: python compare_models.py
"""

from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from model import compute_technical_features, add_fundamental_features
from model_enhanced import EnhancedPredictor, compare_models, HAS_XGB, HAS_LGB
from data import fetch_fundamental_data
from config import MIN_DATA_POINTS


# Configuration
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# Test on Tier 1 stocks (best performers)
TIER1_STOCKS = ["MCD", "QQQ", "MSFT", "MA", "COST", "SPY"]
TIER2_STOCKS = ["V", "AAPL", "PEP", "JPM", "PG", "HD"]
ALL_STOCKS = TIER1_STOCKS + TIER2_STOCKS


def fetch_historical_data(ticker: str, start_date: str = "2019-01-01"):
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


def run_model_comparison(
    stocks: List[str] = None,
    days: int = 20,
    verbose: bool = True,
) -> Dict:
    """
    Compare different model types on the same test data.
    """
    if stocks is None:
        stocks = TIER1_STOCKS

    print("=" * 70)
    print("MODEL COMPARISON: Enhanced ML vs Baseline")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Training: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing: {TEST_START} to {TEST_END}")
    print(f"Horizon: {days} days")
    print(f"Stocks: {stocks}")
    print()
    print(f"Available models:")
    print(f"  - RandomForest (baseline)")
    print(f"  - GradientBoosting (sklearn)")
    print(f"  - XGBoost: {'YES' if HAS_XGB else 'NO (pip install xgboost)'}")
    print(f"  - LightGBM: {'YES' if HAS_LGB else 'NO (pip install lightgbm)'}")
    print(f"  - Ensemble (combines available models)")
    print("=" * 70)

    # Collect all training and test data across stocks
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []
    all_test_returns = []
    feature_names = None

    for ticker in stocks:
        if verbose:
            print(f"\nLoading {ticker}...")

        df = fetch_historical_data(ticker)
        if df is None:
            continue

        features = compute_technical_features(df.reset_index())
        fundamentals = fetch_fundamental_data(ticker)
        features = add_fundamental_features(features, fundamentals)

        if feature_names is None:
            feature_names = features.columns.tolist()

        # Target: direction
        future_return = df["close"].shift(-days) / df["close"] - 1
        target = (future_return > 0).astype(int)

        features.index = df.index
        data = features.copy()
        data["target"] = target
        data["future_return"] = future_return
        data = data.ffill().bfill()

        # Split
        train = data[(data.index >= TRAIN_START) & (data.index <= TRAIN_END)].dropna()
        test = data[(data.index >= TEST_START) & (data.index <= TEST_END)].dropna()

        if len(train) < MIN_DATA_POINTS or len(test) < 10:
            continue

        # Add training data (use all training samples)
        for i in range(len(train)):
            all_X_train.append(train[feature_names].iloc[i].values)
            all_y_train.append(train["target"].iloc[i])

        # Sample test at non-overlapping intervals
        test_indices = list(range(0, len(test) - days, days))

        # For test, use actual test samples
        for idx in test_indices:
            actual_idx = idx + days
            if actual_idx < len(test):
                all_X_test.append(test[feature_names].iloc[idx].values)
                future_ret = test["future_return"].iloc[idx]
                all_y_test.append(1 if future_ret > 0 else 0)
                all_test_returns.append(float(future_ret) if not np.isnan(future_ret) else 0)

    if len(all_X_test) == 0:
        return {"error": "No test data"}

    # Convert to arrays
    X_train = np.array(all_X_train)
    y_train = np.array(all_y_train)
    X_test = np.array(all_X_test)
    y_test = np.array(all_y_test)

    # Replace inf/nan
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    print(f"\nData: {len(X_train)} train samples, {len(X_test)} test samples")
    print(f"Features: {len(feature_names)}")

    # Run comparison
    print("\n" + "=" * 70)
    print("TESTING MODELS...")
    print("=" * 70)

    results = {}
    model_types = ["rf", "gradient_boost"]
    if HAS_XGB:
        model_types.append("xgboost")
    if HAS_LGB:
        model_types.append("lightgbm")
    model_types.append("ensemble")

    for model_type in model_types:
        print(f"\nTesting {model_type}...")

        try:
            predictor = EnhancedPredictor(
                model_type=model_type,
                feature_selection=True,
                calibrate=False,
            )
            predictor.fit(X_train, y_train, feature_names)
            predictions, confidences = predictor.predict(X_test)

            # Metrics
            accuracy = (predictions == y_test).mean()

            # Accuracy at different confidence levels
            conf_70 = confidences >= 0.70
            conf_75 = confidences >= 0.75
            conf_80 = confidences >= 0.80

            acc_70 = (predictions[conf_70] == y_test[conf_70]).mean() if conf_70.sum() > 0 else 0
            acc_75 = (predictions[conf_75] == y_test[conf_75]).mean() if conf_75.sum() > 0 else 0
            acc_80 = (predictions[conf_80] == y_test[conf_80]).mean() if conf_80.sum() > 0 else 0

            # Average return when predicting UP with high confidence
            up_high_conf = (predictions == 1) & conf_75
            if up_high_conf.sum() > 0:
                avg_return = np.array(all_test_returns)[up_high_conf].mean()
            else:
                avg_return = 0

            results[model_type] = {
                "accuracy": round(accuracy * 100, 2),
                "acc_70": round(acc_70 * 100, 2),
                "acc_75": round(acc_75 * 100, 2),
                "acc_80": round(acc_80 * 100, 2),
                "count_75": int(conf_75.sum()),
                "count_80": int(conf_80.sum()),
                "avg_return_75": round(avg_return * 100, 3),
                "avg_confidence": round(confidences.mean() * 100, 2),
            }

            # Show selected features for ensemble
            if model_type == "ensemble" and predictor.selected_features:
                results[model_type]["selected_features"] = len(predictor.selected_features)

            print(f"  Accuracy: {accuracy*100:.1f}%")
            print(f"  @ 75% conf: {acc_75*100:.1f}% ({conf_75.sum()} signals)")

        except Exception as e:
            results[model_type] = {"error": str(e)}
            print(f"  Error: {e}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<20} {'Accuracy':>10} {'@75% Conf':>10} {'Signals':>10} {'Avg Ret':>10}")
    print("-" * 60)

    baseline_acc = 0
    for model_type in model_types:
        if "error" in results[model_type]:
            print(f"{model_type:<20} {'ERROR':>10}")
            continue

        r = results[model_type]
        if model_type == "rf":
            baseline_acc = r["accuracy"]

        delta = ""
        if baseline_acc > 0 and model_type != "rf":
            diff = r["accuracy"] - baseline_acc
            delta = f" ({diff:+.1f})" if diff != 0 else ""

        print(f"{model_type:<20} {r['accuracy']:>9.1f}%{delta:>0} {r['acc_75']:>9.1f}% {r['count_75']:>10} {r['avg_return_75']:>9.2f}%")

    # Find best model
    best_model = max(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1]["acc_75"],
        default=(None, None)
    )

    print("\n" + "-" * 60)
    if best_model[0]:
        print(f"BEST MODEL: {best_model[0]}")
        print(f"  - {best_model[1]['acc_75']}% accuracy at 75% confidence")
        print(f"  - {best_model[1]['count_75']} high-conviction signals")
        print(f"  - {best_model[1]['avg_return_75']}% average return per signal")

        if best_model[0] != "rf":
            improvement = best_model[1]["accuracy"] - baseline_acc
            print(f"\n  vs Baseline RF: {improvement:+.1f}% improvement")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if not HAS_XGB:
        print("\n  - Install XGBoost for better performance: pip install xgboost")
    if not HAS_LGB:
        print("\n  - Install LightGBM for faster training: pip install lightgbm")

    if best_model[0] == "ensemble":
        print("\n  - Ensemble combines multiple models for more robust predictions")
        print("  - Consider using ensemble for production")
    elif best_model[0] in ["xgboost", "lightgbm"]:
        print(f"\n  - {best_model[0]} outperforms RandomForest on this data")
        print(f"  - Consider switching to {best_model[0]} for better accuracy")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20, help="Prediction horizon")
    parser.add_argument("--tier1-only", action="store_true", help="Test only Tier 1 stocks")
    parser.add_argument("--quiet", action="store_true", help="Less output")

    args = parser.parse_args()

    stocks = TIER1_STOCKS if args.tier1_only else ALL_STOCKS

    results = run_model_comparison(
        stocks=stocks,
        days=args.days,
        verbose=not args.quiet,
    )
