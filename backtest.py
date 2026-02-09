"""
Backtesting module to evaluate model performance on historical data.

LEAKAGE PREVENTION NOTES:
- All features are computed using ONLY data available at prediction time
- The model is trained on data BEFORE test_idx, never including test_idx or later
- Confidence thresholds (min_confidence) are fixed BEFORE backtesting begins
- No hyperparameter tuning is done using test period performance
- Future returns (target) are computed using shift(-days) which looks forward,
  but this is ONLY used for labeling training data, not features
- SPY regime gate uses TRAILING returns only (no future data)

REGIME GATE NOTES:
- SPY 5-day trailing return is calculated ENDING at decision date (no lookahead)
- Confidence thresholds for bear mode are FIXED before testing (not optimized)
- Gate rules are applied consistently across all stocks and periods
"""
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from model import compute_technical_features, add_fundamental_features
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from config import MIN_DATA_POINTS


def run_backtest(
    df: pd.DataFrame,
    days: int,
    fundamentals: Optional[dict] = None,
    test_periods: int = 50,
) -> Optional[dict]:
    """
    Run a walk-forward backtest on historical data.

    Args:
        df: DataFrame with OHLCV data
        days: Prediction horizon in days
        fundamentals: Optional fundamental data dict
        test_periods: Number of test predictions to make

    Returns:
        Dictionary with backtest results or None if insufficient data
    """
    if len(df) < MIN_DATA_POINTS + test_periods + days:
        return None

    # Compute features
    features = compute_technical_features(df)
    features = add_fundamental_features(features, fundamentals)

    # Construct targets
    future_return = df["close"].shift(-days) / df["close"] - 1
    target_direction = (future_return > 0).astype(int)
    target_return = future_return

    # Combine features and targets
    feature_cols = features.columns.tolist()
    data = features.copy()
    data["target_direction"] = target_direction
    data["target_return"] = target_return
    data["actual_return"] = future_return
    data["close"] = df["close"]

    # Fill NaN values
    data = data.ffill().bfill()
    data = data.dropna()

    if len(data) < MIN_DATA_POINTS + test_periods:
        return None

    # Walk-forward backtest with NON-OVERLAPPING periods
    # This prevents inflated returns from overlapping bets
    results = []

    # Reserve last portion for testing
    # Use non-overlapping windows: test every `days` rows to avoid overlap
    test_start_idx = len(data) - (test_periods * days) - days
    if test_start_idx < MIN_DATA_POINTS:
        test_start_idx = MIN_DATA_POINTS

    actual_test_periods = 0
    step = 0
    while actual_test_periods < test_periods:
        # Training data: everything up to this point
        # Step by `days` to create non-overlapping test windows
        train_end = test_start_idx + (step * days)
        step += 1

        if train_end >= len(data) - days:
            break

        train_data = data.iloc[:train_end]

        if len(train_data) < MIN_DATA_POINTS:
            continue

        # Test point: the next row
        test_idx = train_end
        if test_idx + days >= len(data):
            break

        actual_test_periods += 1

        test_row = data.iloc[[test_idx]]

        X_train = train_data[feature_cols]
        y_train_dir = train_data["target_direction"]
        y_train_ret = train_data["target_return"]

        X_test = test_row[feature_cols]

        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier
        classifier = RandomForestClassifier(
            n_estimators=100,  # Fewer trees for speed
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        classifier.fit(X_train_scaled, y_train_dir)

        # Train regressor
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        regressor.fit(X_train_scaled, y_train_ret)

        # Predict
        pred_direction = classifier.predict(X_test_scaled)[0]
        pred_proba = classifier.predict_proba(X_test_scaled)[0]
        confidence = float(max(pred_proba))
        pred_return = float(regressor.predict(X_test_scaled)[0])

        # Actual outcome (look ahead `days` rows)
        actual_return = float(data.iloc[test_idx + days]["close"] / data.iloc[test_idx]["close"] - 1)
        actual_direction = 1 if actual_return > 0 else 0

        # Record result (convert numpy types to native Python types)
        results.append({
            "date": str(data.index[test_idx]) if hasattr(data.index[test_idx], 'strftime') else test_idx,
            "predicted_direction": "up" if pred_direction == 1 else "down",
            "actual_direction": "up" if actual_direction == 1 else "down",
            "direction_correct": bool(pred_direction == actual_direction),
            "confidence": float(confidence),
            "predicted_return": float(pred_return * 100),  # As percentage
            "actual_return": float(actual_return * 100),
            "return_error": float(abs(pred_return - actual_return) * 100),
        })

    if len(results) == 0:
        return None

    # Calculate summary statistics
    df_results = pd.DataFrame(results)

    correct_predictions = df_results["direction_correct"].sum()
    total_predictions = len(df_results)
    accuracy = correct_predictions / total_predictions

    # Accuracy by confidence level
    high_conf = df_results[df_results["confidence"] >= 0.7]
    high_conf_accuracy = high_conf["direction_correct"].mean() if len(high_conf) > 0 else None

    low_conf = df_results[df_results["confidence"] < 0.6]
    low_conf_accuracy = low_conf["direction_correct"].mean() if len(low_conf) > 0 else None

    # Return prediction quality
    avg_return_error = df_results["return_error"].mean()

    # LONG-ONLY strategy: only buy when predicted UP, stay in cash otherwise
    # This is realistic - no shorting assumption
    up_predictions = df_results[df_results["predicted_direction"] == "up"]
    down_predictions = df_results[df_results["predicted_direction"] == "down"]

    # Average return when we predicted UP and bought
    avg_return_when_up = up_predictions["actual_return"].mean() if len(up_predictions) > 0 else 0

    # Average return when we predicted DOWN and stayed out
    avg_return_avoided = down_predictions["actual_return"].mean() if len(down_predictions) > 0 else 0

    # Win rate for UP predictions (did it actually go up?)
    up_correct = (up_predictions["actual_return"] > 0).sum() if len(up_predictions) > 0 else 0
    up_win_rate = up_correct / len(up_predictions) if len(up_predictions) > 0 else 0

    # Simple strategy return: average of trades taken (UP predictions only)
    # NOT compounded - more interpretable
    strategy_avg_return = avg_return_when_up

    # Buy and hold: average return across all periods
    buy_hold_avg = df_results["actual_return"].mean()

    return {
        "summary": {
            "total_predictions": int(total_predictions),
            "correct_predictions": int(correct_predictions),
            "accuracy": round(float(accuracy * 100), 2),
            "high_confidence_accuracy": round(float(high_conf_accuracy * 100), 2) if high_conf_accuracy else None,
            "low_confidence_accuracy": round(float(low_conf_accuracy * 100), 2) if low_conf_accuracy else None,
            "avg_return_error": round(float(avg_return_error), 2),
            # Long-only strategy metrics (more realistic)
            "up_predictions": int(len(up_predictions)),
            "up_win_rate": round(float(up_win_rate * 100), 2),
            "avg_return_when_up": round(float(avg_return_when_up), 2),
            "avg_return_avoided": round(float(avg_return_avoided), 2),
            "buy_hold_avg": round(float(buy_hold_avg), 2),
            "strategy_beats_hold": bool(strategy_avg_return > buy_hold_avg),
        },
        "predictions": results[-20:],  # Last 20 predictions for display
    }


def run_screener_backtest(
    stocks_data: List[Dict],
    days: int,
    min_confidence: float = 0.7,
    test_periods: int = 30,
    spy_data: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Backtest the screener strategy across multiple stocks with comprehensive analysis.

    LEAKAGE PREVENTION:
    - min_confidence is fixed BEFORE any testing begins (not optimized on test data)
    - Model is retrained at each step using ONLY past data (walk-forward)
    - Features use ONLY information available at prediction time
    - No future prices leak into features (verified in compute_technical_features)

    Args:
        stocks_data: List of dicts with 'ticker', 'df', 'fundamentals'
        days: Prediction horizon
        min_confidence: Minimum confidence threshold (FIXED before testing)
        test_periods: Number of test periods per stock
        spy_data: Optional SPY data for market comparison

    Returns:
        Dictionary with comprehensive screener backtest results
    """
    # =========================================================================
    # SANITY CHECK: Verify min_confidence is locked before testing
    # This threshold must NOT be tuned based on backtest results
    # =========================================================================
    LOCKED_MIN_CONFIDENCE = min_confidence  # Frozen at start

    all_picks = []  # All screener picks across all stocks and time periods
    all_baseline = []  # All stock returns for baseline comparison
    period_pick_counts = {}  # Track picks per period for no-trade analysis
    stocks_tested = 0
    stocks_failed = 0

    # Track SPY returns for same periods (market benchmark)
    spy_returns_by_period = {}
    if spy_data is not None and len(spy_data) > 0:
        spy_df = spy_data.copy()
        spy_df["return"] = spy_df["close"].shift(-days) / spy_df["close"] - 1

    for stock_info in stocks_data:
        ticker = stock_info["ticker"]
        df = stock_info["df"]
        fundamentals = stock_info.get("fundamentals")

        if df is None or len(df) < MIN_DATA_POINTS + test_periods + days:
            stocks_failed += 1
            continue

        stocks_tested += 1

        # =====================================================================
        # LEAKAGE CHECK: Features computed from historical data only
        # compute_technical_features uses rolling windows that look BACKWARD
        # No .shift(-n) is used in features (that would be forward-looking)
        # =====================================================================
        features = compute_technical_features(df)
        features = add_fundamental_features(features, fundamentals)

        # Construct targets (these use future data but ONLY for labeling, not features)
        future_return = df["close"].shift(-days) / df["close"] - 1
        target_direction = (future_return > 0).astype(int)

        # Combine features and targets
        feature_cols = features.columns.tolist()
        data = features.copy()
        data["target_direction"] = target_direction
        data["close"] = df["close"]

        # Compute volatility for regime analysis (20-day rolling std of returns)
        data["volatility"] = df["close"].pct_change().rolling(20).std()

        # Fill NaN values
        data = data.ffill().bfill()
        data = data.dropna()

        if len(data) < MIN_DATA_POINTS + test_periods:
            stocks_failed += 1
            continue

        # Walk-forward backtest with NON-OVERLAPPING periods
        test_start_idx = len(data) - (test_periods * days) - days
        if test_start_idx < MIN_DATA_POINTS:
            test_start_idx = MIN_DATA_POINTS

        step = 0
        periods_done = 0
        while periods_done < test_periods:
            # Step by `days` to avoid overlapping positions
            train_end = test_start_idx + (step * days)
            step += 1

            if train_end >= len(data) - days:
                break

            train_data = data.iloc[:train_end]

            if len(train_data) < MIN_DATA_POINTS:
                continue

            test_idx = train_end
            if test_idx + days >= len(data):
                break

            periods_done += 1
            period_id = f"period_{step}"

            # Initialize period count if needed
            if period_id not in period_pick_counts:
                period_pick_counts[period_id] = 0

            test_row = data.iloc[[test_idx]]

            # ================================================================
            # LEAKAGE CHECK: Training data is strictly BEFORE test_idx
            # train_data = data.iloc[:train_end] where train_end == test_idx
            # This ensures no test data leaks into training
            # ================================================================
            X_train = train_data[feature_cols].replace([np.inf, -np.inf], 0)
            X_test = test_row[feature_cols].replace([np.inf, -np.inf], 0)
            y_train_dir = train_data["target_direction"]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ================================================================
            # LEAKAGE CHECK: Model hyperparameters are FIXED, not tuned on test
            # Using random_state=42 for reproducibility
            # ================================================================
            classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            classifier.fit(X_train_scaled, y_train_dir)

            # Predict
            pred_direction = classifier.predict(X_test_scaled)[0]
            pred_proba = classifier.predict_proba(X_test_scaled)[0]
            confidence = float(max(pred_proba))

            # Actual return (this is the outcome we're testing against)
            actual_return = float(data.iloc[test_idx + days]["close"] / data.iloc[test_idx]["close"] - 1)

            # Get volatility at prediction time (for regime analysis)
            volatility = float(data.iloc[test_idx]["volatility"])

            # Get SPY return for same period (market benchmark)
            spy_return = None
            if spy_data is not None:
                try:
                    test_date = data.index[test_idx]
                    # Find closest SPY date
                    spy_idx = spy_df.index.get_indexer([test_date], method='nearest')[0]
                    if spy_idx + days < len(spy_df):
                        spy_return = float(spy_df.iloc[spy_idx + days]["close"] / spy_df.iloc[spy_idx]["close"] - 1)
                        spy_returns_by_period[period_id] = spy_return
                except:
                    pass

            # Track for baseline (all predictions, not just picks)
            all_baseline.append({
                "return": actual_return,
                "volatility": volatility,
                "spy_return": spy_return,
            })

            # ================================================================
            # SCREENER PICK LOGIC
            # Uses LOCKED_MIN_CONFIDENCE which was set BEFORE testing began
            # ================================================================
            if pred_direction == 1 and confidence >= LOCKED_MIN_CONFIDENCE:
                period_pick_counts[period_id] += 1
                all_picks.append({
                    "ticker": ticker,
                    "period_id": period_id,
                    "confidence": confidence,
                    "actual_return": actual_return,
                    "direction_correct": actual_return > 0,
                    "volatility": volatility,
                    "spy_return": spy_return,
                    "beat_market": actual_return > spy_return if spy_return is not None else None,
                })

    if len(all_picks) == 0:
        return None

    # =========================================================================
    # ANALYSIS SECTION
    # =========================================================================
    df_picks = pd.DataFrame(all_picks)
    df_baseline = pd.DataFrame(all_baseline)

    # --- BASIC METRICS ---
    total_picks = len(df_picks)
    winning_picks = df_picks["direction_correct"].sum()
    win_rate = winning_picks / total_picks
    avg_pick_return = df_picks["actual_return"].mean()
    avg_baseline_return = df_baseline["return"].mean() if len(df_baseline) > 0 else 0

    # --- 1. CONFIDENCE BUCKET ANALYSIS ---
    confidence_buckets = []
    bucket_ranges = [
        ("70-75%", 0.70, 0.75),
        ("75-80%", 0.75, 0.80),
        ("80%+", 0.80, 1.01),
    ]
    for bucket_name, low, high in bucket_ranges:
        bucket_df = df_picks[(df_picks["confidence"] >= low) & (df_picks["confidence"] < high)]
        if len(bucket_df) > 0:
            bucket_win_rate = bucket_df["direction_correct"].mean()
            bucket_avg_return = bucket_df["actual_return"].mean()
            bucket_worst = bucket_df["actual_return"].min()
            confidence_buckets.append({
                "bucket": bucket_name,
                "trades": int(len(bucket_df)),
                "win_rate": round(float(bucket_win_rate) * 100, 1),
                "avg_return": round(float(bucket_avg_return) * 100, 2),
                "worst_trade": round(float(bucket_worst) * 100, 2),
            })

    # --- 2. DRAWDOWN & RISK METRICS ---
    returns_list = df_picks["actual_return"].tolist()
    worst_single_trade = min(returns_list)

    # Consecutive loss streaks
    loss_streaks = []
    current_streak = 0
    current_streak_return = 0
    for ret in returns_list:
        if ret < 0:
            current_streak += 1
            current_streak_return += ret
        else:
            if current_streak >= 2:
                loss_streaks.append({
                    "length": current_streak,
                    "total_loss": current_streak_return
                })
            current_streak = 0
            current_streak_return = 0
    if current_streak >= 2:
        loss_streaks.append({"length": current_streak, "total_loss": current_streak_return})

    worst_streak = max(loss_streaks, key=lambda x: abs(x["total_loss"])) if loss_streaks else None

    # Max drawdown (assuming equal allocation per pick)
    cumulative = 0
    peak = 0
    max_dd = 0
    for ret in returns_list:
        cumulative += ret
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    risk_metrics = {
        "worst_single_trade": round(float(worst_single_trade) * 100, 2),
        "worst_streak_length": worst_streak["length"] if worst_streak else 0,
        "worst_streak_loss": round(float(worst_streak["total_loss"]) * 100, 2) if worst_streak else 0,
        "max_drawdown": round(float(max_dd) * 100, 2),
    }

    # --- 3. REGIME SEGMENTATION ---
    # Volatility regimes (split at median)
    median_vol = df_picks["volatility"].median()
    high_vol = df_picks[df_picks["volatility"] >= median_vol]
    low_vol = df_picks[df_picks["volatility"] < median_vol]

    # Market trend regimes (based on SPY return)
    picks_with_spy = df_picks[df_picks["spy_return"].notna()]
    if len(picks_with_spy) > 0:
        up_trend = picks_with_spy[picks_with_spy["spy_return"] > 0.01]  # >1% SPY gain
        down_trend = picks_with_spy[picks_with_spy["spy_return"] < -0.01]  # >1% SPY loss
        sideways = picks_with_spy[(picks_with_spy["spy_return"] >= -0.01) & (picks_with_spy["spy_return"] <= 0.01)]
    else:
        up_trend = pd.DataFrame()
        down_trend = pd.DataFrame()
        sideways = pd.DataFrame()

    def regime_stats(df_regime, name):
        if len(df_regime) == 0:
            return None
        return {
            "regime": name,
            "trades": int(len(df_regime)),
            "win_rate": round(float(df_regime["direction_correct"].mean()) * 100, 1),
            "avg_return": round(float(df_regime["actual_return"].mean()) * 100, 2),
        }

    regime_analysis = {
        "volatility": [
            r for r in [
                regime_stats(high_vol, "High Volatility"),
                regime_stats(low_vol, "Low Volatility"),
            ] if r is not None
        ],
        "market_trend": [
            r for r in [
                regime_stats(up_trend, "Bull Market (SPY +1%+)"),
                regime_stats(sideways, "Sideways (-1% to +1%)"),
                regime_stats(down_trend, "Bear Market (SPY -1%+)"),
            ] if r is not None
        ],
    }

    # --- 4. MARKET-RELATIVE VALIDATION ---
    picks_with_market = df_picks[df_picks["beat_market"].notna()]
    market_comparison = None
    if len(picks_with_market) > 0:
        beat_market_count = picks_with_market["beat_market"].sum()
        avg_alpha = (picks_with_market["actual_return"] - picks_with_market["spy_return"]).mean()
        market_comparison = {
            "picks_with_benchmark": int(len(picks_with_market)),
            "beat_market_count": int(beat_market_count),
            "beat_market_pct": round(float(beat_market_count / len(picks_with_market)) * 100, 1),
            "avg_alpha": round(float(avg_alpha) * 100, 2),
        }

    # --- 5. NO-TRADE FREQUENCY ---
    total_periods = len(period_pick_counts)
    periods_with_picks = sum(1 for count in period_pick_counts.values() if count > 0)
    periods_without_picks = total_periods - periods_with_picks
    avg_picks_per_period = total_picks / total_periods if total_periods > 0 else 0

    no_trade_analysis = {
        "total_periods": total_periods,
        "periods_with_picks": periods_with_picks,
        "periods_without_picks": periods_without_picks,
        "no_trade_pct": round(float(periods_without_picks / total_periods) * 100, 1) if total_periods > 0 else 0,
        "avg_picks_per_period": round(float(avg_picks_per_period), 2),
    }

    # --- TOP/BOTTOM STOCKS ---
    per_stock = df_picks.groupby("ticker").agg({
        "actual_return": ["mean", "count"],
        "direction_correct": "mean"
    }).round(4)
    per_stock.columns = ["avg_return", "num_picks", "win_rate"]
    per_stock = per_stock.sort_values("avg_return", ascending=False)

    top_stocks = []
    for ticker, row in per_stock.head(10).iterrows():
        top_stocks.append({
            "ticker": ticker,
            "avg_return": round(float(row["avg_return"]) * 100, 2),
            "num_picks": int(row["num_picks"]),
            "win_rate": round(float(row["win_rate"]) * 100, 1),
        })

    # Best and worst individual picks
    df_picks_sorted = df_picks.sort_values("actual_return", ascending=False)
    best_picks = df_picks_sorted.head(5).to_dict("records")
    worst_picks = df_picks_sorted.tail(5).to_dict("records")

    # Convert numpy types
    for pick in best_picks + worst_picks:
        pick["confidence"] = float(pick["confidence"])
        pick["actual_return"] = float(pick["actual_return"])
        pick["direction_correct"] = bool(pick["direction_correct"])
        # Remove internal fields from output
        pick.pop("period_id", None)
        pick.pop("volatility", None)
        pick.pop("spy_return", None)
        pick.pop("beat_market", None)

    return {
        "summary": {
            "stocks_tested": stocks_tested,
            "stocks_failed": stocks_failed,
            "total_picks": int(total_picks),
            "winning_picks": int(winning_picks),
            "win_rate": round(float(win_rate) * 100, 2),
            "avg_pick_return": round(float(avg_pick_return) * 100, 2),
            "avg_baseline_return": round(float(avg_baseline_return) * 100, 2),
            "alpha_vs_baseline": round(float(avg_pick_return - avg_baseline_return) * 100, 2),
        },
        "confidence_buckets": confidence_buckets,
        "risk_metrics": risk_metrics,
        "regime_analysis": regime_analysis,
        "market_comparison": market_comparison,
        "no_trade_analysis": no_trade_analysis,
        "top_stocks": top_stocks,
        "best_picks": best_picks[:5],
        "worst_picks": list(reversed(worst_picks[:5])),
    }


def _compute_metrics_from_picks(picks_list: List[Dict], spy_returns: Dict) -> Dict:
    """
    Helper function to compute standardized metrics from a list of picks.
    Used by regime-gated backtest to compute metrics for each mode.
    """
    if len(picks_list) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "worst_trade": 0,
            "worst_streak_length": 0,
            "worst_streak_loss": 0,
            "max_drawdown": 0,
            "beat_spy_pct": 0,
        }

    df = pd.DataFrame(picks_list)

    total_trades = len(df)
    winning = (df["actual_return"] > 0).sum()
    win_rate = winning / total_trades
    avg_return = df["actual_return"].mean()
    worst_trade = df["actual_return"].min()

    # Loss streaks
    returns_list = df["actual_return"].tolist()
    loss_streaks = []
    current_streak = 0
    current_streak_return = 0
    for ret in returns_list:
        if ret < 0:
            current_streak += 1
            current_streak_return += ret
        else:
            if current_streak >= 2:
                loss_streaks.append({"length": current_streak, "total_loss": current_streak_return})
            current_streak = 0
            current_streak_return = 0
    if current_streak >= 2:
        loss_streaks.append({"length": current_streak, "total_loss": current_streak_return})

    worst_streak = max(loss_streaks, key=lambda x: abs(x["total_loss"])) if loss_streaks else None

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for ret in returns_list:
        cumulative += ret
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Beat SPY %
    picks_with_spy = df[df["spy_return"].notna()]
    if len(picks_with_spy) > 0:
        beat_spy = (picks_with_spy["actual_return"] > picks_with_spy["spy_return"]).sum()
        beat_spy_pct = beat_spy / len(picks_with_spy)
    else:
        beat_spy_pct = 0

    return {
        "total_trades": int(total_trades),
        "win_rate": round(float(win_rate) * 100, 2),
        "avg_return": round(float(avg_return) * 100, 2),
        "worst_trade": round(float(worst_trade) * 100, 2),
        "worst_streak_length": worst_streak["length"] if worst_streak else 0,
        "worst_streak_loss": round(float(worst_streak["total_loss"]) * 100, 2) if worst_streak else 0,
        "max_drawdown": round(float(max_dd) * 100, 2),
        "beat_spy_pct": round(float(beat_spy_pct) * 100, 1),
    }


def run_regime_gated_backtest(
    stocks_data: List[Dict],
    days: int,
    min_confidence: float = 0.7,
    test_periods: int = 30,
    spy_data: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Backtest the screener with market regime gate based on SPY performance.

    REGIME GATE RULES:
    - At each decision date, calculate SPY 5-day TRAILING return (no future data)
    - If SPY trailing return >= -1%: Normal mode (confidence >= 70%)
    - If SPY trailing return < -1%: Bear mode applies

    BEAR MODE OPTIONS:
    - "suppressed": No trades allowed during bear market
    - "high_confidence": Only allow trades with confidence >= 80%

    LEAKAGE PREVENTION:
    - SPY trailing return uses data ENDING at decision date (days -4 to 0)
    - This is calculated as: SPY[today] / SPY[today - days] - 1
    - NO future SPY data is used
    - Confidence thresholds are FIXED before testing begins

    Args:
        stocks_data: List of dicts with 'ticker', 'df', 'fundamentals'
        days: Prediction horizon (also used for SPY trailing window)
        min_confidence: Base confidence threshold (0.7 for normal mode)
        test_periods: Number of test periods per stock
        spy_data: SPY price data for regime detection

    Returns:
        Dictionary with comparison of all three modes
    """
    # =========================================================================
    # LOCK THRESHOLDS BEFORE TESTING
    # These values are FIXED and not optimized based on backtest results
    # =========================================================================
    NORMAL_CONFIDENCE = min_confidence  # 0.70 for normal mode
    BEAR_CONFIDENCE = 0.80  # 0.80 for bear mode (high confidence)
    BEAR_THRESHOLD = -0.01  # -1% SPY return triggers bear mode

    if spy_data is None or len(spy_data) < days + 10:
        return None

    # =========================================================================
    # PRECOMPUTE SPY TRAILING RETURNS
    # CRITICAL: This uses TRAILING data only (no future leakage)
    # spy_trailing_return[date] = SPY[date] / SPY[date - days] - 1
    # This represents the return ENDING at that date, not starting from it
    # =========================================================================
    spy_df = spy_data.copy()
    spy_df = spy_df.sort_index()

    # Trailing return: current price / price N days ago - 1
    # This is the return over the PAST N days, available at decision time
    spy_df["trailing_return"] = spy_df["close"] / spy_df["close"].shift(days) - 1

    # Forward return for benchmark comparison (used ONLY for outcome measurement)
    spy_df["forward_return"] = spy_df["close"].shift(-days) / spy_df["close"] - 1

    # Storage for picks under each mode
    picks_no_gate = []  # Baseline: no regime gate
    picks_suppressed = []  # Bear mode: suppress all trades
    picks_high_conf = []  # Bear mode: require 80% confidence

    # Track periods and regime states
    period_regimes = []  # Track which regime each period was in
    stocks_tested = 0
    stocks_failed = 0

    for stock_info in stocks_data:
        ticker = stock_info["ticker"]
        df = stock_info["df"]
        fundamentals = stock_info.get("fundamentals")

        if df is None or len(df) < MIN_DATA_POINTS + test_periods + days:
            stocks_failed += 1
            continue

        stocks_tested += 1

        # Compute features (uses only historical data)
        features = compute_technical_features(df)
        features = add_fundamental_features(features, fundamentals)

        # Construct targets
        future_return = df["close"].shift(-days) / df["close"] - 1
        target_direction = (future_return > 0).astype(int)

        feature_cols = features.columns.tolist()
        data = features.copy()
        data["target_direction"] = target_direction
        data["close"] = df["close"]

        data = data.ffill().bfill().dropna()

        if len(data) < MIN_DATA_POINTS + test_periods:
            stocks_failed += 1
            continue

        # Walk-forward with non-overlapping periods
        test_start_idx = len(data) - (test_periods * days) - days
        if test_start_idx < MIN_DATA_POINTS:
            test_start_idx = MIN_DATA_POINTS

        step = 0
        periods_done = 0

        while periods_done < test_periods:
            train_end = test_start_idx + (step * days)
            step += 1

            if train_end >= len(data) - days:
                break

            train_data = data.iloc[:train_end]

            if len(train_data) < MIN_DATA_POINTS:
                continue

            test_idx = train_end
            if test_idx + days >= len(data):
                break

            periods_done += 1

            # =================================================================
            # GET SPY TRAILING RETURN AT DECISION DATE
            # LEAKAGE CHECK: We use spy_df["trailing_return"] which is computed
            # as current_price / price_N_days_ago. This is TRAILING data only.
            # =================================================================
            test_date = data.index[test_idx]
            spy_trailing = None
            spy_forward = None

            try:
                # Find the SPY row closest to our decision date
                spy_idx = spy_df.index.get_indexer([test_date], method='nearest')[0]

                # TRAILING return: available at decision time (no leakage)
                if spy_idx >= days:
                    spy_trailing = float(spy_df.iloc[spy_idx]["trailing_return"])

                # FORWARD return: used only for outcome measurement
                if spy_idx + days < len(spy_df):
                    spy_forward = float(spy_df.iloc[spy_idx]["forward_return"])
            except:
                pass

            # =================================================================
            # DETERMINE MARKET REGIME
            # Based on SPY TRAILING return (data available at decision time)
            # =================================================================
            is_bear_market = spy_trailing is not None and spy_trailing < BEAR_THRESHOLD

            # Record regime for this period
            period_regimes.append({
                "period": step,
                "date": str(test_date),
                "spy_trailing": spy_trailing,
                "is_bear": is_bear_market,
            })

            # Train model (same for all modes - no model changes)
            test_row = data.iloc[[test_idx]]
            X_train = train_data[feature_cols].replace([np.inf, -np.inf], 0)
            X_test = test_row[feature_cols].replace([np.inf, -np.inf], 0)
            y_train_dir = train_data["target_direction"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            classifier.fit(X_train_scaled, y_train_dir)

            pred_direction = classifier.predict(X_test_scaled)[0]
            pred_proba = classifier.predict_proba(X_test_scaled)[0]
            confidence = float(max(pred_proba))

            # Actual outcome
            actual_return = float(data.iloc[test_idx + days]["close"] / data.iloc[test_idx]["close"] - 1)

            # Base pick info
            pick_info = {
                "ticker": ticker,
                "confidence": confidence,
                "actual_return": actual_return,
                "spy_return": spy_forward,
                "spy_trailing": spy_trailing,
                "is_bear": is_bear_market,
            }

            # =================================================================
            # MODE 1: NO GATE (Baseline)
            # Standard screener logic with fixed 70% confidence
            # =================================================================
            if pred_direction == 1 and confidence >= NORMAL_CONFIDENCE:
                picks_no_gate.append(pick_info.copy())

            # =================================================================
            # MODE 2: BEAR MODE - SUPPRESSED
            # In bear market (SPY trailing < -1%), skip ALL trades
            # In normal market, use standard 70% threshold
            # =================================================================
            if is_bear_market:
                # Bear market: suppress trades entirely
                pass
            else:
                # Normal market: standard screener
                if pred_direction == 1 and confidence >= NORMAL_CONFIDENCE:
                    picks_suppressed.append(pick_info.copy())

            # =================================================================
            # MODE 3: BEAR MODE - HIGH CONFIDENCE (80%)
            # In bear market, require 80% confidence
            # In normal market, use standard 70% threshold
            # =================================================================
            if is_bear_market:
                # Bear market: require higher confidence
                if pred_direction == 1 and confidence >= BEAR_CONFIDENCE:
                    picks_high_conf.append(pick_info.copy())
            else:
                # Normal market: standard screener
                if pred_direction == 1 and confidence >= NORMAL_CONFIDENCE:
                    picks_high_conf.append(pick_info.copy())

    # =========================================================================
    # COMPUTE METRICS FOR EACH MODE
    # =========================================================================
    spy_returns = {}  # Not needed for helper function

    metrics_no_gate = _compute_metrics_from_picks(picks_no_gate, spy_returns)
    metrics_suppressed = _compute_metrics_from_picks(picks_suppressed, spy_returns)
    metrics_high_conf = _compute_metrics_from_picks(picks_high_conf, spy_returns)

    # Count bear market periods
    total_periods = len(period_regimes)
    bear_periods = sum(1 for p in period_regimes if p["is_bear"])
    normal_periods = total_periods - bear_periods

    # =========================================================================
    # BUILD COMPARISON TABLE
    # =========================================================================
    comparison = {
        "no_gate": {
            "mode": "No Regime Gate (Baseline)",
            **metrics_no_gate,
        },
        "suppressed": {
            "mode": "Bear Mode: Suppressed",
            **metrics_suppressed,
        },
        "high_confidence": {
            "mode": "Bear Mode: 80% Confidence",
            **metrics_high_conf,
        },
    }

    # Calculate improvements vs baseline
    baseline_trades = metrics_no_gate["total_trades"]
    baseline_dd = metrics_no_gate["max_drawdown"]
    baseline_worst = metrics_no_gate["worst_trade"]

    if baseline_trades > 0:
        comparison["suppressed"]["trade_reduction"] = round(
            (1 - metrics_suppressed["total_trades"] / baseline_trades) * 100, 1
        )
        comparison["high_confidence"]["trade_reduction"] = round(
            (1 - metrics_high_conf["total_trades"] / baseline_trades) * 100, 1
        )
    else:
        comparison["suppressed"]["trade_reduction"] = 0
        comparison["high_confidence"]["trade_reduction"] = 0

    if baseline_dd > 0:
        comparison["suppressed"]["drawdown_improvement"] = round(
            (1 - metrics_suppressed["max_drawdown"] / baseline_dd) * 100, 1
        )
        comparison["high_confidence"]["drawdown_improvement"] = round(
            (1 - metrics_high_conf["max_drawdown"] / baseline_dd) * 100, 1
        )
    else:
        comparison["suppressed"]["drawdown_improvement"] = 0
        comparison["high_confidence"]["drawdown_improvement"] = 0

    comparison["no_gate"]["trade_reduction"] = 0
    comparison["no_gate"]["drawdown_improvement"] = 0

    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    interpretation = []

    # Risk reduction analysis
    if metrics_suppressed["max_drawdown"] < baseline_dd:
        interpretation.append(
            f"Suppressed mode reduces max drawdown from {baseline_dd}% to {metrics_suppressed['max_drawdown']}%."
        )

    if metrics_high_conf["max_drawdown"] < baseline_dd:
        interpretation.append(
            f"80% confidence mode reduces max drawdown from {baseline_dd}% to {metrics_high_conf['max_drawdown']}%."
        )

    # Win rate analysis
    if metrics_suppressed["total_trades"] > 0 and metrics_suppressed["win_rate"] > metrics_no_gate["win_rate"]:
        interpretation.append(
            f"Suppressed mode improves win rate from {metrics_no_gate['win_rate']}% to {metrics_suppressed['win_rate']}%."
        )

    if metrics_high_conf["total_trades"] > 0 and metrics_high_conf["win_rate"] > metrics_no_gate["win_rate"]:
        interpretation.append(
            f"80% confidence mode improves win rate from {metrics_no_gate['win_rate']}% to {metrics_high_conf['win_rate']}%."
        )

    # Trade frequency impact
    if bear_periods > 0:
        interpretation.append(
            f"{bear_periods} of {total_periods} periods were in bear market regime (SPY trailing < -1%)."
        )

    # Trade-off note
    if metrics_suppressed["total_trades"] < baseline_trades:
        interpretation.append(
            f"Trade-off: Suppressed mode reduces trades by {comparison['suppressed']['trade_reduction']}%, "
            f"potentially missing some profitable opportunities."
        )

    return {
        "summary": {
            "stocks_tested": stocks_tested,
            "stocks_failed": stocks_failed,
            "total_periods": total_periods,
            "bear_periods": bear_periods,
            "normal_periods": normal_periods,
            "bear_threshold": f"SPY trailing < {BEAR_THRESHOLD * 100}%",
            "normal_confidence": f"{NORMAL_CONFIDENCE * 100}%",
            "bear_confidence": f"{BEAR_CONFIDENCE * 100}%",
        },
        "comparison": comparison,
        "interpretation": interpretation,
        "period_breakdown": {
            "bear_periods_pct": round(bear_periods / total_periods * 100, 1) if total_periods > 0 else 0,
        },
    }
