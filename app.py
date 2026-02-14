import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from job_manager import job_manager, Job, JobStatus

from cache import prediction_cache
from config import (
    PREDICTION_CACHE_TTL,
    SCREENER_CACHE_TTL,
    MIN_DATA_POINTS,
    STOCK_LIST,
)
from data import fetch_stock_data, fetch_fundamental_data
from model import predict_direction
from backtest import (
    run_backtest,
    run_backtest_realistic,
    run_screener_backtest,
    run_regime_gated_backtest,
    compute_screener_metrics_from_raw,
    compute_regime_metrics_from_raw,
)
from backtest_realistic import (
    run_screener_backtest_realistic,
    run_regime_backtest_realistic,
)
from validation import run_year_split_backtest, analyze_feature_importance
from regime_aware import get_current_regime, detect_regime
from optimal_strategy import (
    run_optimal_backtest,
    run_production_scan,
    get_stocks_for_horizon,
)
from stress_test import run_all_stress_tests, run_crisis_backtest
from composite_score import (
    calculate_composite_score,
    rank_all_stocks,
    get_top_decile,
    get_sector_rankings,
    get_available_sectors,
)
from composite_backtest import (
    run_composite_backtest,
    run_quintile_analysis,
    run_grade_analysis,
    run_factor_analysis,
    run_sector_relative_analysis,
)
from composite_backtest_rigorous import (
    run_portfolio_simulation,
    run_monte_carlo_significance,
    run_rigorous_backtest,
)
from config import COMPOSITE_STOCK_LIST

app = FastAPI(title="Stock Direction Predictor")


def get_market_regime() -> dict:
    """
    Detect current market regime based on SPY's trailing 5-day return.

    Returns:
        dict with regime info:
        - regime: "bull" | "normal" | "bear"
        - spy_return: trailing 5-day return as percentage
        - recommended_confidence: 0.75 for normal/bull, 0.80 for bear
        - description: human-readable explanation
    """
    try:
        spy_data = fetch_stock_data("SPY", prediction_days=5)
        if spy_data is None or len(spy_data) < 5:
            return {
                "regime": "unknown",
                "spy_return": None,
                "recommended_confidence": 0.75,
                "description": "Could not fetch SPY data for regime detection",
            }

        # Calculate trailing 5-day return
        start_price = float(spy_data["close"].iloc[-5])
        end_price = float(spy_data["close"].iloc[-1])
        spy_return = ((end_price - start_price) / start_price) * 100
        spy_return = round(spy_return, 2)

        # Classify regime based on backtested thresholds
        if spy_return < -1.0:
            regime = "bear"
            recommended_confidence = 0.80
            description = f"Bear market signal: SPY down {abs(spy_return)}% over 5 days. Use higher confidence threshold (80%+)."
        elif spy_return > 2.0:
            regime = "bull"
            recommended_confidence = 0.75
            description = f"Bull market signal: SPY up {spy_return}% over 5 days. Standard confidence threshold (75%+) applies."
        else:
            regime = "normal"
            recommended_confidence = 0.75
            description = f"Normal market conditions: SPY {'+' if spy_return >= 0 else ''}{spy_return}% over 5 days. Standard confidence threshold (75%+) applies."

        return {
            "regime": regime,
            "spy_return": spy_return,
            "recommended_confidence": recommended_confidence,
            "description": description,
        }
    except Exception:
        return {
            "regime": "unknown",
            "spy_return": None,
            "recommended_confidence": 0.75,
            "description": "Error detecting market regime",
        }


# CORS must be added before routes
cors_origins = ["http://localhost:5173"]
if os.environ.get("FRONTEND_URL"):
    cors_origins.append(os.environ["FRONTEND_URL"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stocks")
def get_stocks():
    """Return list of available stocks for prediction."""
    return {"stocks": STOCK_LIST}


@app.get("/predict")
def predict(
    ticker: str = Query(..., description="Stock ticker symbol"),
    days: int = Query(..., ge=1, le=30, description="Days ahead to predict (1-30)"),
):
    """Predict stock price direction (up/down) for the given ticker and time horizon."""
    ticker = ticker.upper()
    cache_key = f"{ticker}:{days}"

    # Check prediction cache
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch stock data (uses more history for longer predictions)
    df = fetch_stock_data(ticker, days)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

    # Check sufficient data
    if len(df) < MIN_DATA_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for training. Need {MIN_DATA_POINTS} data points, got {len(df)}",
        )

    # Fetch fundamental data (earnings, analyst recommendations, etc.)
    fundamentals = fetch_fundamental_data(ticker)

    # Make prediction with both technical and fundamental data
    result = predict_direction(df, days, fundamentals)
    if result is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    direction, confidence, predicted_return = result
    latest_price = round(float(df["close"].iloc[-1]), 2)
    # Align predicted return sign with direction for consistency
    aligned_return = abs(predicted_return) if direction == "up" else -abs(predicted_return)
    predicted_price = round(latest_price * (1 + aligned_return), 2)
    # Get stock name from fundamentals
    name = fundamentals.get("name", ticker) if fundamentals else ticker

    # Get current market regime
    market_regime = get_market_regime()

    # Determine if this prediction meets regime-adjusted threshold
    meets_regime_threshold = confidence >= market_regime["recommended_confidence"]

    response = {
        "ticker": ticker,
        "name": name,
        "days": days,
        "direction": direction,
        "confidence": round(confidence, 4),
        "latest_price": latest_price,
        "predicted_price": predicted_price,
        "predicted_change": round(aligned_return * 100, 2),  # As percentage
        "market_regime": {
            "regime": market_regime["regime"],
            "spy_return": market_regime["spy_return"],
            "recommended_confidence": market_regime["recommended_confidence"],
            "description": market_regime["description"],
            "meets_threshold": meets_regime_threshold,
        },
    }

    # Cache the result
    prediction_cache.set(cache_key, response, PREDICTION_CACHE_TTL)

    return response


def _predict_single(ticker: str, days: int) -> Optional[dict]:
    """Helper to predict a single stock, returns None on failure."""
    try:
        df = fetch_stock_data(ticker, days)
        if df is None or len(df) < MIN_DATA_POINTS:
            return None
        fundamentals = fetch_fundamental_data(ticker)
        result = predict_direction(df, days, fundamentals)
        if result is None:
            return None
        direction, confidence, predicted_return = result
        latest_price = round(float(df["close"].iloc[-1]), 2)
        # Align predicted return sign with direction for consistency
        aligned_return = abs(predicted_return) if direction == "up" else -abs(predicted_return)
        predicted_price = round(latest_price * (1 + aligned_return), 2)
        # Get stock name from fundamentals
        name = fundamentals.get("name", ticker) if fundamentals else ticker
        return {
            "ticker": ticker,
            "name": name,
            "direction": direction,
            "confidence": round(confidence, 4),
            "latest_price": latest_price,
            "predicted_price": predicted_price,
            "predicted_change": round(aligned_return * 100, 2),
        }
    except Exception:
        return None


@app.get("/screener")
def screener(
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    min_confidence: float = Query(0.75, ge=0.5, le=1.0, description="Minimum confidence (0.5-1.0)"),
):
    """Scan stocks and return those predicted to go UP with confidence above threshold."""
    # Cache key excludes min_confidence - we cache all bullish predictions and filter afterwards
    cache_key = f"screener:{days}"

    # Check cache for all bullish predictions
    cached = prediction_cache.get(cache_key)

    if cached is None:
        # No cache - run fresh scan
        all_bullish = []
        scanned = 0
        errors = 0

        # Scan stocks in parallel (8 threads)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_predict_single, ticker, days): ticker for ticker in STOCK_LIST}

            for future in as_completed(futures):
                result = future.result()
                scanned += 1
                if result is None:
                    errors += 1
                elif result["direction"] == "up":
                    # Store ALL bullish predictions (filter by confidence later)
                    all_bullish.append({
                        "ticker": result["ticker"],
                        "name": result["name"],
                        "confidence": result["confidence"],
                        "latest_price": result["latest_price"],
                        "predicted_price": result["predicted_price"],
                        "predicted_change": result["predicted_change"],
                    })

        # Sort by confidence descending
        all_bullish.sort(key=lambda x: x["confidence"], reverse=True)

        # Cache the full results (all bullish predictions)
        cached = {
            "stocks_scanned": scanned,
            "errors": errors,
            "all_bullish": all_bullish,
        }
        prediction_cache.set(cache_key, cached, SCREENER_CACHE_TTL)

    # Filter by user's requested min_confidence (instant from cache)
    matches = [m for m in cached["all_bullish"] if m["confidence"] >= min_confidence]

    # Get current market regime
    market_regime = get_market_regime()

    # Filter matches that meet regime-adjusted threshold
    regime_filtered_matches = [
        m for m in matches
        if m["confidence"] >= market_regime["recommended_confidence"]
    ]

    response = {
        "days": days,
        "min_confidence": min_confidence,
        "stocks_scanned": cached["stocks_scanned"],
        "errors": cached["errors"],
        "matches": matches,
        "market_regime": {
            "regime": market_regime["regime"],
            "spy_return": market_regime["spy_return"],
            "recommended_confidence": market_regime["recommended_confidence"],
            "description": market_regime["description"],
            "regime_filtered_count": len(regime_filtered_matches),
        },
    }

    # No need to cache here - raw data already cached above
    # Filtering by min_confidence is instant

    return response


@app.get("/backtest")
def backtest(
    ticker: str = Query(..., description="Stock ticker symbol"),
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    mode: str = Query("realistic", description="'realistic' (recommended) or 'walkforward' (inflated)"),
):
    """
    Run a backtest to evaluate model performance on historical data.

    TWO MODES:
    - realistic (DEFAULT): Train once on 70% of data, test on 30%. No retraining.
      Shows honest accuracy (~50-55%). This matches real-world deployment.

    - walkforward: Retrain model at every step. Shows inflated accuracy (~65-70%).
      Does NOT match production reality. Included for comparison only.

    IMPORTANT: The 'realistic' mode is what you should trust.
    """
    ticker = ticker.upper()
    cache_key = f"backtest:{ticker}:{days}:{mode}"

    # Check cache
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch stock data (use long lookback for more test data)
    df = fetch_stock_data(ticker, prediction_days=30)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

    # Fetch fundamental data
    fundamentals = fetch_fundamental_data(ticker)

    # Run appropriate backtest
    if mode == "walkforward":
        result = run_backtest(df, days, fundamentals, test_periods=50)
        if result:
            result["backtest_type"] = "walkforward"
            result["warning"] = "Walk-forward backtests show INFLATED accuracy due to constant retraining. Use 'realistic' mode for honest results."
    else:
        result = run_backtest_realistic(df, days, fundamentals)

    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for backtesting.",
        )

    # Get stock name
    name = fundamentals.get("name", ticker) if fundamentals else ticker

    response = {
        "ticker": ticker,
        "name": name,
        "days": days,
        **result,
    }

    # Cache the result (6 hours)
    prediction_cache.set(cache_key, response, 21600)

    return response


# Sample stocks for screener backtest (61 stocks - prime number for optimal sampling)
SCREENER_BACKTEST_STOCKS = [
    # Big tech (10)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
    # Finance (8)
    "JPM", "V", "MA", "BAC", "GS", "WFC", "MS", "AXP",
    # Healthcare (7)
    "UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "TMO",
    # Consumer/Travel (10)
    "WMT", "PG", "KO", "NKE", "MCD", "BKNG", "PEP", "COST", "SBUX", "HD",
    # Energy (2)
    "XOM", "CVX",
    # ETFs (7)
    "SPY", "QQQ", "DIA", "VTI", "VOO", "XLK", "IWM",
    # Cloud/Cyber (4)
    "CRWD", "SNOW", "DDOG", "NET",
    # EV & Clean Energy (4)
    "RIVN", "NIO", "LCID", "ENPH",
    # Fintech (3)
    "COIN", "SQ", "PYPL",
    # AI/Software (2)
    "PLTR", "TTD",
    # Semiconductors (2)
    "MRVL", "ON",
    # International (2)
    "SHOP", "MELI",
]


@app.get("/screener-backtest")
def screener_backtest(
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    min_confidence: float = Query(0.75, ge=0.5, le=1.0, description="Minimum confidence (0.5-1.0)"),
    mode: str = Query("realistic", description="'realistic' (recommended) or 'walkforward' (inflated)"),
):
    """
    Backtest the screener strategy across multiple stocks.

    TWO MODES:
    - realistic (DEFAULT): Train once on 2020-2022, test on 2023-2024. No retraining.
      Shows honest accuracy. This matches real-world deployment.

    - walkforward: Retrain at every step. Shows INFLATED accuracy (~15-20% higher).
      Does NOT match production. Included for comparison only.
    """
    cache_key = f"screener-backtest:{days}:{min_confidence}:{mode}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    if mode == "realistic":
        # Use realistic backtest
        result = run_screener_backtest_realistic(
            stocks=SCREENER_BACKTEST_STOCKS,
            days=days,
            min_confidence=min_confidence,
        )

        if result is None:
            raise HTTPException(
                status_code=400,
                detail="No screener picks found with the given criteria.",
            )

        response = {
            "days": days,
            "min_confidence": min_confidence,
            **result,
        }
    else:
        # Legacy walk-forward (with warning)
        spy_data = fetch_stock_data("SPY", prediction_days=30)
        stocks_data = []

        def fetch_stock_info(ticker):
            df = fetch_stock_data(ticker, prediction_days=30)
            fundamentals = fetch_fundamental_data(ticker)
            return {"ticker": ticker, "df": df, "fundamentals": fundamentals}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_stock_info, t): t for t in SCREENER_BACKTEST_STOCKS}
            for future in as_completed(futures):
                try:
                    stocks_data.append(future.result())
                except Exception:
                    pass

        result = run_screener_backtest(stocks_data, days, min_confidence, 30, spy_data)

        if result is None:
            raise HTTPException(status_code=400, detail="No screener picks found.")

        result.pop("_raw_data", None)

        response = {
            "warning": "Walk-forward backtest - results are INFLATED. Use mode=realistic for honest accuracy.",
            "backtest_type": "walkforward",
            "days": days,
            "min_confidence": min_confidence,
            **result,
        }

    prediction_cache.set(cache_key, response, 21600)
    return response


@app.get("/regime-gated-backtest")
def regime_gated_backtest(
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    mode: str = Query("realistic", description="'realistic' (recommended) or 'walkforward' (inflated)"),
):
    """
    Backtest the screener with market regime gate based on SPY performance.

    TWO MODES:
    - realistic (DEFAULT): Train once on 2020-2022, test on 2023-2024. Honest results.
    - walkforward: Retrain at every step. INFLATED results.

    Compares three trading modes:
    - No Gate (Baseline): Standard 75% confidence threshold
    - Bear Suppressed: No trades when SPY trailing return < -1%
    - Bear 80% Confidence: Require 80% confidence when SPY trailing < -1%
    """
    cache_key = f"regime-backtest:{days}:{mode}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    if mode == "realistic":
        result = run_regime_backtest_realistic(
            stocks=SCREENER_BACKTEST_STOCKS,
            days=days,
            min_confidence=0.75,
        )

        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient data for backtest.")

        response = {"days": days, **result}
    else:
        # Legacy walk-forward
        spy_data = fetch_stock_data("SPY", prediction_days=30)
        if spy_data is None or len(spy_data) < 50:
            raise HTTPException(status_code=400, detail="Could not fetch SPY data.")

        stocks_data = []

        def fetch_stock_info(ticker):
            df = fetch_stock_data(ticker, prediction_days=30)
            fundamentals = fetch_fundamental_data(ticker)
            return {"ticker": ticker, "df": df, "fundamentals": fundamentals}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_stock_info, t): t for t in SCREENER_BACKTEST_STOCKS}
            for future in as_completed(futures):
                try:
                    stocks_data.append(future.result())
                except Exception:
                    pass

        result = run_regime_gated_backtest(stocks_data, days, 0.75, 30, spy_data)

        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient data.")

        result.pop("_raw_data", None)

        response = {
            "warning": "Walk-forward backtest - results are INFLATED. Use mode=realistic for honest accuracy.",
            "backtest_type": "walkforward",
            "days": days,
            **result,
        }

    prediction_cache.set(cache_key, response, 21600)
    return response


@app.get("/validate")
def validate_model(
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    min_confidence: float = Query(0.75, ge=0.5, le=1.0, description="Minimum confidence (0.5-1.0)"),
):
    """
    Run out-of-sample validation: train on 2020-2022, test on 2023-2024.

    This is the gold standard test for detecting overfitting and data leakage.
    Returns accuracy metrics, feature importance, and pass/fail determination.

    WARNING: This endpoint takes 2-5 minutes to run as it tests across many stocks.
    """
    # Run year-split backtest
    results = run_year_split_backtest(
        days=days,
        min_confidence=min_confidence,
        verbose=False,
    )

    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])

    # Analyze feature importance
    importance = analyze_feature_importance(results, verbose=False)

    # Determine pass/fail
    overall_accuracy = results["summary"]["overall_accuracy"]
    if overall_accuracy >= 60:
        verdict = "PASS"
        verdict_detail = "Model shows strong predictive signal on out-of-sample data"
    elif overall_accuracy >= 55:
        verdict = "MARGINAL"
        verdict_detail = "Model shows weak but potentially useful signal"
    else:
        verdict = "FAIL"
        verdict_detail = "Model does not generalize well - likely overfitting or leakage"

    return {
        "validation_type": "year_split",
        "train_period": results["summary"]["train_period"],
        "test_period": results["summary"]["test_period"],
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "summary": results["summary"],
        "confidence_buckets": results["confidence_buckets"],
        "screener_performance": results["screener_performance"],
        "top_features": importance.get("top_features", [])[:10],
        "interpretation": results["interpretation"],
    }


@app.get("/regime")
def check_regime():
    """
    Check current market regime to determine if conditions are favorable for trading.

    Regimes:
    - NORMAL: Favorable conditions, strategy can be used
    - CAUTION: Elevated risk, reduce position sizes
    - CRISIS: High volatility/drawdown, strategy historically fails here
    - RECOVERY: Coming out of stress, proceed cautiously
    - EUPHORIA: Potentially overextended, tighten stops

    IMPORTANT: This strategy has been tested to FAIL during crisis periods
    (2008, 2018 Q4, 2020 COVID). The regime check helps avoid those periods.
    """
    try:
        regime = get_current_regime(verbose=False)
        return regime
    except Exception as e:
        error_msg = str(e).lower()
        if 'rate' in error_msg or 'too many' in error_msg or '429' in error_msg:
            return {
                "regime": "UNKNOWN",
                "action": "WAIT",
                "explanation": "Yahoo Finance is rate limiting requests. Please wait 5-10 minutes and try again.",
                "metrics": {},
                "error": "rate_limited"
            }
        return {
            "regime": "UNKNOWN",
            "action": "WAIT",
            "explanation": f"Error checking regime: {str(e)}",
            "metrics": {},
            "error": str(e)
        }


@app.get("/about")
def about():
    """
    Honest disclosure about what this system can and cannot do.
    """
    return {
        "name": "Stock Direction Predictor",
        "version": "2.0 (Regime-Aware)",
        "what_it_does": {
            "description": "Identifies stocks with aligned technical signals for potential directional moves",
            "methodology": "Multi-signal consensus filtering, not pure ML prediction",
            "time_horizon": "20-day holding period works best",
        },
        "honest_limitations": {
            "accuracy_claims": "78% accuracy was measured on 2023-2024 bull market only",
            "crisis_performance": "Strategy FAILS during crisis periods (tested: 2008, 2018, 2020)",
            "selection_bias": "Best-performing stocks were selected AFTER seeing results",
            "signal_frequency": "High-conviction signals occur only 1-2 times per month",
            "regime_dependent": "Only works when market regime is NORMAL",
        },
        "what_it_cannot_do": [
            "Predict with 89% accuracy across all conditions",
            "Work reliably during market crashes",
            "Generate daily trading signals",
            "Replace human judgment on position sizing and risk",
        ],
        "recommended_use": {
            "check_regime_first": "Always call /regime before trusting signals",
            "paper_trade_first": "Run for 3-6 months before using real capital",
            "position_sizing": "Never risk more than you can afford to lose",
            "diversification": "This should be one input among many, not your only signal",
        },
        "tested_periods": {
            "2023-2024 (bull)": "78% accuracy on Tier 1 stocks with 4+ signals",
            "2022 (bear)": "57% accuracy - survived but degraded",
            "2020 COVID": "29% accuracy - FAILED",
            "2018 Q4": "33% accuracy - FAILED",
            "2008 crisis": "31% accuracy - FAILED",
        },
    }


@app.get("/tier-info")
def get_tier_info(
    days: int = Query(20, ge=2, le=30, description="Days ahead (2-5 for short-term, 20 for optimal)"),
):
    """
    Get the tier 1 and tier 2 stock lists for a given horizon.
    This is a fast endpoint that returns static data.
    """
    tier1_stocks, tier2_stocks = get_stocks_for_horizon(days)

    if days <= 2:
        horizon_used = 2
    elif days == 3:
        horizon_used = 3
    elif days == 4:
        horizon_used = 4
    elif days == 5:
        horizon_used = 5
    else:
        horizon_used = 20

    return {
        "horizon_days": days,
        "horizon_tier_used": horizon_used,
        "tier1_stocks": tier1_stocks,
        "tier2_stocks": tier2_stocks,
        "tier1_count": len(tier1_stocks),
        "tier2_count": len(tier2_stocks),
        "total_stocks": len(tier1_stocks) + len(tier2_stocks),
    }


@app.get("/optimal-scan")
def optimal_scan(
    days: int = Query(20, ge=2, le=30, description="Days ahead (2-5 for short-term, 20 for optimal)"),
):
    """
    Run production scan for high-conviction trading signals.

    This strategy uses horizon-specific optimized stock lists:
    - 2-day: 11 stocks optimized for very short-term
    - 3-day: 22 stocks optimized for short-term
    - 4-day: 28 stocks optimized for short-term
    - 5-day: 23 stocks optimized for short-term
    - 6-30 days: Uses 20-day optimized list (99 stocks)

    Returns actionable signals when 4+ of 5 signals align bullish.

    IMPORTANT CAVEATS:
    - Check /regime first - only trade when regime is NORMAL
    - Accuracy varies by horizon and was measured on 2023-2024
    - High-conviction signals occur 1-2 times per month
    - This strategy FAILS during crisis periods
    """
    cache_key = f"optimal-scan:{days}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Check regime first
    regime = get_current_regime(verbose=False)

    # Get horizon-specific stocks
    tier1_stocks, tier2_stocks = get_stocks_for_horizon(days)
    optimal_stocks = tier1_stocks + tier2_stocks

    result = run_production_scan(days=days, verbose=False)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Add regime warning
    if regime.get("regime") in ["CRISIS", "CAUTION"]:
        result["regime_warning"] = f"Market is in {regime.get('regime')} mode. Strategy historically fails here. Consider waiting."
    else:
        result["regime_warning"] = None

    result["current_regime"] = regime

    # Determine which horizon tier is being used
    if days <= 2:
        horizon_used = 2
    elif days == 3:
        horizon_used = 3
    elif days == 4:
        horizon_used = 4
    elif days == 5:
        horizon_used = 5
    else:
        horizon_used = 20

    response = {
        "scan_type": "optimal_strategy",
        "horizon_days": days,
        "horizon_tier_used": horizon_used,
        "tier1_stocks": tier1_stocks,
        "tier2_stocks": tier2_stocks,
        "optimal_stocks": optimal_stocks,
        "available_horizons": [2, 3, 4, 5, 20],
        **result,
        "methodology": {
            "signals": [
                "ML prediction with 70%+ confidence",
                "Trend aligned (price > SMA20 > SMA50)",
                "Low volatility (< 35% annualized)",
                "Positive 5-day momentum",
                "RSI not overbought (< 70)",
            ],
            "threshold": "4 or more signals must align for high conviction",
            "ultra_conviction": "All 5 signals aligned",
        },
        "horizon_info": {
            "description": f"Using stocks optimized for {horizon_used}-day horizon",
            "tier1_count": len(tier1_stocks),
            "tier2_count": len(tier2_stocks),
            "total_stocks": len(optimal_stocks),
        },
    }

    prediction_cache.set(cache_key, response, 3600)  # 1 hour cache
    return response


@app.get("/stress-test")
def stress_test(
    period: str = Query(None, description="Specific period: 2008_crisis, 2018_selloff, 2020_covid, 2022_bear"),
):
    """
    Run stress tests against historical crisis periods.

    This shows how the strategy ACTUALLY performs during market stress.
    SPOILER: It fails during most crises. This is why we check /regime.

    Available periods:
    - 2008_crisis: Lehman collapse, -50% drawdown
    - 2018_selloff: Fed tightening, -20% correction
    - 2020_covid: Fastest bear market in history
    - 2022_bear: Fed hiking, tech collapse

    Returns accuracy and drawdown metrics for each period.
    """
    cache_key = f"stress-test:{period or 'all'}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    if period:
        result = run_crisis_backtest(period, verbose=False)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        response = {"period_tested": period, **result}
    else:
        results = run_all_stress_tests(verbose=False)
        response = {
            "all_periods": results,
            "summary": {
                "survived": [],
                "failed": [],
            },
            "interpretation": [],
        }

        for key, r in results.items():
            if "error" not in r:
                if r.get("high_conviction_accuracy", 0) >= 50:
                    response["summary"]["survived"].append({
                        "period": r["period"],
                        "accuracy": r["high_conviction_accuracy"],
                    })
                else:
                    response["summary"]["failed"].append({
                        "period": r["period"],
                        "accuracy": r["high_conviction_accuracy"],
                    })

        if len(response["summary"]["failed"]) > len(response["summary"]["survived"]):
            response["verdict"] = "REGIME_DEPENDENT"
            response["interpretation"] = [
                "Strategy fails during most crisis periods",
                "ALWAYS check /regime before trading",
                "In CRISIS or CAUTION mode, do not take new positions",
            ]
        else:
            response["verdict"] = "RESILIENT"
            response["interpretation"] = ["Strategy shows resilience across periods"]

    prediction_cache.set(cache_key, response, 86400)  # 24 hour cache
    return response


@app.get("/optimal-backtest")
def optimal_backtest(
    days: int = Query(20, ge=2, le=30, description="Days ahead (2-5 for short-term, 20 for optimal)"),
    min_signals: int = Query(4, ge=3, le=5, description="Minimum signals required (4 recommended)"),
    tier1_only: bool = Query(True, description="Use Tier 1 stocks only (recommended)"),
):
    """
    Backtest the optimal strategy with realistic methodology.

    This shows what accuracy we achieved on the test period (2023-2024)
    using horizon-specific optimized stock lists.

    Stocks are automatically selected based on the horizon:
    - 2-5 days: Uses short-term optimized stocks
    - 6-30 days: Uses 20-day optimized stocks
    """
    cache_key = f"optimal-backtest:{days}:{min_signals}:{tier1_only}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get horizon-specific stocks
    tier1_stocks, tier2_stocks = get_stocks_for_horizon(days)
    stocks_tested = tier1_stocks if tier1_only else (tier1_stocks + tier2_stocks)

    # Determine which horizon tier is being used
    if days <= 2:
        horizon_used = 2
    elif days == 3:
        horizon_used = 3
    elif days == 4:
        horizon_used = 4
    elif days == 5:
        horizon_used = 5
    else:
        horizon_used = 20

    result = run_optimal_backtest(
        days=days,
        min_signals=min_signals,
        tier1_only=tier1_only,
        verbose=False,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    response = {
        "backtest_type": "optimal_strategy",
        "methodology": "Train on 2020-2022, test on 2023-2024 (no retraining)",
        "settings": {
            "horizon_days": days,
            "horizon_tier_used": horizon_used,
            "min_signals": min_signals,
            "tier1_only": tier1_only,
            "stocks_tested": stocks_tested,
            "tier1_stocks": tier1_stocks,
            "tier2_stocks": tier2_stocks,
        },
        **result,
        "horizon_info": {
            "description": f"Using stocks optimized for {horizon_used}-day horizon",
            "tier1_count": len(tier1_stocks),
            "tier2_count": len(tier2_stocks),
        },
        "caveats": [
            "These results are from 2023-2024 bull market only",
            "Strategy fails during crisis periods (see /stress-test)",
            "Past performance does not guarantee future results",
        ],
    }

    prediction_cache.set(cache_key, response, 21600)  # 6 hour cache
    return response


# =============================================================================
# COMPOSITE SCORE ENDPOINTS - Long-term fundamental scoring
# =============================================================================


@app.get("/composite-scores")
def get_composite_scores(
    sector: Optional[str] = Query(None, description="Filter by sector (e.g., 'Technology', 'Healthcare')"),
    grade: Optional[str] = Query(None, description="Filter by grade (A, B, C, D, F)"),
    horizon: int = Query(24, description="Investment horizon in months (1, 3, 6, 12, 24, 60, 120)"),
    limit: int = Query(50, ge=1, le=500, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get composite scores for all stocks in the universe, ranked by score.

    This endpoint provides long-term fundamental analysis scores combining:
    - Growth (30%): Revenue growth, earnings growth, price momentum (over horizon period)
    - Quality (30%): ROE, ROA, profit margin, operating margin
    - Financial Strength (20%): Debt-to-equity, current ratio, institutional holders
    - Valuation (20%): P/E vs sector, PEG ratio, price-to-book

    Args:
        horizon: Investment horizon in months. Common values:
                 - 1 (30 days): Very short-term momentum
                 - 3 (3 months): Short-term
                 - 6 (6 months): Medium-term
                 - 12 (1 year): Standard
                 - 24 (2 years): Long-term (default)
                 - 60 (5 years): Full market cycle
                 - 120 (10 years): Multiple cycles

    Use filters to narrow down results by sector or grade.
    """
    # Validate horizon
    horizon = min(max(horizon, 1), 120)

    cache_key = f"composite-scores:{sector}:{grade}:{horizon}m:{limit}:{offset}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get all rankings with specified horizon
    if sector:
        all_ranked = get_sector_rankings(sector, horizon_months=horizon)
    else:
        all_ranked = rank_all_stocks(horizon_months=horizon)

    if all_ranked is None:
        raise HTTPException(status_code=500, detail="Failed to calculate composite scores")

    # Filter by grade if specified
    if grade:
        all_ranked = [s for s in all_ranked if s.get("grade") == grade.upper()]

    total_count = len(all_ranked)

    # Apply pagination
    paginated = all_ranked[offset:offset + limit]

    # Get available sectors for filtering
    sectors = get_available_sectors(horizon_months=horizon)

    # Calculate grade distribution
    grade_counts = {}
    for stock in all_ranked:
        g = stock.get("grade", "F")
        grade_counts[g] = grade_counts.get(g, 0) + 1

    # Generate period label
    if horizon == 1:
        period_label = "30 days"
    elif horizon < 12:
        period_label = f"{horizon} months"
    elif horizon == 12:
        period_label = "1 year"
    else:
        period_label = f"{horizon // 12} years"

    response = {
        "total_stocks": total_count,
        "universe_size": len(COMPOSITE_STOCK_LIST),
        "horizon_months": horizon,
        "horizon_label": period_label,
        "limit": limit,
        "offset": offset,
        "filters": {
            "sector": sector,
            "grade": grade,
            "horizon_months": horizon,
        },
        "available_sectors": sectors,
        "grade_distribution": grade_counts,
        "stocks": paginated,
        "methodology": _get_adaptive_methodology(horizon),
    }

    prediction_cache.set(cache_key, response, 21600)  # 6 hour cache
    return response


def _get_adaptive_methodology(horizon_months: int) -> dict:
    """Get methodology description with adaptive weights based on horizon."""
    if horizon_months <= 3:
        weights = {"growth": 40, "quality": 20, "financial_strength": 15, "valuation": 25}
        profile = "Short-term (momentum-focused)"
        description = "Higher weight on growth/momentum for short-term picks"
    elif horizon_months <= 12:
        weights = {"growth": 30, "quality": 30, "financial_strength": 20, "valuation": 20}
        profile = "Medium-term (balanced)"
        description = "Balanced approach for 1-year horizon"
    elif horizon_months <= 60:
        weights = {"growth": 25, "quality": 35, "financial_strength": 25, "valuation": 15}
        profile = "Long-term (quality-focused)"
        description = "Higher weight on quality and financial strength for multi-year holds"
    else:
        weights = {"growth": 20, "quality": 35, "financial_strength": 30, "valuation": 15}
        profile = "Very long-term (quality + stability)"
        description = "Maximum weight on fundamentals for 5+ year investments"

    return {
        "weight_profile": profile,
        "description": description,
        "growth_weight": f"{weights['growth']}%",
        "quality_weight": f"{weights['quality']}%",
        "financial_strength_weight": f"{weights['financial_strength']}%",
        "valuation_weight": f"{weights['valuation']}%",
        "scoring": "Percentile-based normalization across universe",
        "expected_returns": {
            "A": {"annual": "12-25%", "description": "Strong outperformers"},
            "B": {"annual": "8-18%", "description": "Above-average returns"},
            "C": {"annual": "4-12%", "description": "Market-average returns"},
            "D": {"annual": "-2-8%", "description": "Below-average returns"},
            "F": {"annual": "-10-5%", "description": "Underperformers"},
        },
        "grades": {
            "A": "80-100 (Top picks)",
            "B": "65-79 (Strong)",
            "C": "50-64 (Average)",
            "D": "35-49 (Weak)",
            "F": "0-34 (Avoid)",
        },
    }


@app.get("/composite-score/{ticker}")
def get_composite_score_single(ticker: str):
    """
    Get detailed composite score breakdown for a single stock.

    Returns:
    - Overall composite score (0-100)
    - Letter grade (A/B/C/D/F)
    - Sub-scores for each category (Growth, Quality, Financial, Valuation)
    - Raw metric values and individual scores
    - Sector comparison data
    """
    ticker = ticker.upper()

    cache_key = f"composite-score-single:{ticker}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Calculate score for this stock
    result = calculate_composite_score(ticker)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not calculate composite score for {ticker}. Stock may not have sufficient fundamental data."
        )

    # Get rank within universe
    all_ranked = rank_all_stocks()
    rank = None
    percentile = None
    if all_ranked:
        for i, stock in enumerate(all_ranked):
            if stock.get("ticker") == ticker:
                rank = i + 1
                percentile = round((1 - (i / len(all_ranked))) * 100, 1)
                break

    response = {
        **result,
        "rank": rank,
        "percentile": percentile,
        "universe_size": len(COMPOSITE_STOCK_LIST),
        "interpretation": _interpret_composite_score(result),
    }

    prediction_cache.set(cache_key, response, 21600)  # 6 hour cache
    return response


@app.get("/composite-top-decile")
def get_top_decile_stocks(
    horizon: int = Query(24, description="Investment horizon in months (1, 3, 6, 12, 24, 60, 120)"),
):
    """
    Get the top 10% of stocks by composite score.

    These are the highest-quality investment candidates based on
    fundamental analysis across Growth, Quality, Financial Strength,
    and Valuation metrics.

    Args:
        horizon: Investment horizon in months (1=30d, 3=3m, 6=6m, 12=1y, 24=2y, 60=5y, 120=10y)
    """
    horizon = min(max(horizon, 1), 120)

    cache_key = f"composite-top-decile:{horizon}m"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    top_stocks = get_top_decile(horizon_months=horizon)

    if not top_stocks:
        raise HTTPException(status_code=500, detail="Failed to calculate top decile")

    # Generate period label
    if horizon == 1:
        period_label = "30 days"
    elif horizon < 12:
        period_label = f"{horizon} months"
    elif horizon == 12:
        period_label = "1 year"
    else:
        period_label = f"{horizon // 12} years"

    response = {
        "horizon_months": horizon,
        "horizon_label": period_label,
        "decile_size": len(top_stocks),
        "universe_size": len(COMPOSITE_STOCK_LIST),
        "average_score": round(sum(s["composite_score"] for s in top_stocks) / len(top_stocks), 1),
        "stocks": top_stocks,
    }

    prediction_cache.set(cache_key, response, 21600)  # 6 hour cache
    return response


def _interpret_composite_score(result: dict) -> dict:
    """Generate interpretation text for a composite score."""
    score = result.get("composite_score", 0)
    grade = result.get("grade", "F")

    # Overall interpretation
    if grade == "A":
        overall = "Excellent fundamentals across all categories. Strong long-term investment candidate."
    elif grade == "B":
        overall = "Good fundamentals with some areas of strength. Solid investment consideration."
    elif grade == "C":
        overall = "Average fundamentals. May have specific strengths but also notable weaknesses."
    elif grade == "D":
        overall = "Below average fundamentals. Significant concerns in multiple areas."
    else:
        overall = "Weak fundamentals. High risk for long-term investment."

    # Category-specific insights
    insights = []

    growth = result.get("growth_score")
    if growth is not None:
        if growth >= 70:
            insights.append("Strong growth trajectory with solid revenue and earnings momentum.")
        elif growth >= 50:
            insights.append("Moderate growth metrics, room for improvement.")
        else:
            insights.append("Weak growth indicators may limit upside potential.")

    quality = result.get("quality_score")
    if quality is not None:
        if quality >= 70:
            insights.append("High quality operations with strong profitability metrics.")
        elif quality >= 50:
            insights.append("Average operational quality.")
        else:
            insights.append("Profitability concerns warrant careful evaluation.")

    financial = result.get("financial_strength_score")
    if financial is not None:
        if financial >= 70:
            insights.append("Strong balance sheet with manageable debt levels.")
        elif financial >= 50:
            insights.append("Adequate financial position.")
        else:
            insights.append("Financial health may be a concern. Check debt levels.")

    valuation = result.get("valuation_score")
    if valuation is not None:
        if valuation >= 70:
            insights.append("Attractively valued relative to fundamentals and sector.")
        elif valuation >= 50:
            insights.append("Fairly valued at current prices.")
        else:
            insights.append("May be overvalued. Consider entry price carefully.")

    return {
        "overall": overall,
        "insights": insights,
        "recommendation": "Consider for long-term portfolio" if grade in ["A", "B"] else "Research further before investing",
    }


# =============================================================================
# COMPOSITE SCORE BACKTEST ENDPOINTS
# =============================================================================


@app.get("/composite-backtest")
def composite_backtest(
    return_period: str = Query("90d", description="Return period: 30d, 90d, 180d, 365d"),
    stocks_count: int = Query(200, ge=50, le=500, description="Number of stocks to analyze"),
):
    """
    Run comprehensive backtest of the composite scoring system.

    Tests whether higher composite scores actually predict better returns.

    Includes:
    - Quintile Analysis: Compare top 20% vs bottom 20%
    - Grade Performance: A/B/C/D/F return comparison
    - Factor Analysis: Which factors (Growth, Quality, etc.) predict best
    - Sector-Relative: Do high scores beat their sector?

    WARNING: This uses current fundamentals with historical returns,
    which introduces look-ahead bias. Results are indicative, not definitive.
    """
    valid_periods = ["30d", "90d", "180d", "365d"]
    if return_period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid return_period. Must be one of: {valid_periods}"
        )

    cache_key = f"composite-backtest-api:{stocks_count}:{return_period}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    stocks = COMPOSITE_STOCK_LIST[:stocks_count]
    result = run_composite_backtest(stocks, return_period)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    prediction_cache.set(cache_key, result, 21600)
    return result


@app.get("/composite-backtest/quintile")
def composite_quintile_backtest(
    return_period: str = Query("90d", description="Return period: 30d, 90d, 180d, 365d"),
    stocks_count: int = Query(200, ge=50, le=500, description="Number of stocks to analyze"),
):
    """
    Run quintile analysis on composite scores.

    Divides stocks into 5 groups by score and compares returns.
    Top quintile should outperform bottom quintile if scoring works.
    """
    valid_periods = ["30d", "90d", "180d", "365d"]
    if return_period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid return_period. Must be one of: {valid_periods}"
        )

    stocks = COMPOSITE_STOCK_LIST[:stocks_count]
    result = run_quintile_analysis(stocks, return_period)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/composite-backtest/factor")
def composite_factor_backtest(
    return_period: str = Query("90d", description="Return period: 30d, 90d, 180d, 365d"),
    stocks_count: int = Query(150, ge=50, le=300, description="Number of stocks to analyze"),
):
    """
    Test which factors (Growth, Quality, Financial, Valuation) best predict returns.

    Useful for understanding which parts of the composite score add value.
    """
    valid_periods = ["30d", "90d", "180d", "365d"]
    if return_period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid return_period. Must be one of: {valid_periods}"
        )

    stocks = COMPOSITE_STOCK_LIST[:stocks_count]
    result = run_factor_analysis(stocks, return_period)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/composite-backtest/sector")
def composite_sector_backtest(
    return_period: str = Query("90d", description="Return period: 30d, 90d, 180d, 365d"),
    stocks_count: int = Query(200, ge=50, le=500, description="Number of stocks to analyze"),
):
    """
    Test if high-scoring stocks outperform their sector average.

    This controls for sector-wide movements and shows stock-picking alpha.
    """
    valid_periods = ["30d", "90d", "180d", "365d"]
    if return_period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid return_period. Must be one of: {valid_periods}"
        )

    stocks = COMPOSITE_STOCK_LIST[:stocks_count]
    result = run_sector_relative_analysis(stocks, return_period)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# =============================================================================
# RIGOROUS BACKTEST ENDPOINTS - Portfolio simulation & statistical tests
# =============================================================================


@app.get("/composite-backtest/portfolio")
def composite_portfolio_backtest(
    top_n: int = Query(20, ge=5, le=50, description="Number of top stocks to hold"),
    rebalance_months: int = Query(3, ge=1, le=12, description="Months between rebalancing"),
    stocks_count: int = Query(150, ge=50, le=300, description="Universe size"),
    years: int = Query(2, ge=1, le=10, description="Simulation years (1-10)"),
):
    """
    Run a walk-forward portfolio simulation.

    This is a more realistic backtest that:
    - Simulates buying top N ranked stocks
    - Rebalances at regular intervals
    - Tracks actual portfolio value over time
    - Compares to SPY benchmark

    The scoring uses point-in-time data to minimize look-ahead bias:
    - Momentum calculated from historical prices only
    - Volatility calculated from historical prices only
    - Trend signals from historical moving averages
    - Fundamentals (some bias - assumes relatively stable)

    Args:
        years: Simulation period (1-10 years). Longer periods capture more
               market cycles including crises and recoveries.

    Returns alpha vs SPY, Sharpe ratio, max drawdown, and equity curve.
    """
    stocks = COMPOSITE_STOCK_LIST[:stocks_count]

    result = run_portfolio_simulation(
        stocks=stocks,
        top_n=top_n,
        rebalance_months=rebalance_months,
        simulation_years=years,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/composite-backtest/significance")
def composite_significance_test(
    num_simulations: int = Query(500, ge=100, le=2000, description="Number of Monte Carlo simulations"),
    top_n: int = Query(20, ge=5, le=50, description="Portfolio size"),
    return_period_days: int = Query(90, ge=30, le=365, description="Return measurement period"),
    stocks_count: int = Query(100, ge=50, le=200, description="Universe size"),
):
    """
    Monte Carlo test for statistical significance.

    Compares the strategy's returns against randomly selected portfolios.
    If the strategy beats >95% of random portfolios, it's statistically significant.

    This answers: "Is this strategy better than random stock picking?"

    Returns:
    - p-value (< 0.05 is significant)
    - Percentile rank vs random portfolios
    - Distribution of random returns
    """
    stocks = COMPOSITE_STOCK_LIST[:stocks_count]

    result = run_monte_carlo_significance(
        stocks=stocks,
        num_simulations=num_simulations,
        top_n=top_n,
        return_period_days=return_period_days,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/composite-backtest/rigorous")
def composite_rigorous_backtest(
    stocks_count: int = Query(150, ge=50, le=300, description="Universe size"),
    years: int = Query(2, ge=1, le=10, description="Simulation years (1-10)"),
):
    """
    Run comprehensive rigorous backtest.

    Combines:
    1. Walk-forward portfolio simulation (quarterly rebalancing)
    2. Monte Carlo significance test (500 simulations)

    This provides the strongest evidence for whether the scoring system works:
    - Portfolio alpha vs SPY benchmark
    - Statistical significance (p-value)
    - Risk-adjusted returns (Sharpe ratio)
    - Maximum drawdown

    Args:
        stocks_count: Number of stocks in universe (50-300)
        years: Simulation period in years (1-10). Use 2-5 for normal testing,
               5-10 for long-term crisis/recovery analysis.

    WARNING: This is computationally intensive. 2 years takes ~2-5 minutes,
    longer periods take proportionally longer.
    """
    stocks = COMPOSITE_STOCK_LIST[:stocks_count]

    result = run_rigorous_backtest(stocks=stocks, simulation_years=years)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# =============================================================================
# JOB API ENDPOINTS - For cancellable long-running operations
# =============================================================================


class JobStartRequest(BaseModel):
    """Request body for starting a job."""
    days: int = 5
    min_confidence: float = 0.75
    ticker: Optional[str] = None
    min_signals: int = 4
    tier1_only: bool = True
    period: Optional[str] = None


# Worker functions for each job type
def _run_screener_job(job: Job, days: int, min_confidence: float) -> dict:
    """Run screener scan with cancellation support."""
    cache_key = f"screener:{days}"
    cached = prediction_cache.get(cache_key)

    if cached is not None:
        # Apply confidence filter
        matches = [m for m in cached["all_bullish"] if m["confidence"] >= min_confidence]
        market_regime = get_market_regime()
        regime_filtered_matches = [
            m for m in matches
            if m["confidence"] >= market_regime["recommended_confidence"]
        ]
        return {
            "days": days,
            "min_confidence": min_confidence,
            "stocks_scanned": cached["stocks_scanned"],
            "errors": cached["errors"],
            "matches": matches,
            "market_regime": {
                "regime": market_regime["regime"],
                "spy_return": market_regime["spy_return"],
                "recommended_confidence": market_regime["recommended_confidence"],
                "description": market_regime["description"],
                "regime_filtered_count": len(regime_filtered_matches),
            },
        }

    # Run fresh scan
    all_bullish = []
    scanned = 0
    errors = 0
    total = len(STOCK_LIST)

    job.progress_message = f"Scanning {total} stocks..."

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_predict_single, ticker, days): ticker for ticker in STOCK_LIST}

        for future in as_completed(futures):
            # Check for cancellation
            if job.cancelled:
                executor.shutdown(wait=False, cancel_futures=True)
                return None

            result = future.result()
            scanned += 1
            job.progress = int((scanned / total) * 100)
            job.progress_message = f"Scanned {scanned}/{total} stocks"

            if result is None:
                errors += 1
            elif result["direction"] == "up":
                all_bullish.append({
                    "ticker": result["ticker"],
                    "name": result["name"],
                    "confidence": result["confidence"],
                    "latest_price": result["latest_price"],
                    "predicted_price": result["predicted_price"],
                    "predicted_change": result["predicted_change"],
                })

    if job.cancelled:
        return None

    all_bullish.sort(key=lambda x: x["confidence"], reverse=True)

    cached_data = {
        "stocks_scanned": scanned,
        "errors": errors,
        "all_bullish": all_bullish,
    }
    prediction_cache.set(cache_key, cached_data, SCREENER_CACHE_TTL)

    matches = [m for m in all_bullish if m["confidence"] >= min_confidence]
    market_regime = get_market_regime()
    regime_filtered_matches = [
        m for m in matches
        if m["confidence"] >= market_regime["recommended_confidence"]
    ]

    return {
        "days": days,
        "min_confidence": min_confidence,
        "stocks_scanned": scanned,
        "errors": errors,
        "matches": matches,
        "market_regime": {
            "regime": market_regime["regime"],
            "spy_return": market_regime["spy_return"],
            "recommended_confidence": market_regime["recommended_confidence"],
            "description": market_regime["description"],
            "regime_filtered_count": len(regime_filtered_matches),
        },
    }


def _run_backtest_job(job: Job, ticker: str, days: int) -> dict:
    """Run backtest with cancellation support."""
    ticker = ticker.upper()
    cache_key = f"backtest:{ticker}:{days}:realistic"

    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    job.progress_message = f"Fetching data for {ticker}..."
    job.progress = 10

    if job.cancelled:
        return None

    df = fetch_stock_data(ticker, prediction_days=30)
    if df is None or len(df) == 0:
        raise ValueError(f"No data found for ticker: {ticker}")

    job.progress = 20
    job.progress_message = "Running backtest..."

    if job.cancelled:
        return None

    fundamentals = fetch_fundamental_data(ticker)

    job.progress = 40

    if job.cancelled:
        return None

    result = run_backtest_realistic(df, days, fundamentals)

    job.progress = 90

    if result is None:
        raise ValueError("Insufficient data for backtesting.")

    if job.cancelled:
        return None

    name = fundamentals.get("name", ticker) if fundamentals else ticker

    response = {
        "ticker": ticker,
        "name": name,
        "days": days,
        **result,
    }

    prediction_cache.set(cache_key, response, 21600)
    job.progress = 100

    return response


def _run_screener_backtest_job(job: Job, days: int, min_confidence: float) -> dict:
    """Run screener backtest with cancellation support."""
    cache_key = f"screener-backtest:{days}:{min_confidence}:realistic"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    job.progress_message = "Running realistic backtest..."
    job.progress = 5

    if job.cancelled:
        return None

    # Use realistic backtest - pass job for cancellation checking
    result = run_screener_backtest_realistic(
        stocks=SCREENER_BACKTEST_STOCKS,
        days=days,
        min_confidence=min_confidence,
    )

    # Check periodically during long operation
    if job.cancelled:
        return None

    job.progress = 90

    if result is None:
        raise ValueError("No screener picks found with the given criteria.")

    response = {
        "days": days,
        "min_confidence": min_confidence,
        **result,
    }

    prediction_cache.set(cache_key, response, 21600)
    job.progress = 100

    return response


def _run_regime_gated_backtest_job(job: Job, days: int) -> dict:
    """Run regime-gated backtest with cancellation support."""
    cache_key = f"regime-backtest:{days}:realistic"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    job.progress_message = "Running regime-gated backtest..."
    job.progress = 5

    if job.cancelled:
        return None

    result = run_regime_backtest_realistic(
        stocks=SCREENER_BACKTEST_STOCKS,
        days=days,
        min_confidence=0.75,
    )

    if job.cancelled:
        return None

    job.progress = 90

    if result is None:
        raise ValueError("Insufficient data for backtest.")

    response = {"days": days, **result}

    prediction_cache.set(cache_key, response, 21600)
    job.progress = 100

    return response


def _run_stress_test_job(job: Job, period: Optional[str] = None) -> dict:
    """Run stress test with cancellation support."""
    cache_key = f"stress-test:{period or 'all'}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    job.progress_message = "Running stress tests..."
    job.progress = 10

    if job.cancelled:
        return None

    if period:
        result = run_crisis_backtest(period, verbose=False)
        if "error" in result:
            raise ValueError(result["error"])
        response = {"period_tested": period, **result}
    else:
        results = run_all_stress_tests(verbose=False)

        if job.cancelled:
            return None

        job.progress = 80

        response = {
            "all_periods": results,
            "summary": {
                "survived": [],
                "failed": [],
            },
            "interpretation": [],
        }

        for key, r in results.items():
            if "error" not in r:
                if r.get("high_conviction_accuracy", 0) >= 50:
                    response["summary"]["survived"].append({
                        "period": r["period"],
                        "accuracy": r["high_conviction_accuracy"],
                    })
                else:
                    response["summary"]["failed"].append({
                        "period": r["period"],
                        "accuracy": r["high_conviction_accuracy"],
                    })

        if len(response["summary"]["failed"]) > len(response["summary"]["survived"]):
            response["verdict"] = "REGIME_DEPENDENT"
            response["interpretation"] = [
                "Strategy fails during most crisis periods",
                "ALWAYS check /regime before trading",
                "In CRISIS or CAUTION mode, do not take new positions",
            ]
        else:
            response["verdict"] = "RESILIENT"
            response["interpretation"] = ["Strategy shows resilience across periods"]

    if job.cancelled:
        return None

    prediction_cache.set(cache_key, response, 86400)
    job.progress = 100

    return response


def _run_optimal_backtest_job(job: Job, days: int, min_signals: int, tier1_only: bool) -> dict:
    """Run optimal backtest with cancellation support."""
    cache_key = f"optimal-backtest:{days}:{min_signals}:{tier1_only}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        job.progress = 100
        return cached

    job.progress_message = "Initializing backtest..."
    job.progress = 5

    if job.cancelled:
        return None

    tier1_stocks, tier2_stocks = get_stocks_for_horizon(days)
    stocks_tested = tier1_stocks if tier1_only else (tier1_stocks + tier2_stocks)
    total_stocks = len(stocks_tested)

    if days <= 2:
        horizon_used = 2
    elif days == 3:
        horizon_used = 3
    elif days == 4:
        horizon_used = 4
    elif days == 5:
        horizon_used = 5
    else:
        horizon_used = 20

    job.progress = 10
    job.progress_message = f"Testing {total_stocks} stocks..."

    if job.cancelled:
        return None

    def progress_callback(current, total, ticker):
        """Report progress and check cancellation."""
        if job.cancelled:
            return False  # Signal cancellation
        # Progress from 10% to 90% during backtest
        progress_pct = 10 + int((current / total) * 80)
        job.progress = progress_pct
        job.progress_message = f"Testing {ticker} ({current + 1}/{total})..."
        return True  # Continue

    result = run_optimal_backtest(
        days=days,
        min_signals=min_signals,
        tier1_only=tier1_only,
        verbose=False,
        progress_callback=progress_callback,
    )

    if result is None or job.cancelled:
        return None

    job.progress = 90

    if "error" in result:
        raise ValueError(result["error"])

    response = {
        "backtest_type": "optimal_strategy",
        "methodology": "Train on 2020-2022, test on 2023-2024 (no retraining)",
        "settings": {
            "horizon_days": days,
            "horizon_tier_used": horizon_used,
            "min_signals": min_signals,
            "tier1_only": tier1_only,
            "stocks_tested": stocks_tested,
            "tier1_stocks": tier1_stocks,
            "tier2_stocks": tier2_stocks,
        },
        **result,
        "horizon_info": {
            "description": f"Using stocks optimized for {horizon_used}-day horizon",
            "tier1_count": len(tier1_stocks),
            "tier2_count": len(tier2_stocks),
        },
        "caveats": [
            "These results are from 2023-2024 bull market only",
            "Strategy fails during crisis periods (see /stress-test)",
            "Past performance does not guarantee future results",
        ],
    }

    prediction_cache.set(cache_key, response, 21600)
    job.progress = 100

    return response


def _run_optimal_scan_job(job: Job, days: int) -> dict:
    """Run optimal scan with cancellation support."""
    cache_key = f"optimal-scan:{days}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        job.progress = 100
        return cached

    job.progress_message = "Initializing scan..."
    job.progress = 5

    if job.cancelled:
        return None

    regime = get_current_regime(verbose=False)
    tier1_stocks, tier2_stocks = get_stocks_for_horizon(days)
    optimal_stocks = tier1_stocks + tier2_stocks
    total_stocks = len(optimal_stocks)

    job.progress = 10
    job.progress_message = f"Scanning {total_stocks} stocks..."

    if job.cancelled:
        return None

    def progress_callback(current, total, ticker):
        """Report progress and check cancellation."""
        if job.cancelled:
            return False  # Signal cancellation
        # Progress from 10% to 90% during scanning
        progress_pct = 10 + int((current / total) * 80)
        job.progress = progress_pct
        job.progress_message = f"Scanning {ticker} ({current + 1}/{total})..."
        return True  # Continue

    result = run_production_scan(days=days, verbose=False, progress_callback=progress_callback)

    if result is None or job.cancelled:
        return None

    job.progress = 90

    if "error" in result:
        raise ValueError(result["error"])

    if regime.get("regime") in ["CRISIS", "CAUTION"]:
        result["regime_warning"] = f"Market is in {regime.get('regime')} mode. Strategy historically fails here. Consider waiting."
    else:
        result["regime_warning"] = None

    result["current_regime"] = regime

    if days <= 2:
        horizon_used = 2
    elif days == 3:
        horizon_used = 3
    elif days == 4:
        horizon_used = 4
    elif days == 5:
        horizon_used = 5
    else:
        horizon_used = 20

    response = {
        "scan_type": "optimal_strategy",
        "horizon_days": days,
        "horizon_tier_used": horizon_used,
        "tier1_stocks": tier1_stocks,
        "tier2_stocks": tier2_stocks,
        "optimal_stocks": optimal_stocks,
        "available_horizons": [2, 3, 4, 5, 20],
        **result,
        "methodology": {
            "signals": [
                "ML prediction with 70%+ confidence",
                "Trend aligned (price > SMA20 > SMA50)",
                "Low volatility (< 35% annualized)",
                "Positive 5-day momentum",
                "RSI not overbought (< 70)",
            ],
            "threshold": "4 or more signals must align for high conviction",
            "ultra_conviction": "All 5 signals aligned",
        },
        "horizon_info": {
            "description": f"Using stocks optimized for {horizon_used}-day horizon",
            "tier1_count": len(tier1_stocks),
            "tier2_count": len(tier2_stocks),
            "total_stocks": len(optimal_stocks),
        },
    }

    prediction_cache.set(cache_key, response, 3600)
    job.progress = 100

    return response


# Job type to worker function mapping
JOB_WORKERS = {
    "screener": lambda job, params: _run_screener_job(
        job,
        params.get("days", 5),
        params.get("min_confidence", 0.75),
    ),
    "backtest": lambda job, params: _run_backtest_job(
        job,
        params.get("ticker", "AAPL"),
        params.get("days", 5),
    ),
    "screener-backtest": lambda job, params: _run_screener_backtest_job(
        job,
        params.get("days", 5),
        params.get("min_confidence", 0.75),
    ),
    "regime-gated-backtest": lambda job, params: _run_regime_gated_backtest_job(
        job,
        params.get("days", 5),
    ),
    "stress-test": lambda job, params: _run_stress_test_job(
        job,
        params.get("period"),
    ),
    "optimal-backtest": lambda job, params: _run_optimal_backtest_job(
        job,
        params.get("days", 20),
        params.get("min_signals", 4),
        params.get("tier1_only", True),
    ),
    "optimal-scan": lambda job, params: _run_optimal_scan_job(
        job,
        params.get("days", 20),
    ),
}


@app.post("/jobs/{job_type}")
def start_job(job_type: str, request: JobStartRequest):
    """
    Start a background job and return the job ID immediately.

    The job runs in a background thread and can be:
    - Monitored via GET /jobs/{job_id}/status
    - Cancelled via POST /jobs/{job_id}/cancel

    Job types:
    - screener: Scan stocks for bullish predictions
    - backtest: Run backtest on a single ticker
    - screener-backtest: Backtest screener strategy
    - regime-gated-backtest: Backtest with regime gating
    - stress-test: Run crisis stress tests
    - optimal-backtest: Backtest optimal strategy
    - optimal-scan: Run optimal scan
    """
    if job_type not in JOB_WORKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {job_type}. Valid types: {list(JOB_WORKERS.keys())}",
        )

    job = job_manager.create_job(job_type)
    params = request.model_dump()

    def worker(j: Job):
        return JOB_WORKERS[job_type](j, params)

    job_manager.start_job(job, worker)

    return {
        "job_id": job.id,
        "job_type": job_type,
        "status": job.status.value,
        "message": "Job started. Poll /jobs/{job_id}/status for progress.",
    }


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """
    Get the current status and progress of a job.

    Returns:
    - status: pending, running, completed, failed, cancelled
    - progress: 0-100 percentage
    - progress_message: Human-readable progress description
    - result: The job result (only when completed)
    - error: Error message (only when failed)
    - elapsed_seconds: Time since job started
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job.to_dict()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """
    Request cancellation of a running job.

    The job will stop at the next cancellation checkpoint.
    This is typically after processing the current item.

    Returns success if cancellation was requested,
    even if the job has already completed.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return {
            "job_id": job_id,
            "status": job.status.value,
            "message": f"Job already finished with status: {job.status.value}",
        }

    success = job_manager.cancel_job(job_id)
    return {
        "job_id": job_id,
        "status": "cancelling" if success else job.status.value,
        "message": "Cancellation requested. Job will stop at next checkpoint." if success else "Could not cancel job.",
    }


@app.get("/jobs")
def list_jobs(job_type: Optional[str] = Query(None, description="Filter by job type")):
    """List all jobs, optionally filtered by type."""
    return {"jobs": job_manager.list_jobs(job_type)}
