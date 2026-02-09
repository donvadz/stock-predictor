import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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
    run_screener_backtest,
    run_regime_gated_backtest,
    compute_screener_metrics_from_raw,
    compute_regime_metrics_from_raw,
)

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
    test_periods: int = Query(50, ge=20, le=200, description="Number of test periods (20-200)"),
):
    """
    Run a walk-forward backtest to evaluate model performance on historical data.

    Tests the model by making predictions at historical points and comparing
    to actual outcomes.
    """
    ticker = ticker.upper()
    cache_key = f"backtest:{ticker}:{days}:{test_periods}"

    # Check cache (backtest results cached for 10 minutes)
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch stock data (use long lookback for more test data)
    df = fetch_stock_data(ticker, prediction_days=30)  # Force long lookback
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

    # Fetch fundamental data
    fundamentals = fetch_fundamental_data(ticker)

    # Run backtest
    result = run_backtest(df, days, fundamentals, test_periods)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for backtesting. Need at least {MIN_DATA_POINTS + test_periods + days} data points.",
        )

    # Get stock name
    name = fundamentals.get("name", ticker) if fundamentals else ticker

    response = {
        "ticker": ticker,
        "name": name,
        "days": days,
        "test_periods": test_periods,
        **result,
    }

    # Cache the result (6 hours - uses daily close prices only)
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
    test_periods: int = Query(30, ge=10, le=100, description="Test periods per stock (10-100)"),
):
    """
    Backtest the screener strategy across multiple stocks.

    Simulates what would have happened if you followed screener picks
    (stocks predicted UP with high confidence) historically.

    Includes:
    - Confidence bucket analysis (70-75%, 75-80%, 80%+)
    - Drawdown & risk metrics
    - Regime segmentation (volatility, market trend)
    - Market-relative validation (vs SPY)
    - No-trade frequency reporting

    CACHE OPTIMIZATION:
    - Raw data is cached by days and min_confidence only
    - If cached data has >= requested test_periods, metrics are recomputed instantly
    - This allows 100 periods to be cached and reused for 30, 50, etc.
    """
    # Cache key for raw data: excludes test_periods (we can recompute for fewer periods)
    raw_cache_key = f"screener-backtest-raw:{days}:{min_confidence}"

    # Check if we have cached raw data with enough periods
    try:
        cached_raw = prediction_cache.get(raw_cache_key)
        if cached_raw is not None and isinstance(cached_raw, dict):
            cached_periods = cached_raw.get("max_periods", 0)
            if cached_periods >= test_periods:
                raw_data = cached_raw.get("_raw_data")
                if raw_data:
                    recomputed = compute_screener_metrics_from_raw(raw_data, test_periods)
                    if recomputed:
                        response = {
                            "days": days,
                            "min_confidence": min_confidence,
                            "test_periods": test_periods,
                            "_cached": True,
                            **recomputed,
                        }
                        return response
    except Exception:
        # If cache fails, continue to run fresh backtest
        pass

    # No suitable cache - run fresh backtest with requested periods
    run_periods = test_periods

    # Fetch SPY data for market benchmark comparison
    spy_data = fetch_stock_data("SPY", prediction_days=30)

    # Fetch data for all test stocks in parallel
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

    # Run screener backtest with max periods
    result = run_screener_backtest(stocks_data, days, min_confidence, run_periods, spy_data)

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="No screener picks found with the given criteria. Try lowering min_confidence.",
        )

    # Cache raw data for future recomputation (requests with fewer periods can reuse)
    raw_data = result.pop("_raw_data", None)
    if raw_data:
        try:
            existing = prediction_cache.get(raw_cache_key)
            existing_periods = existing.get("max_periods", 0) if existing and isinstance(existing, dict) else 0
            if run_periods > existing_periods:
                cache_entry = {
                    "max_periods": run_periods,
                    "_raw_data": raw_data,
                }
                prediction_cache.set(raw_cache_key, cache_entry, 21600)
        except Exception:
            pass  # Caching is best-effort

    response = {
        "days": days,
        "min_confidence": min_confidence,
        "test_periods": test_periods,
        **result,
    }

    return response


@app.get("/regime-gated-backtest")
def regime_gated_backtest(
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    test_periods: int = Query(30, ge=10, le=100, description="Test periods per stock (10-100)"),
):
    """
    Backtest the screener with market regime gate based on SPY performance.

    Compares three modes:
    - No Gate (Baseline): Standard 75% confidence threshold
    - Bear Suppressed: No trades when SPY trailing return < -1%
    - Bear 80% Confidence: Require 80% confidence when SPY trailing < -1%

    Returns comparison table with metrics for each mode:
    - Total trades, win rate, avg return
    - Worst trade, worst streak, max drawdown
    - % of picks beating SPY

    CACHE OPTIMIZATION:
    - Raw data is cached by days only
    - If cached data has >= requested test_periods, metrics are recomputed instantly
    - This allows 100 periods to be cached and reused for 30, 50, etc.
    """
    # Cache key for raw data: excludes test_periods (we can recompute for fewer periods)
    raw_cache_key = f"regime-gated-backtest-raw:{days}"

    # Check if we have cached raw data with enough periods
    try:
        cached_raw = prediction_cache.get(raw_cache_key)
        if cached_raw is not None and isinstance(cached_raw, dict):
            cached_periods = cached_raw.get("max_periods", 0)
            if cached_periods >= test_periods:
                raw_data = cached_raw.get("_raw_data")
                if raw_data:
                    recomputed = compute_regime_metrics_from_raw(raw_data, test_periods)
                    if recomputed:
                        response = {
                            "days": days,
                            "test_periods": test_periods,
                            "_cached": True,
                            **recomputed,
                        }
                        return response
    except Exception:
        pass  # If cache fails, continue to run fresh backtest

    # No suitable cache - run fresh backtest with requested periods
    run_periods = test_periods

    # Fetch SPY data for regime detection
    spy_data = fetch_stock_data("SPY", prediction_days=30)

    if spy_data is None or len(spy_data) < 50:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch SPY data for regime detection.",
        )

    # Fetch data for all test stocks in parallel
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

    # Run regime-gated backtest with max periods
    result = run_regime_gated_backtest(
        stocks_data,
        days,
        min_confidence=0.75,
        test_periods=run_periods,
        spy_data=spy_data,
    )

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data for regime-gated backtest.",
        )

    # Cache raw data for future recomputation (requests with fewer periods can reuse)
    raw_data = result.pop("_raw_data", None)
    if raw_data:
        try:
            existing = prediction_cache.get(raw_cache_key)
            existing_periods = existing.get("max_periods", 0) if existing and isinstance(existing, dict) else 0
            if run_periods > existing_periods:
                cache_entry = {
                    "max_periods": run_periods,
                    "_raw_data": raw_data,
                }
                prediction_cache.set(raw_cache_key, cache_entry, 21600)
        except Exception:
            pass  # Caching is best-effort

    response = {
        "days": days,
        "test_periods": test_periods,
        **result,
    }

    return response
