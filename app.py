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
from data import fetch_stock_data
from model import predict_direction

app = FastAPI(title="Stock Direction Predictor")

cors_origins = ["http://localhost:5173"]
if os.environ.get("FRONTEND_URL"):
    cors_origins.append(os.environ["FRONTEND_URL"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Fetch stock data
    df = fetch_stock_data(ticker)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

    # Check sufficient data
    if len(df) < MIN_DATA_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for training. Need {MIN_DATA_POINTS} data points, got {len(df)}",
        )

    # Make prediction
    result = predict_direction(df, days)
    if result is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    direction, confidence = result
    response = {
        "ticker": ticker,
        "days": days,
        "direction": direction,
        "confidence": round(confidence, 4),
    }

    # Cache the result
    prediction_cache.set(cache_key, response, PREDICTION_CACHE_TTL)

    return response


def _predict_single(ticker: str, days: int) -> Optional[dict]:
    """Helper to predict a single stock, returns None on failure."""
    try:
        df = fetch_stock_data(ticker)
        if df is None or len(df) < MIN_DATA_POINTS:
            return None
        result = predict_direction(df, days)
        if result is None:
            return None
        direction, confidence = result
        return {
            "ticker": ticker,
            "direction": direction,
            "confidence": round(confidence, 4),
        }
    except Exception:
        return None


@app.get("/screener")
def screener(
    days: int = Query(5, ge=1, le=30, description="Days ahead to predict (1-30)"),
    min_confidence: float = Query(0.75, ge=0.5, le=1.0, description="Minimum confidence (0.5-1.0)"),
):
    """Scan stocks and return those predicted to go UP with confidence above threshold."""
    cache_key = f"screener:{days}:{min_confidence}"

    # Check cache
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    matches = []
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
            elif result["direction"] == "up" and result["confidence"] >= min_confidence:
                matches.append({
                    "ticker": result["ticker"],
                    "confidence": result["confidence"],
                })

    # Sort by confidence descending
    matches.sort(key=lambda x: x["confidence"], reverse=True)

    response = {
        "days": days,
        "min_confidence": min_confidence,
        "stocks_scanned": scanned,
        "errors": errors,
        "matches": matches,
    }

    # Cache the result
    prediction_cache.set(cache_key, response, SCREENER_CACHE_TTL)

    return response
