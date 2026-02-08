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

app = FastAPI(title="Stock Direction Predictor")


@app.get("/stocks")
def get_stocks():
    """Return list of available stocks for prediction."""
    return {"stocks": STOCK_LIST}

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
    response = {
        "ticker": ticker,
        "name": name,
        "days": days,
        "direction": direction,
        "confidence": round(confidence, 4),
        "latest_price": latest_price,
        "predicted_price": predicted_price,
        "predicted_change": round(aligned_return * 100, 2),  # As percentage
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
                    "name": result["name"],
                    "confidence": result["confidence"],
                    "latest_price": result["latest_price"],
                    "predicted_price": result["predicted_price"],
                    "predicted_change": result["predicted_change"],
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
