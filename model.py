from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import MIN_DATA_POINTS


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from OHLCV data."""
    features = pd.DataFrame(index=df.index)

    # Price returns
    features["return_1d"] = df["close"].pct_change(1)
    features["return_5d"] = df["close"].pct_change(5)
    features["return_20d"] = df["close"].pct_change(20)

    # Volatility
    features["volatility_5d"] = features["return_1d"].rolling(5).std()
    features["volatility_20d"] = features["return_1d"].rolling(20).std()

    # Moving average ratios
    sma_5 = df["close"].rolling(5).mean()
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    sma_200 = df["close"].rolling(200).mean()
    features["sma_ratio_5_20"] = sma_5 / sma_20
    features["sma_ratio_20_50"] = sma_20 / sma_50
    features["sma_ratio_50_200"] = sma_50 / sma_200  # Golden/death cross indicator
    features["price_to_sma_20"] = df["close"] / sma_20
    features["price_to_sma_50"] = df["close"] / sma_50

    # Volume features
    features["volume_change_5d"] = df["volume"].pct_change(5)
    volume_sma_20 = df["volume"].rolling(20).mean()
    features["volume_ratio"] = df["volume"] / volume_sma_20

    # Price range features
    features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    high_low_diff = df["high"] - df["low"]
    features["close_position"] = np.where(
        high_low_diff > 0,
        (df["close"] - df["low"]) / high_low_diff,
        0.5  # Default when high == low
    )

    # Momentum indicators
    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_histogram"] = features["macd"] - features["macd_signal"]

    # Bollinger Bands
    bb_sma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    features["bb_upper"] = (df["close"] - (bb_sma + 2 * bb_std)) / df["close"]
    features["bb_lower"] = (df["close"] - (bb_sma - 2 * bb_std)) / df["close"]
    features["bb_width"] = (4 * bb_std) / bb_sma

    # Average True Range (ATR)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / df["close"]

    # On-Balance Volume trend
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    features["obv_trend"] = obv.pct_change(20)

    return features


def add_fundamental_features(features: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """Add fundamental data as constant columns to the feature DataFrame."""
    if fundamentals is None:
        return features

    # List of fundamental features to add
    fundamental_cols = [
        "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales",
        "profit_margin", "operating_margin", "roe", "roa",
        "revenue_growth", "earnings_growth",
        "dividend_yield", "payout_ratio",
        "debt_to_equity", "current_ratio", "quick_ratio",
        "beta", "fifty_two_week_change", "short_ratio", "short_percent",
        "sector", "analyst_sentiment", "analyst_count",
        "earnings_surprise", "news_count", "institutional_holders",
    ]

    for col in fundamental_cols:
        value = fundamentals.get(col)
        # Handle None/NaN values with 0 (will be normalized later)
        features[col] = 0 if value is None or (isinstance(value, float) and np.isnan(value)) else value

    # Normalize market cap to a reasonable scale (log transform)
    market_cap = fundamentals.get("market_cap")
    if market_cap and market_cap > 0:
        features["market_cap_log"] = np.log10(market_cap)
    else:
        features["market_cap_log"] = 0

    return features


def predict_direction(
    df: pd.DataFrame,
    days: int,
    fundamentals: Optional[dict] = None
) -> Optional[tuple[str, float, float]]:
    """
    Train model and predict stock direction and expected return.

    Args:
        df: DataFrame with OHLCV data
        days: Number of days ahead to predict
        fundamentals: Optional dict of fundamental data

    Returns:
        Tuple of (direction, confidence, predicted_return) or None if prediction fails
    """
    if len(df) < MIN_DATA_POINTS:
        return None

    # Compute technical features
    features = compute_technical_features(df)

    # Add fundamental features
    features = add_fundamental_features(features, fundamentals)

    # Construct targets
    future_return = df["close"].shift(-days) / df["close"] - 1
    target_direction = (future_return > 0).astype(int)
    target_return = future_return  # Actual percentage return

    # Combine features and targets
    feature_cols = features.columns.tolist()
    data = features.copy()
    data["target_direction"] = target_direction
    data["target_return"] = target_return

    # Fill NaN values from rolling calculations (forward fill, then backward fill remaining)
    data = data.ffill().bfill()

    # Drop any remaining rows with NaN
    data = data.dropna()

    if len(data) < MIN_DATA_POINTS:
        return None

    # Split: all but last row for training, last row for prediction
    train_data = data.iloc[:-1]

    if len(train_data) < MIN_DATA_POINTS:
        return None

    X_train = train_data[feature_cols]
    y_train_direction = train_data["target_direction"]
    y_train_return = train_data["target_return"]

    # For prediction, use the most recent features
    latest_features = features.ffill().bfill().iloc[[-1]]
    X_predict = latest_features[feature_cols]

    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_predict = X_predict.replace([np.inf, -np.inf], 0)

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    # Train classifier for direction
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train_scaled, y_train_direction)

    # Train regressor for return prediction
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train_scaled, y_train_return)

    # Predict direction
    prediction = classifier.predict(X_predict_scaled)[0]
    probabilities = classifier.predict_proba(X_predict_scaled)[0]
    confidence = float(max(probabilities))
    direction = "up" if prediction == 1 else "down"

    # Predict return
    predicted_return = float(regressor.predict(X_predict_scaled)[0])

    return direction, confidence, predicted_return
