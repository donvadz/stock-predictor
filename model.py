from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import MIN_DATA_POINTS


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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
    features["sma_ratio_5_20"] = sma_5 / sma_20
    features["sma_ratio_20_50"] = sma_20 / sma_50

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

    return features


def predict_direction(df: pd.DataFrame, days: int) -> Optional[tuple[str, float]]:
    """
    Train model and predict stock direction.

    Args:
        df: DataFrame with OHLCV data
        days: Number of days ahead to predict

    Returns:
        Tuple of (direction, confidence) or None if prediction fails
    """
    if len(df) < MIN_DATA_POINTS:
        return None

    # Compute features
    features = compute_features(df)

    # Construct target: 1 if price goes up in N days, 0 otherwise
    future_return = df["close"].shift(-days) / df["close"] - 1
    target = (future_return > 0).astype(int)

    # Combine features and target
    feature_cols = features.columns.tolist()
    data = features.copy()
    data["target"] = target

    # Drop rows with NaN (from rolling calculations and future shift)
    data = data.dropna()

    if len(data) < MIN_DATA_POINTS:
        return None

    # Split: all but last row for training, last row for prediction
    # The last `days` rows don't have valid targets, so we use the most recent valid row
    train_data = data.iloc[:-1]  # All rows except the last one with valid target

    if len(train_data) < MIN_DATA_POINTS:
        return None

    X_train = train_data[feature_cols]
    y_train = train_data["target"]

    # For prediction, use the most recent features (from original features df)
    # Get the last row that has valid features
    latest_features = features.dropna().iloc[[-1]]
    X_predict = latest_features[feature_cols]

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict
    prediction = model.predict(X_predict)[0]
    probabilities = model.predict_proba(X_predict)[0]
    confidence = float(max(probabilities))

    direction = "up" if prediction == 1 else "down"

    return direction, confidence
