"""
Enhanced Model Module - Better ML Approaches for Stock Prediction

Improvements over baseline RandomForest:
1. XGBoost/LightGBM - Gradient boosting often outperforms RF
2. Ensemble Voting - Combine multiple models
3. Calibrated Probabilities - Better confidence estimates
4. Feature Selection - Reduce noise

Usage:
    from model_enhanced import EnhancedPredictor

    predictor = EnhancedPredictor(model_type="ensemble")
    predictor.fit(X_train, y_train)
    predictions, confidence = predictor.predict(X_test)
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel

# Try to import XGBoost and LightGBM (optional dependencies)
HAS_XGB = False
HAS_LGB = False

try:
    import xgboost as xgb
    # Test that it actually works (not just importable)
    _ = xgb.XGBClassifier()
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False

try:
    import lightgbm as lgb
    # Test that it actually works
    _ = lgb.LGBMClassifier()
    HAS_LGB = True
except (ImportError, Exception):
    HAS_LGB = False


class EnhancedPredictor:
    """
    Enhanced stock direction predictor with multiple ML approaches.

    Model types:
    - "rf": RandomForest (baseline)
    - "xgboost": XGBoost gradient boosting
    - "lightgbm": LightGBM gradient boosting
    - "gradient_boost": Sklearn GradientBoosting
    - "ensemble": Voting ensemble of multiple models
    - "ensemble_calibrated": Calibrated ensemble with better probabilities
    """

    def __init__(
        self,
        model_type: str = "ensemble",
        n_estimators: int = 200,
        max_depth: int = 8,
        feature_selection: bool = True,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_selection = feature_selection
        self.calibrate = calibrate
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False

    def _create_base_models(self) -> Dict:
        """Create base models for ensemble."""
        models = {}

        # RandomForest - good baseline
        models["rf"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Sklearn GradientBoosting - always available
        models["gb"] = GradientBoostingClassifier(
            n_estimators=min(100, self.n_estimators),  # GB is slower
            max_depth=min(5, self.max_depth),
            min_samples_leaf=5,
            random_state=self.random_state,
        )

        # XGBoost - if available
        if HAS_XGB:
            models["xgb"] = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

        # LightGBM - if available
        if HAS_LGB:
            models["lgb"] = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

        return models

    def _create_model(self):
        """Create the selected model type."""
        base_models = self._create_base_models()

        if self.model_type == "rf":
            return base_models["rf"]

        elif self.model_type == "gradient_boost":
            return base_models["gb"]

        elif self.model_type == "xgboost":
            if not HAS_XGB:
                print("XGBoost not installed, falling back to GradientBoosting")
                return base_models["gb"]
            return base_models["xgb"]

        elif self.model_type == "lightgbm":
            if not HAS_LGB:
                print("LightGBM not installed, falling back to GradientBoosting")
                return base_models["gb"]
            return base_models["lgb"]

        elif self.model_type in ["ensemble", "ensemble_calibrated"]:
            # Build ensemble from available models
            estimators = [("rf", base_models["rf"])]

            if HAS_XGB:
                estimators.append(("xgb", base_models["xgb"]))
            if HAS_LGB:
                estimators.append(("lgb", base_models["lgb"]))

            # If we only have RF, add gradient boosting for diversity
            if len(estimators) == 1:
                estimators.append(("gb", base_models["gb"]))

            return VotingClassifier(
                estimators=estimators,
                voting="soft",  # Use probabilities
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> "EnhancedPredictor":
        """
        Fit the model.

        Args:
            X: Feature matrix
            y: Target labels (0/1)
            feature_names: Optional feature names for selection
        """
        # Handle infinite values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Feature selection (optional)
        if self.feature_selection and X.shape[1] > 15:
            # Use RF to select important features
            selector_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
            )
            selector_model.fit(X_scaled, y)

            # Select features with importance > median
            importances = selector_model.feature_importances_
            threshold = np.median(importances)

            self.feature_selector = SelectFromModel(
                selector_model,
                threshold=threshold,
                prefit=True,
            )
            X_selected = self.feature_selector.transform(X_scaled)

            # Track which features were selected
            if feature_names is not None:
                mask = self.feature_selector.get_support()
                self.selected_features = [f for f, m in zip(feature_names, mask) if m]

            X_train = X_selected
        else:
            X_train = X_scaled

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train, y)

        # Calibrate probabilities (optional)
        if self.calibrate and self.model_type == "ensemble_calibrated":
            self.model = CalibratedClassifierCV(
                self.model,
                method="isotonic",
                cv=3,
            )
            self.model.fit(X_train, y)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict direction and confidence.

        Returns:
            Tuple of (predictions, confidences)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle infinite values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Feature selection
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)

        return predictions, confidences

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances if available."""
        if not self.is_fitted:
            return None

        # Try to get importances from the model
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "estimators_"):
            # Ensemble - average importances
            all_importances = []
            for name, est in self.model.estimators_:
                if hasattr(est, "feature_importances_"):
                    all_importances.append(est.feature_importances_)
            if all_importances:
                importances = np.mean(all_importances, axis=0)
            else:
                return None
        else:
            return None

        if self.selected_features is not None:
            return dict(zip(self.selected_features, importances))
        return {"feature_" + str(i): imp for i, imp in enumerate(importances)}


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str] = None,
) -> Dict:
    """
    Compare all available model types.

    Returns dict with accuracy for each model.
    """
    results = {}

    model_types = ["rf", "gradient_boost", "ensemble"]
    if HAS_XGB:
        model_types.append("xgboost")
    if HAS_LGB:
        model_types.append("lightgbm")

    for model_type in model_types:
        try:
            predictor = EnhancedPredictor(
                model_type=model_type,
                feature_selection=True,
                calibrate=False,
            )
            predictor.fit(X_train, y_train, feature_names)
            predictions, confidences = predictor.predict(X_test)

            accuracy = (predictions == y_test).mean()

            # High confidence accuracy
            high_conf_mask = confidences >= 0.75
            if high_conf_mask.sum() > 0:
                high_conf_acc = (predictions[high_conf_mask] == y_test[high_conf_mask]).mean()
            else:
                high_conf_acc = 0

            results[model_type] = {
                "accuracy": round(accuracy * 100, 2),
                "high_conf_accuracy": round(high_conf_acc * 100, 2),
                "high_conf_count": int(high_conf_mask.sum()),
                "avg_confidence": round(confidences.mean() * 100, 2),
            }
        except Exception as e:
            results[model_type] = {"error": str(e)}

    return results


# Convenience function for quick predictions
def predict_with_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    feature_names: List[str] = None,
) -> Tuple[int, float]:
    """
    Quick ensemble prediction for a single sample.

    Returns:
        Tuple of (direction, confidence)
        direction: 1 for UP, 0 for DOWN
        confidence: 0.0 to 1.0
    """
    predictor = EnhancedPredictor(model_type="ensemble")
    predictor.fit(X_train, y_train, feature_names)
    predictions, confidences = predictor.predict(X_predict)

    return int(predictions[0]), float(confidences[0])
