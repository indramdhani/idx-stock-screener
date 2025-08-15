"""
Signal Enhancer Module for Indonesian Stock Screener

This module provides machine learning enhancements for trading signal generation:
- ML-based signal confidence scoring
- Feature engineering from technical indicators
- Ensemble model predictions
- Signal filtering and ranking
- Model training and validation

Author: IDX Stock Screener Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.pipeline import Pipeline
    import lightgbm as lgb
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..data.models.signal import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class MLModelType(Enum):
    """Machine learning model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    SVM = "svm"
    ENSEMBLE = "ensemble"


class SignalDirection(Enum):
    """Signal direction for classification"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class MLSignal:
    """Enhanced trading signal with ML predictions"""
    # Original signal data
    original_signal: TradingSignal

    # ML enhancements
    ml_confidence: float = 0.0
    ml_probability: float = 0.0
    ensemble_score: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Model predictions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    consensus_prediction: float = 0.0
    prediction_std: float = 0.0

    # Enhanced scoring
    technical_score: float = 0.0
    ml_score: float = 0.0
    combined_score: float = 0.0
    final_confidence: float = 0.0

    # Risk adjustments
    risk_adjusted_confidence: float = 0.0
    volatility_penalty: float = 0.0

    def to_dict(self) -> Dict:
        """Convert ML signal to dictionary"""
        result = self.original_signal.to_dict()
        result.update({
            'ml_confidence': self.ml_confidence,
            'ml_probability': self.ml_probability,
            'ensemble_score': self.ensemble_score,
            'feature_importance': self.feature_importance,
            'model_predictions': self.model_predictions,
            'consensus_prediction': self.consensus_prediction,
            'prediction_std': self.prediction_std,
            'technical_score': self.technical_score,
            'ml_score': self.ml_score,
            'combined_score': self.combined_score,
            'final_confidence': self.final_confidence,
            'risk_adjusted_confidence': self.risk_adjusted_confidence,
            'volatility_penalty': self.volatility_penalty
        })
        return result


@dataclass
class FeatureSet:
    """Technical analysis features for ML models"""
    # Price features
    price_change: float = 0.0
    price_momentum_5d: float = 0.0
    price_momentum_10d: float = 0.0
    price_volatility: float = 0.0

    # Technical indicators
    rsi: float = 50.0
    rsi_change: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Moving averages
    sma_5: float = 0.0
    sma_13: float = 0.0
    sma_21: float = 0.0
    ema_5: float = 0.0
    ema_13: float = 0.0
    ema_21: float = 0.0

    # Price position relative to MAs
    price_vs_sma5: float = 0.0
    price_vs_sma13: float = 0.0
    price_vs_sma21: float = 0.0
    price_vs_ema5: float = 0.0
    price_vs_ema13: float = 0.0
    price_vs_ema21: float = 0.0

    # Volume features
    volume_ratio: float = 1.0
    volume_sma_ratio: float = 1.0
    volume_change: float = 0.0

    # VWAP features
    price_vs_vwap: float = 0.0
    vwap_slope: float = 0.0

    # Volatility features
    atr: float = 0.0
    atr_ratio: float = 1.0
    bollinger_position: float = 0.5

    # Pattern features
    higher_highs: float = 0.0
    lower_lows: float = 0.0
    support_resistance_strength: float = 0.0

    # Market structure
    trend_strength: float = 0.0
    consolidation_score: float = 0.0
    breakout_potential: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models"""
        return np.array(list(self.__dict__.values()))

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary"""
        return dict(self.__dict__)


class SignalEnhancer:
    """Machine Learning Signal Enhancer for trading signals"""

    def __init__(
        self,
        model_types: List[MLModelType] = None,
        ensemble_weights: Dict[str, float] = None,
        feature_importance_threshold: float = 0.01,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize Signal Enhancer

        Args:
            model_types: List of ML models to use
            ensemble_weights: Weights for ensemble averaging
            feature_importance_threshold: Minimum feature importance to consider
            confidence_threshold: Minimum confidence for signal acceptance
        """
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed. Install with: pip install scikit-learn lightgbm xgboost")

        self.model_types = model_types or [
            MLModelType.RANDOM_FOREST,
            MLModelType.LIGHTGBM,
            MLModelType.GRADIENT_BOOSTING
        ]

        self.ensemble_weights = ensemble_weights or {
            'random_forest': 0.3,
            'lightgbm': 0.4,
            'gradient_boosting': 0.3
        }

        self.feature_importance_threshold = feature_importance_threshold
        self.confidence_threshold = confidence_threshold

        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_trained = False

        # Performance tracking
        self.training_scores: Dict[str, float] = {}
        self.validation_scores: Dict[str, float] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Signal Enhancer initialized with models: {[m.value for m in self.model_types]}")

    def extract_features(self, stock_data: pd.DataFrame, signal: TradingSignal) -> FeatureSet:
        """
        Extract technical features from stock data for ML models

        Args:
            stock_data: Historical stock data with OHLCV columns
            signal: Original trading signal

        Returns:
            FeatureSet with extracted features
        """
        try:
            if stock_data.empty or len(stock_data) < 21:
                return FeatureSet()

            features = FeatureSet()

            # Get latest data point
            latest = stock_data.iloc[-1]
            close_prices = stock_data['close']
            volumes = stock_data['volume']

            # Price features
            features.price_change = (latest['close'] - stock_data.iloc[-2]['close']) / stock_data.iloc[-2]['close']
            features.price_momentum_5d = (latest['close'] - stock_data.iloc[-6]['close']) / stock_data.iloc[-6]['close'] if len(stock_data) >= 6 else 0
            features.price_momentum_10d = (latest['close'] - stock_data.iloc[-11]['close']) / stock_data.iloc[-11]['close'] if len(stock_data) >= 11 else 0
            features.price_volatility = close_prices.rolling(10).std().iloc[-1] / close_prices.rolling(10).mean().iloc[-1] if len(stock_data) >= 10 else 0

            # Technical indicators (assuming they exist in signal)
            if hasattr(signal, 'technical_indicators'):
                indicators = signal.technical_indicators
                features.rsi = indicators.get('rsi', 50.0)
                features.rsi_change = features.rsi - indicators.get('rsi_prev', 50.0)
                features.atr = indicators.get('atr', latest['high'] - latest['low'])

            # Moving averages
            if len(stock_data) >= 5:
                features.sma_5 = close_prices.rolling(5).mean().iloc[-1]
                features.ema_5 = close_prices.ewm(span=5).mean().iloc[-1]
                features.price_vs_sma5 = (latest['close'] - features.sma_5) / features.sma_5
                features.price_vs_ema5 = (latest['close'] - features.ema_5) / features.ema_5

            if len(stock_data) >= 13:
                features.sma_13 = close_prices.rolling(13).mean().iloc[-1]
                features.ema_13 = close_prices.ewm(span=13).mean().iloc[-1]
                features.price_vs_sma13 = (latest['close'] - features.sma_13) / features.sma_13
                features.price_vs_ema13 = (latest['close'] - features.ema_13) / features.ema_13

            if len(stock_data) >= 21:
                features.sma_21 = close_prices.rolling(21).mean().iloc[-1]
                features.ema_21 = close_prices.ewm(span=21).mean().iloc[-1]
                features.price_vs_sma21 = (latest['close'] - features.sma_21) / features.sma_21
                features.price_vs_ema21 = (latest['close'] - features.ema_21) / features.ema_21

            # Volume features
            if len(stock_data) >= 5:
                avg_volume = volumes.rolling(20).mean().iloc[-1] if len(stock_data) >= 20 else volumes.mean()
                features.volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1.0
                features.volume_change = (latest['volume'] - volumes.iloc[-2]) / volumes.iloc[-2] if volumes.iloc[-2] > 0 else 0

            # VWAP features
            if hasattr(signal, 'vwap'):
                features.price_vs_vwap = (latest['close'] - signal.vwap) / signal.vwap if signal.vwap > 0 else 0

            # Pattern recognition features
            if len(stock_data) >= 10:
                highs = stock_data['high'].rolling(10)
                lows = stock_data['low'].rolling(10)
                features.higher_highs = 1.0 if latest['high'] > highs.max().iloc[-2] else 0.0
                features.lower_lows = 1.0 if latest['low'] < lows.min().iloc[-2] else 0.0

            # Trend strength
            if len(stock_data) >= 10:
                price_changes = close_prices.pct_change().rolling(10)
                positive_changes = (price_changes > 0).sum().iloc[-1]
                features.trend_strength = positive_changes / 10.0

            return features

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return FeatureSet()

    def prepare_training_data(
        self,
        historical_signals: List[Tuple[TradingSignal, pd.DataFrame, float]],
        lookforward_days: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical signals

        Args:
            historical_signals: List of (signal, stock_data, actual_return) tuples
            lookforward_days: Days to look forward for return calculation

        Returns:
            Tuple of (features, labels) for training
        """
        try:
            features_list = []
            labels = []

            for signal, stock_data, actual_return in historical_signals:
                try:
                    # Extract features
                    feature_set = self.extract_features(stock_data, signal)
                    features_array = feature_set.to_array()

                    # Create label based on actual return
                    if actual_return > 0.02:  # > 2% return
                        label = SignalDirection.BUY.value
                    elif actual_return < -0.01:  # < -1% return
                        label = SignalDirection.SELL.value
                    else:
                        label = SignalDirection.HOLD.value

                    features_list.append(features_array)
                    labels.append(label)

                except Exception as e:
                    self.logger.warning(f"Error processing signal: {e}")
                    continue

            if not features_list:
                raise ValueError("No valid training data prepared")

            features = np.array(features_list)
            labels = np.array(labels)

            # Store feature names for later use
            self.feature_names = list(FeatureSet().__dict__.keys())

            self.logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
            return features, labels

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def train_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        optimize_hyperparameters: bool = True
    ) -> Dict[str, float]:
        """
        Train all ML models

        Args:
            X: Feature matrix
            y: Target labels
            validation_split: Fraction of data for validation
            optimize_hyperparameters: Whether to optimize hyperparameters

        Returns:
            Dictionary of model performance scores
        """
        try:
            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("Empty training data provided")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )

            performance_scores = {}

            for model_type in self.model_types:
                try:
                    self.logger.info(f"Training {model_type.value} model...")

                    # Create and train model
                    model, scaler = self._create_model(model_type, optimize_hyperparameters)

                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)

                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Validate model
                    train_score = model.score(X_train_scaled, y_train)
                    val_score = model.score(X_val_scaled, y_val)

                    # Store model and scaler
                    self.models[model_type.value] = model
                    self.scalers[model_type.value] = scaler

                    # Store scores
                    performance_scores[model_type.value] = val_score
                    self.training_scores[model_type.value] = train_score
                    self.validation_scores[model_type.value] = val_score

                    self.logger.info(
                        f"{model_type.value}: Train={train_score:.3f}, Val={val_score:.3f}"
                    )

                except Exception as e:
                    self.logger.error(f"Error training {model_type.value}: {e}")
                    continue

            if self.models:
                self.is_trained = True
                self.logger.info(f"Successfully trained {len(self.models)} models")
            else:
                raise RuntimeError("No models were successfully trained")

            return performance_scores

        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            return {}

    def _create_model(
        self,
        model_type: MLModelType,
        optimize_hyperparameters: bool = True
    ) -> Tuple[Any, Any]:
        """Create and optionally optimize ML model"""
        try:
            scaler = RobustScaler()

            if model_type == MLModelType.RANDOM_FOREST:
                if optimize_hyperparameters:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                    base_model = RandomForestClassifier(random_state=42)
                    model = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)

            elif model_type == MLModelType.GRADIENT_BOOSTING:
                if optimize_hyperparameters:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                    base_model = GradientBoostingClassifier(random_state=42)
                    model = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
                else:
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

            elif model_type == MLModelType.LIGHTGBM:
                params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'verbose': -1,
                    'random_state': 42
                }
                model = lgb.LGBMClassifier(**params)

            elif model_type == MLModelType.XGBOOST:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric='mlogloss'
                )

            elif model_type == MLModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(random_state=42, max_iter=1000)

            elif model_type == MLModelType.SVM:
                model = SVC(probability=True, random_state=42)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            return model, scaler

        except Exception as e:
            self.logger.error(f"Error creating model {model_type.value}: {e}")
            raise

    def enhance_signal(
        self,
        signal: TradingSignal,
        stock_data: pd.DataFrame
    ) -> MLSignal:
        """
        Enhance trading signal with ML predictions

        Args:
            signal: Original trading signal
            stock_data: Stock price data

        Returns:
            Enhanced ML signal
        """
        try:
            if not self.is_trained:
                self.logger.warning("Models not trained, returning original signal")
                return MLSignal(original_signal=signal)

            # Extract features
            feature_set = self.extract_features(stock_data, signal)
            features = feature_set.to_array().reshape(1, -1)

            # Get predictions from all models
            model_predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                try:
                    scaler = self.scalers[model_name]
                    features_scaled = scaler.transform(features)

                    # Get prediction and probability
                    prediction = model.predict(features_scaled)[0]
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0]
                        # Get probability of positive class (BUY)
                        buy_proba = proba[2] if len(proba) > 2 else proba[-1]  # Assuming BUY=1 is last class
                    else:
                        buy_proba = 0.5

                    model_predictions[model_name] = prediction
                    probabilities[model_name] = buy_proba

                except Exception as e:
                    self.logger.warning(f"Error getting prediction from {model_name}: {e}")
                    continue

            # Calculate ensemble predictions
            consensus_prediction = np.mean(list(model_predictions.values())) if model_predictions else 0
            prediction_std = np.std(list(model_predictions.values())) if len(model_predictions) > 1 else 0

            # Calculate weighted ensemble probability
            weighted_proba = 0.0
            total_weight = 0.0
            for model_name, proba in probabilities.items():
                weight = self.ensemble_weights.get(model_name, 1.0)
                weighted_proba += proba * weight
                total_weight += weight

            ensemble_probability = weighted_proba / total_weight if total_weight > 0 else 0.5

            # Calculate ML confidence
            ml_confidence = abs(ensemble_probability - 0.5) * 2  # Scale to 0-1

            # Calculate technical score (from original signal)
            technical_score = signal.confidence_score / 100.0 if hasattr(signal, 'confidence_score') else 0.5

            # Combine scores
            ml_score = ml_confidence
            combined_score = (technical_score * 0.4 + ml_score * 0.6)

            # Risk adjustment based on volatility
            volatility_penalty = min(feature_set.price_volatility * 0.5, 0.3)  # Max 30% penalty
            risk_adjusted_confidence = combined_score * (1 - volatility_penalty)

            # Final confidence
            final_confidence = risk_adjusted_confidence * 100  # Scale to 0-100

            # Get feature importance if available
            feature_importance = {}
            if model_predictions:
                try:
                    # Use first available model for feature importance
                    first_model_name = list(self.models.keys())[0]
                    first_model = self.models[first_model_name]

                    if hasattr(first_model, 'feature_importances_'):
                        importances = first_model.feature_importances_
                        feature_importance = dict(zip(self.feature_names, importances))
                    elif hasattr(first_model, 'coef_'):
                        importances = np.abs(first_model.coef_[0])
                        feature_importance = dict(zip(self.feature_names, importances))

                except Exception as e:
                    self.logger.warning(f"Error getting feature importance: {e}")

            # Create enhanced signal
            ml_signal = MLSignal(
                original_signal=signal,
                ml_confidence=ml_confidence * 100,
                ml_probability=ensemble_probability * 100,
                ensemble_score=consensus_prediction,
                feature_importance=feature_importance,
                model_predictions=model_predictions,
                consensus_prediction=consensus_prediction,
                prediction_std=prediction_std,
                technical_score=technical_score * 100,
                ml_score=ml_score * 100,
                combined_score=combined_score * 100,
                final_confidence=final_confidence,
                risk_adjusted_confidence=risk_adjusted_confidence * 100,
                volatility_penalty=volatility_penalty * 100
            )

            return ml_signal

        except Exception as e:
            self.logger.error(f"Error enhancing signal: {e}")
            return MLSignal(original_signal=signal)

    def save_models(self, filepath: Path) -> bool:
        """Save trained models to file"""
        try:
            if not self.is_trained:
                self.logger.warning("No trained models to save")
                return False

            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'ensemble_weights': self.ensemble_weights,
                'training_scores': self.training_scores,
                'validation_scores': self.validation_scores,
                'model_types': [m.value for m in self.model_types]
            }

            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False

    def load_models(self, filepath: Path) -> bool:
        """Load trained models from file"""
        try:
            if not filepath.exists():
                self.logger.error(f"Model file not found: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.ensemble_weights = model_data['ensemble_weights']
            self.training_scores = model_data.get('training_scores', {})
            self.validation_scores = model_data.get('validation_scores', {})

            self.is_trained = True
            self.logger.info(f"Models loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False

    def get_model_summary(self) -> str:
        """Get summary of trained models"""
        try:
            if not self.is_trained:
                return "No models trained"

            summary = []
            summary.append("=" * 50)
            summary.append("ML MODEL SUMMARY")
            summary.append("=" * 50)

            for model_name in self.models.keys():
                train_score = self.training_scores.get(model_name, 0)
                val_score = self.validation_scores.get(model_name, 0)
                weight = self.ensemble_weights.get(model_name, 1.0)

                summary.append(f"{model_name.upper()}:")
                summary.append(f"  Training Score: {train_score:.3f}")
                summary.append(f"  Validation Score: {val_score:.3f}")
                summary.append(f"  Ensemble Weight: {weight:.2f}")
                summary.append("")

            summary.append(f"Total Models: {len(self.models)}")
            summary.append(f"Feature Count: {len(self.feature_names)}")
            summary.append("=" * 50)

            return "\n".join(summary)

        except Exception as e:
            self.logger.error(f"Error generating model summary: {e}")
            return f"Error generating summary: {e}"
