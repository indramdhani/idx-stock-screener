"""
Machine Learning Module for Indonesian Stock Screener

This module provides machine learning enhancements for signal generation and analysis:
- Signal enhancement using ML models
- Feature engineering for technical indicators
- Model training and prediction
- Signal confidence scoring
- Pattern recognition
- Ensemble methods

Author: IDX Stock Screener Team
Version: 1.0.0
"""

from .signal_enhancer import SignalEnhancer, MLSignal
from .feature_engineer import FeatureEngineer, TechnicalFeatures
from .model_trainer import ModelTrainer, ModelConfig
from .pattern_recognition import PatternRecognizer, PatternType
from .ensemble_predictor import EnsemblePredictor, EnsembleConfig

__all__ = [
    'SignalEnhancer',
    'MLSignal',
    'FeatureEngineer',
    'TechnicalFeatures',
    'ModelTrainer',
    'ModelConfig',
    'PatternRecognizer',
    'PatternType',
    'EnsemblePredictor',
    'EnsembleConfig'
]

__version__ = '1.0.0'
