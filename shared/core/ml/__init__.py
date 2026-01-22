"""
Machine Learning Module
Price and trend prediction models using Scikit-Learn
"""

from .price_predictor import PricePredictor
from .trend_predictor import TrendPredictor
from .model_trainer import ModelTrainer

__all__ = [
    'PricePredictor',
    'TrendPredictor',
    'ModelTrainer',
]
