"""
Tests for ML modules
"""

import pytest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared.core.ml.price_predictor import PricePredictor
from shared.core.ml.trend_predictor import TrendPredictor
from shared.core.ml.model_trainer import ModelTrainer


def create_sample_data(n_bars=200):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_bars), periods=n_bars, freq='1H')
    
    # Simulate price movement
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.rand(n_bars) * 0.5,
        'high': close_prices + np.random.rand(n_bars) * 1.0,
        'low': close_prices - np.random.rand(n_bars) * 1.0,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    
    return df


class TestPricePredictor:
    """Tests for PricePredictor"""
    
    def test_initialization(self):
        """Test predictor initialization"""
        predictor = PricePredictor(model_type='random_forest')
        assert predictor is not None
        assert predictor.model_type == 'random_forest'
        assert predictor.is_trained == False
    
    def test_train(self):
        """Test model training"""
        predictor = PricePredictor(model_type='random_forest')
        data = create_sample_data(n_bars=200)
        
        metrics = predictor.train(data)
        
        assert predictor.is_trained == True
        assert 'train_score' in metrics
        assert 'test_score' in metrics


class TestTrendPredictor:
    """Tests for TrendPredictor"""
    
    def test_initialization(self):
        """Test predictor initialization"""
        predictor = TrendPredictor()
        assert predictor is not None
        assert predictor.is_trained == False
    
    def test_train(self):
        """Test model training"""
        predictor = TrendPredictor()
        data = create_sample_data(n_bars=200)
        
        metrics = predictor.train(data, horizon=5)
        
        assert predictor.is_trained == True
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics


class TestModelTrainer:
    """Tests for ModelTrainer"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        trainer = ModelTrainer(models_dir='/tmp/test_models')
        assert trainer is not None
        assert str(trainer.models_dir) == '/tmp/test_models'
    
    def test_statistics(self):
        """Test getting statistics"""
        trainer = ModelTrainer()
        stats = trainer.get_statistics()
        
        assert 'models_dir' in stats
        assert 'price_predictor_trained' in stats
        assert 'trend_predictor_trained' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
