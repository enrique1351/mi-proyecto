"""
Model Trainer
Unified interface for training and managing ML models
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd

from .price_predictor import PricePredictor
from .trend_predictor import TrendPredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified model training and management system
    """
    
    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize model trainer
        
        Args:
            models_dir: Directory to save/load models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.price_predictor: Optional[PricePredictor] = None
        self.trend_predictor: Optional[TrendPredictor] = None
        
        # Training history
        self.training_history = []
        
        logger.info(f"✅ ModelTrainer initialized (models_dir: {models_dir})")
    
    def train_price_predictor(
        self,
        data: pd.DataFrame,
        model_type: str = "random_forest",
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Train price prediction model
        
        Args:
            data: DataFrame with OHLCV data
            model_type: 'random_forest' or 'gradient_boosting'
            save: Save model to disk after training
        
        Returns:
            Training metrics
        """
        try:
            logger.info(f"Training price predictor ({model_type})...")
            
            # Initialize model
            self.price_predictor = PricePredictor(model_type=model_type)
            
            # Train
            metrics = self.price_predictor.train(data)
            
            # Save if requested
            if save:
                model_path = self.models_dir / f"price_predictor_{model_type}.joblib"
                self.price_predictor.save_model(str(model_path))
                metrics['model_path'] = str(model_path)
            
            # Record in history
            self.training_history.append({
                'model': 'price_predictor',
                'model_type': model_type,
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training price predictor: {e}")
            raise
    
    def train_trend_predictor(
        self,
        data: pd.DataFrame,
        horizon: int = 5,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Train trend prediction model
        
        Args:
            data: DataFrame with OHLCV data
            horizon: Prediction horizon (periods ahead)
            save: Save model to disk after training
        
        Returns:
            Training metrics
        """
        try:
            logger.info(f"Training trend predictor (horizon={horizon})...")
            
            # Initialize model
            self.trend_predictor = TrendPredictor()
            
            # Train
            metrics = self.trend_predictor.train(data, horizon=horizon)
            
            # Save if requested
            if save:
                model_path = self.models_dir / f"trend_predictor_h{horizon}.joblib"
                self.trend_predictor.save_model(str(model_path))
                metrics['model_path'] = str(model_path)
            
            # Record in history
            self.training_history.append({
                'model': 'trend_predictor',
                'horizon': horizon,
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training trend predictor: {e}")
            raise
    
    def train_all_models(
        self,
        data: pd.DataFrame,
        price_model_type: str = "random_forest",
        trend_horizon: int = 5
    ) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            data: DataFrame with OHLCV data
            price_model_type: Type for price predictor
            trend_horizon: Horizon for trend predictor
        
        Returns:
            Combined training metrics
        """
        results = {}
        
        # Train price predictor
        try:
            results['price_predictor'] = self.train_price_predictor(
                data, model_type=price_model_type
            )
        except Exception as e:
            logger.error(f"Failed to train price predictor: {e}")
            results['price_predictor'] = {'error': str(e)}
        
        # Train trend predictor
        try:
            results['trend_predictor'] = self.train_trend_predictor(
                data, horizon=trend_horizon
            )
        except Exception as e:
            logger.error(f"Failed to train trend predictor: {e}")
            results['trend_predictor'] = {'error': str(e)}
        
        return results
    
    def load_price_predictor(
        self,
        model_type: str = "random_forest"
    ) -> bool:
        """Load price predictor from disk"""
        try:
            model_path = self.models_dir / f"price_predictor_{model_type}.joblib"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return False
            
            self.price_predictor = PricePredictor(model_type=model_type)
            self.price_predictor.load_model(str(model_path))
            
            logger.info(f"✅ Loaded price predictor from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading price predictor: {e}")
            return False
    
    def load_trend_predictor(self, horizon: int = 5) -> bool:
        """Load trend predictor from disk"""
        try:
            model_path = self.models_dir / f"trend_predictor_h{horizon}.joblib"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return False
            
            self.trend_predictor = TrendPredictor()
            self.trend_predictor.load_model(str(model_path))
            
            logger.info(f"✅ Loaded trend predictor from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading trend predictor: {e}")
            return False
    
    def get_price_prediction(self, data: pd.DataFrame) -> Optional[float]:
        """Get price prediction"""
        if not self.price_predictor or not self.price_predictor.is_trained:
            logger.warning("Price predictor not trained")
            return None
        
        try:
            return self.price_predictor.predict(data)
        except Exception as e:
            logger.error(f"Error getting price prediction: {e}")
            return None
    
    def get_trend_prediction(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> Optional[tuple]:
        """Get trend prediction"""
        if not self.trend_predictor or not self.trend_predictor.is_trained:
            logger.warning("Trend predictor not trained")
            return None
        
        try:
            return self.trend_predictor.predict(data, horizon)
        except Exception as e:
            logger.error(f"Error getting trend prediction: {e}")
            return None
    
    def get_training_history(self) -> list:
        """Get training history"""
        return self.training_history
    
    def get_statistics(self) -> dict:
        """Get model statistics"""
        return {
            'models_dir': str(self.models_dir),
            'price_predictor_trained': (
                self.price_predictor.is_trained if self.price_predictor else False
            ),
            'trend_predictor_trained': (
                self.trend_predictor.is_trained if self.trend_predictor else False
            ),
            'training_history_count': len(self.training_history)
        }
