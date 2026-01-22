"""
Price Predictor
Machine learning model for price prediction using Scikit-Learn
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Price prediction model using ensemble methods
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize price predictor
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn not installed. Install with: pip install scikit-learn joblib"
            )
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"✅ PricePredictor initialized ({model_type})")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (features, targets)
        """
        # Technical indicators as features
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=10).std()
        
        # Price position
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Target: next period's return
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Drop NaN
        df = df.dropna()
        
        # Select features
        feature_columns = [
            'returns', 'log_returns',
            'sma_5', 'sma_10', 'sma_20',
            'volatility', 'high_low_ratio', 'close_open_ratio',
            'volume_ratio'
        ]
        
        X = df[feature_columns].values
        y = df['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the model
        
        Args:
            df: DataFrame with OHLCV data
            test_size: Proportion of data for testing
        
        Returns:
            Dict with training metrics
        """
        try:
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training price prediction model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.is_trained = True
            
            metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Model trained - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict next price change
        
        Args:
            df: DataFrame with recent OHLCV data
        
        Returns:
            Predicted return (as decimal, e.g., 0.02 = 2% increase)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        try:
            # Prepare features from recent data
            X, _ = self.prepare_features(df)
            
            # Use last row for prediction
            X_last = X[-1:, :]
            X_scaled = self.scaler.transform(X_last)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }, filepath)
        
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.is_trained = True
        
        logger.info(f"✅ Model loaded from {filepath}")
