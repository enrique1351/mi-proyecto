"""
Trend Predictor
Binary classification model for trend direction prediction
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrendPredictor:
    """
    Trend prediction model using classification
    Predicts: UP (1) or DOWN (0)
    """
    
    def __init__(self):
        """Initialize trend predictor"""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn not installed. Install with: pip install scikit-learn joblib"
            )
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info("✅ TrendPredictor initialized")
    
    def prepare_features(self, df: pd.DataFrame, horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Periods ahead to predict trend
        
        Returns:
            Tuple of (features, labels)
        """
        df = df.copy()
        
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['volatility'] = df['returns'].rolling(window=10).std()
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Price position relative to MA
        df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        
        # Target: 1 if price goes up in next 'horizon' periods, 0 otherwise
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Drop NaN
        df = df.dropna()
        
        # Select features
        feature_columns = [
            'returns', 'sma_5', 'sma_10', 'sma_20',
            'rsi', 'volatility', 'volume_ratio',
            'momentum_5', 'momentum_10',
            'price_vs_sma5', 'price_vs_sma20'
        ]
        
        X = df[feature_columns].values
        y = df['target'].values
        
        return X, y
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, df: pd.DataFrame, horizon: int = 5, test_size: float = 0.2) -> dict:
        """
        Train the model
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Periods ahead to predict
            test_size: Proportion of data for testing
        
        Returns:
            Dict with training metrics
        """
        try:
            # Prepare features
            X, y = self.prepare_features(df, horizon)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training trend prediction model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            
            self.is_trained = True
            
            metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Model trained - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, df: pd.DataFrame, horizon: int = 5) -> Tuple[int, float]:
        """
        Predict trend direction
        
        Args:
            df: DataFrame with recent OHLCV data
            horizon: Prediction horizon
        
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 for UP, 0 for DOWN
            confidence: probability of predicted class
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        try:
            # Prepare features
            X, _ = self.prepare_features(df, horizon)
            
            # Use last row
            X_last = X[-1:, :]
            X_scaled = self.scaler.transform(X_last)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = probabilities[prediction]
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0, 0.5
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
        
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        
        logger.info(f"✅ Model loaded from {filepath}")
