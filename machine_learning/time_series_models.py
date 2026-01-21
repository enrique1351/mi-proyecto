"""
Time Series Models
==================

ML models for time series prediction.
Includes RNN, LSTM, and ARIMA models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path


logger = logging.getLogger(__name__)


class TimeSeriesPredictor:
    """
    Base class for time series prediction models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for training.
        
        Args:
            data: DataFrame with OHLCV data
            sequence_length: Number of time steps to look back
            target_col: Column to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            # Scale data
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data[[target_col]])
            
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return np.array([]), np.array([])
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model to disk."""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load model from disk."""
        raise NotImplementedError


class LSTMPredictor(TimeSeriesPredictor):
    """
    LSTM neural network for time series prediction.
    """
    
    def __init__(self, sequence_length: int = 60, units: int = 50):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length: Number of time steps
            units: Number of LSTM units
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.units = units
        
        try:
            from tensorflow import keras
            self.keras = keras
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        try:
            model = self.keras.Sequential([
                self.keras.layers.LSTM(
                    units=self.units,
                    return_sequences=True,
                    input_shape=input_shape
                ),
                self.keras.layers.Dropout(0.2),
                self.keras.layers.LSTM(units=self.units, return_sequences=True),
                self.keras.layers.Dropout(0.2),
                self.keras.layers.LSTM(units=self.units),
                self.keras.layers.Dropout(0.2),
                self.keras.layers.Dense(units=1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            self.model = model
            logger.info("✅ LSTM model built")
            
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Train LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data split ratio
        """
        try:
            # Reshape X for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Build model if not exists
            if self.model is None:
                self.build_model((X.shape[1], 1))
            
            # Train
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            logger.info("✅ Model training completed")
            return history
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        try:
            # Reshape for LSTM
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Predict
            predictions = self.model.predict(X)
            
            # Inverse scale
            if self.scaler:
                predictions = self.scaler.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            return np.array([])
    
    def save(self, path: str):
        """Save model to disk."""
        try:
            self.model.save(path)
            logger.info(f"✅ Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load(self, path: str):
        """Load model from disk."""
        try:
            self.model = self.keras.models.load_model(path)
            logger.info(f"✅ Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


class ARIMAPredictor(TimeSeriesPredictor):
    """
    ARIMA model for time series prediction.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Initialize ARIMA predictor.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        super().__init__()
        self.order = order
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.ARIMA = ARIMA
        except ImportError:
            logger.error("statsmodels not installed. Install with: pip install statsmodels")
            raise
    
    def train(self, data: pd.Series):
        """
        Train ARIMA model.
        
        Args:
            data: Time series data
        """
        try:
            self.model = self.ARIMA(data, order=self.order)
            self.model_fit = self.model.fit()
            
            logger.info(f"✅ ARIMA model trained with order {self.order}")
            logger.info(f"AIC: {self.model_fit.aic:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to train ARIMA: {e}")
            raise
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions array
        """
        try:
            forecast = self.model_fit.forecast(steps=steps)
            return forecast.values
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            return np.array([])
    
    def save(self, path: str):
        """Save model to disk."""
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model_fit, f)
            logger.info(f"✅ Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load(self, path: str):
        """Load model from disk."""
        try:
            import pickle
            with open(path, 'rb') as f:
                self.model_fit = pickle.load(f)
            logger.info(f"✅ Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
