"""
Model Training Module
=====================

Handles training of ML models:
- Data preparation
- Training pipeline
- Hyperparameter tuning
- Model evaluation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepares data for ML model training"""
    
    def __init__(self):
        self.scaler = None
        logger.info("DataPreparator initialized")
    
    def prepare_timeseries_data(self, df: pd.DataFrame, 
                                sequence_length: int = 60,
                                target_col: str = 'close',
                                feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for LSTM/RNN
        
        Args:
            df: DataFrame with OHLCV data
            sequence_length: Lookback period
            target_col: Column to predict
            feature_cols: Columns to use as features
            
        Returns:
            Tuple of (X, y) arrays
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return np.array([]), np.array([])
        
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        try:
            # Extract features
            features = df[feature_cols].values
            target = df[target_col].values
            
            # Create sequences
            X, y = [], []
            for i in range(len(features) - sequence_length):
                X.append(features[i:i+sequence_length])
                y.append(target[i+sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} sequences (shape: {X.shape})")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare timeseries data: {e}")
            return np.array([]), np.array([])
    
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize data using min-max scaling
        
        Args:
            data: Input data
            fit: Whether to fit the scaler (True) or use existing (False)
            
        Returns:
            Normalized data
            
        Note:
            This is a simple implementation. For production use, consider
            using sklearn.preprocessing.StandardScaler or MinMaxScaler
            with proper fit/transform pattern.
        """
        try:
            if fit or self.scaler is None:
                # Fit the scaler
                data_min = data.min(axis=0)
                data_max = data.max(axis=0)
                self.scaler = {'min': data_min, 'max': data_max}
            
            # Transform using saved scaler
            normalized = (data - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'] + 1e-8)
            
            logger.info("Data normalized")
            return normalized
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return data
    
    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse normalize data back to original scale
        
        Args:
            data: Normalized data
            
        Returns:
            Data in original scale
        """
        if self.scaler is None:
            logger.warning("No scaler fitted - returning data as is")
            return data
        
        try:
            denormalized = data * (self.scaler['max'] - self.scaler['min'] + 1e-8) + self.scaler['min']
            logger.info("Data denormalized")
            return denormalized
        except Exception as e:
            logger.error(f"Denormalization failed: {e}")
            return data
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple:
        """
        Split data into train/val/test sets
        
        Args:
            X: Features
            y: Targets
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        logger.info(f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelTrainer:
    """Manages model training process"""
    
    def __init__(self):
        self.training_history: List[Dict[str, Any]] = []
        self.best_model = None
        self.best_score = float('inf')
        logger.info("ModelTrainer initialized")
    
    def train_model(self, model, X_train, y_train,
                    X_val=None, y_val=None,
                    **training_kwargs) -> Dict[str, Any]:
        """
        Train a model
        
        Args:
            model: ML model instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **training_kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        try:
            logger.info(f"Starting training for {model.name}")
            start_time = datetime.now()
            
            # Train the model
            history = model.train(X_train, y_train, **training_kwargs)
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_predictions = model.predict(X_val)
                val_loss = self._calculate_loss(y_val, val_predictions)
                history['val_loss'] = val_loss
                
                # Update best model
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_model = model
                    logger.info(f"New best model with val_loss: {val_loss:.4f}")
            
            training_time = (datetime.now() - start_time).total_seconds()
            history['training_time'] = training_time
            history['timestamp'] = datetime.now()
            
            self.training_history.append(history)
            
            logger.info(f"Training completed in {training_time:.2f}s")
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}
    
    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss (MSE)"""
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            return float(mse)
        except:
            return float('inf')
    
    def hyperparameter_search(self, model_class, X_train, y_train,
                             param_grid: Dict[str, List],
                             X_val=None, y_val=None) -> Dict[str, Any]:
        """
        Perform hyperparameter search
        
        Args:
            model_class: Model class to instantiate
            X_train: Training features
            y_train: Training targets
            param_grid: Grid of parameters to search
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting hyperparameter search with {len(param_grid)} parameters")
        
        best_params = {}
        best_score = float('inf')
        results = []
        
        # TODO: Implement grid search or random search
        # For now, just log the intent
        
        logger.info(f"Hyperparameter search completed. Best score: {best_score}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training runs"""
        return {
            'total_runs': len(self.training_history),
            'best_score': self.best_score,
            'best_model': self.best_model.name if self.best_model else None,
            'history': self.training_history
        }


class ModelEvaluator:
    """Evaluates trained models"""
    
    def __init__(self):
        logger.info("ModelEvaluator initialized")
    
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        try:
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - predictions))
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'n_samples': len(y_test)
            }
            
            logger.info(f"Model evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def calculate_prediction_accuracy(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     threshold: float = 0.05) -> float:
        """
        Calculate directional prediction accuracy
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Threshold for considering direction correct
            
        Returns:
            Accuracy ratio
        """
        try:
            # Direction accuracy (up/down prediction)
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            
            accuracy = np.mean(true_direction == pred_direction)
            
            logger.info(f"Directional accuracy: {accuracy:.4f}")
            return float(accuracy)
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("=== Data Preparation ===")
    preparator = DataPreparator()
    
    # Mock OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    X, y = preparator.prepare_timeseries_data(df, sequence_length=60)
    print(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_data(X, y)
    
    print("\n=== Model Training ===")
    trainer = ModelTrainer()
    
    # Mock model
    class MockModel:
        def __init__(self):
            self.name = "MockModel"
            self.is_trained = False
        
        def train(self, X, y, epochs=10):
            self.is_trained = True
            return {'loss': [0.1] * epochs}
        
        def predict(self, X):
            return np.random.randn(len(X))
    
    model = MockModel()
    history = trainer.train_model(model, X_train, y_train, X_val, y_val, epochs=10)
    print(f"Training completed in {history['training_time']:.2f}s")
    
    print("\n=== Model Evaluation ===")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)
    print(f"Test metrics: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
