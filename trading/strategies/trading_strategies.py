"""
Trading Strategies Module
=========================

Enhanced trading strategies with ML integration:
- Traditional strategies (trend following, mean reversion)
- ML-enhanced strategies
- Multi-asset strategies
"""

import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategySignal:
    """Trading signal"""
    
    def __init__(self, symbol: str, action: str, strength: float, 
                 confidence: float, reason: str):
        self.symbol = symbol
        self.action = action  # 'buy', 'sell', 'hold'
        self.strength = strength  # 0.0 to 1.0
        self.confidence = confidence  # 0.0 to 1.0
        self.reason = reason
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'strength': self.strength,
            'confidence': self.confidence,
            'reason': self.reason,
            'timestamp': self.timestamp
        }


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.signals_generated = 0
        self.last_signal_time: Optional[datetime] = None
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[StrategySignal]:
        """Generate trading signal"""
        pass
    
    def activate(self):
        """Activate strategy"""
        self.is_active = True
        logger.info(f"Strategy '{self.name}' activated")
    
    def deactivate(self):
        """Deactivate strategy"""
        self.is_active = False
        logger.info(f"Strategy '{self.name}' deactivated")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'signals_generated': self.signals_generated,
            'last_signal_time': self.last_signal_time
        }


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy
    
    Uses moving averages and momentum indicators
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("TrendFollowing")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[StrategySignal]:
        """Generate signal based on trend"""
        if not self.is_active:
            return None
        
        if len(data) < self.slow_period:
            logger.warning("Insufficient data for trend following")
            return None
        
        try:
            symbol = kwargs.get('symbol', 'UNKNOWN')
            
            # Calculate moving averages
            fast_ma = data['close'].rolling(window=self.fast_period).mean()
            slow_ma = data['close'].rolling(window=self.slow_period).mean()
            
            # Get latest values
            current_fast = fast_ma.iloc[-1]
            current_slow = slow_ma.iloc[-1]
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]
            
            # Generate signal
            signal = None
            
            if prev_fast <= prev_slow and current_fast > current_slow:
                # Bullish crossover
                strength = min((current_fast - current_slow) / current_slow, 1.0)
                signal = StrategySignal(
                    symbol=symbol,
                    action='buy',
                    strength=abs(strength),
                    confidence=0.7,
                    reason=f"Bullish MA crossover: fast={current_fast:.2f}, slow={current_slow:.2f}"
                )
            elif prev_fast >= prev_slow and current_fast < current_slow:
                # Bearish crossover
                strength = min((current_slow - current_fast) / current_fast, 1.0)
                signal = StrategySignal(
                    symbol=symbol,
                    action='sell',
                    strength=abs(strength),
                    confidence=0.7,
                    reason=f"Bearish MA crossover: fast={current_fast:.2f}, slow={current_slow:.2f}"
                )
            
            if signal:
                self.signals_generated += 1
                self.last_signal_time = datetime.now()
                logger.info(f"Signal generated: {signal.action} {symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate signal: {e}")
            return None


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    
    Trades based on deviation from mean
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__("MeanReversion")
        self.window = window
        self.num_std = num_std
    
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[StrategySignal]:
        """Generate signal based on mean reversion"""
        if not self.is_active:
            return None
        
        if len(data) < self.window:
            logger.warning("Insufficient data for mean reversion")
            return None
        
        try:
            symbol = kwargs.get('symbol', 'UNKNOWN')
            
            # Calculate Bollinger Bands
            sma = data['close'].rolling(window=self.window).mean()
            std = data['close'].rolling(window=self.window).std()
            
            upper_band = sma + (std * self.num_std)
            lower_band = sma - (std * self.num_std)
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_sma = sma.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # Generate signal
            signal = None
            
            if current_price <= current_lower:
                # Oversold - buy signal
                strength = (current_sma - current_price) / current_sma
                signal = StrategySignal(
                    symbol=symbol,
                    action='buy',
                    strength=min(strength, 1.0),
                    confidence=0.65,
                    reason=f"Oversold: price={current_price:.2f}, lower_band={current_lower:.2f}"
                )
            elif current_price >= current_upper:
                # Overbought - sell signal
                strength = (current_price - current_sma) / current_sma
                signal = StrategySignal(
                    symbol=symbol,
                    action='sell',
                    strength=min(strength, 1.0),
                    confidence=0.65,
                    reason=f"Overbought: price={current_price:.2f}, upper_band={current_upper:.2f}"
                )
            
            if signal:
                self.signals_generated += 1
                self.last_signal_time = datetime.now()
                logger.info(f"Signal generated: {signal.action} {symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate signal: {e}")
            return None


class MLEnhancedStrategy(BaseStrategy):
    """
    ML-Enhanced Strategy
    
    Combines traditional indicators with ML predictions
    """
    
    def __init__(self, prediction_engine=None):
        super().__init__("MLEnhanced")
        self.prediction_engine = prediction_engine
    
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[StrategySignal]:
        """Generate signal using ML predictions"""
        if not self.is_active:
            return None
        
        if self.prediction_engine is None:
            logger.warning("No prediction engine configured")
            return None
        
        try:
            symbol = kwargs.get('symbol', 'UNKNOWN')
            
            # Get ML prediction
            # TODO: Integrate with actual prediction engine
            predicted_change = np.random.randn() * 0.02  # Mock prediction
            confidence = 0.75
            
            # Generate signal based on prediction
            signal = None
            
            if predicted_change > 0.01:
                signal = StrategySignal(
                    symbol=symbol,
                    action='buy',
                    strength=min(abs(predicted_change) * 10, 1.0),
                    confidence=confidence,
                    reason=f"ML predicts {predicted_change*100:.2f}% increase"
                )
            elif predicted_change < -0.01:
                signal = StrategySignal(
                    symbol=symbol,
                    action='sell',
                    strength=min(abs(predicted_change) * 10, 1.0),
                    confidence=confidence,
                    reason=f"ML predicts {predicted_change*100:.2f}% decrease"
                )
            
            if signal:
                self.signals_generated += 1
                self.last_signal_time = datetime.now()
                logger.info(f"ML signal generated: {signal.action} {symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate ML signal: {e}")
            return None


class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        logger.info("StrategyManager initialized")
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Strategy '{strategy.name}' registered")
    
    def activate_strategy(self, name: str) -> bool:
        """Activate a strategy"""
        strategy = self.strategies.get(name)
        if strategy:
            strategy.activate()
            return True
        logger.warning(f"Strategy '{name}' not found")
        return False
    
    def deactivate_strategy(self, name: str) -> bool:
        """Deactivate a strategy"""
        strategy = self.strategies.get(name)
        if strategy:
            strategy.deactivate()
            return True
        logger.warning(f"Strategy '{name}' not found")
        return False
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[StrategySignal]:
        """Generate signals from all active strategies"""
        signals = []
        
        for strategy in self.strategies.values():
            if strategy.is_active:
                signal = strategy.generate_signal(data, symbol=symbol)
                if signal:
                    signals.append(signal)
        
        logger.info(f"Generated {len(signals)} signals for {symbol}")
        return signals
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all strategies"""
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': sum(1 for s in self.strategies.values() if s.is_active),
            'strategies': {name: s.get_statistics() for name, s in self.strategies.items()}
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    manager = StrategyManager()
    
    # Register strategies
    trend = TrendFollowingStrategy(fast_period=20, slow_period=50)
    mean_rev = MeanReversionStrategy(window=20, num_std=2.0)
    ml_strat = MLEnhancedStrategy()
    
    manager.register_strategy(trend)
    manager.register_strategy(mean_rev)
    manager.register_strategy(ml_strat)
    
    # Activate strategies
    manager.activate_strategy("TrendFollowing")
    manager.activate_strategy("MeanReversion")
    
    # Mock data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
    })
    
    # Generate signals
    signals = manager.generate_signals(data, symbol="BTCUSDT")
    
    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal.action.upper()} - Strength: {signal.strength:.2f}, "
              f"Confidence: {signal.confidence:.2f}")
        print(f"  Reason: {signal.reason}")
    
    print("\nStatistics:", manager.get_statistics())
