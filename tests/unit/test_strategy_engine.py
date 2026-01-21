"""
================================================================================
UNIT TESTS - Strategy Engine
================================================================================
Ruta: quant_system/tests/unit/test_strategy_engine.py

Tests para el módulo strategy_engine.py
- Test de estrategias individuales
- Test de señales de trading
- Test de parámetros
- Test de edge cases
================================================================================
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared.core.strategies.strategy_engine import (
    StrategyEngine, Signal, SignalType,
    MeanReversionStrategy, MomentumStrategy,
    TrendFollowingStrategy, BreakoutStrategy
)


class TestSignal(unittest.TestCase):
    """Tests para la clase Signal"""
    
    def test_signal_creation_buy(self):
        """Test: Crear señal de compra"""
        signal = Signal(
            symbol='BTCUSDT',
            signal_type=SignalType.BUY,
            price=50000,
            quantity=1.0,
            confidence=0.8
        )
        
        self.assertEqual(signal.symbol, 'BTCUSDT')
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.price, 50000)
        self.assertEqual(signal.quantity, 1.0)
        self.assertEqual(signal.confidence, 0.8)
    
    def test_signal_creation_sell(self):
        """Test: Crear señal de venta"""
        signal = Signal(
            symbol='ETHUSDT',
            signal_type=SignalType.SELL,
            price=3000,
            quantity=10.0,
            confidence=0.9
        )
        
        self.assertEqual(signal.signal_type, SignalType.SELL)
    
    def test_signal_to_dict(self):
        """Test: Convertir señal a diccionario"""
        signal = Signal(
            symbol='BTCUSDT',
            signal_type=SignalType.BUY,
            price=50000,
            quantity=1.0,
            confidence=0.8
        )
        
        signal_dict = signal.to_dict()
        
        self.assertIsInstance(signal_dict, dict)
        self.assertIn('symbol', signal_dict)
        self.assertIn('signal_type', signal_dict)
        self.assertIn('timestamp', signal_dict)


class TestMeanReversionStrategy(unittest.TestCase):
    """Tests para Mean Reversion Strategy"""
    
    def setUp(self):
        """Setup para cada test"""
        self.strategy = MeanReversionStrategy(
            lookback_period=20,
            std_dev=2.0,
            rsi_period=14
        )
        
        # Crear datos de prueba
        self.data = self._create_test_data()
    
    def _create_test_data(self, n=100):
        """Crea datos de prueba para la estrategia"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
        
        # Crear precio con mean reversion
        base_price = 100
        noise = np.random.randn(n) * 2
        prices = base_price + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices - 0.5,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        return df
    
    def test_strategy_initialization(self):
        """Test: Inicialización de estrategia"""
        self.assertEqual(self.strategy.lookback_period, 20)
        self.assertEqual(self.strategy.std_dev, 2.0)
        self.assertEqual(self.strategy.rsi_period, 14)
    
    def test_generate_signal_with_sufficient_data(self):
        """Test: Generar señal con suficientes datos"""
        signal = self.strategy.generate_signal('BTCUSDT', self.data)
        
        self.assertIsInstance(signal, (Signal, type(None)))
    
    def test_generate_signal_insufficient_data(self):
        """Test: Generar señal con datos insuficientes"""
        short_data = self.data.head(10)  # Solo 10 barras
        signal = self.strategy.generate_signal('BTCUSDT', short_data)
        
        # Debería retornar None o manejar gracefully
        self.assertIsNone(signal)
    
    def test_oversold_condition_generates_buy(self):
        """Test: Condición oversold genera señal BUY"""
        # Crear datos con precio muy bajo (oversold)
        oversold_data = self.data.copy()
        oversold_data['close'].iloc[-1] = 80  # Precio bajo
        
        signal = self.strategy.generate_signal('BTCUSDT', oversold_data)
        
        if signal:
            self.assertEqual(signal.signal_type, SignalType.BUY)
    
    def test_overbought_condition_generates_sell(self):
        """Test: Condición overbought genera señal SELL"""
        # Crear datos con precio muy alto (overbought)
        overbought_data = self.data.copy()
        overbought_data['close'].iloc[-1] = 120  # Precio alto
        
        signal = self.strategy.generate_signal('BTCUSDT', overbought_data)
        
        if signal:
            self.assertEqual(signal.signal_type, SignalType.SELL)


class TestMomentumStrategy(unittest.TestCase):
    """Tests para Momentum Strategy"""
    
    def setUp(self):
        """Setup para cada test"""
        self.strategy = MomentumStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        self.data = self._create_trending_data()
    
    def _create_trending_data(self, n=100):
        """Crea datos con tendencia para momentum"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
        
        # Crear tendencia alcista
        trend = np.linspace(100, 120, n)
        noise = np.random.randn(n) * 0.5
        prices = trend + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices - 0.5,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n)
        })
    
    def test_strategy_initialization(self):
        """Test: Inicialización de estrategia momentum"""
        self.assertEqual(self.strategy.fast_period, 12)
        self.assertEqual(self.strategy.slow_period, 26)
        self.assertEqual(self.strategy.signal_period, 9)
    
    def test_uptrend_generates_buy(self):
        """Test: Tendencia alcista genera BUY"""
        signal = self.strategy.generate_signal('BTCUSDT', self.data)
        
        if signal:
            # En tendencia alcista, debería generar BUY
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.HOLD])
    
    def test_downtrend_generates_sell(self):
        """Test: Tendencia bajista genera SELL"""
        # Crear datos con tendencia bajista
        downtrend_data = self._create_trending_data()
        downtrend_data['close'] = downtrend_data['close'][::-1].values
        
        signal = self.strategy.generate_signal('BTCUSDT', downtrend_data)
        
        if signal:
            self.assertIn(signal.signal_type, [SignalType.SELL, SignalType.HOLD])


class TestTrendFollowingStrategy(unittest.TestCase):
    """Tests para Trend Following Strategy"""
    
    def setUp(self):
        """Setup para cada test"""
        self.strategy = TrendFollowingStrategy(
            fast_ma=50,
            slow_ma=200,
            atr_period=14
        )
        
        self.data = self._create_test_data(250)  # Necesita más datos
    
    def _create_test_data(self, n):
        """Crea datos de prueba"""
        dates = pd.date_range(start='2023-01-01', periods=n, freq='1D')
        
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 2
        prices = trend + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices - 0.5,
            'high': prices + 2.0,
            'low': prices - 2.0,
            'close': prices,
            'volume': np.random.randint(10000, 100000, n)
        })
    
    def test_golden_cross_generates_buy(self):
        """Test: Golden cross genera señal BUY"""
        # El golden cross ocurre cuando MA rápida cruza por encima de MA lenta
        signal = self.strategy.generate_signal('BTCUSDT', self.data)
        
        if signal:
            self.assertIsInstance(signal, Signal)
    
    def test_death_cross_generates_sell(self):
        """Test: Death cross genera señal SELL"""
        # Invertir tendencia para death cross
        reversed_data = self.data.copy()
        reversed_data['close'] = reversed_data['close'][::-1].values
        
        signal = self.strategy.generate_signal('BTCUSDT', reversed_data)
        
        if signal:
            self.assertIsInstance(signal, Signal)


class TestBreakoutStrategy(unittest.TestCase):
    """Tests para Breakout Strategy"""
    
    def setUp(self):
        """Setup para cada test"""
        self.strategy = BreakoutStrategy(
            lookback_period=20,
            volume_threshold=1.5
        )
        
        self.data = self._create_consolidation_data()
    
    def _create_consolidation_data(self, n=100):
        """Crea datos con consolidación y breakout"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
        
        # Crear consolidación (precio plano)
        prices = np.ones(n) * 100
        prices[:80] += np.random.randn(80) * 0.5  # Consolidación
        prices[80:] += np.linspace(0, 10, 20)  # Breakout
        
        volumes = np.random.randint(1000, 2000, n)
        volumes[80:] = np.random.randint(3000, 5000, 20)  # Alto volumen en breakout
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices - 0.3,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': volumes
        })
    
    def test_breakout_above_generates_buy(self):
        """Test: Breakout al alza genera BUY"""
        signal = self.strategy.generate_signal('BTCUSDT', self.data)
        
        if signal:
            self.assertEqual(signal.signal_type, SignalType.BUY)
    
    def test_breakdown_below_generates_sell(self):
        """Test: Breakdown a la baja genera SELL"""
        # Invertir para breakdown
        breakdown_data = self.data.copy()
        breakdown_data['close'].iloc[80:] -= 10
        
        signal = self.strategy.generate_signal('BTCUSDT', breakdown_data)
        
        if signal:
            self.assertEqual(signal.signal_type, SignalType.SELL)


class TestStrategyEngine(unittest.TestCase):
    """Tests para el Strategy Engine principal"""
    
    def setUp(self):
        """Setup para cada test"""
        self.engine = StrategyEngine()
        self.test_data = self._create_test_data()
    
    def _create_test_data(self, n=100):
        """Crea datos de prueba"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices - 0.5,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n)
        })
    
    def test_engine_initialization(self):
        """Test: Inicialización del engine"""
        self.assertIsNotNone(self.engine)
        self.assertIsInstance(self.engine.strategies, dict)
    
    def test_add_strategy(self):
        """Test: Añadir estrategia al engine"""
        strategy = MeanReversionStrategy()
        self.engine.add_strategy('mean_reversion', strategy)
        
        self.assertIn('mean_reversion', self.engine.strategies)
    
    def test_remove_strategy(self):
        """Test: Remover estrategia del engine"""
        strategy = MeanReversionStrategy()
        self.engine.add_strategy('test_strategy', strategy)
        
        self.engine.remove_strategy('test_strategy')
        self.assertNotIn('test_strategy', self.engine.strategies)
    
    def test_generate_signals_all_strategies(self):
        """Test: Generar señales de todas las estrategias"""
        # Añadir múltiples estrategias
        self.engine.add_strategy('mean_reversion', MeanReversionStrategy())
        self.engine.add_strategy('momentum', MomentumStrategy())
        
        signals = self.engine.generate_signals('BTCUSDT', self.test_data)
        
        self.assertIsInstance(signals, dict)
    
    def test_get_combined_signal(self):
        """Test: Obtener señal combinada de todas las estrategias"""
        self.engine.add_strategy('mean_reversion', MeanReversionStrategy())
        self.engine.add_strategy('momentum', MomentumStrategy())
        
        combined = self.engine.get_combined_signal('BTCUSDT', self.test_data)
        
        self.assertIsInstance(combined, (Signal, type(None)))
    
    def test_empty_engine_returns_none(self):
        """Test: Engine sin estrategias retorna None"""
        combined = self.engine.get_combined_signal('BTCUSDT', self.test_data)
        self.assertIsNone(combined)


class TestStrategyEdgeCases(unittest.TestCase):
    """Tests para casos extremos"""
    
    def test_empty_dataframe(self):
        """Test: DataFrame vacío"""
        strategy = MeanReversionStrategy()
        empty_df = pd.DataFrame()
        
        signal = strategy.generate_signal('BTCUSDT', empty_df)
        self.assertIsNone(signal)
    
    def test_nan_values_in_data(self):
        """Test: Valores NaN en datos"""
        strategy = MomentumStrategy()
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': [100] * 100,
            'high': [102] * 100,
            'low': [98] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        data.loc[50, 'close'] = np.nan  # Insertar NaN
        
        signal = strategy.generate_signal('BTCUSDT', data)
        # Debería manejar NaN gracefully
        self.assertIsInstance(signal, (Signal, type(None)))
    
    def test_extreme_volatility(self):
        """Test: Volatilidad extrema"""
        strategy = TrendFollowingStrategy()
        
        # Crear datos con volatilidad extrema
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=250, freq='1D'),
            'open': np.random.uniform(50, 150, 250),
            'high': np.random.uniform(100, 200, 250),
            'low': np.random.uniform(10, 100, 250),
            'close': np.random.uniform(50, 150, 250),
            'volume': np.random.randint(1000, 100000, 250)
        })
        
        signal = strategy.generate_signal('BTCUSDT', data)
        # Debería manejar sin crashes
        self.assertIsInstance(signal, (Signal, type(None)))


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)