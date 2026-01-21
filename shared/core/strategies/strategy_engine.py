# shared/core/strategies/strategy_engine.py

import logging
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(Enum):
    """Tipos de señales."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class MarketRegime(Enum):
    """Regímenes de mercado."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# ============================================================================
# BASE STRATEGY CLASS (Abstract)
# ============================================================================

class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias.
    Todas las estrategias deben heredar de esta clase.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Inicializa la estrategia.
        
        Args:
            name: Nombre de la estrategia
            params: Parámetros personalizados
        """
        self.name = name
        self.params = params or {}
        self.enabled = True
        self.performance_score = 0.0
        
        # Metadata
        self.description = ""
        self.asset_classes = []  # Clases de activos soportadas
        self.timeframes = []     # Timeframes recomendados
        self.min_data_points = 100  # Mínimo de datos necesarios
        
        # Estadísticas
        self.signals_generated = 0
        self.successful_signals = 0
        self.failed_signals = 0
    
    @abstractmethod
    def generate_signal(
        self,
        asset: str,
        data: pd.DataFrame,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        """
        Genera una señal de trading.
        
        Args:
            asset: Símbolo del asset
            data: DataFrame con datos OHLCV + indicadores
            regime: Régimen de mercado actual
        
        Returns:
            Dict con: {
                'signal': SignalType,
                'confidence': float (0-1),
                'quantity': float,
                'stop_loss': float (opcional),
                'take_profit': float (opcional),
                'metadata': dict
            }
        """
        pass
    
    def should_trade(self, regime: Optional[MarketRegime] = None) -> bool:
        """
        Determina si la estrategia debe operar en el régimen actual.
        
        Args:
            regime: Régimen de mercado actual
        
        Returns:
            True si debe operar, False si no
        """
        return True
    
    def calculate_position_size(
        self,
        capital: float,
        confidence: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calcula el tamaño de posición basado en confianza y riesgo.
        
        Args:
            capital: Capital disponible
            confidence: Confianza de la señal (0-1)
            risk_per_trade: Porcentaje de riesgo por trade
        
        Returns:
            Tamaño de posición en USD
        """
        base_size = capital * risk_per_trade
        adjusted_size = base_size * confidence
        return adjusted_size
    
    def update_performance(self, success: bool):
        """Actualiza métricas de performance."""
        self.signals_generated += 1
        if success:
            self.successful_signals += 1
        else:
            self.failed_signals += 1
        
        # Calcular score
        if self.signals_generated > 0:
            self.performance_score = self.successful_signals / self.signals_generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de la estrategia."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'signals_generated': self.signals_generated,
            'successful_signals': self.successful_signals,
            'failed_signals': self.failed_signals,
            'performance_score': self.performance_score,
            'win_rate': self.performance_score * 100
        }


# ============================================================================
# ESTRATEGIAS CONCRETAS
# ============================================================================

class TrendFollowingStrategy(BaseStrategy):
    """Estrategia de seguimiento de tendencia usando EMAs."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("TrendFollowing", params)
        self.description = "Sigue tendencias usando EMAs cruzadas"
        self.asset_classes = ["crypto", "stocks", "forex"]
        self.timeframes = ["1h", "4h", "1d"]
        
        # Parámetros
        self.fast_period = self.params.get('fast_period', 12)
        self.slow_period = self.params.get('slow_period', 26)
        self.signal_period = self.params.get('signal_period', 9)
    
    def generate_signal(
        self,
        asset: str,
        data: pd.DataFrame,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        
        if len(data) < self.slow_period:
            return self._no_signal()
        
        # Calcular EMAs
        data['ema_fast'] = data['close'].ewm(span=self.fast_period).mean()
        data['ema_slow'] = data['close'].ewm(span=self.slow_period).mean()
        
        # Última fila
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Detectar cruce
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Golden cross (EMA rápida cruza hacia arriba)
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            signal = SignalType.BUY
            confidence = 0.8
        
        # Death cross (EMA rápida cruza hacia abajo)
        elif prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            signal = SignalType.SELL
            confidence = 0.8
        
        # Si ya estamos en tendencia
        elif last['ema_fast'] > last['ema_slow']:
            # Tendencia alcista pero sin cruce
            distance = (last['ema_fast'] - last['ema_slow']) / last['ema_slow']
            if distance > 0.02:  # 2% de distancia
                signal = SignalType.BUY
                confidence = min(0.6, distance * 10)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': last['close'] * 0.97 if signal == SignalType.BUY else None,
            'take_profit': last['close'] * 1.06 if signal == SignalType.BUY else None,
            'metadata': {
                'ema_fast': last['ema_fast'],
                'ema_slow': last['ema_slow'],
                'price': last['close']
            }
        }
    
    def should_trade(self, regime: Optional[MarketRegime] = None) -> bool:
        """Solo operar en tendencias claras."""
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return True
        return False
    
    def _no_signal(self) -> Dict[str, Any]:
        """Retorna señal vacía."""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'metadata': {}
        }


class MeanReversionStrategy(BaseStrategy):
    """Estrategia de reversión a la media usando Bollinger Bands."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("MeanReversion", params)
        self.description = "Compra en sobreventa y vende en sobrecompra"
        self.asset_classes = ["crypto", "stocks", "forex"]
        self.timeframes = ["5m", "15m", "1h"]
        
        # Parámetros
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.rsi_period = self.params.get('rsi_period', 14)
    
    def generate_signal(
        self,
        asset: str,
        data: pd.DataFrame,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        
        if len(data) < self.bb_period:
            return self._no_signal()
        
        # Calcular Bollinger Bands
        data['sma'] = data['close'].rolling(window=self.bb_period).mean()
        data['std'] = data['close'].rolling(window=self.bb_period).std()
        data['bb_upper'] = data['sma'] + (data['std'] * self.bb_std)
        data['bb_lower'] = data['sma'] - (data['std'] * self.bb_std)
        
        # Calcular RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        last = data.iloc[-1]
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Comprar si está en banda inferior y RSI < 30 (sobreventa)
        if last['close'] <= last['bb_lower'] and last['rsi'] < 30:
            signal = SignalType.BUY
            confidence = 0.85
        
        # Vender si está en banda superior y RSI > 70 (sobrecompra)
        elif last['close'] >= last['bb_upper'] and last['rsi'] > 70:
            signal = SignalType.SELL
            confidence = 0.85
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': last['bb_lower'] * 0.98 if signal == SignalType.BUY else None,
            'take_profit': last['sma'] if signal == SignalType.BUY else None,
            'metadata': {
                'rsi': last['rsi'],
                'bb_upper': last['bb_upper'],
                'bb_lower': last['bb_lower'],
                'price': last['close']
            }
        }
    
    def should_trade(self, regime: Optional[MarketRegime] = None) -> bool:
        """Solo operar en mercados ranging."""
        if regime == MarketRegime.RANGING:
            return True
        return False
    
    def _no_signal(self) -> Dict[str, Any]:
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'metadata': {}
        }


class BreakoutStrategy(BaseStrategy):
    """Estrategia de ruptura de niveles clave."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("Breakout", params)
        self.description = "Opera rupturas de resistencia/soporte"
        self.asset_classes = ["crypto", "stocks"]
        self.timeframes = ["15m", "1h", "4h"]
        
        self.lookback = self.params.get('lookback', 50)
        self.volume_multiplier = self.params.get('volume_multiplier', 1.5)
    
    def generate_signal(
        self,
        asset: str,
        data: pd.DataFrame,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        
        if len(data) < self.lookback:
            return self._no_signal()
        
        # Calcular niveles
        data['resistance'] = data['high'].rolling(window=self.lookback).max()
        data['support'] = data['low'].rolling(window=self.lookback).min()
        data['avg_volume'] = data['volume'].rolling(window=self.lookback).mean()
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Ruptura alcista
        if last['close'] > prev['resistance'] and last['volume'] > last['avg_volume'] * self.volume_multiplier:
            signal = SignalType.BUY
            confidence = 0.9
        
        # Ruptura bajista
        elif last['close'] < prev['support'] and last['volume'] > last['avg_volume'] * self.volume_multiplier:
            signal = SignalType.SELL
            confidence = 0.9
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': prev['support'] if signal == SignalType.BUY else None,
            'take_profit': last['close'] * 1.1 if signal == SignalType.BUY else None,
            'metadata': {
                'resistance': last['resistance'],
                'support': last['support'],
                'volume': last['volume'],
                'avg_volume': last['avg_volume']
            }
        }
    
    def _no_signal(self) -> Dict[str, Any]:
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'metadata': {}
        }


class MomentumStrategy(BaseStrategy):
    """Estrategia de momentum usando MACD y RSI."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("Momentum", params)
        self.description = "Opera momentum fuerte con confirmación"
        self.asset_classes = ["crypto", "stocks"]
        self.timeframes = ["5m", "15m", "1h"]
        
        self.fast_period = self.params.get('fast_period', 12)
        self.slow_period = self.params.get('slow_period', 26)
        self.signal_period = self.params.get('signal_period', 9)
        self.rsi_period = self.params.get('rsi_period', 14)
    
    def generate_signal(
        self,
        asset: str,
        data: pd.DataFrame,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        
        if len(data) < self.slow_period:
            return self._no_signal()
        
        # MACD
        ema_fast = data['close'].ewm(span=self.fast_period).mean()
        ema_slow = data['close'].ewm(span=self.slow_period).mean()
        data['macd'] = ema_fast - ema_slow
        data['signal_line'] = data['macd'].ewm(span=self.signal_period).mean()
        data['macd_hist'] = data['macd'] - data['signal_line']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Momentum alcista: MACD cruza hacia arriba y RSI > 50
        if (prev['macd'] <= prev['signal_line'] and 
            last['macd'] > last['signal_line'] and 
            last['rsi'] > 50):
            signal = SignalType.BUY
            confidence = 0.8
        
        # Momentum bajista: MACD cruza hacia abajo y RSI < 50
        elif (prev['macd'] >= prev['signal_line'] and 
              last['macd'] < last['signal_line'] and 
              last['rsi'] < 50):
            signal = SignalType.SELL
            confidence = 0.8
        
        return {
            'signal': signal,
            'confidence': confidence,
            'metadata': {
                'macd': last['macd'],
                'signal_line': last['signal_line'],
                'rsi': last['rsi']
            }
        }
    
    def _no_signal(self) -> Dict[str, Any]:
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'metadata': {}
        }


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

class StrategyRegistry:
    """Registro centralizado de estrategias."""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
    
    def register(self, strategy: BaseStrategy):
        """Registra una estrategia."""
        self.strategies[strategy.name] = strategy
        logger.info(f"Estrategia '{strategy.name}' registrada")
    
    def unregister(self, name: str):
        """Desregistra una estrategia."""
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Estrategia '{name}' desregistrada")
    
    def get(self, name: str) -> Optional[BaseStrategy]:
        """Obtiene una estrategia por nombre."""
        return self.strategies.get(name)
    
    def get_all(self) -> List[BaseStrategy]:
        """Obtiene todas las estrategias registradas."""
        return list(self.strategies.values())
    
    def get_enabled(self) -> List[BaseStrategy]:
        """Obtiene solo estrategias habilitadas."""
        return [s for s in self.strategies.values() if s.enabled]
    
    def get_by_asset_class(self, asset_class: str) -> List[BaseStrategy]:
        """Obtiene estrategias para una clase de activo."""
        return [
            s for s in self.strategies.values() 
            if asset_class in s.asset_classes and s.enabled
        ]


# ============================================================================
# STRATEGY ENGINE (Mejorado)
# ============================================================================

class StrategyEngine:
    """
    Motor de estrategias mejorado que:
    - Soporta múltiples estrategias simultáneas
    - Scoring y ranking de señales
    - Adaptación a régimen de mercado
    - Gestión dinámica de estrategias
    """
    
    def __init__(
        self,
        assets: List[str],
        timeframes: List[str],
        data_manager: Any,
        regime_detector: Optional[Any] = None
    ):
        """
        Inicializa el motor de estrategias.
        
        Args:
            assets: Lista de activos a operar
            timeframes: Timeframes a analizar
            data_manager: Gestor de datos
            regime_detector: Detector de régimen de mercado
        """
        self.assets = assets
        self.timeframes = timeframes
        self.data_manager = data_manager
        self.regime_detector = regime_detector
        
        # Registry de estrategias
        self.registry = StrategyRegistry()
        
        # Estado
        self.current_regime: Optional[MarketRegime] = None
        self.signals_cache: Dict[str, Dict] = {}
        
        # Inicializar estrategias por defecto
        self._initialize_default_strategies()
        
        logger.info(f"StrategyEngine inicializado con {len(self.assets)} assets")
    
    def _initialize_default_strategies(self):
        """Registra estrategias por defecto."""
        self.registry.register(TrendFollowingStrategy())
        self.registry.register(MeanReversionStrategy())
        self.registry.register(BreakoutStrategy())
        self.registry.register(MomentumStrategy())
    
    def add_strategy(self, strategy: BaseStrategy):
        """Agrega una estrategia personalizada."""
        self.registry.register(strategy)
    
    def remove_strategy(self, name: str):
        """Elimina una estrategia."""
        self.registry.unregister(name)
    
    def set_regime(self, regime: MarketRegime):
        """Actualiza el régimen de mercado actual."""
        self.current_regime = regime
        logger.info(f"Régimen actualizado a: {regime.value}")
    
    def run_cycle(self) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta un ciclo completo de generación de señales.
        
        Returns:
            Dict con formato: {
                'BTCUSDT': {
                    'best_signal': SignalType,
                    'confidence': float,
                    'strategy': str,
                    'all_signals': List[Dict]
                }
            }
        """
        all_signals = {}
        
        for asset in self.assets:
            asset_signals = self._generate_signals_for_asset(asset)
            
            if asset_signals:
                # Obtener la mejor señal
                best_signal = self._select_best_signal(asset_signals)
                
                all_signals[asset] = {
                    'best_signal': best_signal['signal'],
                    'confidence': best_signal['confidence'],
                    'strategy': best_signal['strategy_name'],
                    'stop_loss': best_signal.get('stop_loss'),
                    'take_profit': best_signal.get('take_profit'),
                    'all_signals': asset_signals,
                    'metadata': best_signal.get('metadata', {})
                }
        
        self.signals_cache = all_signals
        return all_signals
    
    def _generate_signals_for_asset(self, asset: str) -> List[Dict[str, Any]]:
        """Genera señales de todas las estrategias para un asset."""
        signals = []
        
        # Obtener estrategias habilitadas
        strategies = self.registry.get_enabled()
        
        for strategy in strategies:
            # Verificar si la estrategia debe operar en este régimen
            if not strategy.should_trade(self.current_regime):
                continue
            
            # Obtener datos para el asset
            data = self._get_market_data(asset, strategy.timeframes[0] if strategy.timeframes else "1h")
            
            if data is None or len(data) < strategy.min_data_points:
                continue
            
            try:
                # Generar señal
                signal_data = strategy.generate_signal(asset, data, self.current_regime)
                
                if signal_data['signal'] != SignalType.HOLD:
                    signal_data['strategy_name'] = strategy.name
                    signal_data['asset'] = asset
                    signals.append(signal_data)
                    
                    logger.debug(
                        f"Señal generada: {strategy.name} -> {asset} "
                        f"{signal_data['signal'].value} (conf: {signal_data['confidence']:.2f})"
                    )
            
            except Exception as e:
                logger.error(f"Error generando señal con {strategy.name} para {asset}: {e}")
        
        return signals
    
    def _select_best_signal(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selecciona la mejor señal basada en confianza y performance histórico."""
        if not signals:
            return {'signal': SignalType.HOLD, 'confidence': 0.0, 'strategy_name': 'None'}
        
        # Ordenar por confianza * performance_score de la estrategia
        scored_signals = []
        
        for signal in signals:
            strategy = self.registry.get(signal['strategy_name'])
            if strategy:
                score = signal['confidence'] * (0.5 + strategy.performance_score * 0.5)
                scored_signals.append((score, signal))
        
        # Retornar la de mayor score
        if scored_signals:
            scored_signals.sort(key=lambda x: x[0], reverse=True)
            return scored_signals[0][1]
        
        return signals[0]
    
    def _get_market_data(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos de mercado del data manager."""
        try:
            # Aquí deberías llamar a tu data_manager real
            # Por ahora retornamos datos mock
            return self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error obteniendo datos para {asset}: {e}")
            return None
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Genera datos mock para testing."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1h')
        
        # Generar precios simulados
        np.random.seed(42)
        price = 50000
        prices = [price]
        
        for _ in range(199):
            change = np.random.randn() * 0.02  # 2% std
            price = price * (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 200)
        })
        
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del engine."""
        strategies = self.registry.get_all()
        
        return {
            'total_strategies': len(strategies),
            'enabled_strategies': len([s for s in strategies if s.enabled]),
            'current_regime': self.current_regime.value if self.current_regime else None,
            'strategies_stats': [s.get_stats() for s in strategies]
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock data manager
    class MockDataManager:
        pass
    
    # Crear engine
    engine = StrategyEngine(
        assets=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframes=["1h", "4h"],
        data_manager=MockDataManager()
    )
    
    # Establecer régimen
    engine.set_regime(MarketRegime.TRENDING_UP)
    
    # Ejecutar ciclo
    signals = engine.run_cycle()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("SEÑALES GENERADAS:")
    print("="*60)
    
    for asset, signal_data in signals.items():
        print(f"\n{asset}:")
        print(f"  Señal: {signal_data['best_signal'].value}")
        print(f"  Confianza: {signal_data['confidence']:.2%}")
        print(f"  Estrategia: {signal_data['strategy']}")
        print(f"  Total señales: {len(signal_data['all_signals'])}")
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS:")
    print("="*60)
    stats = engine.get_statistics()
    print(f"Estrategias totales: {stats['total_strategies']}")
    print(f"Estrategias activas: {stats['enabled_strategies']}")
    print(f"Régimen actual: {stats['current_regime']}")