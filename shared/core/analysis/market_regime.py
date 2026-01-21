# shared/core/analysis/market_regime.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class MarketRegime(Enum):
    """Regímenes de mercado identificables."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNCERTAIN = "uncertain"


class TrendStrength(Enum):
    """Fuerza de la tendencia."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


# ============================================================================
# REGIME INDICATORS
# ============================================================================

class RegimeIndicators:
    """Indicadores para detección de régimen."""
    
    @staticmethod
    def calculate_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calcula Average Directional Index (ADX).
        ADX mide la fuerza de la tendencia (no la dirección).
        
        Args:
            high, low, close: Series de precios
            period: Período de cálculo
        
        Returns:
            Serie con valores ADX (0-100)
        """
        # True Range
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_trend_intensity(
        close: pd.Series,
        period: int = 20
    ) -> float:
        """
        Calcula intensidad de tendencia usando regresión lineal.
        
        Args:
            close: Serie de precios de cierre
            period: Período de análisis
        
        Returns:
            Intensidad de tendencia (-1 a 1)
        """
        if len(close) < period:
            return 0.0
        
        prices = close.tail(period).values
        x = np.arange(len(prices))
        
        # Regresión lineal
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalizar por precio promedio
        avg_price = prices.mean()
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        # Limitar entre -1 y 1
        intensity = np.clip(normalized_slope * 100, -1, 1)
        
        return float(intensity)
    
    @staticmethod
    def calculate_volatility_regime(
        returns: pd.Series,
        short_window: int = 10,
        long_window: int = 50
    ) -> str:
        """
        Determina régimen de volatilidad.
        
        Args:
            returns: Serie de retornos
            short_window: Ventana corta
            long_window: Ventana larga
        
        Returns:
            'high', 'normal', o 'low'
        """
        if len(returns) < long_window:
            return 'normal'
        
        short_vol = returns.tail(short_window).std()
        long_vol = returns.tail(long_window).std()
        
        if long_vol == 0:
            return 'normal'
        
        vol_ratio = short_vol / long_vol
        
        if vol_ratio > 1.5:
            return 'high'
        elif vol_ratio < 0.7:
            return 'low'
        else:
            return 'normal'
    
    @staticmethod
    def calculate_hurst_exponent(
        prices: pd.Series,
        max_lag: int = 20
    ) -> float:
        """
        Calcula exponente de Hurst.
        H < 0.5 = Mean reverting
        H = 0.5 = Random walk
        H > 0.5 = Trending
        
        Args:
            prices: Serie de precios
            max_lag: Lag máximo
        
        Returns:
            Exponente de Hurst (0-1)
        """
        if len(prices) < max_lag * 2:
            return 0.5
        
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Calcular diferencias
            diff = prices.diff(lag).dropna()
            # Desviación estándar
            tau.append(np.sqrt(np.mean(diff ** 2)))
        
        # Regresión log-log
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0]
            return float(np.clip(hurst, 0, 1))
        except:
            return 0.5
    
    @staticmethod
    def detect_market_structure(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 50
    ) -> Dict[str, Any]:
        """
        Detecta estructura de mercado (HH/HL para uptrend, LH/LL para downtrend).
        
        Returns:
            Dict con información de estructura
        """
        if len(close) < lookback:
            return {'structure': 'unknown', 'confidence': 0.0}
        
        # Encontrar pivots
        data = close.tail(lookback)
        
        # Máximos y mínimos locales
        highs = data[data == data.rolling(5, center=True).max()]
        lows = data[data == data.rolling(5, center=True).min()]
        
        if len(highs) < 2 or len(lows) < 2:
            return {'structure': 'insufficient_data', 'confidence': 0.0}
        
        # Analizar últimos 2 pivots
        last_2_highs = highs.tail(2).values
        last_2_lows = lows.tail(2).values
        
        # Higher highs y higher lows = uptrend
        hh = last_2_highs[-1] > last_2_highs[0] if len(last_2_highs) == 2 else False
        hl = last_2_lows[-1] > last_2_lows[0] if len(last_2_lows) == 2 else False
        
        # Lower highs y lower lows = downtrend
        lh = last_2_highs[-1] < last_2_highs[0] if len(last_2_highs) == 2 else False
        ll = last_2_lows[-1] < last_2_lows[0] if len(last_2_lows) == 2 else False
        
        if hh and hl:
            return {'structure': 'uptrend', 'confidence': 0.8}
        elif lh and ll:
            return {'structure': 'downtrend', 'confidence': 0.8}
        else:
            return {'structure': 'ranging', 'confidence': 0.6}


# ============================================================================
# MARKET REGIME DETECTOR (Principal)
# ============================================================================

class MarketRegimeDetector:
    """
    Detector de régimen de mercado.
    
    Identifica:
    - Tendencias (up/down/ranging)
    - Volatilidad (high/low)
    - Fuerza de tendencia
    - Estructura de mercado
    """
    
    def __init__(
        self,
        data_manager: Any,
        lookback_period: int = 100
    ):
        """
        Inicializa el detector.
        
        Args:
            data_manager: Instancia de DataManager
            lookback_period: Período de análisis
        """
        self.data_manager = data_manager
        self.lookback_period = lookback_period
        
        # Estado
        self.current_regime: Dict[str, MarketRegime] = {}
        self.regime_confidence: Dict[str, float] = {}
        self.regime_history: deque = deque(maxlen=100)
        
        # Configuración
        self.adx_threshold_trending = 25  # ADX > 25 = trending
        self.adx_threshold_strong = 40    # ADX > 40 = strong trend
        self.hurst_threshold = 0.55       # Hurst > 0.55 = trending
        
        logger.info("MarketRegimeDetector inicializado")
    
    def detect(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h"
    ) -> MarketRegime:
        """
        Detecta régimen de mercado actual.
        
        Args:
            symbol: Símbolo del asset
            timeframe: Timeframe
        
        Returns:
            MarketRegime detectado
        """
        # Obtener datos
        data = self.data_manager.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=self.lookback_period
        )
        
        if data.empty or len(data) < 50:
            logger.warning(f"Datos insuficientes para {symbol}")
            return MarketRegime.UNCERTAIN
        
        # Calcular indicadores
        indicators = self._calculate_indicators(data)
        
        # Detectar régimen
        regime = self._determine_regime(indicators)
        
        # Calcular confianza
        confidence = self._calculate_confidence(indicators)
        
        # Guardar estado
        self.current_regime[symbol] = regime
        self.regime_confidence[symbol] = confidence
        
        # Historial
        self.regime_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'regime': regime.value,
            'confidence': confidence
        })
        
        logger.info(
            f"Régimen detectado para {symbol}: {regime.value} "
            f"(confianza: {confidence:.2%})"
        )
        
        return regime
    
    def detect_multi_asset(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h"
    ) -> Dict[str, MarketRegime]:
        """
        Detecta régimen para múltiples assets.
        
        Args:
            symbols: Lista de símbolos (usa assets registrados si None)
            timeframe: Timeframe
        
        Returns:
            Dict {symbol: regime}
        """
        symbols = symbols or self.data_manager.get_assets()
        
        regimes = {}
        
        for symbol in symbols:
            try:
                regime = self.detect(symbol, timeframe)
                regimes[symbol] = regime
            except Exception as e:
                logger.error(f"Error detectando régimen para {symbol}: {e}")
                regimes[symbol] = MarketRegime.UNCERTAIN
        
        return regimes
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula todos los indicadores necesarios."""
        indicators = {}
        
        try:
            # Retornos
            data['returns'] = data['close'].pct_change()
            
            # ADX (fuerza de tendencia)
            indicators['adx'] = RegimeIndicators.calculate_adx(
                data['high'],
                data['low'],
                data['close']
            ).iloc[-1]
            
            # Intensidad de tendencia
            indicators['trend_intensity'] = RegimeIndicators.calculate_trend_intensity(
                data['close']
            )
            
            # Volatilidad
            indicators['volatility_regime'] = RegimeIndicators.calculate_volatility_regime(
                data['returns'].dropna()
            )
            
            # Exponente de Hurst
            indicators['hurst'] = RegimeIndicators.calculate_hurst_exponent(
                data['close']
            )
            
            # Estructura de mercado
            indicators['market_structure'] = RegimeIndicators.detect_market_structure(
                data['high'],
                data['low'],
                data['close']
            )
            
            # EMAs para tendencia
            data['ema_20'] = data['close'].ewm(span=20).mean()
            data['ema_50'] = data['close'].ewm(span=50).mean()
            
            last = data.iloc[-1]
            indicators['price_above_ema20'] = last['close'] > last['ema_20']
            indicators['price_above_ema50'] = last['close'] > last['ema_50']
            indicators['ema20_above_ema50'] = last['ema_20'] > last['ema_50']
            
            # Volatilidad actual
            indicators['current_volatility'] = data['returns'].tail(20).std()
            indicators['avg_volatility'] = data['returns'].std()
            
            # Precio actual vs rango
            price_range = data['high'].max() - data['low'].min()
            current_position = (last['close'] - data['low'].min()) / price_range if price_range > 0 else 0.5
            indicators['price_position_in_range'] = current_position
            
        except Exception as e:
            logger.error(f"Error calculando indicadores: {e}")
            # Retornar valores por defecto
            indicators = {
                'adx': 20,
                'trend_intensity': 0,
                'volatility_regime': 'normal',
                'hurst': 0.5,
                'market_structure': {'structure': 'unknown', 'confidence': 0},
                'price_above_ema20': False,
                'price_above_ema50': False,
                'ema20_above_ema50': False,
                'current_volatility': 0.02,
                'avg_volatility': 0.02,
                'price_position_in_range': 0.5
            }
        
        return indicators
    
    def _determine_regime(self, indicators: Dict[str, Any]) -> MarketRegime:
        """Determina régimen basado en indicadores."""
        
        adx = indicators.get('adx', 20)
        trend_intensity = indicators.get('trend_intensity', 0)
        hurst = indicators.get('hurst', 0.5)
        volatility = indicators.get('volatility_regime', 'normal')
        structure = indicators.get('market_structure', {})
        
        # Volatilidad extrema tiene prioridad
        if volatility == 'high':
            return MarketRegime.HIGH_VOLATILITY
        
        # ADX bajo = ranging (sin tendencia clara)
        if adx < self.adx_threshold_trending:
            if volatility == 'low':
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.RANGING
        
        # ADX alto = trending
        # Determinar dirección
        if trend_intensity > 0.3:
            # Tendencia alcista
            if structure.get('structure') == 'uptrend':
                return MarketRegime.BULLISH
            else:
                return MarketRegime.TRENDING_UP
        
        elif trend_intensity < -0.3:
            # Tendencia bajista
            if structure.get('structure') == 'downtrend':
                return MarketRegime.BEARISH
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Hurst para confirmar
        if hurst > self.hurst_threshold:
            # Trending pero dirección incierta
            if indicators.get('ema20_above_ema50'):
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Default
        return MarketRegime.RANGING
    
    def _calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calcula confianza en la detección."""
        confidence_factors = []
        
        # Factor ADX (0-1)
        adx = indicators.get('adx', 20)
        adx_confidence = min(adx / 50, 1.0)
        confidence_factors.append(adx_confidence)
        
        # Factor trend intensity
        trend_intensity = abs(indicators.get('trend_intensity', 0))
        intensity_confidence = min(trend_intensity, 1.0)
        confidence_factors.append(intensity_confidence)
        
        # Factor estructura de mercado
        structure_confidence = indicators.get('market_structure', {}).get('confidence', 0.5)
        confidence_factors.append(structure_confidence)
        
        # Factor Hurst
        hurst = indicators.get('hurst', 0.5)
        hurst_distance = abs(hurst - 0.5)  # Qué tan lejos está de random walk
        hurst_confidence = min(hurst_distance * 2, 1.0)
        confidence_factors.append(hurst_confidence)
        
        # Promedio ponderado
        confidence = np.average(confidence_factors, weights=[0.3, 0.3, 0.2, 0.2])
        
        return float(confidence)
    
    def get_trend_strength(self, symbol: str) -> TrendStrength:
        """
        Evalúa fuerza de la tendencia actual.
        
        Args:
            symbol: Símbolo del asset
        
        Returns:
            TrendStrength
        """
        data = self.data_manager.get_market_data(symbol, "1h", limit=50)
        
        if data.empty:
            return TrendStrength.NONE
        
        # Calcular ADX
        adx = RegimeIndicators.calculate_adx(
            data['high'],
            data['low'],
            data['close']
        ).iloc[-1]
        
        if np.isnan(adx):
            return TrendStrength.NONE
        
        if adx > 40:
            return TrendStrength.STRONG
        elif adx > 25:
            return TrendStrength.MODERATE
        elif adx > 15:
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE
    
    def should_trade(
        self,
        symbol: str,
        strategy_type: str
    ) -> bool:
        """
        Determina si una estrategia debe operar en el régimen actual.
        
        Args:
            symbol: Símbolo del asset
            strategy_type: Tipo de estrategia (trend_following, mean_reversion, etc.)
        
        Returns:
            True si debe operar
        """
        regime = self.current_regime.get(symbol, MarketRegime.UNCERTAIN)
        confidence = self.regime_confidence.get(symbol, 0.0)
        
        # Requiere confianza mínima
        if confidence < 0.5:
            return False
        
        # Estrategias de tendencia
        if strategy_type in ['trend_following', 'momentum', 'breakout']:
            return regime in [
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BULLISH,
                MarketRegime.BEARISH
            ]
        
        # Estrategias de reversión
        elif strategy_type in ['mean_reversion', 'scalping']:
            return regime in [
                MarketRegime.RANGING,
                MarketRegime.LOW_VOLATILITY
            ]
        
        # Estrategias de volatilidad
        elif strategy_type == 'volatility_based':
            return regime == MarketRegime.HIGH_VOLATILITY
        
        # Default: permitir trading
        return True
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del detector."""
        if not self.regime_history:
            return {}
        
        # Convertir a DataFrame
        df = pd.DataFrame(list(self.regime_history))
        
        # Contar regímenes
        regime_counts = df['regime'].value_counts().to_dict()
        
        # Confianza promedio
        avg_confidence = df['confidence'].mean()
        
        # Régimen más común
        most_common_regime = df['regime'].mode()[0] if not df.empty else 'unknown'
        
        # Tiempo en cada régimen
        regime_durations = {}
        if len(df) > 1:
            for regime in df['regime'].unique():
                regime_df = df[df['regime'] == regime]
                if len(regime_df) > 1:
                    duration = (regime_df['timestamp'].max() - regime_df['timestamp'].min()).total_seconds() / 3600
                    regime_durations[regime] = duration
        
        return {
            'total_detections': len(df),
            'regime_counts': regime_counts,
            'avg_confidence': avg_confidence,
            'most_common_regime': most_common_regime,
            'regime_durations_hours': regime_durations,
            'current_regimes': {k: v.value for k, v in self.current_regime.items()}
        }
    
    def get_market_summary(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Genera resumen del mercado.
        
        Args:
            symbols: Lista de símbolos a analizar
        
        Returns:
            Dict con resumen del mercado
        """
        symbols = symbols or list(self.current_regime.keys())
        
        if not symbols:
            return {'error': 'No hay símbolos para analizar'}
        
        # Detectar regímenes
        regimes = self.detect_multi_asset(symbols)
        
        # Contar por tipo
        regime_distribution = {}
        for regime in regimes.values():
            regime_distribution[regime.value] = regime_distribution.get(regime.value, 0) + 1
        
        # Régimen dominante
        dominant_regime = max(regime_distribution, key=regime_distribution.get)
        
        # Porcentaje trending vs ranging
        trending = sum(
            1 for r in regimes.values()
            if r in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, 
                    MarketRegime.BULLISH, MarketRegime.BEARISH]
        )
        
        ranging = sum(
            1 for r in regimes.values()
            if r in [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY]
        )
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(symbols),
            'regime_distribution': regime_distribution,
            'dominant_regime': dominant_regime,
            'trending_pct': trending / len(symbols) * 100,
            'ranging_pct': ranging / len(symbols) * 100,
            'individual_regimes': {s: r.value for s, r in regimes.items()}
        }
        
        return summary


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock DataManager
    class MockDataManager:
        def get_assets(self):
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        def get_market_data(self, symbol, timeframe, limit):
            # Generar datos mock
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='1h')
            
            # Simular trending up
            np.random.seed(42)
            prices = [50000]
            for _ in range(limit - 1):
                change = np.random.randn() * 0.01 + 0.001  # Bias alcista
                prices.append(prices[-1] * (1 + change))
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': np.random.uniform(100, 1000, limit)
            })
            
            return df
    
    # Crear detector
    dm = MockDataManager()
    detector = MarketRegimeDetector(dm)
    
    # Detectar régimen
    print("\n" + "="*60)
    print("DETECCIÓN DE RÉGIMEN:")
    print("="*60)
    regime = detector.detect("BTCUSDT", "1h")
    print(f"Régimen: {regime.value}")
    print(f"Confianza: {detector.regime_confidence.get('BTCUSDT', 0):.2%}")
    
    # Fuerza de tendencia
    print("\n" + "="*60)
    print("FUERZA DE TENDENCIA:")
    print("="*60)
    strength = detector.get_trend_strength("BTCUSDT")
    print(f"Fuerza: {strength.value}")
    
    # Resumen de mercado
    print("\n" + "="*60)
    print("RESUMEN DE MERCADO:")
    print("="*60)
    import json
    summary = detector.get_market_summary()
    print(json.dumps(summary, indent=2))