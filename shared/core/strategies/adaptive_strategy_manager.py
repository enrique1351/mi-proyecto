# shared/core/strategies/adaptive_strategy_manager.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# STATISTICS LAYER
# ============================================================================

class StatisticsLayer:
    """
    Capa estadística para análisis de rendimiento.
    Calcula métricas avanzadas y proporciona insights.
    """
    
    def __init__(self):
        self.historical_performance: Dict[str, List[Dict]] = defaultdict(list)
        self.current_regime = None
    
    def update_regime(self, regime):
        """Actualiza el régimen de mercado actual."""
        self.current_regime = regime
    
    def record_trade(
        self,
        strategy_name: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        duration_hours: float
    ):
        """
        Registra un trade completado.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo del asset
            entry_price: Precio de entrada
            exit_price: Precio de salida
            quantity: Cantidad operada
            side: BUY o SELL
            duration_hours: Duración del trade en horas
        """
        # Calcular PnL
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
            return_pct = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
            return_pct = (entry_price - exit_price) / entry_price
        
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'side': side,
            'pnl': pnl,
            'return_pct': return_pct,
            'duration_hours': duration_hours,
            'regime': self.current_regime.value if self.current_regime else 'unknown',
            'success': pnl > 0
        }
        
        key = f"{strategy_name}_{symbol}"
        self.historical_performance[key].append(trade_record)
    
    def calculate_strategy_metrics(
        self,
        strategy_name: str,
        symbol: Optional[str] = None,
        lookback_trades: int = 50
    ) -> Dict[str, float]:
        """
        Calcula métricas de performance de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Filtrar por símbolo (opcional)
            lookback_trades: Número de trades a analizar
        
        Returns:
            Dict con métricas
        """
        # Filtrar trades
        if symbol:
            key = f"{strategy_name}_{symbol}"
            trades = self.historical_performance.get(key, [])
        else:
            # Todos los trades de la estrategia
            trades = []
            for key, trade_list in self.historical_performance.items():
                if key.startswith(strategy_name):
                    trades.extend(trade_list)
        
        if not trades:
            return self._empty_metrics()
        
        # Tomar últimos N trades
        recent_trades = trades[-lookback_trades:]
        
        # Calcular métricas
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t['success'])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL
        total_pnl = sum(t['pnl'] for t in recent_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        wins = [t['pnl'] for t in recent_trades if t['success']]
        losses = [t['pnl'] for t in recent_trades if not t['success']]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Sharpe Ratio (simplificado)
        returns = [t['return_pct'] for t in recent_trades]
        if returns and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown de equity curve
        equity_curve = []
        equity = 0
        for t in recent_trades:
            equity += t['pnl']
            equity_curve.append(equity)
        
        if equity_curve:
            cummax = np.maximum.accumulate(equity_curve)
            drawdowns = (np.array(equity_curve) - cummax) / (cummax + 1)  # +1 para evitar div/0
            max_drawdown = abs(np.min(drawdowns))
        else:
            max_drawdown = 0
        
        # Consistency score (últimos 10 trades)
        if len(recent_trades) >= 10:
            last_10 = recent_trades[-10:]
            last_10_winrate = sum(1 for t in last_10 if t['success']) / len(last_10)
            consistency = 1 - abs(win_rate - last_10_winrate)  # Qué tan consistente es el win rate
        else:
            consistency = 0.5
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'consistency_score': consistency
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Retorna métricas vacías."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'consistency_score': 0
        }
    
    def get_regime_performance(
        self,
        strategy_name: str,
        regime: str
    ) -> Dict[str, float]:
        """
        Analiza performance de estrategia en un régimen específico.
        
        Args:
            strategy_name: Nombre de la estrategia
            regime: Régimen de mercado
        
        Returns:
            Dict con métricas en ese régimen
        """
        # Filtrar trades por estrategia y régimen
        trades = []
        for key, trade_list in self.historical_performance.items():
            if key.startswith(strategy_name):
                trades.extend([t for t in trade_list if t['regime'] == regime])
        
        if not trades:
            return self._empty_metrics()
        
        # Calcular métricas (reusar lógica)
        total = len(trades)
        wins = sum(1 for t in trades if t['success'])
        
        return {
            'total_trades': total,
            'win_rate': wins / total if total > 0 else 0,
            'avg_pnl': np.mean([t['pnl'] for t in trades]),
            'total_pnl': sum(t['pnl'] for t in trades)
        }


# ============================================================================
# ADAPTIVE STRATEGY MANAGER (Principal)
# ============================================================================

class AdaptiveStrategyManager:
    """
    Gestor adaptativo de estrategias.
    
    Coordina:
    - Strategy Engine (genera señales)
    - Market Regime Detector (identifica contexto)
    - Risk Manager (valida riesgo)
    - Statistics Layer (analiza performance)
    
    Adapta:
    - Selección de estrategias según régimen
    - Tamaño de posiciones según confianza
    - Pesos de estrategias según performance
    """
    
    def __init__(
        self,
        data_manager: Any,
        strategy_engine: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
        risk_manager: Optional[Any] = None
    ):
        """
        Inicializa el manager adaptativo.
        
        Args:
            data_manager: Instancia de DataManager
            strategy_engine: Instancia de StrategyEngine
            regime_detector: Instancia de MarketRegimeDetector
            risk_manager: Instancia de RiskManager
        """
        self.data_manager = data_manager
        self.strategy_engine = strategy_engine
        self.regime_detector = regime_detector
        self.risk_manager = risk_manager
        
        # Statistics Layer
        self.stats_layer = StatisticsLayer()
        
        # Pesos de estrategias (dinámicos)
        self.strategy_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Configuración
        self.min_confidence = 0.5  # Confianza mínima para operar
        self.min_trades_for_adaptation = 10  # Trades mínimos antes de adaptar
        self.adaptation_speed = 0.1  # Velocidad de adaptación (0-1)
        
        # Estado
        self.active_strategies: Dict[str, bool] = {}
        self.regime_strategy_map: Dict[str, List[str]] = self._initialize_regime_map()
        
        # Persistencia
        self.state_file = Path("data/adaptive_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
        
        logger.info("AdaptiveStrategyManager inicializado")
    
    def _initialize_regime_map(self) -> Dict[str, List[str]]:
        """
        Inicializa mapeo de regímenes a estrategias óptimas.
        
        Returns:
            Dict {regime: [strategies]}
        """
        return {
            'trending_up': ['TrendFollowing', 'Momentum', 'Breakout'],
            'trending_down': ['TrendFollowing', 'Momentum'],
            'bullish': ['TrendFollowing', 'Breakout', 'Momentum'],
            'bearish': ['TrendFollowing', 'Momentum'],
            'ranging': ['MeanReversion', 'Scalping'],
            'high_volatility': ['Breakout', 'Volatility'],
            'low_volatility': ['MeanReversion', 'Scalping'],
            'uncertain': ['MeanReversion']  # Conservador
        }
    
    def _load_state(self):
        """Carga estado persistente."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.strategy_weights = defaultdict(lambda: 1.0, state.get('weights', {}))
                    self.active_strategies = state.get('active_strategies', {})
                
                logger.info("Estado adaptativo cargado")
            except Exception as e:
                logger.error(f"Error cargando estado: {e}")
    
    def _save_state(self):
        """Guarda estado persistente."""
        try:
            state = {
                'weights': dict(self.strategy_weights),
                'active_strategies': self.active_strategies,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
    
    def adapt_signals(
        self,
        raw_signals: Dict[str, Dict[str, Any]],
        available_capital: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Adapta señales según régimen, riesgo y performance.
        
        Args:
            raw_signals: Señales crudas del StrategyEngine
                        {asset: {strategy, side, confidence, ...}}
            available_capital: Capital disponible por asset
        
        Returns:
            Señales adaptadas con position sizing
        """
        adapted_signals = {}
        
        for asset, signal_data in raw_signals.items():
            try:
                # 1️⃣ Obtener datos del asset
                strategy_name = signal_data.get('strategy', 'Unknown')
                confidence = signal_data.get('confidence', 0.5)
                side = signal_data.get('side', 'BUY')
                
                # 2️⃣ Verificar si la estrategia debe operar en el régimen actual
                if self.regime_detector and not self._should_trade_in_regime(asset, strategy_name):
                    logger.debug(f"Estrategia {strategy_name} no debe operar en régimen actual para {asset}")
                    continue
                
                # 3️⃣ Ajustar confianza por performance histórico
                adjusted_confidence = self._adjust_confidence(strategy_name, asset, confidence)
                
                # 4️⃣ Verificar confianza mínima
                if adjusted_confidence < self.min_confidence:
                    logger.debug(f"Confianza {adjusted_confidence:.2f} < mínimo {self.min_confidence}")
                    continue
                
                # 5️⃣ Obtener precio actual
                price = self.data_manager.get_latest_price(asset, "1m")
                if not price or price <= 0:
                    logger.warning(f"No hay precio válido para {asset}")
                    continue
                
                # 6️⃣ Calcular position size con Risk Manager
                if self.risk_manager:
                    quantity, stop_loss, take_profit = self.risk_manager.calculate_position_size(
                        symbol=asset,
                        entry_price=price,
                        confidence=adjusted_confidence
                    )
                else:
                    # Fallback: usar capital disponible y confianza
                    capital = available_capital.get(asset, 1000)
                    position_value = capital * adjusted_confidence * 0.02  # 2% del capital
                    quantity = position_value / price
                    stop_loss = price * 0.97  # 3% stop loss
                    take_profit = price * 1.06  # 6% take profit
                
                # 7️⃣ Crear señal adaptada
                adapted_signal = {
                    'strategy': strategy_name,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': adjusted_confidence,
                    'original_confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
                
                adapted_signals[asset] = adapted_signal
                
                logger.debug(
                    f"Señal adaptada: {asset} {strategy_name} "
                    f"{side} qty={quantity:.4f} conf={adjusted_confidence:.2f}"
                )
            
            except Exception as e:
                logger.error(f"Error adaptando señal para {asset}: {e}")
        
        return adapted_signals
    
    def _should_trade_in_regime(self, asset: str, strategy_name: str) -> bool:
        """Verifica si la estrategia debe operar en el régimen actual."""
        if not self.regime_detector:
            return True
        
        # Detectar régimen actual
        current_regime = self.regime_detector.current_regime.get(asset)
        
        if not current_regime:
            # Si no hay régimen detectado, detectar ahora
            current_regime = self.regime_detector.detect(asset)
        
        regime_value = current_regime.value if current_regime else 'uncertain'
        
        # Obtener estrategias óptimas para este régimen
        optimal_strategies = self.regime_strategy_map.get(regime_value, [])
        
        # Verificar si la estrategia está en la lista óptima
        return strategy_name in optimal_strategies
    
    def _adjust_confidence(
        self,
        strategy_name: str,
        symbol: str,
        base_confidence: float
    ) -> float:
        """
        Ajusta confianza basada en performance histórico.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo del asset
            base_confidence: Confianza base de la señal
        
        Returns:
            Confianza ajustada (0-1)
        """
        # Obtener métricas históricas
        metrics = self.stats_layer.calculate_strategy_metrics(strategy_name, symbol)
        
        if metrics['total_trades'] < self.min_trades_for_adaptation:
            # No hay suficiente historial, usar confianza base
            return base_confidence
        
        # Factores de ajuste
        win_rate_factor = metrics['win_rate']  # 0-1
        sharpe_factor = min(max(metrics['sharpe_ratio'] / 2, 0), 1)  # Normalizar sharpe
        consistency_factor = metrics['consistency_score']
        expectancy_factor = 1.0 if metrics['expectancy'] > 0 else 0.5
        
        # Peso de estrategia (performance global)
        weight_factor = self.strategy_weights.get(strategy_name, 1.0)
        
        # Ajuste combinado (promedio ponderado)
        adjustment = np.average(
            [win_rate_factor, sharpe_factor, consistency_factor, expectancy_factor, weight_factor],
            weights=[0.3, 0.2, 0.2, 0.2, 0.1]
        )
        
        # Aplicar ajuste con velocidad de adaptación
        adjusted = base_confidence * (1 - self.adaptation_speed) + adjustment * self.adaptation_speed
        
        # Limitar entre 0 y 1
        adjusted = max(0, min(adjusted, 1))
        
        logger.debug(
            f"Confianza ajustada para {strategy_name} en {symbol}: "
            f"{base_confidence:.2f} -> {adjusted:.2f}"
        )
        
        return adjusted
    
    def update_performance(
        self,
        strategy_name: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        duration_hours: float = 1.0
    ):
        """
        Actualiza performance después de un trade.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo del asset
            entry_price: Precio de entrada
            exit_price: Precio de salida
            quantity: Cantidad
            side: BUY o SELL
            duration_hours: Duración del trade
        """
        # Registrar trade en statistics layer
        self.stats_layer.record_trade(
            strategy_name=strategy_name,
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side=side,
            duration_hours=duration_hours
        )
        
        # Recalcular peso de la estrategia
        self._recalculate_strategy_weight(strategy_name)
        
        # Guardar estado
        self._save_state()
    
    def _recalculate_strategy_weight(self, strategy_name: str):
        """Recalcula peso de una estrategia basado en performance."""
        metrics = self.stats_layer.calculate_strategy_metrics(strategy_name)
        
        if metrics['total_trades'] < self.min_trades_for_adaptation:
            return
        
        # Calcular peso basado en métricas
        # Peso = (win_rate * 0.3) + (profit_factor * 0.3) + (sharpe * 0.2) + (consistency * 0.2)
        
        win_rate = metrics['win_rate']
        
        # Normalizar profit factor (0-1)
        profit_factor = min(metrics['profit_factor'] / 3, 1.0)
        
        # Normalizar sharpe (0-1)
        sharpe = min(max(metrics['sharpe_ratio'] / 2, 0), 1)
        
        consistency = metrics['consistency_score']
        
        new_weight = np.average(
            [win_rate, profit_factor, sharpe, consistency],
            weights=[0.3, 0.3, 0.2, 0.2]
        )
        
        # Limitar peso entre 0.1 y 2.0
        new_weight = max(0.1, min(new_weight * 2, 2.0))
        
        # Actualizar con smoothing
        old_weight = self.strategy_weights.get(strategy_name, 1.0)
        smoothed_weight = old_weight * 0.7 + new_weight * 0.3
        
        self.strategy_weights[strategy_name] = smoothed_weight
        
        logger.info(
            f"Peso de {strategy_name} actualizado: {old_weight:.2f} -> {smoothed_weight:.2f}"
        )
    
    def get_best_strategies(
        self,
        regime: Optional[str] = None,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Obtiene las mejores estrategias.
        
        Args:
            regime: Filtrar por régimen (opcional)
            top_n: Número de estrategias a retornar
        
        Returns:
            Lista de tuplas (strategy_name, weight)
        """
        if regime:
            # Filtrar por régimen
            optimal = self.regime_strategy_map.get(regime, [])
            strategies = {
                name: self.strategy_weights.get(name, 1.0)
                for name in optimal
            }
        else:
            strategies = dict(self.strategy_weights)
        
        # Ordenar por peso
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_strategies[:top_n]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Genera reporte completo de performance."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'strategy_weights': dict(self.strategy_weights),
            'active_strategies': self.active_strategies,
            'strategy_metrics': {}
        }
        
        # Métricas por estrategia
        all_strategies = set()
        for trades in self.stats_layer.historical_performance.values():
            for trade in trades:
                all_strategies.add(trade['strategy'])
        
        for strategy in all_strategies:
            metrics = self.stats_layer.calculate_strategy_metrics(strategy)
            report['strategy_metrics'][strategy] = metrics
        
        # Métricas agregadas
        total_trades = sum(m['total_trades'] for m in report['strategy_metrics'].values())
        total_pnl = sum(m['total_pnl'] for m in report['strategy_metrics'].values())
        
        report['aggregate'] = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
        }
        
        return report
    
    def optimize_regime_mapping(self):
        """
        Optimiza el mapeo de regímenes a estrategias basado en datos históricos.
        Aprende qué estrategias funcionan mejor en cada régimen.
        """
        # Analizar performance por régimen
        for regime in self.regime_strategy_map.keys():
            regime_performance = {}
            
            # Calcular performance de cada estrategia en este régimen
            for strategy in self.strategy_weights.keys():
                perf = self.stats_layer.get_regime_performance(strategy, regime)
                if perf['total_trades'] >= 5:  # Mínimo 5 trades
                    regime_performance[strategy] = perf['win_rate']
            
            # Si hay datos, actualizar mapeo
            if regime_performance:
                # Top 3 estrategias para este régimen
                top_strategies = sorted(
                    regime_performance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                new_strategies = [s[0] for s in top_strategies]
                
                # Actualizar con smoothing (mantener algo de la configuración original)
                old_strategies = set(self.regime_strategy_map[regime])
                combined = list(old_strategies.union(set(new_strategies)))
                
                self.regime_strategy_map[regime] = combined[:5]  # Máximo 5 estrategias
                
                logger.info(f"Mapeo optimizado para régimen {regime}: {combined}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock components
    class MockDataManager:
        def get_latest_price(self, symbol, timeframe):
            return 50000.0
    
    class MockRiskManager:
        def calculate_position_size(self, symbol, entry_price, confidence):
            return 0.1, entry_price * 0.97, entry_price * 1.06
    
    # Crear manager
    dm = MockDataManager()
    rm = MockRiskManager()
    manager = AdaptiveStrategyManager(dm, risk_manager=rm)
    
    # Señales raw
    raw_signals = {
        'BTCUSDT': {
            'strategy': 'TrendFollowing',
            'side': 'BUY',
            'confidence': 0.8
        },
        'ETHUSDT': {
            'strategy': 'MeanReversion',
            'side': 'SELL',
            'confidence': 0.6
        }
    }
    
    # Capital disponible
    capital = {
        'BTCUSDT': 5000,
        'ETHUSDT': 3000
    }
    
    # Adaptar señales
    print("\n" + "="*60)
    print("SEÑALES ADAPTADAS:")
    print("="*60)
    adapted = manager.adapt_signals(raw_signals, capital)
    import json
    print(json.dumps(adapted, indent=2, default=str))
    
    # Simular actualización de performance
    manager.update_performance(
        strategy_name='TrendFollowing',
        symbol='BTCUSDT',
        entry_price=50000,
        exit_price=51000,
        quantity=0.1,
        side='BUY',
        duration_hours=2.0
    )
    
    # Reporte
    print("\n" + "="*60)
    print("REPORTE DE PERFORMANCE:")
    print("="*60)
    report = manager.get_performance_report()
    print(json.dumps(report, indent=2, default=str))