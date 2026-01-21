import logging
import numpy as np
import pandas as pd
from core_local.statistics_layer import StatisticsLayer

logger = logging.getLogger("StrategyOptimizer")

class StrategyOptimizer:
    """
    Módulo de comparación de estrategias y reescritura adaptativa:
    - Compara diversas estrategias basadas en métricas de rendimiento
    - Reescribe las estrategias en función del panorama actual del mercado
    """

    def __init__(self, data_manager, adaptive_manager, regime_detector: 'MarketRegimeDetector'):
        self.data_manager = data_manager
        self.adaptive_manager = adaptive_manager
        self.regime_detector = regime_detector

    def compare_strategies(self, asset, strategies):
        """
        Compara varias estrategias para un activo basado en métricas clave.
        asset: el activo a comparar
        strategies: lista de estrategias para comparar
        Devuelve un diccionario con la evaluación de cada estrategia
        """

        # Obtener los históricos de cada estrategia
        results = {}
        for strategy in strategies:
            results[strategy] = self.evaluate_strategy(asset, strategy)

        return results

    def evaluate_strategy(self, asset, strategy):
        """
        Evalúa una estrategia para un activo.
        Devuelve un diccionario con métricas de la estrategia
        """
        # Obtener datos históricos de trades
        trade_history = self.data_manager.get_strategy_history(strategy, asset)

        if trade_history.empty:
            return {"pnl": 0, "drawdown": 0, "sharpe_ratio": 0, "volatility": 0}

        # Calcular PnL
        pnl = trade_history["pnl"].sum()

        # Calcular Drawdown
        drawdown = self.calculate_drawdown(trade_history)

        # Calcular Volatilidad
        volatility = trade_history["pnl"].std()

        # Calcular Ratio de Sharpe (simplificado)
        sharpe_ratio = pnl / volatility if volatility != 0 else 0

        return {
            "pnl": pnl,
            "drawdown": drawdown,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility
        }

    def calculate_drawdown(self, trade_history):
        """
        Calcula el drawdown basado en la serie histórica de PnL.
        """
        cumulative_pnl = trade_history["pnl"].cumsum()
        peak = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - peak) / peak
        return drawdown.min()  # Valor mínimo del drawdown (el peor punto)

    def adapt_strategy(self, asset, strategies, current_regime):
        """
        Adapta las estrategias en función del régimen del mercado y el rendimiento
        actual, modificando las estrategias que no sean rentables.
        """
        # Comparar las estrategias basadas en el rendimiento
        strategy_comparisons = self.compare_strategies(asset, strategies)
        
        # Obtener el régimen actual del mercado
        regime = self.regime_detector.detect()

        adapted_strategies = {}

        for strategy, metrics in strategy_comparisons.items():
            if metrics["sharpe_ratio"] < 0.5:  # Si el Sharpe Ratio es bajo, no es eficiente
                logger.info(f"Reescribiendo estrategia {strategy} para {asset} debido a bajo rendimiento.")
                adapted_strategies[strategy] = self.rewrite_strategy(strategy, asset, regime)
            else:
                adapted_strategies[strategy] = strategy

        return adapted_strategies

    def rewrite_strategy(self, strategy, asset, regime):
        """
        Reescribe una estrategia en función del régimen actual y el activo.
        """
        logger.info(f"Reescribiendo la estrategia {strategy} para {asset} en régimen {regime}.")
        
        # Lógica para ajustar la estrategia (esto puede ser el ajuste de parámetros de una estrategia)
        new_strategy = strategy + "_adapted_" + regime
        
        # Ajuste de parámetros según el régimen del mercado
        if regime == "bull":
            new_strategy += "_bullish"
        elif regime == "bear":
            new_strategy += "_bearish"
        elif regime == "neutral":
            new_strategy += "_neutral"
        
        return new_strategy
