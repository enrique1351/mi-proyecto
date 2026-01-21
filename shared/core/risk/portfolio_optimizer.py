# shared/core/risk/portfolio_optimizer.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Métodos de optimización disponibles."""
    MEAN_VARIANCE = "mean_variance"  # Markowitz
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hrp"


class PortfolioOptimizer:
    """
    Optimizador de cartera con múltiples métodos.
    
    Implementa:
    - Mean-Variance (Markowitz)
    - Minimum Variance
    - Maximum Sharpe Ratio
    - Risk Parity
    - Black-Litterman
    - Hierarchical Risk Parity (HRP)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        logger.info("PortfolioOptimizer inicializado")
    
    def optimize(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Optimiza pesos de cartera.
        
        Args:
            returns: DataFrame con retornos de assets (cada columna = asset)
            method: Método de optimización
            constraints: Restricciones adicionales
            **kwargs: Parámetros específicos del método
        
        Returns:
            Dict con pesos óptimos {symbol: weight}
        """
        
        if returns.empty:
            logger.error("Returns DataFrame está vacío")
            return {}
        
        logger.info(f"Optimizando cartera con método: {method.value}")
        logger.info(f"Assets: {len(returns.columns)}")
        
        # Calcular matriz de covarianza y retornos esperados
        cov_matrix = returns.cov() * 252  # Anualizar
        expected_returns = returns.mean() * 252  # Anualizar
        
        # Optimizar según método
        if method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._mean_variance_optimization(
                expected_returns, cov_matrix, constraints
            )
        
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance_optimization(cov_matrix, constraints)
        
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = self._max_sharpe_optimization(
                expected_returns, cov_matrix, constraints
            )
        
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity_optimization(cov_matrix, constraints)
        
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            market_caps = kwargs.get('market_caps')
            views = kwargs.get('views')
            weights = self._black_litterman_optimization(
                returns, expected_returns, cov_matrix, market_caps, views
            )
        
        elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
            weights = self._hrp_optimization(returns, cov_matrix)
        
        else:
            logger.error(f"Método {method} no implementado")
            return {}
        
        # Convertir a dict
        weights_dict = dict(zip(returns.columns, weights))
        
        # Log resultados
        self._log_portfolio_metrics(weights, expected_returns, cov_matrix)
        
        return weights_dict
    
    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Optimización Mean-Variance (Markowitz)."""
        
        n_assets = len(expected_returns)
        
        # Función objetivo: minimizar varianza para retorno objetivo
        target_return = constraints.get('target_return', expected_returns.mean()) if constraints else expected_returns.mean()
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Restricciones
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Suma = 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}  # Retorno objetivo
        ]
        
        # Límites (0 <= w <= 1, no short)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Punto inicial (equal weight)
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimizar
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if not result.success:
            logger.warning(f"Optimización no convergió: {result.message}")
        
        return result.x
    
    def _min_variance_optimization(
        self,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Minimiza varianza sin restricción de retorno."""
        
        n_assets = len(cov_matrix)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _max_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Maximiza Sharpe Ratio."""
        
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Negativo porque minimize busca mínimo
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _risk_parity_optimization(
        self,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Risk Parity: igual contribución al riesgo."""
        
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights):
            # Calcular contribución marginal al riesgo
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Queremos que todas las contribuciones sean iguales
            target = np.mean(risk_contrib)
            return np.sum((risk_contrib - target) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        views: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Black-Litterman Model.
        
        Args:
            market_caps: Capitalización de mercado de cada asset
            views: Dict con views del inversionista
        """
        
        n_assets = len(expected_returns)
        
        # Si no hay market caps, usar equal weight
        if market_caps is None:
            market_caps = pd.Series([1/n_assets] * n_assets, index=expected_returns.index)
        else:
            market_caps = market_caps / market_caps.sum()
        
        # Implied equilibrium returns (reverse optimization)
        risk_aversion = 2.5  # Lambda típico
        pi = risk_aversion * np.dot(cov_matrix, market_caps)
        
        # Si no hay views, usar equilibrium returns
        if not views:
            posterior_returns = pi
        else:
            # Incorporar views (simplified)
            # En producción, usar matriz P y Q completa
            posterior_returns = pi  # Placeholder
        
        # Optimizar con returns posteriores
        return self._max_sharpe_optimization(
            pd.Series(posterior_returns, index=expected_returns.index),
            cov_matrix
        )
    
    def _hrp_optimization(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity.
        
        Método más robusto que no requiere inversión de matriz.
        """
        
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convertir covarianza a matriz de distancias
        corr_matrix = returns.corr()
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # Clustering jerárquico
        link = linkage(squareform(dist_matrix), method='single')
        
        # Recursive bisection para asignar pesos
        weights = self._recursive_bisection(cov_matrix, link)
        
        return weights
    
    def _recursive_bisection(
        self,
        cov_matrix: pd.DataFrame,
        link: np.ndarray,
        items: Optional[List] = None
    ) -> np.ndarray:
        """Recursive bisection para HRP."""
        
        n = len(cov_matrix)
        
        if items is None:
            items = list(range(n))
        
        if len(items) == 1:
            weights = np.zeros(n)
            weights[items[0]] = 1.0
            return weights
        
        # Split cluster
        left = [i for i in items if i < n//2]
        right = [i for i in items if i >= n//2]
        
        # Calcular pesos recursivamente
        left_weights = self._recursive_bisection(cov_matrix, link, left)
        right_weights = self._recursive_bisection(cov_matrix, link, right)
        
        # Variance de cada sub-portfolio
        left_var = np.dot(left_weights.T, np.dot(cov_matrix, left_weights))
        right_var = np.dot(right_weights.T, np.dot(cov_matrix, right_weights))
        
        # Allocation factor
        alpha = 1 - left_var / (left_var + right_var)
        
        # Combine
        weights = alpha * left_weights + (1 - alpha) * right_weights
        
        return weights
    
    def _log_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ):
        """Log métricas del portfolio optimizado."""
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        logger.info(f"\n📊 Portfolio Optimizado:")
        logger.info(f"  Expected Return: {portfolio_return:.2%}")
        logger.info(f"  Volatility: {portfolio_vol:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        
        # Top holdings
        top_n = 5
        weights_series = pd.Series(weights, index=expected_returns.index)
        top_weights = weights_series.nlargest(top_n)
        
        logger.info(f"  Top {top_n} Holdings:")
        for symbol, weight in top_weights.items():
            logger.info(f"    {symbol}: {weight:.2%}")
    
    def calculate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Calcula la frontera eficiente.
        
        Returns:
            DataFrame con (return, volatility, sharpe) para cada punto
        """
        
        cov_matrix = returns.cov() * 252
        expected_returns = returns.mean() * 252
        
        # Rango de retornos
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        
        for target_ret in target_returns:
            try:
                weights = self._mean_variance_optimization(
                    expected_returns,
                    cov_matrix,
                    {'target_return': target_ret}
                )
                
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                
                frontier.append({
                    'return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe': sharpe
                })
            
            except:
                continue
        
        return pd.DataFrame(frontier)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generar datos de prueba
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    
    # Retornos simulados
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        columns=[f'ASSET_{i}' for i in range(n_assets)]
    )
    
    # Crear optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Optimizar con diferentes métodos
    print("\n" + "="*60)
    print("MAX SHARPE:")
    weights_sharpe = optimizer.optimize(returns, OptimizationMethod.MAX_SHARPE)
    for symbol, weight in weights_sharpe.items():
        print(f"  {symbol}: {weight:.2%}")
    
    print("\n" + "="*60)
    print("RISK PARITY:")
    weights_rp = optimizer.optimize(returns, OptimizationMethod.RISK_PARITY)
    for symbol, weight in weights_rp.items():
        print(f"  {symbol}: {weight:.2%}")