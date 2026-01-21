# shared/core/risk/risk_manager.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class RiskLevel(Enum):
    """Niveles de riesgo del sistema."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class PositionSizingMethod(Enum):
    """Métodos de position sizing."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"


# ============================================================================
# RISK METRICS CALCULATOR
# ============================================================================

class RiskMetrics:
    """Calculadora de métricas de riesgo."""
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> float:
        """
        Calcula volatilidad histórica.
        
        Args:
            returns: Serie de retornos
            window: Ventana de cálculo
        
        Returns:
            Volatilidad anualizada
        """
        if len(returns) < window:
            return 0.0
        
        vol = returns.tail(window).std()
        annualized_vol = vol * np.sqrt(252)  # Anualizar (252 días trading)
        
        return float(annualized_vol)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula Sharpe Ratio.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anual
        
        Returns:
            Sharpe Ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        sharpe = excess_returns / returns.std()
        annualized_sharpe = sharpe * np.sqrt(252)
        
        return float(annualized_sharpe)
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula Sortino Ratio (solo penaliza volatilidad negativa).
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo
        
        Returns:
            Sortino Ratio
        """
        if returns.empty:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        sortino = excess_returns / downside_std
        annualized_sortino = sortino * np.sqrt(252)
        
        return float(annualized_sortino)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
        """
        Calcula máximo drawdown.
        
        Args:
            equity_curve: Serie con valores de equity
        
        Returns:
            Tuple (max_drawdown_pct, duration_days)
        """
        if equity_curve.empty:
            return 0.0, 0
        
        # Calcular peak acumulativo
        cummax = equity_curve.cummax()
        
        # Calcular drawdown
        drawdown = (equity_curve - cummax) / cummax
        
        max_dd = abs(drawdown.min())
        
        # Duración del drawdown
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            dd_periods = in_drawdown.astype(int).groupby(
                (in_drawdown != in_drawdown.shift()).cumsum()
            ).sum()
            max_duration = dd_periods.max()
        else:
            max_duration = 0
        
        return float(max_dd), int(max_duration)
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calcula Value at Risk (VaR).
        
        Args:
            returns: Serie de retornos
            confidence_level: Nivel de confianza (0.95 = 95%)
        
        Returns:
            VaR (pérdida máxima esperada)
        """
        if returns.empty:
            return 0.0
        
        var = returns.quantile(1 - confidence_level)
        return abs(float(var))
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calcula Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Serie de retornos
            confidence_level: Nivel de confianza
        
        Returns:
            CVaR
        """
        if returns.empty:
            return 0.0
        
        var = returns.quantile(1 - confidence_level)
        cvar = returns[returns <= var].mean()
        
        return abs(float(cvar))


# ============================================================================
# POSITION SIZER
# ============================================================================

class PositionSizer:
    """Calculadora de tamaño de posiciones."""
    
    def __init__(self, risk_per_trade: float = 0.02):
        """
        Inicializa el position sizer.
        
        Args:
            risk_per_trade: Porcentaje de riesgo por trade (default 2%)
        """
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        method: PositionSizingMethod = PositionSizingMethod.PERCENTAGE
    ) -> float:
        """
        Calcula tamaño de posición.
        
        Args:
            capital: Capital disponible
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            method: Método de cálculo
        
        Returns:
            Cantidad a operar (en unidades del asset)
        """
        if method == PositionSizingMethod.FIXED:
            return self._fixed_size(capital)
        
        elif method == PositionSizingMethod.PERCENTAGE:
            return self._percentage_size(capital, entry_price)
        
        elif method == PositionSizingMethod.KELLY:
            # Requiere win_rate y avg_win/loss - por ahora usa percentage
            return self._percentage_size(capital, entry_price)
        
        elif method == PositionSizingMethod.VOLATILITY:
            return self._volatility_based_size(capital, entry_price, stop_loss)
        
        else:
            return self._percentage_size(capital, entry_price)
    
    def _fixed_size(self, capital: float) -> float:
        """Tamaño fijo basado en capital."""
        return capital * self.risk_per_trade
    
    def _percentage_size(self, capital: float, entry_price: float) -> float:
        """Tamaño basado en porcentaje del capital."""
        position_value = capital * self.risk_per_trade
        quantity = position_value / entry_price
        return quantity
    
    def _volatility_based_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """Tamaño basado en riesgo hasta stop loss."""
        if stop_loss <= 0 or entry_price <= 0:
            return self._percentage_size(capital, entry_price)
        
        # Calcular riesgo por unidad
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Capital a arriesgar
        risk_capital = capital * self.risk_per_trade
        
        # Cantidad
        quantity = risk_capital / risk_per_unit
        
        return quantity
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calcula Kelly Criterion.
        
        Args:
            win_rate: Tasa de éxito (0-1)
            avg_win: Ganancia promedio
            avg_loss: Pérdida promedia
        
        Returns:
            Fracción Kelly (porcentaje a arriesgar)
        """
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss  # Ratio win/loss
        p = win_rate
        q = 1 - win_rate
        
        kelly = (p * b - q) / b
        
        # Kelly conservador (usar 1/2 o 1/4 del Kelly completo)
        conservative_kelly = kelly * 0.5
        
        # Limitar entre 0 y 0.25 (25% máximo)
        return max(0, min(conservative_kelly, 0.25))


# ============================================================================
# RISK MANAGER (Principal)
# ============================================================================

class RiskManager:
    """
    Gestor principal de riesgo del sistema.
    
    Funcionalidades:
    - Cálculo de position sizing
    - Stop loss y take profit automáticos
    - Gestión de drawdown
    - Límites de exposición
    - Correlación entre posiciones
    - Kill-switch automático
    """
    
    def __init__(
        self,
        data_manager: Any,
        initial_capital: float = 10000.0,
        max_drawdown: float = 0.20,
        max_position_size: float = 0.10,
        risk_per_trade: float = 0.02
    ):
        """
        Inicializa el Risk Manager.
        
        Args:
            data_manager: Instancia de DataManager
            initial_capital: Capital inicial
            max_drawdown: Máximo drawdown permitido (20%)
            max_position_size: Tamaño máximo de posición por asset (10%)
            risk_per_trade: Riesgo por trade (2%)
        """
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Límites
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.max_leverage = 3.0
        self.max_open_positions = 20
        self.max_correlation = 0.7
        
        # Position sizer
        self.position_sizer = PositionSizer(risk_per_trade)
        
        # Estado
        self.current_positions: Dict[str, Dict] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_pnl: List[float] = []
        
        # Risk metrics
        self.current_drawdown = 0.0
        self.peak_equity = initial_capital
        self.risk_level = RiskLevel.SAFE
        
        # Archivo de estado
        self.state_file = Path("data/risk_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
        
        logger.info(f"RiskManager inicializado con capital: ${initial_capital:,.2f}")
    
    def _load_state(self):
        """Carga estado persistente."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_capital = state.get('current_capital', self.initial_capital)
                    self.peak_equity = state.get('peak_equity', self.initial_capital)
                    self.current_positions = state.get('positions', {})
                
                logger.info(f"Estado cargado: Capital ${self.current_capital:,.2f}")
            except Exception as e:
                logger.error(f"Error cargando estado: {e}")
    
    def _save_state(self):
        """Guarda estado persistente."""
        try:
            state = {
                'current_capital': self.current_capital,
                'peak_equity': self.peak_equity,
                'positions': self.current_positions,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        confidence: float = 0.5,
        method: PositionSizingMethod = PositionSizingMethod.VOLATILITY
    ) -> Tuple[float, float, float]:
        """
        Calcula tamaño de posición con stop loss y take profit.
        
        Args:
            symbol: Símbolo del asset
            entry_price: Precio de entrada
            confidence: Confianza de la señal (0-1)
            method: Método de position sizing
        
        Returns:
            Tuple (quantity, stop_loss, take_profit)
        """
        # Obtener volatilidad del asset
        volatility = self.data_manager.get_volatility(symbol, "1h", window=20)
        
        if volatility == 0:
            volatility = 0.02  # Default 2%
        
        # Calcular stop loss (2x ATR o 2% mínimo)
        atr_multiplier = 2.0
        stop_loss_pct = max(volatility * atr_multiplier, 0.02)
        stop_loss = entry_price * (1 - stop_loss_pct)
        
        # Calcular take profit (risk-reward ratio 2:1)
        risk_reward_ratio = 2.0
        take_profit = entry_price * (1 + stop_loss_pct * risk_reward_ratio)
        
        # Capital disponible
        available_capital = self.get_available_capital()
        
        # Ajustar por confianza
        adjusted_capital = available_capital * confidence
        
        # Calcular tamaño de posición
        quantity = self.position_sizer.calculate_position_size(
            capital=adjusted_capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            method=method
        )
        
        # Aplicar límites
        max_position_value = self.current_capital * self.max_position_size
        max_quantity = max_position_value / entry_price
        
        quantity = min(quantity, max_quantity)
        
        logger.info(
            f"Position sizing para {symbol}: "
            f"Qty={quantity:.4f}, SL=${stop_loss:.2f}, TP=${take_profit:.2f}"
        )
        
        return quantity, stop_loss, take_profit
    
    def validate_signal(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Tuple[bool, str]:
        """
        Valida una señal antes de ejecución.
        
        Args:
            symbol: Símbolo del asset
            side: BUY o SELL
            quantity: Cantidad
            price: Precio
        
        Returns:
            Tuple (is_valid, reason)
        """
        # 1️⃣ Verificar drawdown
        if self.current_drawdown >= self.max_drawdown:
            return False, f"Drawdown máximo alcanzado: {self.current_drawdown:.2%}"
        
        # 2️⃣ Verificar número de posiciones
        if len(self.current_positions) >= self.max_open_positions:
            return False, f"Máximo de posiciones abiertas: {self.max_open_positions}"
        
        # 3️⃣ Verificar tamaño de posición
        position_value = quantity * price
        max_allowed = self.current_capital * self.max_position_size
        
        if position_value > max_allowed:
            return False, f"Posición demasiado grande: ${position_value:.2f} > ${max_allowed:.2f}"
        
        # 4️⃣ Verificar capital disponible
        available = self.get_available_capital()
        if position_value > available:
            return False, f"Capital insuficiente: ${position_value:.2f} > ${available:.2f}"
        
        # 5️⃣ Verificar correlación (si ya hay posiciones)
        if self.current_positions:
            correlation_risk = self._check_correlation_risk(symbol)
            if correlation_risk > self.max_correlation:
                return False, f"Correlación muy alta: {correlation_risk:.2f}"
        
        # 6️⃣ Verificar nivel de riesgo
        if self.risk_level in [RiskLevel.DANGER, RiskLevel.CRITICAL]:
            return False, f"Nivel de riesgo: {self.risk_level.value}"
        
        return True, "Signal validated"
    
    def filter_signals(
        self,
        signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Filtra señales según reglas de riesgo.
        
        Args:
            signals: Dict con señales {symbol: {side, quantity, price, ...}}
        
        Returns:
            Dict con señales filtradas
        """
        filtered = {}
        
        for symbol, signal in signals.items():
            side = signal.get('side', 'BUY')
            quantity = signal.get('quantity', 0)
            price = signal.get('price', 0)
            
            # Si no hay precio, obtener del data manager
            if price == 0:
                price = self.data_manager.get_latest_price(symbol, "1m") or 0
            
            if price == 0:
                logger.warning(f"No hay precio para {symbol}, señal ignorada")
                continue
            
            # Validar
            is_valid, reason = self.validate_signal(symbol, side, quantity, price)
            
            if is_valid:
                filtered[symbol] = signal
                logger.debug(f"✅ Señal {symbol} aprobada")
            else:
                logger.warning(f"❌ Señal {symbol} rechazada: {reason}")
        
        logger.info(f"Señales filtradas: {len(filtered)}/{len(signals)}")
        
        return filtered
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Registra apertura de posición."""
        position = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'opened_at': datetime.now().isoformat(),
            'pnl': 0.0
        }
        
        self.current_positions[symbol] = position
        
        # Actualizar capital
        position_value = quantity * entry_price
        if side == 'BUY':
            self.current_capital -= position_value
        
        self._save_state()
        
        logger.info(f"Posición abierta: {side} {quantity} {symbol} @ ${entry_price:.2f}")
    
    def close_position(
        self,
        symbol: str,
        exit_price: float
    ) -> float:
        """
        Cierra posición y calcula PnL.
        
        Args:
            symbol: Símbolo del asset
            exit_price: Precio de salida
        
        Returns:
            PnL de la posición
        """
        if symbol not in self.current_positions:
            logger.warning(f"No hay posición abierta para {symbol}")
            return 0.0
        
        position = self.current_positions[symbol]
        
        # Calcular PnL
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        if position['side'] == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
        
        # Actualizar capital
        position_value = quantity * exit_price
        self.current_capital += position_value + pnl
        
        # Actualizar equity curve
        self.equity_curve.append((datetime.now(), self.current_capital))
        self.daily_pnl.append(pnl)
        
        # Actualizar peak
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        
        # Calcular drawdown
        self.current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
        
        # Actualizar risk level
        self._update_risk_level()
        
        # Eliminar posición
        del self.current_positions[symbol]
        
        self._save_state()
        
        logger.info(
            f"Posición cerrada: {symbol} @ ${exit_price:.2f} | "
            f"PnL: ${pnl:.2f} | Capital: ${self.current_capital:.2f}"
        )
        
        return pnl
    
    def _check_correlation_risk(self, new_symbol: str) -> float:
        """Verifica riesgo de correlación con posiciones existentes."""
        if not self.current_positions:
            return 0.0
        
        # Obtener símbolos actuales
        current_symbols = list(self.current_positions.keys())
        
        # Calcular correlación
        try:
            correlation_matrix = self.data_manager.get_correlation_matrix(
                symbols=current_symbols + [new_symbol],
                timeframe="1d",
                limit=30
            )
            
            if correlation_matrix.empty:
                return 0.0
            
            # Correlación máxima con posiciones existentes
            new_symbol_corr = correlation_matrix.loc[new_symbol, current_symbols]
            max_correlation = abs(new_symbol_corr).max()
            
            return float(max_correlation)
        
        except Exception as e:
            logger.error(f"Error calculando correlación: {e}")
            return 0.0
    
    def _update_risk_level(self):
        """Actualiza nivel de riesgo del sistema."""
        if self.current_drawdown >= 0.15:
            self.risk_level = RiskLevel.CRITICAL
        elif self.current_drawdown >= 0.10:
            self.risk_level = RiskLevel.DANGER
        elif self.current_drawdown >= 0.05:
            self.risk_level = RiskLevel.WARNING
        elif self.current_drawdown >= 0.03:
            self.risk_level = RiskLevel.CAUTION
        else:
            self.risk_level = RiskLevel.SAFE
    
    def get_available_capital(self) -> float:
        """Retorna capital disponible para nuevas posiciones."""
        # Capital en posiciones
        capital_in_positions = sum(
            pos['quantity'] * pos['entry_price']
            for pos in self.current_positions.values()
        )
        
        # Capital disponible
        available = self.current_capital - (capital_in_positions * 0.1)  # 10% buffer
        
        return max(0, available)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Genera reporte completo de riesgo."""
        # Calcular métricas
        equity_series = pd.Series([eq for _, eq in self.equity_curve])
        
        if not equity_series.empty:
            returns = equity_series.pct_change().dropna()
            max_dd, dd_duration = RiskMetrics.calculate_max_drawdown(equity_series)
            sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
            sortino = RiskMetrics.calculate_sortino_ratio(returns)
            var_95 = RiskMetrics.calculate_var(returns, 0.95)
        else:
            max_dd, dd_duration = 0.0, 0
            sharpe, sortino, var_95 = 0.0, 0.0, 0.0
        
        report = {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'available': self.get_available_capital(),
                'in_positions': sum(
                    pos['quantity'] * pos['entry_price']
                    for pos in self.current_positions.values()
                ),
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
            },
            'positions': {
                'open': len(self.current_positions),
                'max_allowed': self.max_open_positions,
                'details': list(self.current_positions.values())
            },
            'risk_metrics': {
                'current_drawdown': self.current_drawdown,
                'max_drawdown': max_dd,
                'drawdown_duration_days': dd_duration,
                'risk_level': self.risk_level.value,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'var_95': var_95
            },
            'limits': {
                'max_drawdown': self.max_drawdown,
                'max_position_size': self.max_position_size,
                'risk_per_trade': self.risk_per_trade,
                'max_leverage': self.max_leverage
            }
        }
        
        return report
    
    def should_trigger_killswitch(self) -> Tuple[bool, str]:
        """
        Determina si debe activarse el kill-switch.
        
        Returns:
            Tuple (should_trigger, reason)
        """
        # Drawdown crítico
        if self.current_drawdown >= self.max_drawdown * 0.9:
            return True, f"Drawdown crítico: {self.current_drawdown:.2%}"
        
        # Pérdidas consecutivas
        if len(self.daily_pnl) >= 5:
            last_5 = self.daily_pnl[-5:]
            if all(pnl < 0 for pnl in last_5):
                total_loss = sum(last_5)
                return True, f"5 días consecutivos de pérdidas: ${total_loss:.2f}"
        
        # Capital mínimo
        min_capital = self.initial_capital * 0.5
        if self.current_capital < min_capital:
            return True, f"Capital por debajo del 50%: ${self.current_capital:.2f}"
        
        return False, ""
    
    def reset(self, new_capital: Optional[float] = None):
        """Resetea el risk manager."""
        if new_capital:
            self.initial_capital = new_capital
            self.current_capital = new_capital
        else:
            self.current_capital = self.initial_capital
        
        self.peak_equity = self.current_capital
        self.current_positions = {}
        self.equity_curve = []
        self.daily_pnl = []
        self.current_drawdown = 0.0
        self.risk_level = RiskLevel.SAFE
        
        self._save_state()
        
        logger.info(f"RiskManager reseteado con capital: ${self.current_capital:,.2f}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock DataManager
    class MockDataManager:
        def get_volatility(self, symbol, timeframe, window):
            return 0.03  # 3% volatility
        
        def get_latest_price(self, symbol, timeframe):
            return 50000.0
        
        def get_correlation_matrix(self, symbols, timeframe, limit):
            return pd.DataFrame()
    
    # Crear RiskManager
    dm = MockDataManager()
    rm = RiskManager(dm, initial_capital=10000)
    
    # Calcular position size
    print("\n" + "="*60)
    print("POSITION SIZING:")
    print("="*60)
    qty, sl, tp = rm.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=50000,
        confidence=0.8
    )
    print(f"Quantity: {qty:.4f}")
    print(f"Stop Loss: ${sl:.2f}")
    print(f"Take Profit: ${tp:.2f}")
    
    # Validar señal
    print("\n" + "="*60)
    print("VALIDACIÓN DE SEÑAL:")
    print("="*60)
    is_valid, reason = rm.validate_signal("BTCUSDT", "BUY", qty, 50000)
    print(f"Válida: {is_valid}")
    print(f"Razón: {reason}")
    
    # Abrir posición
    if is_valid:
        rm.open_position("BTCUSDT", "BUY", qty, 50000, sl, tp)
    
    # Reporte de riesgo
    print("\n" + "="*60)
    print("REPORTE DE RIESGO:")
    print("="*60)
    report = rm.get_risk_report()
    print(json.dumps(report, indent=2, default=str))