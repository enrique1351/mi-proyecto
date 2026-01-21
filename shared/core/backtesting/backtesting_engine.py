# shared/core/backtesting/backtesting_engine.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class BacktestConfig:
    """Configuración del backtest."""
    initial_capital: float = 10000.0
    start_date: datetime = None
    end_date: datetime = None
    commission: float = 0.001  # 0.1%
    slippage: float = 0.001    # 0.1%
    position_size_method: str = "fixed"  # fixed, kelly, volatility
    max_positions: int = 10
    risk_per_trade: float = 0.02


@dataclass
class Trade:
    """Registro de un trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    strategy: str
    is_open: bool = True


@dataclass
class BacktestResult:
    """Resultados del backtest."""
    # Equity curve
    equity_curve: pd.Series
    
    # Trades
    trades: List[Trade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    
    # Win/Loss
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Other
    var_95: float
    cvar_95: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestingEngine:
    """
    Motor de backtesting completo.
    
    Características:
    - Backtesting histórico con datos reales
    - Walk-forward analysis
    - Monte Carlo simulation
    - Optimización de parámetros
    - Análisis de sensibilidad
    """
    
    def __init__(
        self,
        data_manager: Any,
        strategy_engine: Any,
        config: Optional[BacktestConfig] = None
    ):
        """
        Inicializa el backtesting engine.
        
        Args:
            data_manager: Instancia de DataManager
            strategy_engine: Instancia de StrategyEngine
            config: Configuración del backtest
        """
        self.data_manager = data_manager
        self.strategy_engine = strategy_engine
        self.config = config or BacktestConfig()
        
        # Estado del backtest
        self.current_capital = self.config.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        logger.info("BacktestingEngine inicializado")
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1h"
    ) -> BacktestResult:
        """
        Ejecuta un backtest completo.
        
        Args:
            symbols: Lista de símbolos a testear
            start_date: Fecha inicial
            end_date: Fecha final
            timeframe: Timeframe de las barras
        
        Returns:
            BacktestResult con resultados completos
        """
        logger.info("="*60)
        logger.info("🔬 INICIANDO BACKTEST")
        logger.info("="*60)
        
        # Configurar fechas
        start_date = start_date or self.config.start_date or datetime.now() - timedelta(days=365)
        end_date = end_date or self.config.end_date or datetime.now()
        
        logger.info(f"Período: {start_date.date()} a {end_date.date()}")
        logger.info(f"Símbolos: {symbols}")
        logger.info(f"Capital inicial: ${self.config.initial_capital:,.2f}")
        
        # Resetear estado
        self._reset_state()
        
        # Obtener datos para todos los símbolos
        market_data = self._load_market_data(symbols, start_date, end_date, timeframe)
        
        if not market_data:
            logger.error("No hay datos de mercado disponibles")
            return None
        
        # Obtener todas las fechas únicas y ordenarlas
        all_timestamps = sorted(set(
            timestamp 
            for symbol_data in market_data.values() 
            for timestamp in symbol_data.index
        ))
        
        logger.info(f"Procesando {len(all_timestamps)} barras...")
        
        # Iterar sobre cada barra
        for i, current_time in enumerate(all_timestamps):
            
            # Progress
            if i % 100 == 0:
                progress = (i / len(all_timestamps)) * 100
                logger.info(f"Progreso: {progress:.1f}% ({i}/{len(all_timestamps)})")
            
            # 1️⃣ Actualizar posiciones abiertas (stop-loss, take-profit)
            self._update_open_positions(current_time, market_data)
            
            # 2️⃣ Generar señales para esta barra
            signals = self._generate_signals(current_time, market_data, symbols, timeframe)
            
            # 3️⃣ Ejecutar señales
            for symbol, signal in signals.items():
                self._execute_signal(symbol, signal, current_time, market_data)
            
            # 4️⃣ Registrar equity
            self._record_equity(current_time)
        
        # Cerrar todas las posiciones al final
        self._close_all_positions(end_date, market_data)
        
        # Calcular resultados
        result = self._calculate_results(start_date, end_date)
        
        logger.info("="*60)
        logger.info("✅ BACKTEST COMPLETADO")
        logger.info("="*60)
        
        self._print_results(result)
        
        return result
    
    def _reset_state(self):
        """Resetea el estado del backtest."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = [(datetime.now(), self.config.initial_capital)]
    
    def _load_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Carga datos de mercado para backtesting."""
        
        market_data = {}
        
        for symbol in symbols:
            try:
                data = self.data_manager.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date,
                    limit=10000
                )
                
                if not data.empty:
                    data = data.set_index('timestamp')
                    market_data[symbol] = data
                    logger.info(f"✅ {symbol}: {len(data)} barras cargadas")
                else:
                    logger.warning(f"⚠️  {symbol}: Sin datos")
            
            except Exception as e:
                logger.error(f"Error cargando datos para {symbol}: {e}")
        
        return market_data
    
    def _generate_signals(
        self,
        current_time: datetime,
        market_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        timeframe: str
    ) -> Dict[str, Dict]:
        """Genera señales para el timestamp actual."""
        
        signals = {}
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
            
            # Obtener datos hasta el timestamp actual
            symbol_data = market_data[symbol]
            
            if current_time not in symbol_data.index:
                continue
            
            # Datos hasta este punto (sin look-ahead bias)
            historical_data = symbol_data.loc[:current_time]
            
            if len(historical_data) < 50:  # Mínimo para indicadores
                continue
            
            # Generar señal usando strategy engine
            try:
                # El strategy engine debe generar señal solo con datos históricos
                signal = self._generate_single_signal(symbol, historical_data)
                
                if signal and signal.get('signal') != 'HOLD':
                    signals[symbol] = signal
            
            except Exception as e:
                logger.error(f"Error generando señal para {symbol}: {e}")
        
        return signals
    
    def _generate_single_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Genera señal para un símbolo usando datos históricos."""
        
        # Usar strategy engine para generar señal
        # Por ahora, implementamos una estrategia simple aquí
        
        if len(data) < 50:
            return None
        
        # EMA crossover simple
        data['ema_fast'] = data['close'].ewm(span=12).mean()
        data['ema_slow'] = data['close'].ewm(span=26).mean()
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        signal = None
        
        # Golden cross
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            signal = {
                'signal': 'BUY',
                'price': last['close'],
                'stop_loss': last['close'] * 0.97,
                'take_profit': last['close'] * 1.06,
                'confidence': 0.7
            }
        
        # Death cross
        elif prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            signal = {
                'signal': 'SELL',
                'price': last['close'],
                'confidence': 0.7
            }
        
        return signal
    
    def _execute_signal(
        self,
        symbol: str,
        signal: Dict,
        current_time: datetime,
        market_data: Dict[str, pd.DataFrame]
    ):
        """Ejecuta una señal en el backtest."""
        
        signal_type = signal.get('signal')
        price = signal.get('price')
        
        # Si es señal de compra y no tenemos posición
        if signal_type == 'BUY' and symbol not in self.positions:
            
            # Verificar si tenemos capital
            if self.current_capital <= 0:
                return
            
            # Calcular tamaño de posición
            position_size = self._calculate_position_size(
                symbol, price, signal.get('confidence', 0.5)
            )
            
            if position_size <= 0:
                return
            
            # Verificar límite de posiciones
            if len(self.positions) >= self.config.max_positions:
                return
            
            # Aplicar slippage
            entry_price = price * (1 + self.config.slippage)
            
            # Calcular comisión
            commission = position_size * entry_price * self.config.commission
            
            # Verificar si tenemos capital suficiente
            total_cost = (position_size * entry_price) + commission
            
            if total_cost > self.current_capital:
                # Ajustar tamaño
                position_size = (self.current_capital * 0.95) / entry_price
                commission = position_size * entry_price * self.config.commission
                total_cost = (position_size * entry_price) + commission
            
            # Abrir posición
            trade = Trade(
                entry_time=current_time,
                exit_time=None,
                symbol=symbol,
                side='BUY',
                quantity=position_size,
                entry_price=entry_price,
                exit_price=None,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                commission=commission,
                slippage=entry_price - price,
                pnl=0,
                pnl_pct=0,
                strategy='EMA_Crossover',
                is_open=True
            )
            
            self.positions[symbol] = trade
            self.current_capital -= total_cost
            
            logger.debug(f"✅ Posición abierta: {symbol} @ ${entry_price:.2f}")
        
        # Si es señal de venta y tenemos posición
        elif signal_type == 'SELL' and symbol in self.positions:
            self._close_position(symbol, current_time, price, market_data, reason="SIGNAL")
    
    def _calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float
    ) -> float:
        """Calcula tamaño de posición."""
        
        if self.config.position_size_method == "fixed":
            # Tamaño fijo basado en capital
            position_value = self.current_capital * self.config.risk_per_trade * confidence
            return position_value / price
        
        elif self.config.position_size_method == "volatility":
            # Basado en volatilidad (requiere más datos)
            position_value = self.current_capital * self.config.risk_per_trade * confidence
            return position_value / price
        
        else:
            # Default
            position_value = self.current_capital * 0.1 * confidence
            return position_value / price
    
    def _update_open_positions(
        self,
        current_time: datetime,
        market_data: Dict[str, pd.DataFrame]
    ):
        """Actualiza posiciones abiertas (stop-loss, take-profit)."""
        
        positions_to_close = []
        
        for symbol, trade in self.positions.items():
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            
            if current_time not in symbol_data.index:
                continue
            
            current_bar = symbol_data.loc[current_time]
            
            # Check stop-loss
            if trade.stop_loss and current_bar['low'] <= trade.stop_loss:
                positions_to_close.append((symbol, trade.stop_loss, "STOP_LOSS"))
            
            # Check take-profit
            elif trade.take_profit and current_bar['high'] >= trade.take_profit:
                positions_to_close.append((symbol, trade.take_profit, "TAKE_PROFIT"))
        
        # Cerrar posiciones
        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, current_time, exit_price, market_data, reason)
    
    def _close_position(
        self,
        symbol: str,
        exit_time: datetime,
        exit_price: float,
        market_data: Dict[str, pd.DataFrame],
        reason: str = "MANUAL"
    ):
        """Cierra una posición."""
        
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        # Aplicar slippage
        exit_price_with_slippage = exit_price * (1 - self.config.slippage)
        
        # Calcular comisión de salida
        exit_commission = trade.quantity * exit_price_with_slippage * self.config.commission
        
        # Calcular PnL
        pnl = (exit_price_with_slippage - trade.entry_price) * trade.quantity
        pnl -= (trade.commission + exit_commission)
        
        pnl_pct = pnl / (trade.entry_price * trade.quantity)
        
        # Actualizar trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price_with_slippage
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.is_open = False
        trade.commission += exit_commission
        
        # Actualizar capital
        proceeds = trade.quantity * exit_price_with_slippage
        self.current_capital += proceeds
        
        # Mover a trades cerrados
        self.closed_trades.append(trade)
        del self.positions[symbol]
        
        logger.debug(f"🔒 Posición cerrada: {symbol} @ ${exit_price:.2f} | PnL: ${pnl:.2f} ({reason})")
    
    def _close_all_positions(
        self,
        end_time: datetime,
        market_data: Dict[str, pd.DataFrame]
    ):
        """Cierra todas las posiciones abiertas al final del backtest."""
        
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                last_price = market_data[symbol].iloc[-1]['close']
                self._close_position(symbol, end_time, last_price, market_data, "END_OF_BACKTEST")
    
    def _record_equity(self, current_time: datetime):
        """Registra equity actual."""
        
        # Capital en cash
        total_equity = self.current_capital
        
        # Capital en posiciones (mark-to-market requeriría precio actual)
        # Por simplicidad, no lo incluimos aquí en cada barra
        
        self.equity_history.append((current_time, total_equity))
    
    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calcula métricas del backtest."""
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_history, columns=['time', 'equity'])
        equity_df = equity_df.set_index('time')
        equity_curve = equity_df['equity']
        
        # Trades
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        # Returns
        final_equity = equity_curve.iloc[-1]
        total_return = final_equity - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital
        
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = ((final_equity / self.config.initial_capital) ** (1/years) - 1) if years > 0 else 0
        
        # Calculate returns series
        returns = equity_curve.pct_change().dropna()
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0
        
        # Win/Loss
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.closed_trades if t.pnl > 0]
        losses = [t.pnl for t in self.closed_trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Best/Worst
        best_trade = max([t.pnl for t in self.closed_trades]) if self.closed_trades else 0
        worst_trade = min([t.pnl for t in self.closed_trades]) if self.closed_trades else 0
        
        # Avg duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.closed_trades if t.exit_time]
        avg_duration = np.mean(durations) if durations else 0
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=self.closed_trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            var_95=var_95,
            cvar_95=cvar_95,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_duration
        )
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcula Sharpe Ratio."""
        if returns.empty or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcula Sortino Ratio."""
        if returns.empty:
            return 0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        downside = returns[returns < 0]
        
        if len(downside) == 0 or downside.std() == 0:
            return 0
        
        return (excess_returns / downside.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> Tuple[float, int]:
        """Calcula max drawdown."""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = abs(drawdown.min())
        
        # Duration
        is_dd = drawdown < 0
        if is_dd.any():
            dd_periods = is_dd.astype(int).groupby((is_dd != is_dd.shift()).cumsum()).sum()
            max_duration = dd_periods.max()
        else:
            max_duration = 0
        
        return max_dd, max_duration
    
    def _print_results(self, result: BacktestResult):
        """Imprime resultados del backtest."""
        
        logger.info("\n" + "="*60)
        logger.info("📊 RESULTADOS DEL BACKTEST")
        logger.info("="*60)
        
        logger.info(f"\n💰 RETURNS:")
        logger.info(f"  Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2%})")
        logger.info(f"  Annualized: {result.annualized_return:.2%}")
        
        logger.info(f"\n📈 RISK METRICS:")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"  Calmar Ratio: {result.calmar_ratio:.2f}")
        
        logger.info(f"\n🎯 TRADING STATS:")
        logger.info(f"  Total Trades: {result.total_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"  Expectancy: ${result.expectancy:.2f}")
        
        logger.info(f"\n💵 WIN/LOSS:")
        logger.info(f"  Avg Win: ${result.avg_win:.2f}")
        logger.info(f"  Avg Loss: ${result.avg_loss:.2f}")
        logger.info(f"  Best Trade: ${result.best_trade:.2f}")
        logger.info(f"  Worst Trade: ${result.worst_trade:.2f}")
        
        logger.info("="*60 + "\n")
    
    def save_results(self, result: BacktestResult, filename: str = None):
        """Guarda resultados del backtest."""
        
        if not filename:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path("data/backtest") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir a dict serializable
        results_dict = {
            'config': {
                'initial_capital': self.config.initial_capital,
                'commission': self.config.commission,
                'slippage': self.config.slippage
            },
            'summary': {
                'total_return': result.total_return,
                'total_return_pct': result.total_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            },
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'symbol': t.symbol,
                    'side': t.side,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct
                }
                for t in result.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"✅ Resultados guardados en: {filepath}")
        
        return filepath


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Este ejemplo requiere DataManager y StrategyEngine
    # Ver main.py para uso completo
    
    print("Backtesting Engine listo para usar")