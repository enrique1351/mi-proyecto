# shared/core/execution/execution_interface.py

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class OrderSide(Enum):
    """Tipo de orden."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Tipos de órdenes soportadas."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Estados de una orden."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class ExecutionMode(Enum):
    """Modo de ejecución."""
    PAPER = "paper"
    REAL = "real"
    SIMULATION = "simulation"


# ============================================================================
# KILL SWITCH (Mejorado)
# ============================================================================

class KillSwitch:
    """
    Sistema de emergencia mejorado con múltiples niveles de protección.
    """
    
    def __init__(self, state_file: str = "data/killswitch_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.active = False
        self.reason = None
        self.triggered_at = None
        self.trigger_count = 0
        self.max_auto_reset = 3  # Máximo 3 auto-resets por sesión
        
        # Cargar estado persistente
        self._load_state()
    
    def _load_state(self):
        """Carga el estado del kill-switch desde disco."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.active = data.get('active', False)
                    self.reason = data.get('reason')
                    self.triggered_at = data.get('triggered_at')
                    self.trigger_count = data.get('trigger_count', 0)
                
                if self.active:
                    logger.warning(f"⚠️  KillSwitch estaba activo desde: {self.triggered_at}")
            except Exception as e:
                logger.error(f"Error cargando estado kill-switch: {e}")
    
    def _save_state(self):
        """Guarda el estado del kill-switch en disco."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'active': self.active,
                    'reason': self.reason,
                    'triggered_at': self.triggered_at,
                    'trigger_count': self.trigger_count
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando estado kill-switch: {e}")
    
    def trigger(self, reason: str = "Unknown"):
        """
        Activa el kill-switch.
        
        Args:
            reason: Razón de la activación
        """
        if not self.active:
            self.active = True
            self.reason = reason
            self.triggered_at = datetime.utcnow().isoformat()
            self.trigger_count += 1
            
            logger.critical(f"🛑 KILL-SWITCH ACTIVADO!")
            logger.critical(f"Razón: {reason}")
            logger.critical(f"Timestamp: {self.triggered_at}")
            logger.critical(f"Trigger count: {self.trigger_count}")
            
            self._save_state()
            
            # TODO: Enviar alertas (Telegram, Email, SMS)
            self._send_alerts(reason)
    
    def reset(self, manual: bool = True):
        """
        Resetea el kill-switch.
        
        Args:
            manual: Si es reset manual (True) o automático (False)
        """
        if self.active:
            self.active = False
            reset_type = "MANUAL" if manual else "AUTOMÁTICO"
            
            logger.info(f"✅ KillSwitch reseteado ({reset_type})")
            logger.info(f"Había sido activado por: {self.reason}")
            
            self.reason = None
            self.triggered_at = None
            
            self._save_state()
    
    def is_active(self) -> bool:
        """Verifica si el kill-switch está activo."""
        return self.active
    
    def can_auto_reset(self) -> bool:
        """Verifica si se puede hacer auto-reset."""
        return self.trigger_count < self.max_auto_reset
    
    def _send_alerts(self, reason: str):
        """Envía alertas por múltiples canales."""
        # TODO: Implementar Telegram, Email, Slack
        logger.warning(f"📢 Alerta: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del kill-switch."""
        return {
            'active': self.active,
            'reason': self.reason,
            'triggered_at': self.triggered_at,
            'trigger_count': self.trigger_count,
            'can_auto_reset': self.can_auto_reset()
        }


# ============================================================================
# ORDER CLASS
# ============================================================================

class Order:
    """Representa una orden de trading."""
    
    def __init__(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",  # GTC, IOC, FOK, DAY
        client_order_id: Optional[str] = None
    ):
        self.asset = asset
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        
        # IDs
        self.client_order_id = client_order_id or self._generate_order_id()
        self.broker_order_id = None
        
        # Estado
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.average_fill_price = 0.0
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.submitted_at = None
        self.filled_at = None
        
        # Metadata
        self.strategy_name = None
        self.signal_confidence = None
        self.metadata = {}
    
    def _generate_order_id(self) -> str:
        """Genera un ID único para la orden."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"ORD_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la orden a diccionario."""
        return {
            'client_order_id': self.client_order_id,
            'broker_order_id': self.broker_order_id,
            'asset': self.asset,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'strategy_name': self.strategy_name,
            'signal_confidence': self.signal_confidence
        }


# ============================================================================
# POSITION TRACKER
# ============================================================================

class PositionTracker:
    """Rastrea posiciones abiertas."""
    
    def __init__(self):
        self.positions: Dict[str, Dict[str, Any]] = {}
    
    def update_position(self, asset: str, side: OrderSide, quantity: float, price: float):
        """Actualiza una posición."""
        if asset not in self.positions:
            self.positions[asset] = {
                'quantity': 0.0,
                'average_price': 0.0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0
            }
        
        pos = self.positions[asset]
        
        if side == OrderSide.BUY:
            # Calcular nuevo precio promedio
            total_cost = (pos['quantity'] * pos['average_price']) + (quantity * price)
            new_quantity = pos['quantity'] + quantity
            pos['average_price'] = total_cost / new_quantity if new_quantity > 0 else 0
            pos['quantity'] = new_quantity
        
        elif side == OrderSide.SELL:
            # Calcular PnL realizado
            if pos['quantity'] > 0:
                pnl = (price - pos['average_price']) * min(quantity, pos['quantity'])
                pos['realized_pnl'] += pnl
            
            pos['quantity'] -= quantity
            
            # Si la posición se cierra completamente
            if pos['quantity'] <= 0:
                pos['quantity'] = 0
                pos['average_price'] = 0
    
    def get_position(self, asset: str) -> Optional[Dict[str, Any]]:
        """Obtiene la posición de un asset."""
        return self.positions.get(asset)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene todas las posiciones."""
        return self.positions
    
    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]):
        """Calcula PnL no realizado."""
        for asset, pos in self.positions.items():
            if pos['quantity'] > 0 and asset in current_prices:
                current_price = current_prices[asset]
                pos['unrealized_pnl'] = (current_price - pos['average_price']) * pos['quantity']


# ============================================================================
# EXECUTION INTERFACE (Mejorado)
# ============================================================================

class ExecutionInterface:
    """
    Interfaz de ejecución mejorada con:
    - Multi-broker support
    - Security checks
    - Position tracking
    - Slippage modeling
    - Smart order routing
    """
    
    def __init__(
        self,
        brokers: List[Any],
        mode: ExecutionMode = ExecutionMode.PAPER,
        security_guard: Optional[Any] = None
    ):
        """
        Inicializa la interfaz de ejecución.
        
        Args:
            brokers: Lista de brokers disponibles
            mode: Modo de ejecución (paper/real)
            security_guard: Guard de seguridad (opcional)
        """
        self.brokers = {broker.name: broker for broker in brokers}
        self.mode = mode
        self.security_guard = security_guard
        
        # Componentes
        self.killswitch = KillSwitch()
        self.position_tracker = PositionTracker()
        
        # Estado
        self.order_history: List[Order] = []
        self.pending_orders: Dict[str, Order] = {}
        
        # Configuración
        self.slippage_pct = 0.001  # 0.1% slippage por defecto
        self.commission_pct = 0.001  # 0.1% comisión
        
        # Logs
        self.trade_log_file = Path("data/trades/trade_log.csv")
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExecutionInterface inicializado en modo {mode.value}")
        logger.info(f"Brokers disponibles: {list(self.brokers.keys())}")
    
    def execute_signal(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        strategy_name: str = "Unknown",
        confidence: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Ejecuta una señal de trading.
        
        Args:
            asset: Símbolo del asset
            side: BUY o SELL
            quantity: Cantidad a operar
            strategy_name: Nombre de la estrategia
            confidence: Confianza de la señal (0-1)
            order_type: Tipo de orden
            limit_price: Precio límite (para órdenes limit)
            stop_price: Precio stop (para órdenes stop)
        
        Returns:
            Order ejecutada o None si falla
        """
        
        # 1️⃣ Verificar kill-switch
        if self.killswitch.is_active():
            logger.warning("🛑 KillSwitch activo. Orden rechazada.")
            return None
        
        # 2️⃣ Crear orden
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        order.strategy_name = strategy_name
        order.signal_confidence = confidence
        
        # 3️⃣ Validaciones de seguridad
        if not self._validate_order(order):
            logger.error(f"❌ Orden rechazada: validación fallida")
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return None
        
        # 4️⃣ Seleccionar broker
        broker = self._select_broker(asset)
        if not broker:
            logger.error(f"❌ No hay broker disponible para {asset}")
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return None
        
        # 5️⃣ Ejecutar según modo
        try:
            if self.mode == ExecutionMode.PAPER:
                executed_order = self._execute_paper(order, broker)
            elif self.mode == ExecutionMode.REAL:
                executed_order = self._execute_real(order, broker)
            else:
                executed_order = self._execute_simulation(order, broker)
            
            # 6️⃣ Actualizar posiciones
            if executed_order and executed_order.status == OrderStatus.FILLED:
                self.position_tracker.update_position(
                    asset=executed_order.asset,
                    side=executed_order.side,
                    quantity=executed_order.filled_quantity,
                    price=executed_order.average_fill_price
                )
            
            # 7️⃣ Log trade
            self._log_trade(executed_order)
            
            # 8️⃣ Agregar al historial
            self.order_history.append(executed_order)
            
            return executed_order
        
        except Exception as e:
            logger.error(f"❌ Error ejecutando orden: {e}")
            logger.exception("Traceback:")
            
            # Activar kill-switch en caso de error crítico
            if "critical" in str(e).lower():
                self.killswitch.trigger(f"Error crítico: {e}")
            
            order.status = OrderStatus.FAILED
            self.order_history.append(order)
            return None
    
    def _validate_order(self, order: Order) -> bool:
        """Valida una orden antes de ejecutarla."""
        
        # Validación básica
        if order.quantity <= 0:
            logger.error("Cantidad debe ser positiva")
            return False
        
        # Validación de security guard
        if self.security_guard:
            if not self.security_guard.validate_order(order):
                logger.error("Security guard rechazó la orden")
                return False
        
        # Validación de límites de posición
        current_pos = self.position_tracker.get_position(order.asset)
        if current_pos and order.side == OrderSide.BUY:
            # TODO: Verificar límites de posición máxima
            pass
        
        return True
    
    def _select_broker(self, asset: str) -> Optional[Any]:
        """Selecciona el mejor broker para un asset."""
        
        # Estrategia simple: usar el primer broker disponible
        # TODO: Implementar smart routing basado en:
        # - Liquidez
        # - Comisiones
        # - Velocidad
        # - Disponibilidad
        
        for broker in self.brokers.values():
            if broker.supports_asset(asset):
                return broker
        
        return None
    
    def _execute_paper(self, order: Order, broker: Any) -> Order:
        """Ejecuta orden en modo paper."""
        
        logger.info(f"📝 [PAPER] {order.side.value.upper()} {order.quantity} {order.asset}")
        
        # Simular ejecución
        order.submitted_at = datetime.utcnow()
        order.status = OrderStatus.SUBMITTED
        
        # Obtener precio actual (simulado)
        current_price = self._get_simulated_price(order.asset, broker)
        
        # Aplicar slippage
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)
        
        # Simular fill
        order.filled_at = datetime.utcnow()
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        
        logger.info(f"✅ [PAPER] Orden ejecutada a ${fill_price:.4f}")
        
        return order
    
    def _execute_real(self, order: Order, broker: Any) -> Order:
        """Ejecuta orden en modo real."""
        
        logger.info(f"⚡ [REAL] {order.side.value.upper()} {order.quantity} {order.asset}")
        
        try:
            # Enviar orden al broker
            order.submitted_at = datetime.utcnow()
            order.status = OrderStatus.SUBMITTED
            
            broker_response = broker.place_order(
                symbol=order.asset,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                limit_price=order.limit_price,
                stop_price=order.stop_price
            )
            
            # Actualizar orden con respuesta del broker
            order.broker_order_id = broker_response.get('order_id')
            order.filled_at = datetime.utcnow()
            order.status = OrderStatus.FILLED
            order.filled_quantity = broker_response.get('filled_quantity', order.quantity)
            order.average_fill_price = broker_response.get('average_price', 0.0)
            
            logger.info(f"✅ [REAL] Orden ejecutada a ${order.average_fill_price:.4f}")
            
            return order
        
        except Exception as e:
            logger.error(f"❌ Error en ejecución real: {e}")
            order.status = OrderStatus.FAILED
            
            # Activar kill-switch si es error crítico
            if "insufficient" in str(e).lower() or "connection" in str(e).lower():
                self.killswitch.trigger(f"Error broker: {e}")
            
            raise
    
    def _execute_simulation(self, order: Order, broker: Any) -> Order:
        """Ejecuta orden en modo simulación (para backtesting)."""
        # Similar a paper pero con datos históricos exactos
        return self._execute_paper(order, broker)
    
    def _get_simulated_price(self, asset: str, broker: Any) -> float:
        """Obtiene precio simulado del asset."""
        try:
            return broker.get_current_price(asset)
        except:
            # Fallback: precio por defecto
            return 100.0  # TODO: Mejorar con data real
    
    def _log_trade(self, order: Order):
        """Registra el trade en archivo CSV."""
        
        if not order:
            return
        
        # Crear DataFrame
        trade_data = {
            'timestamp': [order.filled_at or order.created_at],
            'client_order_id': [order.client_order_id],
            'broker_order_id': [order.broker_order_id],
            'asset': [order.asset],
            'side': [order.side.value],
            'quantity': [order.quantity],
            'filled_quantity': [order.filled_quantity],
            'price': [order.average_fill_price],
            'order_type': [order.order_type.value],
            'status': [order.status.value],
            'strategy': [order.strategy_name],
            'confidence': [order.signal_confidence],
            'mode': [self.mode.value]
        }
        
        df = pd.DataFrame(trade_data)
        
        # Escribir en CSV (append)
        if not self.trade_log_file.exists():
            df.to_csv(self.trade_log_file, index=False)
        else:
            df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
    
    def execute_signals(self, signals: Dict[str, Dict[str, Any]]) -> List[Order]:
        """
        Ejecuta múltiples señales.
        
        Args:
            signals: Dict con formato {asset: {side, quantity, strategy, confidence}}
        
        Returns:
            Lista de órdenes ejecutadas
        """
        executed_orders = []
        
        for asset, signal_data in signals.items():
            order = self.execute_signal(
                asset=asset,
                side=OrderSide(signal_data.get('side', 'buy')),
                quantity=signal_data.get('quantity', 0),
                strategy_name=signal_data.get('strategy', 'Unknown'),
                confidence=signal_data.get('confidence', 0.0)
            )
            
            if order:
                executed_orders.append(order)
        
        return executed_orders
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Retorna posiciones abiertas."""
        return {
            asset: pos for asset, pos in self.position_tracker.get_all_positions().items()
            if pos['quantity'] > 0
        }
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna historial de órdenes."""
        return [order.to_dict() for order in self.order_history[-limit:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de ejecución."""
        total_orders = len(self.order_history)
        filled_orders = sum(1 for o in self.order_history if o.status == OrderStatus.FILLED)
        rejected_orders = sum(1 for o in self.order_history if o.status == OrderStatus.REJECTED)
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'rejected_orders': rejected_orders,
            'success_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'killswitch_status': self.killswitch.get_status(),
            'open_positions': len(self.get_open_positions())
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Mock broker para testing
    class MockBroker:
        def __init__(self, name: str):
            self.name = name
        
        def supports_asset(self, asset: str) -> bool:
            return True
        
        def get_current_price(self, asset: str) -> float:
            return 50000.0  # Mock price
        
        def place_order(self, **kwargs):
            return {
                'order_id': 'MOCK_12345',
                'filled_quantity': kwargs.get('quantity'),
                'average_price': 50000.0
            }
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear execution interface
    mock_broker = MockBroker("MockBroker")
    exec_interface = ExecutionInterface(
        brokers=[mock_broker],
        mode=ExecutionMode.PAPER
    )
    
    # Ejecutar señal
    order = exec_interface.execute_signal(
        asset="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.1,
        strategy_name="TrendFollowing",
        confidence=0.85
    )
    
    print(f"\n{'='*60}")
    print("ORDEN EJECUTADA:")
    print(json.dumps(order.to_dict(), indent=2))
    print(f"{'='*60}\n")
    
    # Ver estadísticas
    stats = exec_interface.get_statistics()
    print("ESTADÍSTICAS:")
    print(json.dumps(stats, indent=2))