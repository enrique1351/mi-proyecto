# shared/core/execution/advanced_orders.py

import logging
import time
from typing import Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AdvancedOrderType(Enum):
    """Tipos de órdenes avanzadas."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Orden oculta
    OCO = "oco"  # One-Cancels-Other
    BRACKET = "bracket"  # Entry + SL + TP
    TRAILING_STOP = "trailing_stop"
    POST_ONLY = "post_only"  # Solo maker


@dataclass
class TWAPOrder:
    """Orden TWAP - divide orden en el tiempo."""
    symbol: str
    side: str
    total_quantity: float
    duration_minutes: int
    num_slices: int = 10
    
    def execute(self, broker, execution_callback: Optional[Callable] = None):
        """Ejecuta orden TWAP."""
        slice_size = self.total_quantity / self.num_slices
        interval = (self.duration_minutes * 60) / self.num_slices
        
        logger.info(f"🕐 Ejecutando TWAP: {self.total_quantity} {self.symbol} en {self.duration_minutes}min")
        
        for i in range(self.num_slices):
            try:
                # Ejecutar slice
                order = broker.place_order(
                    symbol=self.symbol,
                    side=self.side,
                    quantity=slice_size,
                    order_type="MARKET"
                )
                
                logger.info(f"  Slice {i+1}/{self.num_slices}: {slice_size} @ Market")
                
                if execution_callback:
                    execution_callback(order)
                
                # Esperar intervalo
                if i < self.num_slices - 1:
                    time.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error en TWAP slice {i+1}: {e}")
                break
        
        logger.info("✅ TWAP completado")


@dataclass
class VWAPOrder:
    """Orden VWAP - divide según volumen."""
    symbol: str
    side: str
    total_quantity: float
    duration_minutes: int
    
    def execute(self, broker, get_volume_profile: Callable):
        """Ejecuta orden VWAP."""
        logger.info(f"📊 Ejecutando VWAP: {self.total_quantity} {self.symbol}")
        
        # Obtener perfil de volumen histórico
        volume_profile = get_volume_profile(self.symbol, self.duration_minutes)
        
        # Distribuir según volumen esperado
        for time_slice, volume_pct in volume_profile.items():
            slice_size = self.total_quantity * volume_pct
            
            try:
                broker.place_order(
                    symbol=self.symbol,
                    side=self.side,
                    quantity=slice_size,
                    order_type="MARKET"
                )
                
                logger.info(f"  Slice @ {time_slice}: {slice_size} ({volume_pct:.1%} vol)")
            
            except Exception as e:
                logger.error(f"Error en VWAP slice: {e}")


@dataclass
class IcebergOrder:
    """Orden Iceberg - oculta cantidad total."""
    symbol: str
    side: str
    total_quantity: float
    visible_quantity: float
    limit_price: float
    
    def execute(self, broker):
        """Ejecuta orden Iceberg."""
        logger.info(f"🧊 Ejecutando Iceberg: {self.total_quantity} {self.symbol} (visible: {self.visible_quantity})")
        
        remaining = self.total_quantity
        
        while remaining > 0:
            current_slice = min(self.visible_quantity, remaining)
            
            try:
                order = broker.place_order(
                    symbol=self.symbol,
                    side=self.side,
                    quantity=current_slice,
                    order_type="LIMIT",
                    limit_price=self.limit_price
                )
                
                # Esperar fill
                while order['status'] != 'FILLED':
                    time.sleep(1)
                    order = broker.get_order_status(order['order_id'])
                
                remaining -= current_slice
                logger.info(f"  Slice ejecutada. Remaining: {remaining}")
            
            except Exception as e:
                logger.error(f"Error en Iceberg: {e}")
                break


@dataclass
class OCOOrder:
    """One-Cancels-Other - dos órdenes, una cancela la otra."""
    symbol: str
    side: str
    quantity: float
    stop_price: float
    limit_price: float
    
    def execute(self, broker):
        """Ejecuta orden OCO."""
        logger.info(f"🔀 Ejecutando OCO: {self.symbol}")
        
        # Crear stop-loss
        stop_order = broker.place_order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type="STOP_LOSS",
            stop_price=self.stop_price
        )
        
        # Crear take-profit
        limit_order = broker.place_order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type="LIMIT",
            limit_price=self.limit_price
        )
        
        logger.info(f"  Stop: ${self.stop_price} | Limit: ${self.limit_price}")
        
        # Monitorear ambas
        while True:
            stop_status = broker.get_order_status(stop_order['order_id'])
            limit_status = broker.get_order_status(limit_order['order_id'])
            
            # Si alguna se ejecuta, cancelar la otra
            if stop_status['status'] == 'FILLED':
                broker.cancel_order(limit_order['order_id'])
                logger.info("✅ Stop-loss ejecutado, limit cancelado")
                break
            
            elif limit_status['status'] == 'FILLED':
                broker.cancel_order(stop_order['order_id'])
                logger.info("✅ Take-profit ejecutado, stop cancelado")
                break
            
            time.sleep(1)


@dataclass
class BracketOrder:
    """Bracket Order - Entry + SL + TP simultáneos."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    
    def execute(self, broker):
        """Ejecuta orden Bracket."""
        logger.info(f"📍 Ejecutando Bracket: {self.symbol}")
        logger.info(f"  Entry: ${self.entry_price} | SL: ${self.stop_loss} | TP: ${self.take_profit}")
        
        # 1. Entry order
        entry_order = broker.place_order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type="LIMIT",
            limit_price=self.entry_price
        )
        
        # 2. Esperar fill de entry
        while entry_order['status'] != 'FILLED':
            time.sleep(1)
            entry_order = broker.get_order_status(entry_order['order_id'])
        
        logger.info("✅ Entry ejecutado")
        
        # 3. Crear OCO con SL y TP
        opposite_side = 'SELL' if self.side == 'BUY' else 'BUY'
        
        oco = OCOOrder(
            symbol=self.symbol,
            side=opposite_side,
            quantity=self.quantity,
            stop_price=self.stop_loss,
            limit_price=self.take_profit
        )
        
        oco.execute(broker)


@dataclass
class TrailingStopOrder:
    """Trailing Stop - stop que sigue el precio."""
    symbol: str
    side: str
    quantity: float
    trail_percent: float  # % de trailing
    
    def execute(self, broker, get_current_price: Callable):
        """Ejecuta trailing stop."""
        logger.info(f"📈 Ejecutando Trailing Stop: {self.symbol} (trail: {self.trail_percent}%)")
        
        # Precio inicial
        entry_price = get_current_price(self.symbol)
        highest_price = entry_price
        stop_price = entry_price * (1 - self.trail_percent / 100)
        
        logger.info(f"  Entry: ${entry_price:.2f} | Initial Stop: ${stop_price:.2f}")
        
        while True:
            current_price = get_current_price(self.symbol)
            
            # Actualizar highest
            if current_price > highest_price:
                highest_price = current_price
                stop_price = highest_price * (1 - self.trail_percent / 100)
                logger.info(f"  🔼 New High: ${highest_price:.2f} | Stop: ${stop_price:.2f}")
            
            # Check si hit stop
            if current_price <= stop_price:
                logger.info(f"🛑 Trailing stop triggered @ ${current_price:.2f}")
                
                broker.place_order(
                    symbol=self.symbol,
                    side=self.side,
                    quantity=self.quantity,
                    order_type="MARKET"
                )
                
                break
            
            time.sleep(1)


class AdvancedOrderManager:
    """Gestor de órdenes avanzadas."""
    
    def __init__(self, broker):
        self.broker = broker
    
    def place_twap(self, symbol: str, side: str, quantity: float, duration_min: int):
        """Coloca orden TWAP."""
        order = TWAPOrder(symbol, side, quantity, duration_min)
        order.execute(self.broker)
    
    def place_vwap(self, symbol: str, side: str, quantity: float, duration_min: int):
        """Coloca orden VWAP."""
        order = VWAPOrder(symbol, side, quantity, duration_min)
        # Necesita implementar get_volume_profile
        order.execute(self.broker, self._get_volume_profile)
    
    def place_iceberg(self, symbol: str, side: str, total: float, visible: float, price: float):
        """Coloca orden Iceberg."""
        order = IcebergOrder(symbol, side, total, visible, price)
        order.execute(self.broker)
    
    def place_oco(self, symbol: str, side: str, qty: float, stop: float, limit: float):
        """Coloca orden OCO."""
        order = OCOOrder(symbol, side, qty, stop, limit)
        order.execute(self.broker)
    
    def place_bracket(self, symbol: str, side: str, qty: float, entry: float, sl: float, tp: float):
        """Coloca orden Bracket."""
        order = BracketOrder(symbol, side, qty, entry, sl, tp)
        order.execute(self.broker)
    
    def place_trailing_stop(self, symbol: str, side: str, qty: float, trail_pct: float):
        """Coloca trailing stop."""
        order = TrailingStopOrder(symbol, side, qty, trail_pct)
        order.execute(self.broker, self.broker.get_current_price)
    
    def _get_volume_profile(self, symbol: str, duration_min: int):
        """Obtiene perfil de volumen (placeholder)."""
        # En producción, obtener del historical data
        return {
            'morning': 0.3,
            'midday': 0.4,
            'afternoon': 0.3
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Advanced Order Types listo")