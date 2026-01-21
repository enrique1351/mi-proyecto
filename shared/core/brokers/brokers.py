# shared/core/brokers/brokers.py

import os
import time
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class BrokerType(Enum):
    """Tipos de brokers soportados."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    MOCK = "mock"


class OrderType(Enum):
    """Tipos de órdenes."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


# ============================================================================
# BASE BROKER (Abstract)
# ============================================================================

class BaseBroker(ABC):
    """
    Clase base abstracta para todos los brokers.
    Define la interfaz común que todos los brokers deben implementar.
    """
    
    def __init__(self, name: str, broker_type: BrokerType, credential_vault: Any = None):
        """
        Inicializa el broker base.
        
        Args:
            name: Nombre del broker
            broker_type: Tipo de broker
            credential_vault: Vault de credenciales (opcional)
        """
        self.name = name
        self.broker_type = broker_type
        self.credential_vault = credential_vault
        
        # Estado
        self.connected = False
        self.last_request_time = 0
        self.rate_limit_per_second = 10  # Por defecto
        
        # Estadísticas
        self.total_requests = 0
        self.failed_requests = 0
        self.total_orders = 0
        self.successful_orders = 0
        
        # Asset classes soportadas
        self.supported_asset_classes = []
        
        logger.info(f"Broker {name} inicializado")
    
    @abstractmethod
    def connect(self):
        """Conecta al broker y autentica."""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """
        Obtiene balance de la cuenta.
        
        Returns:
            Dict con formato: {'USDT': 10000.0, 'BTC': 0.5, ...}
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Obtiene el precio actual de un símbolo.
        
        Args:
            symbol: Símbolo del asset (ej: 'BTCUSDT')
        
        Returns:
            Precio actual
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Coloca una orden en el broker.
        
        Args:
            symbol: Símbolo del asset
            side: 'BUY' o 'SELL'
            quantity: Cantidad a operar
            order_type: Tipo de orden
            limit_price: Precio límite (para órdenes LIMIT)
            stop_price: Precio stop (para órdenes STOP)
        
        Returns:
            Dict con información de la orden ejecutada
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden.
        
        Args:
            order_id: ID de la orden a cancelar
        
        Returns:
            True si se canceló exitosamente
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de una orden.
        
        Args:
            order_id: ID de la orden
        
        Returns:
            Dict con información del estado
        """
        pass
    
    def supports_asset(self, asset: str) -> bool:
        """
        Verifica si el broker soporta un asset.
        
        Args:
            asset: Símbolo del asset
        
        Returns:
            True si lo soporta
        """
        # Implementación por defecto - override en subclases
        return True
    
    def _rate_limit(self):
        """Implementa rate limiting para evitar baneos."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit_per_second
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _handle_error(self, error: Exception, operation: str):
        """Manejo centralizado de errores."""
        self.failed_requests += 1
        logger.error(f"[{self.name}] Error en {operation}: {error}")
        logger.exception("Traceback completo:")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del broker."""
        return {
            'name': self.name,
            'type': self.broker_type.value,
            'connected': self.connected,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (
                (self.total_requests - self.failed_requests) / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders
        }


# ============================================================================
# BINANCE BROKER
# ============================================================================

class BinanceBroker(BaseBroker):
    """Broker para Binance."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 credential_vault: Any = None):
        super().__init__("Binance", BrokerType.BINANCE, credential_vault)
        
        self.supported_asset_classes = ["crypto"]
        self.rate_limit_per_second = 20  # Binance permite ~1200/min
        
        # Obtener credenciales
        if credential_vault:
            self.api_key = credential_vault.get_credential("BINANCE_API_KEY")
            self.api_secret = credential_vault.get_credential("BINANCE_API_SECRET")
        else:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials no encontradas")
        
        self.client = None
        self.connect()
    
    def connect(self):
        """Conecta a Binance."""
        try:
            from binance.client import Client
            self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
            
            # Test connection
            self.client.get_account()
            self.connected = True
            logger.info("✅ Binance conectado exitosamente")
        
        except Exception as e:
            self.connected = False
            logger.error(f"❌ Error conectando a Binance: {e}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """Obtiene balance de Binance."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            account_info = self.client.get_account()
            balances = {}
            
            for balance in account_info['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    balances[asset] = total
            
            return balances
        
        except Exception as e:
            self._handle_error(e, "get_balance")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Obtiene precio actual de Binance."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        
        except Exception as e:
            self._handle_error(e, f"get_current_price({symbol})")
            return 0.0
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Coloca orden en Binance."""
        self._rate_limit()
        self.total_requests += 1
        self.total_orders += 1
        
        try:
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if order_type == "LIMIT" and limit_price:
                order_params['timeInForce'] = 'GTC'
                order_params['price'] = limit_price
            
            if order_type == "STOP_LOSS" and stop_price:
                order_params['stopPrice'] = stop_price
            
            response = self.client.create_order(**order_params)
            
            self.successful_orders += 1
            logger.info(f"✅ [Binance] Orden ejecutada: {response['orderId']}")
            
            return {
                'order_id': str(response['orderId']),
                'symbol': response['symbol'],
                'side': response['side'],
                'quantity': float(response['executedQty']),
                'filled_quantity': float(response['executedQty']),
                'average_price': float(response.get('price', 0)),
                'status': response['status'],
                'timestamp': response['transactTime']
            }
        
        except Exception as e:
            self._handle_error(e, f"place_order({symbol}, {side})")
            raise
    
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancela orden en Binance."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            if not symbol:
                logger.error("Symbol requerido para cancelar orden en Binance")
                return False
            
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"✅ [Binance] Orden {order_id} cancelada")
            return True
        
        except Exception as e:
            self._handle_error(e, f"cancel_order({order_id})")
            return False
    
    def get_order_status(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """Obtiene estado de orden en Binance."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            if not symbol:
                return {}
            
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            
            return {
                'order_id': str(order['orderId']),
                'status': order['status'],
                'filled_quantity': float(order['executedQty']),
                'remaining_quantity': float(order['origQty']) - float(order['executedQty'])
            }
        
        except Exception as e:
            self._handle_error(e, f"get_order_status({order_id})")
            return {}


# ============================================================================
# COINBASE BROKER
# ============================================================================

class CoinbaseBroker(BaseBroker):
    """Broker para Coinbase."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 passphrase: Optional[str] = None, credential_vault: Any = None):
        super().__init__("Coinbase", BrokerType.COINBASE, credential_vault)
        
        self.supported_asset_classes = ["crypto"]
        self.rate_limit_per_second = 3  # Coinbase es más restrictivo
        
        # Obtener credenciales
        if credential_vault:
            self.api_key = credential_vault.get_credential("COINBASE_API_KEY")
            self.api_secret = credential_vault.get_credential("COINBASE_API_SECRET")
            self.passphrase = credential_vault.get_credential("COINBASE_PASSPHRASE")
        else:
            self.api_key = api_key or os.getenv("COINBASE_API_KEY")
            self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")
            self.passphrase = passphrase or os.getenv("COINBASE_PASSPHRASE")
        
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError("Coinbase API credentials incompletas")
        
        self.client = None
        self.connect()
    
    def connect(self):
        """Conecta a Coinbase."""
        try:
            # Nota: Usar coinbasepro o cbpro para API real
            # from coinbase.wallet.client import Client
            # self.client = Client(self.api_key, self.api_secret)
            
            # Por ahora, mock connection
            self.connected = True
            logger.info("✅ Coinbase conectado exitosamente")
        
        except Exception as e:
            self.connected = False
            logger.error(f"❌ Error conectando a Coinbase: {e}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """Obtiene balance de Coinbase."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            # Mock implementation
            logger.info("[Coinbase] get_balance llamado")
            return {'USD': 10000.0, 'BTC': 0.0}
        
        except Exception as e:
            self._handle_error(e, "get_balance")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Obtiene precio actual de Coinbase."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            # Mock implementation
            logger.info(f"[Coinbase] get_current_price({symbol}) llamado")
            return 50000.0
        
        except Exception as e:
            self._handle_error(e, f"get_current_price({symbol})")
            return 0.0
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Coloca orden en Coinbase."""
        self._rate_limit()
        self.total_requests += 1
        self.total_orders += 1
        
        try:
            # Mock implementation
            logger.info(f"[Coinbase] Orden: {side} {quantity} {symbol}")
            
            self.successful_orders += 1
            
            return {
                'order_id': f"CB_{int(time.time())}",
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'filled_quantity': quantity,
                'average_price': limit_price or 50000.0,
                'status': 'FILLED',
                'timestamp': int(time.time() * 1000)
            }
        
        except Exception as e:
            self._handle_error(e, f"place_order({symbol}, {side})")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancela orden en Coinbase."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            logger.info(f"[Coinbase] Cancelando orden {order_id}")
            return True
        
        except Exception as e:
            self._handle_error(e, f"cancel_order({order_id})")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Obtiene estado de orden en Coinbase."""
        self._rate_limit()
        self.total_requests += 1
        
        try:
            return {
                'order_id': order_id,
                'status': 'FILLED',
                'filled_quantity': 1.0,
                'remaining_quantity': 0.0
            }
        
        except Exception as e:
            self._handle_error(e, f"get_order_status({order_id})")
            return {}


# ============================================================================
# MOCK BROKER (Para Testing)
# ============================================================================

class MockBroker(BaseBroker):
    """Broker simulado para testing sin conexión real."""
    
    def __init__(self, initial_balance: Dict[str, float] = None):
        super().__init__("MockBroker", BrokerType.MOCK)
        
        self.supported_asset_classes = ["crypto", "stocks", "forex"]
        self.balance = initial_balance or {'USD': 10000.0}
        self.orders: Dict[str, Dict] = {}
        self.order_counter = 0
        
        self.connected = True
        logger.info("✅ MockBroker inicializado")
    
    def connect(self):
        """Mock connection."""
        self.connected = True
    
    def get_balance(self) -> Dict[str, float]:
        """Retorna balance simulado."""
        return self.balance.copy()
    
    def get_current_price(self, symbol: str) -> float:
        """Retorna precio simulado."""
        # Precios mock basados en símbolo
        mock_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'SOLUSDT': 100.0,
            'AAPL': 180.0,
            'TSLA': 250.0,
            'EURUSD': 1.10
        }
        return mock_prices.get(symbol, 100.0)
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simula colocación de orden."""
        self.total_orders += 1
        self.order_counter += 1
        
        order_id = f"MOCK_{self.order_counter}"
        current_price = self.get_current_price(symbol)
        fill_price = limit_price if limit_price else current_price
        
        # Simular slippage
        if order_type == "MARKET":
            slippage = 0.001  # 0.1%
            if side == "BUY":
                fill_price *= (1 + slippage)
            else:
                fill_price *= (1 - slippage)
        
        # Guardar orden
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'filled_quantity': quantity,
            'average_price': fill_price,
            'status': 'FILLED',
            'timestamp': int(time.time() * 1000),
            'order_type': order_type
        }
        
        self.orders[order_id] = order
        self.successful_orders += 1
        
        # Actualizar balance simulado
        base_asset = symbol[:3]  # Ej: BTC de BTCUSDT
        quote_asset = symbol[3:]  # Ej: USDT de BTCUSDT
        
        if side == "BUY":
            cost = quantity * fill_price
            if quote_asset in self.balance:
                self.balance[quote_asset] -= cost
            if base_asset not in self.balance:
                self.balance[base_asset] = 0
            self.balance[base_asset] += quantity
        else:
            if base_asset in self.balance:
                self.balance[base_asset] -= quantity
            if quote_asset not in self.balance:
                self.balance[quote_asset] = 0
            self.balance[quote_asset] += quantity * fill_price
        
        logger.info(f"✅ [MockBroker] Orden {order_id} ejecutada a ${fill_price:.2f}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Simula cancelación de orden."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            logger.info(f"✅ [MockBroker] Orden {order_id} cancelada")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Retorna estado de orden simulada."""
        return self.orders.get(order_id, {})


# ============================================================================
# BROKER FACTORY
# ============================================================================

class BrokerFactory:
    """Factory para crear brokers."""
    
    @staticmethod
    def create_broker(
        broker_type: BrokerType,
        credential_vault: Any = None,
        **kwargs
    ) -> BaseBroker:
        """
        Crea una instancia de broker.
        
        Args:
            broker_type: Tipo de broker a crear
            credential_vault: Vault de credenciales
            **kwargs: Argumentos adicionales
        
        Returns:
            Instancia de broker
        """
        if broker_type == BrokerType.BINANCE:
            return BinanceBroker(credential_vault=credential_vault, **kwargs)
        
        elif broker_type == BrokerType.COINBASE:
            return CoinbaseBroker(credential_vault=credential_vault, **kwargs)
        
        elif broker_type == BrokerType.MOCK:
            return MockBroker(**kwargs)
        
        else:
            raise ValueError(f"Broker type {broker_type} no soportado")


# ============================================================================
# BROKER MANAGER
# ============================================================================

class BrokerManager:
    """Gestor centralizado de múltiples brokers."""
    
    def __init__(self):
        self.brokers: Dict[str, BaseBroker] = {}
        logger.info("BrokerManager inicializado")
    
    def add_broker(self, broker: BaseBroker):
        """Agrega un broker al manager."""
        self.brokers[broker.name] = broker
        logger.info(f"Broker '{broker.name}' agregado")
    
    def remove_broker(self, name: str):
        """Elimina un broker."""
        if name in self.brokers:
            del self.brokers[name]
            logger.info(f"Broker '{name}' eliminado")
    
    def get_broker(self, name: str) -> Optional[BaseBroker]:
        """Obtiene un broker por nombre."""
        return self.brokers.get(name)
    
    def get_all_brokers(self) -> List[BaseBroker]:
        """Retorna todos los brokers."""
        return list(self.brokers.values())
    
    def get_total_balance(self) -> Dict[str, float]:
        """Obtiene balance total de todos los brokers."""
        total_balance = {}
        
        for broker in self.brokers.values():
            try:
                balance = broker.get_balance()
                for asset, amount in balance.items():
                    if asset not in total_balance:
                        total_balance[asset] = 0
                    total_balance[asset] += amount
            except Exception as e:
                logger.error(f"Error obteniendo balance de {broker.name}: {e}")
        
        return total_balance
    
    def get_statistics(self) -> List[Dict[str, Any]]:
        """Retorna estadísticas de todos los brokers."""
        return [broker.get_statistics() for broker in self.brokers.values()]


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear brokers usando factory
    mock_broker = BrokerFactory.create_broker(
        BrokerType.MOCK,
        initial_balance={'USD': 10000, 'BTC': 0}
    )
    
    # Usar el broker
    print("\n" + "="*60)
    print("BALANCE INICIAL:")
    print("="*60)
    print(mock_broker.get_balance())
    
    # Colocar orden
    print("\n" + "="*60)
    print("EJECUTANDO ORDEN:")
    print("="*60)
    order = mock_broker.place_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        order_type="MARKET"
    )
    print(order)
    
    # Balance final
    print("\n" + "="*60)
    print("BALANCE FINAL:")
    print("="*60)
    print(mock_broker.get_balance())
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS:")
    print("="*60)
    print(mock_broker.get_statistics())