"""
================================================================================
WEBSOCKET MANAGER - Real-time Market Data Connections
================================================================================
Autor: Sistema de Trading Cuantitativo
Versión: 1.0.0
Fecha: Enero 2025

Características:
- Conexiones persistentes WebSocket
- Auto-reconnection con backoff exponencial
- Multi-exchange support (Binance, Kraken, etc.)
- Message queuing y buffering
- Thread-safe operations
- Health monitoring
- Rate limiting compliance
================================================================================
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from enum import Enum
from collections import deque
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


class ExchangeType(Enum):
    """Tipos de exchanges soportados"""
    BINANCE = "binance"
    KRAKEN = "kraken"
    MOCK = "mock"


class SubscriptionType(Enum):
    """Tipos de suscripciones disponibles"""
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    KLINE = "kline"
    DEPTH = "depth"


class ConnectionStatus(Enum):
    """Estados de conexión"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class WebSocketConnection:
    """Gestiona una conexión WebSocket individual"""
    
    def __init__(self, exchange: ExchangeType, url: str, 
                 on_message: Callable, on_error: Callable = None):
        self.exchange = exchange
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        
        self.ws = None
        self.status = ConnectionStatus.DISCONNECTED
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0  # segundos
        self.max_reconnect_delay = 60.0
        
        self.subscriptions: List[Dict] = []
        self.message_queue = deque(maxlen=10000)
        self.last_message_time = None
        self.connection_time = None
        
        self._running = False
        self._task = None
    
    async def connect(self):
        """Establece la conexión WebSocket"""
        try:
            self.status = ConnectionStatus.CONNECTING
            self.ws = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self.status = ConnectionStatus.CONNECTED
            self.connection_time = datetime.now()
            self.reconnect_attempts = 0
            print(f"✓ WebSocket conectado: {self.exchange.value}")
            
            # Resubscribir si hay suscripciones previas
            for sub in self.subscriptions:
                await self._send_subscription(sub)
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            print(f"✗ Error conectando WebSocket {self.exchange.value}: {e}")
            if self.on_error:
                self.on_error(e)
            raise
    
    async def disconnect(self):
        """Cierra la conexión WebSocket"""
        self._running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.status = ConnectionStatus.DISCONNECTED
        print(f"✓ WebSocket desconectado: {self.exchange.value}")
    
    async def subscribe(self, subscription: Dict):
        """Suscribe a un canal de datos"""
        self.subscriptions.append(subscription)
        if self.status == ConnectionStatus.CONNECTED:
            await self._send_subscription(subscription)
    
    async def _send_subscription(self, subscription: Dict):
        """Envía mensaje de suscripción al exchange"""
        try:
            message = self._format_subscription(subscription)
            await self.ws.send(json.dumps(message))
            print(f"✓ Suscrito a: {subscription}")
        except Exception as e:
            print(f"✗ Error suscribiendo: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _format_subscription(self, subscription: Dict) -> Dict:
        """Formatea mensaje de suscripción según el exchange"""
        if self.exchange == ExchangeType.BINANCE:
            return {
                "method": "SUBSCRIBE",
                "params": [subscription['channel']],
                "id": int(time.time())
            }
        else:
            return subscription
    
    async def listen(self):
        """Loop principal de escucha de mensajes"""
        self._running = True
        
        while self._running:
            try:
                if self.status != ConnectionStatus.CONNECTED:
                    await self._reconnect()
                
                # Recibir mensaje
                message = await asyncio.wait_for(
                    self.ws.recv(), 
                    timeout=30.0
                )
                
                self.last_message_time = datetime.now()
                
                # Procesar mensaje
                data = json.loads(message)
                self.message_queue.append({
                    'timestamp': self.last_message_time,
                    'data': data
                })
                
                # Callback con el mensaje
                if self.on_message:
                    await self.on_message(data)
                
            except asyncio.TimeoutError:
                print(f"⚠ Timeout en WebSocket {self.exchange.value}")
                await self._reconnect()
                
            except ConnectionClosed:
                print(f"⚠ Conexión cerrada {self.exchange.value}")
                await self._reconnect()
                
            except Exception as e:
                print(f"✗ Error en WebSocket {self.exchange.value}: {e}")
                if self.on_error:
                    self.on_error(e)
                await self._reconnect()
    
    async def _reconnect(self):
        """Reconecta con backoff exponencial"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"✗ Máximo de reintentos alcanzado para {self.exchange.value}")
            self.status = ConnectionStatus.ERROR
            return
        
        self.status = ConnectionStatus.RECONNECTING
        self.reconnect_attempts += 1
        
        # Backoff exponencial
        delay = min(
            self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
            self.max_reconnect_delay
        )
        
        print(f"⟳ Reconectando {self.exchange.value} en {delay}s (intento {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
        except Exception as e:
            print(f"✗ Fallo al reconectar: {e}")
    
    def get_health_status(self) -> Dict:
        """Retorna el estado de salud de la conexión"""
        return {
            'exchange': self.exchange.value,
            'status': self.status.value,
            'connected_since': self.connection_time.isoformat() if self.connection_time else None,
            'last_message': self.last_message_time.isoformat() if self.last_message_time else None,
            'reconnect_attempts': self.reconnect_attempts,
            'subscriptions': len(self.subscriptions),
            'queued_messages': len(self.message_queue)
        }


class WebSocketManager:
    """
    Manager principal para gestionar múltiples conexiones WebSocket
    """
    
    # URLs de los exchanges
    EXCHANGE_URLS = {
        ExchangeType.BINANCE: "wss://stream.binance.com:9443/ws",
        ExchangeType.KRAKEN: "wss://ws.kraken.com",
    }
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.loop = None
        self.thread = None
        self._running = False
        
        # Estadísticas
        self.total_messages = 0
        self.start_time = None
    
    def add_connection(self, name: str, exchange: ExchangeType,
                      url: Optional[str] = None) -> WebSocketConnection:
        """Añade una nueva conexión WebSocket"""
        if name in self.connections:
            print(f"⚠ Conexión '{name}' ya existe")
            return self.connections[name]
        
        ws_url = url or self.EXCHANGE_URLS.get(exchange)
        if not ws_url:
            raise ValueError(f"URL no disponible para {exchange.value}")
        
        connection = WebSocketConnection(
            exchange=exchange,
            url=ws_url,
            on_message=lambda msg: self._handle_message(name, msg),
            on_error=lambda err: self._handle_error(name, err)
        )
        
        self.connections[name] = connection
        print(f"✓ Conexión '{name}' añadida para {exchange.value}")
        return connection
    
    def register_callback(self, connection_name: str, callback: Callable):
        """Registra un callback para mensajes de una conexión"""
        if connection_name not in self.callbacks:
            self.callbacks[connection_name] = []
        self.callbacks[connection_name].append(callback)
    
    async def _handle_message(self, connection_name: str, message: Dict):
        """Maneja mensajes recibidos"""
        self.total_messages += 1
        
        # Ejecutar callbacks registrados
        if connection_name in self.callbacks:
            for callback in self.callbacks[connection_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    print(f"✗ Error en callback: {e}")
    
    def _handle_error(self, connection_name: str, error: Exception):
        """Maneja errores de conexión"""
        print(f"✗ Error en conexión '{connection_name}': {error}")
    
    def start(self):
        """Inicia el manager en un thread separado"""
        if self._running:
            print("⚠ WebSocketManager ya está corriendo")
            return
        
        self._running = True
        self.start_time = datetime.now()
        
        # Crear thread con event loop
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("✓ WebSocketManager iniciado")
    
    def _run_loop(self):
        """Ejecuta el event loop en el thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._run_connections())
        except Exception as e:
            print(f"✗ Error en event loop: {e}")
        finally:
            self.loop.close()
    
    async def _run_connections(self):
        """Ejecuta todas las conexiones concurrentemente"""
        # Conectar todas las conexiones
        connect_tasks = [
            conn.connect() 
            for conn in self.connections.values()
        ]
        await asyncio.gather(*connect_tasks, return_exceptions=True)
        
        # Escuchar en todas las conexiones
        listen_tasks = [
            conn.listen() 
            for conn in self.connections.values()
        ]
        await asyncio.gather(*listen_tasks, return_exceptions=True)
    
    def stop(self):
        """Detiene el manager y todas las conexiones"""
        if not self._running:
            return
        
        self._running = False
        
        # Desconectar todas las conexiones
        if self.loop:
            for conn in self.connections.values():
                asyncio.run_coroutine_threadsafe(
                    conn.disconnect(), 
                    self.loop
                )
        
        # Esperar thread
        if self.thread:
            self.thread.join(timeout=5)
        
        print("✓ WebSocketManager detenido")
    
    def subscribe(self, connection_name: str, subscription: Dict):
        """Suscribe a un canal en una conexión específica"""
        if connection_name not in self.connections:
            raise ValueError(f"Conexión '{connection_name}' no existe")
        
        conn = self.connections[connection_name]
        
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                conn.subscribe(subscription),
                self.loop
            )
        else:
            print("⚠ Manager no iniciado, suscripción guardada para cuando conecte")
            conn.subscriptions.append(subscription)
    
    def get_status(self) -> Dict:
        """Retorna el estado completo del manager"""
        return {
            'running': self._running,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'total_messages': self.total_messages,
            'total_connections': len(self.connections),
            'connections': {
                name: conn.get_health_status()
                for name, conn in self.connections.items()
            }
        }
    
    def get_connection(self, name: str) -> Optional[WebSocketConnection]:
        """Obtiene una conexión por nombre"""
        return self.connections.get(name)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

async def handle_binance_trade(message: Dict):
    """Callback para trades de Binance"""
    if 'e' in message and message['e'] == 'trade':
        print(f"TRADE: {message['s']} @ {message['p']} Vol: {message['q']}")


if __name__ == "__main__":
    # Crear manager
    manager = WebSocketManager()
    
    # Añadir conexión Binance
    binance_conn = manager.add_connection('binance_main', ExchangeType.BINANCE)
    manager.register_callback('binance_main', handle_binance_trade)
    
    # Iniciar manager
    manager.start()
    
    # Suscribir a canales
    time.sleep(2)  # Esperar conexión
    
    # Binance: Trade stream
    manager.subscribe('binance_main', {
        'channel': 'btcusdt@trade'
    })
    
    # Dejar corriendo
    try:
        while True:
            time.sleep(10)
            status = manager.get_status()
            print(f"\n📊 Status: {status['total_messages']} mensajes recibidos")
            print(f"   Uptime: {status['uptime_seconds']:.0f}s")
            
    except KeyboardInterrupt:
        print("\n⚠ Deteniendo...")
        manager.stop()
        print("✓ Sistema detenido")