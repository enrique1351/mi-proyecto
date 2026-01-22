"""
CCXT Broker Integration
Unified interface for multiple cryptocurrency exchanges
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from .brokers import BaseBroker, BrokerType, OrderType

logger = logging.getLogger(__name__)


class CCXTBroker(BaseBroker):
    """
    CCXT broker integration for cryptocurrency exchanges
    Supports: Binance, Kraken, and 100+ exchanges
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        credential_vault: Any = None,
        testnet: bool = True
    ):
        """
        Initialize CCXT broker
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'kraken')
            credential_vault: Vault with API credentials
            testnet: Use testnet/sandbox (default: True)
        """
        super().__init__(f"CCXT-{exchange_id.upper()}", BrokerType.BINANCE, credential_vault)
        
        if not CCXT_AVAILABLE:
            raise ImportError(
                "CCXT not installed. Install with: pip install ccxt"
            )
        
        self.exchange_id = exchange_id.lower()
        self.testnet = testnet
        self.exchange: Optional[ccxt.Exchange] = None
        self.supported_asset_classes = ['crypto']
        
        # Exchange-specific rate limits
        self.rate_limit_per_second = 10
        
    def connect(self):
        """Connect to exchange and authenticate"""
        try:
            # Get exchange class
            if not hasattr(ccxt, self.exchange_id):
                raise ValueError(f"Exchange '{self.exchange_id}' not supported by CCXT")
            
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Get credentials
            if self.credential_vault:
                api_key = self.credential_vault.get_credential(f"{self.exchange_id}_api_key")
                api_secret = self.credential_vault.get_credential(f"{self.exchange_id}_api_secret")
            else:
                # Fallback to environment variables
                import os
                api_key = os.getenv(f"{self.exchange_id.upper()}_API_KEY")
                api_secret = os.getenv(f"{self.exchange_id.upper()}_API_SECRET")
            
            # Initialize exchange
            config = {
                'enableRateLimit': True,
                'timeout': 30000,
            }
            
            if api_key and api_secret:
                config['apiKey'] = api_key
                config['secret'] = api_secret
            
            # Set testnet if supported
            if self.testnet:
                if self.exchange_id == 'binance':
                    config['options'] = {'defaultType': 'spot'}
                    config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api',
                            'private': 'https://testnet.binance.vision/api',
                        }
                    }
            
            self.exchange = exchange_class(config)
            
            # Load markets
            self.exchange.load_markets()
            
            # Test connection
            if api_key and api_secret:
                balance = self.exchange.fetch_balance()
                self.connected = True
                logger.info(f"✅ Connected to {self.exchange_id.upper()} ({'Testnet' if self.testnet else 'Live'})")
                logger.info(f"   Markets loaded: {len(self.exchange.markets)}")
            else:
                self.connected = False
                logger.warning(f"⚠️  Connected to {self.exchange_id.upper()} (Public API only - no credentials)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.exchange_id}: {e}")
            self.connected = False
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        try:
            balance = self.exchange.fetch_balance()
            
            # Extract free and total balances
            result = {}
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and 'total' in amounts:
                    if amounts['total'] > 0:
                        result[currency] = amounts['total']
                        result[f"{currency}_free"] = amounts.get('free', 0)
                        result[f"{currency}_used"] = amounts.get('used', 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            self.failed_requests += 1
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        try:
            # Normalize symbol format (e.g., BTC/USDT)
            if '/' not in symbol:
                # Try to guess quote currency
                if symbol.endswith('USDT'):
                    symbol = f"{symbol[:-4]}/USDT"
                elif symbol.endswith('USD'):
                    symbol = f"{symbol[:-3]}/USD"
                elif symbol.endswith('BTC'):
                    symbol = f"{symbol[:-3]}/BTC"
            
            ticker = self.exchange.fetch_ticker(symbol)
            
            return ticker['last'] if ticker['last'] else ticker['close']
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            self.failed_requests += 1
            raise
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Place an order"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        if not self.connected:
            raise RuntimeError("Authentication required to place orders")
        
        try:
            # Normalize symbol
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    symbol = f"{symbol[:-4]}/USDT"
                elif symbol.endswith('USD'):
                    symbol = f"{symbol[:-3]}/USD"
            
            # Map order type
            ccxt_type = 'market'
            if order_type == OrderType.LIMIT:
                ccxt_type = 'limit'
                if price is None:
                    raise ValueError("Limit orders require a price")
            elif order_type == OrderType.STOP_LOSS:
                ccxt_type = 'stop_loss'
            
            # Place order
            order = self.exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=side.lower(),
                amount=quantity,
                price=price
            )
            
            self.total_orders += 1
            self.successful_orders += 1
            
            logger.info(f"✅ Order placed: {side} {quantity} {symbol} @ {order_type.value}")
            
            return {
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type.value,
                'status': order['status'],
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', quantity),
                'price': order.get('price'),
                'average': order.get('average'),
                'timestamp': order.get('timestamp'),
                'datetime': order.get('datetime')
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.total_orders += 1
            self.failed_requests += 1
            raise
    
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            self.failed_requests += 1
            return False
    
    def get_order_status(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """Get order status"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            
            return {
                'order_id': order['id'],
                'symbol': order['symbol'],
                'status': order['status'],
                'side': order['side'],
                'type': order['type'],
                'quantity': order['amount'],
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', 0),
                'price': order.get('price'),
                'average': order.get('average'),
                'timestamp': order.get('timestamp'),
                'datetime': order.get('datetime')
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            self.failed_requests += 1
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        try:
            balance = self.get_balance()
            
            # For spot trading, positions are just non-zero balances
            positions = []
            for currency, amount in balance.items():
                if not currency.endswith('_free') and not currency.endswith('_used'):
                    if amount > 0:
                        # Try to get current price in USDT
                        try:
                            if currency == 'USDT' or currency == 'USD':
                                price = 1.0
                            else:
                                price = self.get_current_price(f"{currency}/USDT")
                            
                            positions.append({
                                'symbol': currency,
                                'quantity': amount,
                                'side': 'long',
                                'current_price': price,
                                'market_value': amount * price
                            })
                        except:
                            # If we can't get price, still include the position
                            positions.append({
                                'symbol': currency,
                                'quantity': amount,
                                'side': 'long',
                                'current_price': None,
                                'market_value': None
                            })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            self.failed_requests += 1
            raise
    
    def get_markets(self) -> List[str]:
        """Get list of available markets"""
        if not self.exchange:
            raise RuntimeError(f"Not connected to {self.exchange_id}")
        
        try:
            return list(self.exchange.markets.keys())
        except Exception as e:
            logger.error(f"Error getting markets: {e}")
            return []
    
    def disconnect(self):
        """Disconnect from exchange"""
        self.connected = False
        self.exchange = None
        logger.info(f"Disconnected from {self.exchange_id}")
