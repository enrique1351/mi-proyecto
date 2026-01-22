"""
Alpaca Broker Integration
Supports US stocks and ETFs trading
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from .brokers import BaseBroker, BrokerType, OrderType

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker integration for US stocks and ETFs
    """
    
    def __init__(self, credential_vault: Any = None, paper_trading: bool = True):
        """
        Initialize Alpaca broker
        
        Args:
            credential_vault: Vault with API credentials
            paper_trading: Use paper trading account (default: True)
        """
        super().__init__("Alpaca", BrokerType.ALPACA, credential_vault)
        
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not installed. Install with: pip install alpaca-py"
            )
        
        self.paper_trading = paper_trading
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self.supported_asset_classes = ['stocks', 'etfs']
        
        # Rate limiting
        self.rate_limit_per_second = 200  # Alpaca allows 200 requests/minute
        
    def connect(self):
        """Connect to Alpaca and authenticate"""
        try:
            if self.credential_vault:
                api_key = self.credential_vault.get_credential("alpaca_api_key")
                api_secret = self.credential_vault.get_credential("alpaca_api_secret")
            else:
                # Fallback to environment variables
                import os
                api_key = os.getenv("ALPACA_API_KEY")
                api_secret = os.getenv("ALPACA_SECRET_KEY")
            
            if not api_key or not api_secret:
                raise ValueError("Alpaca API credentials not found")
            
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=self.paper_trading
            )
            
            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=api_secret
            )
            
            # Test connection by getting account info
            account = self.trading_client.get_account()
            self.connected = True
            
            logger.info(f"✅ Connected to Alpaca ({'Paper' if self.paper_trading else 'Live'} trading)")
            logger.info(f"   Account Status: {account.status}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alpaca: {e}")
            self.connected = False
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if not self.connected:
            raise RuntimeError("Not connected to Alpaca")
        
        try:
            account = self.trading_client.get_account()
            
            return {
                'USD': float(account.cash),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            self.failed_requests += 1
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if not self.connected:
            raise RuntimeError("Not connected to Alpaca")
        
        try:
            # Get latest trade
            from alpaca.data.requests import StockLatestTradeRequest
            
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trades = self.data_client.get_stock_latest_trade(request)
            
            if symbol in trades:
                return float(trades[symbol].price)
            else:
                raise ValueError(f"No price data for {symbol}")
                
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
        if not self.connected:
            raise RuntimeError("Not connected to Alpaca")
        
        try:
            # Convert side to Alpaca format
            alpaca_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            # Create order request based on type
            if order_type == OrderType.MARKET:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY
                )
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Limit orders require a price")
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            self.total_orders += 1
            self.successful_orders += 1
            
            logger.info(f"✅ Order placed: {side} {quantity} {symbol} @ {order_type.value}")
            
            return {
                'order_id': order.id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type.value,
                'status': order.status.value,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.total_orders += 1
            self.failed_requests += 1
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.connected:
            raise RuntimeError("Not connected to Alpaca")
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            self.failed_requests += 1
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if not self.connected:
            raise RuntimeError("Not connected to Alpaca")
        
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'status': order.status.value,
                'side': order.side.value,
                'quantity': float(order.qty),
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at,
                'updated_at': order.updated_at
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            self.failed_requests += 1
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        if not self.connected:
            raise RuntimeError("Not connected to Alpaca")
        
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'quantity': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            self.failed_requests += 1
            raise
    
    def disconnect(self):
        """Disconnect from Alpaca"""
        self.connected = False
        self.trading_client = None
        self.data_client = None
        logger.info("Disconnected from Alpaca")
