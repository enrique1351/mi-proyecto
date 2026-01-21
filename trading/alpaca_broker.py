"""
Alpaca Broker Integration
==========================

Integration with Alpaca Markets API for stocks and bonds trading.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Alpaca Markets broker integration.
    
    Supports:
    - US stocks
    - ETFs
    - Paper and live trading
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets"
    ):
        """
        Initialize Alpaca broker.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: API base URL (paper or live)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        # Initialize Alpaca client
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                api_key,
                api_secret,
                base_url,
                api_version='v2'
            )
            logger.info("✅ Alpaca broker initialized")
            
            # Verify account
            account = self.api.get_account()
            logger.info(f"Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
            
        except ImportError:
            logger.error("alpaca-trade-api package not installed. Install with: pip install alpaca-trade-api")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca broker: {e}")
            raise
    
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dict with account details
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'status': account.status,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'pattern_day_trader': account.pattern_day_trader,
                'daytrade_count': account.daytrade_count
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of positions
        """
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str = 'buy',
        order_type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            qty: Number of shares (optional if notional provided)
            notional: Dollar amount (optional if qty provided)
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order details dict or None if failed
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                notional=notional,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            logger.info(f"✅ Order placed: {side.upper()} {qty or notional} {symbol}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty) if order.qty else None,
                'notional': float(order.notional) if order.notional else None,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at,
                'updated_at': order.updated_at
            }
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"✅ Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_orders(self, status: str = 'open') -> List[Dict]:
        """
        Get orders.
        
        Args:
            status: Order status ('open', 'closed', 'all')
            
        Returns:
            List of orders
        """
        try:
            orders = self.api.list_orders(status=status)
            return [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty) if order.qty else None,
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'created_at': order.created_at
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Day',
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get historical price bars.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day', etc.)
            start: Start date (ISO format)
            end: End date (ISO format)
            limit: Max number of bars
            
        Returns:
            List of OHLCV bars
        """
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                limit=limit
            ).df
            
            return bars.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get bars: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if closed successfully
        """
        try:
            self.api.close_position(symbol)
            logger.info(f"✅ Position {symbol} closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all positions.
        
        Returns:
            True if all closed successfully
        """
        try:
            self.api.close_all_positions()
            logger.info("✅ All positions closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
