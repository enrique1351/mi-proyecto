"""
Forex Broker Integration (Oanda)
=================================

Integration with Oanda v20 API for forex trading.
"""

import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class OandaBroker:
    """
    Oanda forex broker integration.
    
    Supports:
    - Major forex pairs (EUR/USD, GBP/USD, etc.)
    - CFDs
    - Practice and live accounts
    """
    
    def __init__(
        self,
        api_key: str,
        account_id: str,
        practice: bool = True
    ):
        """
        Initialize Oanda broker.
        
        Args:
            api_key: Oanda API key
            account_id: Oanda account ID
            practice: Use practice (demo) account
        """
        self.api_key = api_key
        self.account_id = account_id
        self.practice = practice
        
        # Set base URL
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"✅ Oanda broker initialized ({'practice' if practice else 'live'} mode)")
    
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dict with account details
        """
        try:
            import requests
            
            response = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()['account']
                return {
                    'id': data.get('id'),
                    'balance': float(data.get('balance', 0)),
                    'pl': float(data.get('pl', 0)),
                    'unrealized_pl': float(data.get('unrealizedPL', 0)),
                    'nav': float(data.get('NAV', 0)),
                    'margin_used': float(data.get('marginUsed', 0)),
                    'margin_available': float(data.get('marginAvailable', 0)),
                    'position_value': float(data.get('positionValue', 0)),
                    'open_positions': int(data.get('openPositionCount', 0)),
                    'open_trades': int(data.get('openTradeCount', 0)),
                    'open_orders': int(data.get('pendingOrderCount', 0))
                }
            else:
                logger.error(f"Failed to get account: {response.status_code}")
                return {}
                
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
            response = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/openPositions",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                positions = response.json().get('positions', [])
                return [
                    {
                        'instrument': pos['instrument'],
                        'long_units': float(pos['long']['units']),
                        'long_pl': float(pos['long']['pl']),
                        'long_unrealized_pl': float(pos['long']['unrealizedPL']),
                        'short_units': float(pos['short']['units']),
                        'short_pl': float(pos['short']['pl']),
                        'short_unrealized_pl': float(pos['short']['unrealizedPL']),
                        'pl': float(pos['pl']),
                        'unrealized_pl': float(pos['unrealizedPL'])
                    }
                    for pos in positions
                ]
            else:
                logger.error(f"Failed to get positions: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        units: int,
        order_type: str = 'MARKET',
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Currency pair (e.g., 'EUR_USD')
            units: Number of units (positive for buy, negative for sell)
            order_type: 'MARKET', 'LIMIT', 'STOP'
            take_profit_price: Optional take profit price
            stop_loss_price: Optional stop loss price
            
        Returns:
            Order details or None
        """
        try:
            order_data = {
                'order': {
                    'type': order_type,
                    'instrument': symbol,
                    'units': str(units),
                    'timeInForce': 'FOK'  # Fill or Kill
                }
            }
            
            # Add take profit
            if take_profit_price:
                order_data['order']['takeProfitOnFill'] = {
                    'price': str(take_profit_price)
                }
            
            # Add stop loss
            if stop_loss_price:
                order_data['order']['stopLossOnFill'] = {
                    'price': str(stop_loss_price)
                }
            
            response = requests.post(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            if response.status_code == 201:
                data = response.json()
                logger.info(f"✅ Order placed: {units} units of {symbol}")
                return data
            else:
                logger.error(f"Failed to place order: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        granularity: str = 'H1',
        count: int = 100
    ) -> List[Dict]:
        """
        Get historical candles.
        
        Args:
            symbol: Currency pair (e.g., 'EUR_USD')
            granularity: Candle granularity ('M1', 'M5', 'H1', 'D', etc.)
            count: Number of candles
            
        Returns:
            List of OHLCV candles
        """
        try:
            params = {
                'granularity': granularity,
                'count': count
            }
            
            response = requests.get(
                f"{self.base_url}/v3/instruments/{symbol}/candles",
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                candles = response.json().get('candles', [])
                return [
                    {
                        'time': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    }
                    for candle in candles
                    if candle.get('complete', False)
                ]
            else:
                logger.error(f"Failed to get historical data: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Currency pair
            
        Returns:
            True if closed successfully
        """
        try:
            # Close long position
            response_long = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close",
                headers=self.headers,
                json={'longUnits': 'ALL'},
                timeout=10
            )
            
            # Close short position
            response_short = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close",
                headers=self.headers,
                json={'shortUnits': 'ALL'},
                timeout=10
            )
            
            success = response_long.status_code == 200 or response_short.status_code == 200
            
            if success:
                logger.info(f"✅ Position {symbol} closed")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
