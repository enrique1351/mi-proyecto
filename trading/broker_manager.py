"""
Broker Manager
==============

Unified interface for managing multiple brokers.
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum


logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    BINANCE = "binance"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    OANDA = "oanda"  # Forex
    FOREX_COM = "forex_com"  # Forex


class BrokerManager:
    """
    Manages multiple broker connections.
    
    Provides unified interface for:
    - Account management
    - Order execution
    - Position tracking
    - Market data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize broker manager.
        
        Args:
            config: Dict with broker configurations
        """
        self.config = config
        self.brokers: Dict[BrokerType, Any] = {}
        
        # Initialize brokers
        self._initialize_brokers()
        
        logger.info(f"BrokerManager initialized with {len(self.brokers)} broker(s)")
    
    def _initialize_brokers(self):
        """Initialize all configured brokers."""
        
        # Binance (crypto)
        if 'binance' in self.config:
            try:
                from shared.core.brokers.brokers import BrokerFactory, BrokerType as CoreBrokerType
                from shared.core.security.credential_vault import CredentialVault
                
                vault = CredentialVault()
                broker = BrokerFactory.create_broker(
                    CoreBrokerType.BINANCE,
                    credential_vault=vault
                )
                self.brokers[BrokerType.BINANCE] = broker
                logger.info("✅ Binance broker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Binance: {e}")
        
        # Alpaca (stocks)
        if 'alpaca' in self.config:
            try:
                from .alpaca_broker import AlpacaBroker
                
                alpaca_config = self.config['alpaca']
                broker = AlpacaBroker(
                    api_key=alpaca_config['api_key'],
                    api_secret=alpaca_config['api_secret'],
                    base_url=alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
                )
                self.brokers[BrokerType.ALPACA] = broker
                logger.info("✅ Alpaca broker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca: {e}")
        
        # Oanda (forex)
        if 'oanda' in self.config:
            try:
                from .forex_broker import OandaBroker
                
                oanda_config = self.config['oanda']
                broker = OandaBroker(
                    api_key=oanda_config['api_key'],
                    account_id=oanda_config['account_id'],
                    practice=oanda_config.get('practice', True)
                )
                self.brokers[BrokerType.OANDA] = broker
                logger.info("✅ Oanda broker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Oanda: {e}")
        
        # Interactive Brokers
        if 'interactive_brokers' in self.config:
            logger.warning("Interactive Brokers integration not yet implemented")
        
        # Coinbase
        if 'coinbase' in self.config:
            logger.warning("Coinbase integration not yet implemented")
    
    def get_broker(self, broker_type: BrokerType) -> Optional[Any]:
        """
        Get a specific broker instance.
        
        Args:
            broker_type: Type of broker to get
            
        Returns:
            Broker instance or None
        """
        return self.brokers.get(broker_type)
    
    def get_all_accounts(self) -> Dict[BrokerType, Dict]:
        """
        Get account information from all brokers.
        
        Returns:
            Dict mapping broker type to account info
        """
        accounts = {}
        
        for broker_type, broker in self.brokers.items():
            try:
                if hasattr(broker, 'get_account'):
                    accounts[broker_type] = broker.get_account()
                elif hasattr(broker, 'get_balance'):
                    accounts[broker_type] = broker.get_balance()
            except Exception as e:
                logger.error(f"Failed to get account for {broker_type.value}: {e}")
        
        return accounts
    
    def get_all_positions(self) -> Dict[BrokerType, List[Dict]]:
        """
        Get positions from all brokers.
        
        Returns:
            Dict mapping broker type to list of positions
        """
        positions = {}
        
        for broker_type, broker in self.brokers.items():
            try:
                if hasattr(broker, 'get_positions'):
                    positions[broker_type] = broker.get_positions()
            except Exception as e:
                logger.error(f"Failed to get positions for {broker_type.value}: {e}")
        
        return positions
    
    def place_order(
        self,
        broker_type: BrokerType,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = 'market',
        **kwargs
    ) -> Optional[Dict]:
        """
        Place an order with a specific broker.
        
        NOTE: Different brokers have different parameter names:
        - Alpaca: qty, side
        - Oanda: units (positive=buy, negative=sell)
        - Binance: quantity, side
        
        This method normalizes the interface. For broker-specific parameters,
        use the broker instance directly via get_broker().
        
        Args:
            broker_type: Broker to use
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type
            **kwargs: Additional broker-specific parameters
            
        Returns:
            Order details or None
        """
        broker = self.get_broker(broker_type)
        
        if not broker:
            logger.error(f"Broker {broker_type.value} not available")
            return None
        
        try:
            if hasattr(broker, 'place_order'):
                return broker.place_order(
                    symbol=symbol,
                    side=side,
                    qty=quantity,
                    order_type=order_type,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Failed to place order with {broker_type.value}: {e}")
            return None
    
    def get_market_data(
        self,
        broker_type: BrokerType,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> List[Dict]:
        """
        Get market data from a broker.
        
        Args:
            broker_type: Broker to query
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of bars
            
        Returns:
            List of OHLCV bars
        """
        broker = self.get_broker(broker_type)
        
        if not broker:
            logger.error(f"Broker {broker_type.value} not available")
            return []
        
        try:
            if hasattr(broker, 'get_bars'):
                return broker.get_bars(symbol, timeframe, limit=limit)
            elif hasattr(broker, 'get_historical_data'):
                return broker.get_historical_data(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Failed to get market data from {broker_type.value}: {e}")
            return []
    
    def health_check(self) -> Dict[BrokerType, bool]:
        """
        Check health of all brokers.
        
        Returns:
            Dict mapping broker type to health status
        """
        health = {}
        
        for broker_type, broker in self.brokers.items():
            try:
                # Try to get account as health check
                if hasattr(broker, 'get_account'):
                    broker.get_account()
                    health[broker_type] = True
                else:
                    health[broker_type] = True
            except Exception as e:
                logger.error(f"Health check failed for {broker_type.value}: {e}")
                health[broker_type] = False
        
        return health
