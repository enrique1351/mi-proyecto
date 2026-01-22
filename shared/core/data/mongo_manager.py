"""
MongoDB Database Manager
NoSQL database support for flexible data storage and analytics
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, OperationFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

logger = logging.getLogger(__name__)


class MongoDBManager:
    """
    MongoDB database manager for trading system
    Ideal for storing unstructured data, logs, and analytics
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "trading_db",
        username: Optional[str] = None,
        password: Optional[str] = None,
        credential_vault: Optional[any] = None
    ):
        """
        Initialize MongoDB manager
        
        Args:
            host: MongoDB host
            port: MongoDB port
            database: Database name
            username: Username (optional)
            password: Password (optional)
            credential_vault: Vault with credentials
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError(
                "PyMongo not installed. Install with: pip install pymongo"
            )
        
        # Get credentials
        if credential_vault:
            self.username = credential_vault.get_credential("mongo_username")
            self.password = credential_vault.get_credential("mongo_password")
            self.host = credential_vault.get_credential("mongo_host") or host
            self.port = int(credential_vault.get_credential("mongo_port") or port)
            self.database = credential_vault.get_credential("mongo_database") or database
        else:
            import os
            self.username = username or os.getenv("MONGO_USER")
            self.password = password or os.getenv("MONGO_PASSWORD")
            self.host = host
            self.port = port
            self.database = database
        
        # Build connection string
        if self.username and self.password:
            self.connection_string = (
                f"mongodb://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/"
            )
        else:
            self.connection_string = f"mongodb://{self.host}:{self.port}/"
        
        # MongoDB client and database
        self.client: Optional[MongoClient] = None
        self.db = None
        self.connected = False
        
        # Statistics
        self.operations_count = 0
        self.failed_operations = 0
        
    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            self.db = self.client[self.database]
            
            # Create indexes
            self._create_indexes()
            
            self.connected = True
            logger.info(f"✅ Connected to MongoDB at {self.host}:{self.port}/{self.database}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            self.connected = False
            return False
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            # OHLCV collection indexes
            ohlcv = self.db['ohlcv_data']
            ohlcv.create_index([('symbol', ASCENDING), ('timeframe', ASCENDING), ('timestamp', ASCENDING)])
            ohlcv.create_index([('timestamp', DESCENDING)])
            
            # Trades collection indexes
            trades = self.db['trades']
            trades.create_index([('order_id', ASCENDING)], unique=True)
            trades.create_index([('symbol', ASCENDING), ('timestamp', DESCENDING)])
            trades.create_index([('strategy', ASCENDING)])
            
            # Events/logs collection indexes
            events = self.db['events']
            events.create_index([('timestamp', DESCENDING)])
            events.create_index([('level', ASCENDING)])
            
            logger.info("✅ MongoDB indexes created/verified")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> bool:
        """
        Save OHLCV data to MongoDB
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: DataFrame with OHLCV data
        
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            collection = self.db['ohlcv_data']
            
            # Convert DataFrame to records
            records = []
            for idx, row in data.iterrows():
                record = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row.get('timestamp', idx),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                records.append(record)
            
            # Insert or update records
            if records:
                # Use bulk operations for efficiency
                from pymongo import UpdateOne
                operations = []
                for record in records:
                    filter_query = {
                        'symbol': record['symbol'],
                        'timeframe': record['timeframe'],
                        'timestamp': record['timestamp']
                    }
                    operations.append(
                        UpdateOne(filter_query, {'$set': record}, upsert=True)
                    )
                
                collection.bulk_write(operations, ordered=False)
                
                self.operations_count += 1
                logger.debug(f"✅ Saved {len(records)} OHLCV bars for {symbol} ({timeframe})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error saving OHLCV data: {e}")
            self.failed_operations += 1
            return False
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get OHLCV data from MongoDB"""
        if not self.connected:
            logger.error("Not connected to MongoDB")
            return pd.DataFrame()
        
        try:
            collection = self.db['ohlcv_data']
            
            # Build query
            query = {
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', ASCENDING)
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to DataFrame
            records = list(cursor)
            if records:
                df = pd.DataFrame(records)
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                self.operations_count += 1
                logger.debug(f"✅ Retrieved {len(df)} OHLCV bars for {symbol} ({timeframe})")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            self.failed_operations += 1
            return pd.DataFrame()
    
    def save_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        pnl: Optional[float] = None,
        strategy: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save trade to MongoDB"""
        if not self.connected:
            return False
        
        try:
            collection = self.db['trades']
            
            trade = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'pnl': pnl,
                'strategy': strategy,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            
            collection.update_one(
                {'order_id': order_id},
                {'$set': trade},
                upsert=True
            )
            
            self.operations_count += 1
            logger.debug(f"✅ Trade saved: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            self.failed_operations += 1
            return False
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get trades from MongoDB"""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            collection = self.db['trades']
            
            # Build query
            query = {}
            if symbol:
                query['symbol'] = symbol
            if strategy:
                query['strategy'] = strategy
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', DESCENDING).limit(limit)
            
            # Convert to DataFrame
            records = list(cursor)
            if records:
                df = pd.DataFrame(records)
                self.operations_count += 1
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            self.failed_operations += 1
            return pd.DataFrame()
    
    def save_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save system event/log to MongoDB"""
        if not self.connected:
            return False
        
        try:
            collection = self.db['events']
            
            event = {
                'type': event_type,
                'level': level,
                'message': message,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            
            collection.insert_one(event)
            self.operations_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error saving event: {e}")
            self.failed_operations += 1
            return False
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        level: Optional[str] = None,
        start_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get system events from MongoDB"""
        if not self.connected:
            return []
        
        try:
            collection = self.db['events']
            
            # Build query
            query = {}
            if event_type:
                query['type'] = event_type
            if level:
                query['level'] = level
            if start_date:
                query['timestamp'] = {'$gte': start_date}
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', DESCENDING).limit(limit)
            
            self.operations_count += 1
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            self.failed_operations += 1
            return []
    
    def aggregate(self, collection_name: str, pipeline: List[Dict]) -> List[Dict]:
        """Execute aggregation pipeline"""
        if not self.connected:
            return []
        
        try:
            collection = self.db[collection_name]
            result = list(collection.aggregate(pipeline))
            self.operations_count += 1
            return result
        except Exception as e:
            logger.error(f"Error executing aggregation: {e}")
            self.failed_operations += 1
            return []
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
        self.connected = False
        logger.info("Disconnected from MongoDB")
    
    def get_statistics(self) -> dict:
        """Get database statistics"""
        stats = {
            'connected': self.connected,
            'host': self.host,
            'database': self.database,
            'operations_count': self.operations_count,
            'failed_operations': self.failed_operations,
            'success_rate': (
                self.operations_count / (self.operations_count + self.failed_operations)
                if (self.operations_count + self.failed_operations) > 0
                else 0.0
            )
        }
        
        if self.connected and self.db:
            try:
                stats['collections'] = self.db.list_collection_names()
                stats['ohlcv_count'] = self.db['ohlcv_data'].count_documents({})
                stats['trades_count'] = self.db['trades'].count_documents({})
                stats['events_count'] = self.db['events'].count_documents({})
            except:
                pass
        
        return stats
