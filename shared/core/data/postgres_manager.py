"""
PostgreSQL Database Manager
Enterprise-grade database support for market data and trading history
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

try:
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Index
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """
    PostgreSQL database manager for trading system
    Provides high-performance storage and querying for market data
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "trading_db",
        username: Optional[str] = None,
        password: Optional[str] = None,
        credential_vault: Optional[Any] = None
    ):
        """
        Initialize PostgreSQL manager
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            credential_vault: Vault with credentials
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy not installed. Install with: pip install SQLAlchemy psycopg2-binary"
            )
        
        # Get credentials
        if credential_vault:
            self.username = credential_vault.get_credential("postgres_username")
            self.password = credential_vault.get_credential("postgres_password")
            self.host = credential_vault.get_credential("postgres_host") or host
            self.port = int(credential_vault.get_credential("postgres_port") or port)
            self.database = credential_vault.get_credential("postgres_database") or database
        else:
            import os
            self.username = username or os.getenv("POSTGRES_USER", "trading_user")
            self.password = password or os.getenv("POSTGRES_PASSWORD")
            self.host = host
            self.port = port
            self.database = database
        
        if not self.password:
            raise ValueError("PostgreSQL password is required")
        
        # Connection string
        self.connection_string = (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        
        # SQLAlchemy engine
        self.engine = None
        self.metadata = None
        self.Session = None
        self.connected = False
        
        # Statistics
        self.queries_executed = 0
        self.failed_queries = 0
        
    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self.metadata = MetaData()
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            self._create_tables()
            
            self.connected = True
            logger.info(f"✅ Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            self.connected = False
            return False
    
    def _create_tables(self):
        """Create necessary tables"""
        from sqlalchemy import Table, Column, Integer, String, Float, DateTime, BigInteger, Index
        
        # OHLCV data table
        ohlcv_table = Table(
            'ohlcv_data',
            self.metadata,
            Column('id', BigInteger, primary_key=True, autoincrement=True),
            Column('symbol', String(20), nullable=False),
            Column('timeframe', String(10), nullable=False),
            Column('timestamp', DateTime, nullable=False),
            Column('open', Float, nullable=False),
            Column('high', Float, nullable=False),
            Column('low', Float, nullable=False),
            Column('close', Float, nullable=False),
            Column('volume', Float, nullable=False),
            Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
            Index('idx_timestamp', 'timestamp'),
        )
        
        # Trades table
        trades_table = Table(
            'trades',
            self.metadata,
            Column('id', BigInteger, primary_key=True, autoincrement=True),
            Column('order_id', String(100), nullable=False, unique=True),
            Column('symbol', String(20), nullable=False),
            Column('side', String(10), nullable=False),
            Column('quantity', Float, nullable=False),
            Column('price', Float, nullable=False),
            Column('commission', Float, default=0.0),
            Column('pnl', Float, nullable=True),
            Column('strategy', String(50), nullable=True),
            Column('timestamp', DateTime, nullable=False, default=datetime.utcnow),
            Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
            Index('idx_strategy', 'strategy'),
        )
        
        # Create all tables
        self.metadata.create_all(self.engine)
        logger.info("✅ PostgreSQL tables created/verified")
    
    def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> bool:
        """
        Save OHLCV data to PostgreSQL
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            data: DataFrame with OHLCV data
        
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Not connected to PostgreSQL")
            return False
        
        try:
            # Prepare data
            df = data.copy()
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                df['timestamp'] = df.index
            
            # Save to database (replace duplicates)
            df.to_sql(
                'ohlcv_data',
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            self.queries_executed += 1
            logger.debug(f"✅ Saved {len(df)} OHLCV bars for {symbol} ({timeframe})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving OHLCV data: {e}")
            self.failed_queries += 1
            return False
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data from PostgreSQL
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            logger.error("Not connected to PostgreSQL")
            return pd.DataFrame()
        
        try:
            # Build parameterized query to prevent SQL injection
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data
                WHERE symbol = %(symbol)s AND timeframe = %(timeframe)s
            """
            
            params = {'symbol': symbol, 'timeframe': timeframe}
            
            if start_date:
                query += " AND timestamp >= %(start_date)s"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= %(end_date)s"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += " LIMIT %(limit)s"
                params['limit'] = limit
            
            # Execute query with parameters
            df = pd.read_sql(query, self.engine, params=params)
            
            self.queries_executed += 1
            logger.debug(f"✅ Retrieved {len(df)} OHLCV bars for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            self.failed_queries += 1
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
        strategy: Optional[str] = None
    ) -> bool:
        """Save trade to database"""
        if not self.connected:
            return False
        
        try:
            trade_data = pd.DataFrame([{
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'pnl': pnl,
                'strategy': strategy,
                'timestamp': datetime.utcnow()
            }])
            
            trade_data.to_sql(
                'trades',
                self.engine,
                if_exists='append',
                index=False
            )
            
            self.queries_executed += 1
            logger.debug(f"✅ Trade saved: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            self.failed_queries += 1
            return False
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get trades from database"""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = {}
            
            if symbol:
                query += " AND symbol = %(symbol)s"
                params['symbol'] = symbol
            
            if strategy:
                query += " AND strategy = %(strategy)s"
                params['strategy'] = strategy
            
            if start_date:
                query += " AND timestamp >= %(start_date)s"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= %(end_date)s"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql(query, self.engine, params=params if params else None)
            self.queries_executed += 1
            return df
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            self.failed_queries += 1
            return pd.DataFrame()
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query"""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            df = pd.read_sql(query, self.engine)
            self.queries_executed += 1
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.failed_queries += 1
            return pd.DataFrame()
    
    def disconnect(self):
        """Disconnect from PostgreSQL"""
        if self.engine:
            self.engine.dispose()
        self.connected = False
        logger.info("Disconnected from PostgreSQL")
    
    def get_statistics(self) -> dict:
        """Get database statistics"""
        return {
            'connected': self.connected,
            'host': self.host,
            'database': self.database,
            'queries_executed': self.queries_executed,
            'failed_queries': self.failed_queries,
            'success_rate': (
                self.queries_executed / (self.queries_executed + self.failed_queries)
                if (self.queries_executed + self.failed_queries) > 0
                else 0.0
            )
        }
