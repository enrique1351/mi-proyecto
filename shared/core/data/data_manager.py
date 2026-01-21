"""
================================================================================
DATA MANAGER - SQLite + Pandas Storage
================================================================================
Ruta: quant_system/shared/core/data/data_manager.py

Gestión centralizada de datos de mercado
- Almacenamiento SQLite optimizado
- Cache en memoria
- Queries optimizadas
- Time-series ready
================================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging


class DataManager:
    """
    Gestiona almacenamiento y recuperación de datos de mercado
    """
    
    def __init__(self, db_path: str = "data/db/trading.db"):
        """
        Inicializa el Data Manager
        
        Args:
            db_path: Ruta a la base de datos SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.cache = {}
        
        # Conectar y crear tablas
        self._connect()
        self._create_tables()
        
        logging.info(f"✓ DataManager inicializado: {self.db_path}")
    
    def _connect(self):
        """Establece conexión con la base de datos"""
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Optimizaciones SQLite
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=10000")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            
        except Exception as e:
            logging.error(f"✗ Error conectando a DB: {e}")
            raise
    
    def _create_tables(self):
        """Crea las tablas necesarias"""
        cursor = self.conn.cursor()
        
        # Tabla OHLCV
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        
        # Índices para búsquedas rápidas
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
            ON ohlcv_data(symbol, timeframe)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON ohlcv_data(timestamp)
        """)
        
        # Tabla de trades ejecutados
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                commission REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                strategy TEXT,
                notes TEXT
            )
        """)
        
        # Tabla de balance/portfolio
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions TEXT,
                metrics TEXT
            )
        """)
        
        self.conn.commit()
        logging.info("✓ Tablas de base de datos creadas/verificadas")
    
    def save_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Guarda datos OHLCV en la base de datos
        
        Args:
            symbol: Símbolo del activo
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
            data: DataFrame con columnas [timestamp, open, high, low, close, volume]
            
        Returns:
            True si se guardó correctamente
        """
        try:
            if data.empty:
                logging.warning(f"⚠ DataFrame vacío para {symbol}")
                return False
            
            # Validar columnas
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                logging.error(f"✗ Columnas faltantes. Se requiere: {required_cols}")
                return False
            
            # Añadir symbol y timeframe
            df = data.copy()
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Convertir timestamp a datetime si es necesario
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Guardar en DB (replace si existe)
            df.to_sql(
                'ohlcv_data',
                self.conn,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            self.conn.commit()
            
            # Limpiar duplicados
            self._remove_duplicates('ohlcv_data')
            
            logging.info(f"✓ Guardados {len(df)} registros: {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logging.error(f"✗ Error guardando OHLCV: {e}")
            self.conn.rollback()
            return False
    
    def get_ohlcv(self, symbol: str, timeframe: str, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """
        Recupera datos OHLCV de la base de datos
        
        Args:
            symbol: Símbolo del activo
            timeframe: Timeframe
            start_date: Fecha inicio (opcional)
            end_date: Fecha fin (opcional)
            limit: Límite de registros (opcional)
            
        Returns:
            DataFrame con datos OHLCV
        """
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logging.error(f"✗ Error recuperando OHLCV: {e}")
            return pd.DataFrame()
    
    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Obtiene la última barra OHLCV
        
        Args:
            symbol: Símbolo del activo
            timeframe: Timeframe
            
        Returns:
            Diccionario con datos de la última barra o None
        """
        df = self.get_ohlcv(symbol, timeframe, limit=1)
        
        if df.empty:
            return None
        
        return df.iloc[-1].to_dict()
    
    def get_latest_bars(self, symbol: str, timeframe: str, n: int = 100) -> pd.DataFrame:
        """
        Obtiene las últimas N barras
        
        Args:
            symbol: Símbolo
            timeframe: Timeframe
            n: Número de barras
            
        Returns:
            DataFrame con las últimas N barras
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=[symbol, timeframe, n])
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def save_trade(self, trade: Dict) -> bool:
        """
        Guarda un trade ejecutado
        
        Args:
            trade: Diccionario con datos del trade
            
        Returns:
            True si se guardó correctamente
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades 
                (timestamp, symbol, side, price, quantity, commission, pnl, strategy, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('timestamp', datetime.now()),
                trade.get('symbol'),
                trade.get('side'),
                trade.get('price'),
                trade.get('quantity'),
                trade.get('commission', 0),
                trade.get('pnl', 0),
                trade.get('strategy', ''),
                trade.get('notes', '')
            ))
            
            self.conn.commit()
            logging.info(f"✓ Trade guardado: {trade.get('symbol')} {trade.get('side')}")
            return True
            
        except Exception as e:
            logging.error(f"✗ Error guardando trade: {e}")
            return False
    
    def get_trades(self, symbol: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Recupera trades históricos
        
        Args:
            symbol: Filtrar por símbolo (opcional)
            start_date: Fecha inicio (opcional)
            end_date: Fecha fin (opcional)
            
        Returns:
            DataFrame con trades
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def calculate_returns(self, symbol: str, timeframe: str, 
                         periods: int = 1) -> pd.DataFrame:
        """
        Calcula retornos del precio de cierre
        
        Args:
            symbol: Símbolo
            timeframe: Timeframe
            periods: Períodos para calcular retorno
            
        Returns:
            DataFrame con columna de retornos
        """
        df = self.get_ohlcv(symbol, timeframe)
        
        if df.empty:
            return pd.DataFrame()
        
        df['returns'] = df['close'].pct_change(periods=periods)
        return df
    
    def calculate_volatility(self, symbol: str, timeframe: str, 
                           window: int = 20) -> float:
        """
        Calcula volatilidad histórica
        
        Args:
            symbol: Símbolo
            timeframe: Timeframe
            window: Ventana para cálculo
            
        Returns:
            Volatilidad anualizada
        """
        returns = self.calculate_returns(symbol, timeframe)
        
        if returns.empty:
            return 0.0
        
        vol = returns['returns'].std() * np.sqrt(252)  # Anualizada
        return vol
    
    def _remove_duplicates(self, table: str):
        """Elimina registros duplicados de una tabla"""
        try:
            if table == 'ohlcv_data':
                self.conn.execute("""
                    DELETE FROM ohlcv_data
                    WHERE id NOT IN (
                        SELECT MIN(id)
                        FROM ohlcv_data
                        GROUP BY symbol, timeframe, timestamp
                    )
                """)
                self.conn.commit()
        except Exception as e:
            logging.warning(f"⚠ Error eliminando duplicados: {e}")
    
    def _get_table_names(self) -> List[str]:
        """Obtiene lista de tablas en la DB"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas de la base de datos
        
        Returns:
            Diccionario con estadísticas
        """
        cursor = self.conn.cursor()
        
        # Contar registros OHLCV
        cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
        ohlcv_count = cursor.fetchone()[0]
        
        # Contar trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        trades_count = cursor.fetchone()[0]
        
        # Símbolos únicos
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_data")
        symbols_count = cursor.fetchone()[0]
        
        return {
            'db_path': str(self.db_path),
            'db_size_mb': self.db_path.stat().st_size / 1024 / 1024,
            'ohlcv_records': ohlcv_count,
            'trades_count': trades_count,
            'unique_symbols': symbols_count,
            'tables': self._get_table_names()
        }
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        if self.conn:
            self.conn.close()
            logging.info("✓ Conexión a DB cerrada")
    
    def get_assets(self) -> List[str]:
        """
        Retorna lista de símbolos únicos disponibles en la base de datos.
        
        Returns:
            Lista de símbolos (strings). Si no hay datos, retorna una lista default.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM ohlcv_data")
            symbols = [row[0] for row in cursor.fetchall()]
            if not symbols:
                # Retornar lista default si la BD está vacía
                return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            return symbols
        except Exception as e:
            logging.error(f"Error getting assets: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear DataManager
    dm = DataManager()
    
    # Crear datos de ejemplo
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50100 + np.random.randn(100) * 100,
        'low': 49900 + np.random.randn(100) * 100,
        'close': 50000 + np.random.randn(100) * 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Guardar datos
    dm.save_ohlcv('BTCUSDT', '1h', sample_data)
    
    # Recuperar datos
    data = dm.get_ohlcv('BTCUSDT', '1h')
    print(f"\nRecuperados {len(data)} registros")
    
    # Última barra
    latest = dm.get_latest_bar('BTCUSDT', '1h')
    print(f"\nÚltima barra: {latest}")
    
    # Estadísticas
    stats = dm.get_statistics()
    print(f"\nEstadísticas DB:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cerrar
    dm.close()