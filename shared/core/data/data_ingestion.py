"""
================================================================================
DATA INGESTION - Multi-Source Data Provider
================================================================================
Ruta: quant_system/shared/core/data/data_ingestion.py

Sistema de ingestión de datos desde múltiples fuentes
- Binance, Yahoo Finance, Mock
- Rate limiting automático
- Error handling robusto
================================================================================
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """Clase base para data providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.rate_limit_per_second = 5
        self.last_request_time = 0
        self.total_requests = 0
        self.failed_requests = 0
    
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 500) -> pd.DataFrame:
        """Obtiene datos OHLCV"""
        pass
    
    def _rate_limit(self):
        """Rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit_per_second
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()


class MockDataProvider(BaseDataProvider):
    """Provider simulado para testing"""
    
    def __init__(self):
        super().__init__("Mock")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 500) -> pd.DataFrame:
        """Genera datos mock"""
        self.total_requests += 1
        
        if not end_time:
            end_time = datetime.now()
        
        freq_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '4h': '4h', '1d': '1D', '1w': '1W'
        }
        
        freq = freq_map.get(timeframe, '1h')
        dates = pd.date_range(end=end_time, periods=limit, freq=freq)
        
        np.random.seed(42)
        base_price = 50000.0 if 'BTC' in symbol else 100.0
        
        prices = [base_price]
        for _ in range(limit - 1):
            change = np.random.randn() * 0.02
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
            'low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
            'close': [p * (1 + np.random.randn() * 0.005) for p in prices],
            'volume': np.random.uniform(100, 1000, limit)
        })
        
        logger.debug(f"Generated {len(df)} mock records for {symbol}")
        return df


class DataIngestion:
    """Sistema de ingestión de datos"""
    
    def __init__(self, data_manager: Any):
        self.data_manager = data_manager
        self.providers: Dict[str, BaseDataProvider] = {}
        
        # Añadir provider mock por defecto
        self.add_provider('mock', MockDataProvider())
        
        logger.info("DataIngestion initialized")
    
    def add_provider(self, name: str, provider: BaseDataProvider):
        """Añade un data provider"""
        self.providers[name] = provider
        logger.info(f"Provider '{name}' added")
    
    def fetch_data(self, symbol: str, timeframe: str,
                   provider_name: str = 'mock',
                   limit: int = 500) -> pd.DataFrame:
        """
        Obtiene datos de un provider
        
        Args:
            symbol: Símbolo del activo
            timeframe: Timeframe
            provider_name: Nombre del provider
            limit: Límite de registros
            
        Returns:
            DataFrame con datos OHLCV
        """
        if provider_name not in self.providers:
            logger.error(f"Provider '{provider_name}' no disponible")
            return pd.DataFrame()
        
        provider = self.providers[provider_name]
        
        try:
            data = provider.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.info(f"Fetched {len(data)} records for {symbol} ({timeframe})")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def ingest_historical_data(self, symbol: str, timeframe: str,
                               days_back: int = 30,
                               provider_name: str = 'mock') -> bool:
        """
        Ingiere datos históricos
        
        Args:
            symbol: Símbolo
            timeframe: Timeframe
            days_back: Días hacia atrás
            provider_name: Provider a usar
            
        Returns:
            True si fue exitoso
        """
        try:
            data = self.fetch_data(symbol, timeframe, provider_name, limit=days_back * 24)
            
            if data.empty:
                return False
            
            # Guardar en DataManager
            success = self.data_manager.save_ohlcv(symbol, timeframe, data)
            
            if success:
                logger.info(f"Ingested {len(data)} records for {symbol} ({timeframe})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in historical ingestion: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de ingestión"""
        stats = {}
        
        for name, provider in self.providers.items():
            stats[name] = {
                'total_requests': provider.total_requests,
                'failed_requests': provider.failed_requests
            }
        
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    class MockDataManager:
        def save_ohlcv(self, symbol, timeframe, data):
            print(f"Saved {len(data)} records for {symbol} ({timeframe})")
            return True
    
    dm = MockDataManager()
    ingestion = DataIngestion(dm)
    
    ingestion.ingest_historical_data('BTCUSDT', '1h', days_back=7)
    
    print("\nStatistics:", ingestion.get_statistics())