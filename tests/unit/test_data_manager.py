"""
================================================================================
UNIT TESTS - Data Manager
================================================================================
Ruta: quant_system/tests/unit/test_data_manager.py

Tests para el módulo data_manager.py
- Test de inicialización
- Test de almacenamiento OHLCV
- Test de consultas
- Test de timeframe conversions
- Test de edge cases
================================================================================
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import os
import sys

# Agregar path del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared.core.data.data_manager import DataManager


class TestDataManagerInit(unittest.TestCase):
    """Tests de inicialización del DataManager"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
    
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init_creates_database(self):
        """Test: Inicialización crea la base de datos"""
        dm = DataManager(self.db_path)
        self.assertTrue(os.path.exists(self.db_path))
        dm.close()
    
    def test_init_creates_tables(self):
        """Test: Inicialización crea las tablas necesarias"""
        dm = DataManager(self.db_path)
        
        # Verificar que existen las tablas
        tables = dm._get_table_names()
        self.assertIn('ohlcv_data', tables)
        self.assertIn('trades', tables)
        dm.close()
    
    def test_multiple_instances_same_db(self):
        """Test: Múltiples instancias pueden acceder a la misma DB"""
        dm1 = DataManager(self.db_path)
        dm2 = DataManager(self.db_path)
        
        self.assertEqual(dm1.db_path, dm2.db_path)
        
        dm1.close()
        dm2.close()


class TestDataManagerOHLCV(unittest.TestCase):
    """Tests para operaciones OHLCV"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.dm = DataManager(self.db_path)
        
        # Datos de prueba
        self.sample_data = self._create_sample_ohlcv()
    
    def tearDown(self):
        """Cleanup después de cada test"""
        self.dm.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_sample_ohlcv(self, n_bars=100):
        """Crea datos OHLCV de prueba"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_bars),
            periods=n_bars,
            freq='h'  # Changed from '1H' to 'h' for pandas 2.0+
        )
        
        # Simular precios realistas
        close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.rand(n_bars) * 0.5,
            'high': close_prices + np.random.rand(n_bars) * 1.0,
            'low': close_prices - np.random.rand(n_bars) * 1.0,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        return df
    
    def test_save_ohlcv_basic(self):
        """Test: Guardar datos OHLCV básicos"""
        result = self.dm.save_ohlcv('BTCUSDT', '1h', self.sample_data)
        self.assertTrue(result)
    
    def test_save_ohlcv_duplicate_timestamps(self):
        """Test: Manejar timestamps duplicados"""
        # Guardar primera vez
        self.dm.save_ohlcv('BTCUSDT', '1h', self.sample_data)
        
        # Intentar guardar de nuevo (deberían actualizarse)
        result = self.dm.save_ohlcv('BTCUSDT', '1h', self.sample_data)
        self.assertTrue(result)
    
    def test_get_ohlcv_all_data(self):
        """Test: Recuperar todos los datos OHLCV"""
        self.dm.save_ohlcv('BTCUSDT', '1h', self.sample_data)
        
        retrieved = self.dm.get_ohlcv('BTCUSDT', '1h')
        
        self.assertEqual(len(retrieved), len(self.sample_data))
        self.assertListEqual(
            list(retrieved.columns),
            ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
    
    def test_get_ohlcv_date_range(self):
        """Test: Recuperar datos en rango de fechas"""
        self.dm.save_ohlcv('BTCUSDT', '1h', self.sample_data)
        
        start = self.sample_data['timestamp'].iloc[10]
        end = self.sample_data['timestamp'].iloc[20]
        
        retrieved = self.dm.get_ohlcv('BTCUSDT', '1h', start_date=start, end_date=end)
        
        self.assertGreaterEqual(len(retrieved), 10)
        self.assertLessEqual(len(retrieved), 11)
    
    def test_get_ohlcv_nonexistent_symbol(self):
        """Test: Consultar símbolo inexistente retorna DataFrame vacío"""
        retrieved = self.dm.get_ohlcv('NONEXISTENT', '1h')
        
        self.assertTrue(retrieved.empty)
    
    def test_save_multiple_symbols(self):
        """Test: Guardar múltiples símbolos"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in symbols:
            result = self.dm.save_ohlcv(symbol, '1h', self.sample_data)
            self.assertTrue(result)
        
        # Verificar que se guardaron todos
        for symbol in symbols:
            retrieved = self.dm.get_ohlcv(symbol, '1h')
            self.assertFalse(retrieved.empty)
    
    def test_save_multiple_timeframes(self):
        """Test: Guardar múltiples timeframes"""
        timeframes = ['1m', '5m', '1h', '1d']
        
        for tf in timeframes:
            result = self.dm.save_ohlcv('BTCUSDT', tf, self.sample_data)
            self.assertTrue(result)
        
        # Verificar que se guardaron todos
        for tf in timeframes:
            retrieved = self.dm.get_ohlcv('BTCUSDT', tf)
            self.assertFalse(retrieved.empty)


class TestDataManagerQueries(unittest.TestCase):
    """Tests para consultas avanzadas"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.dm = DataManager(self.db_path)
        
        # Crear y guardar datos de prueba
        self.sample_data = self._create_sample_ohlcv()
        self.dm.save_ohlcv('BTCUSDT', '1h', self.sample_data)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        self.dm.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_sample_ohlcv(self, n_bars=100):
        """Crea datos OHLCV de prueba"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_bars),
            periods=n_bars,
            freq='h'  # Changed from '1H' to 'h' for pandas 2.0+
        )
        
        close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.rand(n_bars) * 0.5,
            'high': close_prices + np.random.rand(n_bars) * 1.0,
            'low': close_prices - np.random.rand(n_bars) * 1.0,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
    
    def test_get_latest_bar(self):
        """Test: Obtener la barra más reciente"""
        latest = self.dm.get_latest_bar('BTCUSDT', '1h')
        
        self.assertIsNotNone(latest)
        self.assertEqual(
            latest['timestamp'], 
            self.sample_data['timestamp'].max()
        )
    
    def test_get_latest_n_bars(self):
        """Test: Obtener las últimas N barras"""
        n = 10
        latest = self.dm.get_latest_bars('BTCUSDT', '1h', n=n)
        
        self.assertEqual(len(latest), n)
        self.assertTrue(latest['timestamp'].is_monotonic_increasing)
    
    def test_calculate_returns(self):
        """Test: Calcular retornos"""
        returns = self.dm.calculate_returns('BTCUSDT', '1h')
        
        self.assertEqual(len(returns), len(self.sample_data) - 1)
        self.assertIn('returns', returns.columns)
    
    def test_calculate_volatility(self):
        """Test: Calcular volatilidad"""
        vol = self.dm.calculate_volatility('BTCUSDT', '1h', window=20)
        
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)


class TestDataManagerEdgeCases(unittest.TestCase):
    """Tests para casos extremos y errores"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.dm = DataManager(self.db_path)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        self.dm.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_empty_dataframe(self):
        """Test: Intentar guardar DataFrame vacío"""
        empty_df = pd.DataFrame()
        result = self.dm.save_ohlcv('BTCUSDT', '1h', empty_df)
        
        # Debería manejar gracefully
        self.assertFalse(result)
    
    def test_save_invalid_columns(self):
        """Test: Intentar guardar DataFrame con columnas inválidas"""
        invalid_df = pd.DataFrame({
            'wrong_col1': [1, 2, 3],
            'wrong_col2': [4, 5, 6]
        })
        
        result = self.dm.save_ohlcv('BTCUSDT', '1h', invalid_df)
        self.assertFalse(result)
    
    def test_save_null_values(self):
        """Test: Manejar valores NULL en datos"""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='h'),  # Changed from '1H' to 'h'
            'open': [100, 101, None, 103, 104],
            'high': [102, 103, 104, 105, None],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, None, 1400]
        })
        
        # Debería manejar o rechazar NULL values
        result = self.dm.save_ohlcv('BTCUSDT', '1h', df)
        # El comportamiento específico depende de la implementación
        self.assertIsNotNone(result)
    
    def test_concurrent_access(self):
        """Test: Acceso concurrente a la DB"""
        import threading
        
        def write_data(symbol):
            dm = DataManager(self.db_path)
            data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='h'),  # Changed from '1H' to 'h'
                'open': np.random.rand(10) * 100,
                'high': np.random.rand(10) * 100,
                'low': np.random.rand(10) * 100,
                'close': np.random.rand(10) * 100,
                'volume': np.random.randint(1000, 10000, 10)
            })
            dm.save_ohlcv(symbol, '1h', data)
            dm.close()
        
        # Crear múltiples threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_data, args=(f'TEST{i}',))
            threads.append(t)
            t.start()
        
        # Esperar a que terminen
        for t in threads:
            t.join()
        
        # Verificar que todos los datos se guardaron
        for i in range(5):
            retrieved = self.dm.get_ohlcv(f'TEST{i}', '1h')
            self.assertFalse(retrieved.empty)


class TestDataManagerPerformance(unittest.TestCase):
    """Tests de performance"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.dm = DataManager(self.db_path)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        self.dm.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_large_dataset_save(self):
        """Test: Guardar dataset grande (10,000 barras)"""
        large_data = self._create_large_dataset(10000)
        
        import time
        start = time.time()
        result = self.dm.save_ohlcv('BTCUSDT', '1h', large_data)
        elapsed = time.time() - start
        
        self.assertTrue(result)
        self.assertLess(elapsed, 5.0)  # Debería tomar menos de 5 segundos
    
    def test_large_dataset_query(self):
        """Test: Consultar dataset grande"""
        large_data = self._create_large_dataset(10000)
        self.dm.save_ohlcv('BTCUSDT', '1h', large_data)
        
        import time
        start = time.time()
        retrieved = self.dm.get_ohlcv('BTCUSDT', '1h')
        elapsed = time.time() - start
        
        self.assertEqual(len(retrieved), 10000)
        self.assertLess(elapsed, 2.0)  # Debería tomar menos de 2 segundos
    
    def _create_large_dataset(self, n_bars):
        """Crea un dataset grande para tests de performance"""
        dates = pd.date_range(
            start=datetime(2020, 1, 1),
            periods=n_bars,
            freq='h'  # Changed from '1H' to 'h' for pandas 2.0+
        )
        
        close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.rand(n_bars) * 0.5,
            'high': close_prices + np.random.rand(n_bars) * 1.0,
            'low': close_prices - np.random.rand(n_bars) * 1.0,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })


class TestDataManagerGetAssets(unittest.TestCase):
    """Tests para el método get_assets"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.dm = DataManager(self.db_path)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        self.dm.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_sample_ohlcv(self, n_bars=10):
        """Crea datos OHLCV de prueba"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_bars),
            periods=n_bars,
            freq='h'  # Changed from 'H' to lowercase 'h' for pandas 2.0+
        )
        
        close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.rand(n_bars) * 0.5,
            'high': close_prices + np.random.rand(n_bars) * 1.0,
            'low': close_prices - np.random.rand(n_bars) * 1.0,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
    
    def test_get_assets_empty_database(self):
        """Test: get_assets retorna lista default cuando DB está vacía"""
        assets = self.dm.get_assets()
        
        self.assertIsInstance(assets, list)
        self.assertGreater(len(assets), 0)
        # Debe retornar la lista default
        self.assertIn('BTCUSDT', assets)
        self.assertIn('ETHUSDT', assets)
    
    def test_get_assets_with_data(self):
        """Test: get_assets retorna símbolos guardados"""
        # Guardar datos para varios símbolos
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
        sample_data = self._create_sample_ohlcv()
        
        for symbol in symbols:
            self.dm.save_ohlcv(symbol, '1h', sample_data)
        
        # Obtener assets
        assets = self.dm.get_assets()
        
        self.assertIsInstance(assets, list)
        self.assertEqual(len(assets), len(symbols))
        
        # Verificar que todos los símbolos estén presentes
        for symbol in symbols:
            self.assertIn(symbol, assets)
    
    def test_get_assets_unique_symbols(self):
        """Test: get_assets retorna símbolos únicos (sin duplicados)"""
        sample_data = self._create_sample_ohlcv()
        
        # Guardar el mismo símbolo con diferentes timeframes
        self.dm.save_ohlcv('BTCUSDT', '1h', sample_data)
        self.dm.save_ohlcv('BTCUSDT', '5m', sample_data)
        self.dm.save_ohlcv('BTCUSDT', '1d', sample_data)
        
        # Obtener assets
        assets = self.dm.get_assets()
        
        # Debe retornar solo un símbolo (sin duplicados)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0], 'BTCUSDT')
    
    def test_get_assets_returns_list(self):
        """Test: get_assets siempre retorna una lista"""
        assets = self.dm.get_assets()
        
        self.assertIsInstance(assets, list)


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Ejecutar todos los tests
    unittest.main(verbosity=2)