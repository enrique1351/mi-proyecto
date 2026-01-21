# bootstrap.py

import logging
import time
from core_local.data_manager import DataManager
from core_local.data_ingestion import DataIngestion
from core_local.strategy_engine import StrategyEngine
from core_local.adaptive_strategy_manager import AdaptiveStrategyManager
from core_local.executor import Executor

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BootstrapFinal")

def bootstrap_hybrid_system(mode="paper", cycles=1, sleep_interval=5):
    """
    Inicializa todo el sistema híbrido adaptativo:
    - DataManager
    - DataIngestion
    - StrategyEngine
    - AdaptiveStrategyManager
    - Executor
    - Operación automática con ingestión de datos en tiempo real
    """

    logger.info("=== Inicializando DataManager ===")
    data_manager = DataManager()

    logger.info("=== Inicializando DataIngestion ===")
    data_ingestion = DataIngestion(data_manager)

    logger.info("=== Inicializando StrategyEngine ===")
    engine = StrategyEngine()

    logger.info("=== Inicializando AdaptiveStrategyManager ===")
    adaptive_mana_
