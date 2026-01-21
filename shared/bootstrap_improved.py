# shared/bootstrap_improved.py

import logging
import time
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Imports de módulos core
from core_local.data_manager import DataManager
from core_local.data_ingestion import DataIngestion
from core_local.strategy_engine import StrategyEngine
from core_local.adaptive_strategy_manager import AdaptiveStrategyManager
from core_local.statistics_layer import StatisticsLayer
from core_local.executor import Executor
from core_local.brokers import BinanceBroker, CoinbaseBroker
from core_local.ai_auditor import AIAuditor
from core_local.risk_manager import RiskManager
from core_local.market_regime import MarketRegimeDetector
from core_local.system_reporter import SystemReporter

# Imports de seguridad
from shared.core.security.credential_vault import CredentialVault

# Configuración de logging mejorado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BootstrapImproved")


class SystemState:
    """Gestión de estado persistente del sistema entre reinicios."""
    
    def __init__(self, state_file: str = "data/system_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Carga el estado desde disco."""
        import json
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error cargando estado: {e}")
                return self._default_state()
        return self._default_state()
    
    def _default_state(self) -> Dict:
        """Estado por defecto del sistema."""
        return {
            "last_cycle": 0,
            "total_cycles_completed": 0,
            "last_execution_time": None,
            "system_health": "healthy",
            "kill_switch_active": False,
            "paused": False,
            "emergency_stop": False
        }
    
    def save_state(self):
        """Guarda el estado en disco."""
        import json
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
    
    def update(self, key: str, value: Any):
        """Actualiza un valor del estado y guarda."""
        self.state[key] = value
        self.save_state()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor del estado."""
        return self.state.get(key, default)


class TradingSystem:
    """
    Sistema de Trading Mejorado con:
    - Gestión segura de credenciales
    - Manejo robusto de errores
    - Recovery automático
    - Persistencia de estado
    - Health checks
    """
    
    def __init__(self, environment: str = "local", mode: str = "paper"):
        """
        Inicializa el sistema de trading.
        
        Args:
            environment: "local" o "cloud"
            mode: "paper" o "real"
        """
        self.environment = environment
        self.mode = mode
        self.state = SystemState()
        
        # Inicializar vault de credenciales
        logger.info("🔐 Inicializando CredentialVault...")
        self.vault = CredentialVault(environment=environment)
        
        # Inicializar componentes
        self.data_manager: Optional[DataManager] = None
        self.data_ingestion: Optional[DataIngestion] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.adaptive_manager: Optional[AdaptiveStrategyManager] = None
        self.executor: Optional[Executor] = None
        self.brokers: List = []
        self.risk_manager: Optional[RiskManager] = None
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.ai_auditor: Optional[AIAuditor] = None
        self.reporter: Optional[SystemReporter] = None
        
        # Estadísticas del sistema
        self.stats = {
            "successful_cycles": 0,
            "failed_cycles": 0,
            "total_signals_generated": 0,
            "total_orders_executed": 0,
            "uptime_start": datetime.now()
        }
        
        logger.info(f"✅ Sistema inicializado en modo {mode.upper()} - Entorno: {environment.upper()}")
    
    def initialize_components(self):
        """Inicializa todos los componentes del sistema con manejo de errores."""
        
        try:
            # 1️⃣ DataManager
            logger.info("📊 Inicializando DataManager...")
            self.data_manager = DataManager()
            
            # 2️⃣ DataIngestion
            logger.info("📥 Inicializando DataIngestion...")
            self.data_ingestion = DataIngestion(self.data_manager)
            
            # 3️⃣ StrategyEngine
            logger.info("🎯 Inicializando StrategyEngine...")
            assets = self.data_manager.get_assets()
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            self.strategy_engine = StrategyEngine(assets=assets, timeframes=timeframes)
            
            # 4️⃣ AdaptiveStrategyManager
            logger.info("🧠 Inicializando AdaptiveStrategyManager...")
            self.adaptive_manager = AdaptiveStrategyManager(self.data_manager)
            
            # 5️⃣ Brokers (con credenciales seguras)
            logger.info("🏦 Inicializando Brokers...")
            self.brokers = self._initialize_brokers()
            
            # 6️⃣ Executor
            logger.info("⚡ Inicializando Executor...")
            self.executor = Executor(brokers=self.brokers)
            
            # 7️⃣ RiskManager
            logger.info("🛡️ Inicializando RiskManager...")
            self.risk_manager = RiskManager(self.data_manager)
            
            # 8️⃣ MarketRegimeDetector
            logger.info("📈 Inicializando MarketRegimeDetector...")
            self.regime_detector = MarketRegimeDetector(self.data_manager)
            
            # 9️⃣ AI Auditor
            logger.info("🤖 Inicializando AI Auditor...")
            self.ai_auditor = AIAuditor(
                self.data_manager,
                self.strategy_engine,
                self.adaptive_manager
            )
            
            # 🔟 SystemReporter
            logger.info("📋 Inicializando SystemReporter...")
            self.reporter = SystemReporter()
            
            logger.info("✅ Todos los componentes inicializados correctamente")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            logger.exception("Traceback completo:")
            return False
    
    def _initialize_brokers(self) -> List:
        """Inicializa brokers con credenciales seguras del vault."""
        brokers = []
        
        try:
            # Binance
            binance_key = self.vault.get_credential("BINANCE_API_KEY")
            binance_secret = self.vault.get_credential("BINANCE_API_SECRET")
            
            if binance_key and binance_secret:
                binance = BinanceBroker(api_key=binance_key, api_secret=binance_secret)
                brokers.append(binance)
                logger.info("✅ Binance broker inicializado")
            else:
                logger.warning("⚠️  Credenciales de Binance no encontradas")
            
            # Coinbase
            coinbase_key = self.vault.get_credential("COINBASE_API_KEY")
            coinbase_secret = self.vault.get_credential("COINBASE_API_SECRET")
            
            if coinbase_key and coinbase_secret:
                coinbase = CoinbaseBroker(api_key=coinbase_key, api_secret=coinbase_secret)
                brokers.append(coinbase)
                logger.info("✅ Coinbase broker inicializado")
            else:
                logger.warning("⚠️  Credenciales de Coinbase no encontradas")
        
        except Exception as e:
            logger.error(f"❌ Error inicializando brokers: {e}")
        
        if not brokers:
            logger.error("❌ No se pudo inicializar ningún broker")
            raise RuntimeError("No hay brokers disponibles para trading")
        
        return brokers
    
    def health_check(self) -> bool:
        """Verifica el estado de salud del sistema."""
        
        checks = {
            "data_manager": self.data_manager is not None,
            "strategy_engine": self.strategy_engine is not None,
            "executor": self.executor is not None,
            "brokers": len(self.brokers) > 0,
            "kill_switch": not self.state.get("kill_switch_active", False)
        }
        
        all_healthy = all(checks.values())
        
        if not all_healthy:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.warning(f"⚠️  Health check falló: {failed_checks}")
        
        return all_healthy
    
    def run_single_cycle(self, cycle_number: int) -> bool:
        """
        Ejecuta un ciclo completo de trading con manejo de errores.
        
        Args:
            cycle_number: Número del ciclo actual
        
        Returns:
            True si el ciclo fue exitoso, False si falló
        """
        
        logger.info(f"{'='*60}")
        logger.info(f"🔄 INICIANDO CICLO {cycle_number}")
        logger.info(f"{'='*60}")
        
        try:
            # Health check
            if not self.health_check():
                logger.error("❌ Health check falló. Saltando ciclo.")
                return False
            
            # 1️⃣ Ingestión de datos
            logger.info(f"[{cycle_number}] 📥 Ingiriendo datos...")
            self.data_ingestion.run_ingestion_cycle()
            self.stats["total_signals_generated"] += 1  # Placeholder
            
            # 2️⃣ Detectar régimen de mercado
            logger.info(f"[{cycle_number}] 📊 Detectando régimen de mercado...")
            current_regime = self.regime_detector.detect()
            self.strategy_engine.set_regime(current_regime)
            self.adaptive_manager.stats_layer.update_regime(current_regime)
            logger.info(f"📈 Régimen detectado: {current_regime}")
            
            # 3️⃣ Generar señales
            logger.info(f"[{cycle_number}] 🎯 Generando señales...")
            signals = self.strategy_engine.run_cycle()
            logger.info(f"✅ {len(signals)} señales generadas")
            
            # 4️⃣ Ajuste adaptativo
            logger.info(f"[{cycle_number}] 🧠 Ajustando señales adaptativamente...")
            available_capital = self._get_available_capital()
            adjusted_signals = self.adaptive_manager.adapt_signals(signals, available_capital)
            logger.info(f"✅ {len(adjusted_signals)} señales ajustadas")
            
            # 5️⃣ Filtrado por riesgo
            logger.info(f"[{cycle_number}] 🛡️ Filtrando señales por riesgo...")
            safe_signals = self.risk_manager.filter_signals(adjusted_signals)
            
            if not safe_signals:
                logger.warning("⚠️  No hay señales seguras. Kill-switch activado o riesgo elevado.")
                return True  # No es un error, solo no hay trading
            
            logger.info(f"✅ {len(safe_signals)} señales seguras")
            
            # 6️⃣ Ejecutar órdenes
            if self.mode == "real":
                logger.info(f"[{cycle_number}] ⚡ Ejecutando órdenes en modo REAL...")
                executed_orders = self.executor.execute_signals(safe_signals)
            else:
                logger.info(f"[{cycle_number}] 📝 Simulando órdenes en modo PAPER...")
                executed_orders = self._simulate_orders(safe_signals)
            
            logger.info(f"✅ {len(executed_orders)} órdenes ejecutadas")
            self.stats["total_orders_executed"] += len(executed_orders)
            
            # 7️⃣ Auditoría IA
            logger.info(f"[{cycle_number}] 🤖 Ejecutando auditoría IA...")
            self.ai_auditor.review_cycle(executed_orders, current_regime)
            
            # 8️⃣ Actualizar estado
            self.state.update("last_cycle", cycle_number)
            self.state.update("total_cycles_completed", self.state.get("total_cycles_completed", 0) + 1)
            self.state.update("last_execution_time", datetime.now().isoformat())
            
            self.stats["successful_cycles"] += 1
            
            logger.info(f"✅ CICLO {cycle_number} COMPLETADO EXITOSAMENTE")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error en ciclo {cycle_number}: {e}")
            logger.exception("Traceback completo:")
            
            self.stats["failed_cycles"] += 1
            
            # Intentar recovery
            self._attempt_recovery(e)
            
            return False
    
    def _get_available_capital(self) -> Dict[str, float]:
        """Obtiene capital disponible por asset (desde brokers o config)."""
        # Placeholder: en producción, consultar balances reales
        assets = self.data_manager.get_assets()
        return {asset: 1000.0 for asset in assets}  # TODO: Implementar lógica real
    
    def _simulate_orders(self, signals: Dict) -> List:
        """Simula ejecución de órdenes en modo paper."""
        # Placeholder para simulación
        simulated_orders = []
        for asset, signal_data in signals.items():
            simulated_orders.append({
                "asset": asset,
                "side": signal_data.get("side", "buy"),
                "quantity": signal_data.get("quantity", 0.1),
                "status": "simulated"
            })
        return simulated_orders
    
    def _attempt_recovery(self, error: Exception):
        """Intenta recuperar el sistema después de un error."""
        logger.info("🔧 Intentando recuperación del sistema...")
        
        # Estrategias de recovery
        try:
            # 1. Reinicializar componentes fallidos
            if not self.health_check():
                logger.info("Reinicializando componentes...")
                self.initialize_components()
            
            # 2. Limpiar caché si es necesario
            # ...
            
            logger.info("✅ Recuperación exitosa")
        
        except Exception as recovery_error:
            logger.error(f"❌ Recovery falló: {recovery_error}")
            # Activar kill-switch si recovery falla
            self.state.update("kill_switch_active", True)
    
    def run(self, cycles: int = 10, sleep_interval: int = 30):
        """
        Ejecuta el sistema por N ciclos.
        
        Args:
            cycles: Número de ciclos a ejecutar
            sleep_interval: Segundos entre ciclos
        """
        
        logger.info("="*60)
        logger.info("🚀 INICIANDO SISTEMA DE TRADING")
        logger.info(f"Modo: {self.mode.upper()} | Entorno: {self.environment.upper()}")
        logger.info(f"Ciclos: {cycles} | Intervalo: {sleep_interval}s")
        logger.info("="*60)
        
        # Inicializar componentes
        if not self.initialize_components():
            logger.error("❌ No se pudieron inicializar los componentes. Abortando.")
            return
        
        # Ejecutar ciclos
        for cycle in range(1, cycles + 1):
            
            # Verificar kill-switch
            if self.state.get("kill_switch_active", False):
                logger.error("🛑 KILL-SWITCH ACTIVADO. Deteniendo sistema.")
                break
            
            # Ejecutar ciclo
            success = self.run_single_cycle(cycle)
            
            # Pausa entre ciclos (excepto el último)
            if cycle < cycles:
                logger.info(f"⏸️  Pausando {sleep_interval} segundos antes del siguiente ciclo...")
                time.sleep(sleep_interval)
        
        # Reporte final
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Genera reporte final del sistema."""
        logger.info("="*60)
        logger.info("📊 REPORTE FINAL DEL SISTEMA")
        logger.info("="*60)
        
        uptime = datetime.now() - self.stats["uptime_start"]
        
        logger.info(f"⏱️  Uptime: {uptime}")
        logger.info(f"✅ Ciclos exitosos: {self.stats['successful_cycles']}")
        logger.info(f"❌ Ciclos fallidos: {self.stats['failed_cycles']}")
        logger.info(f"📊 Total señales: {self.stats['total_signals_generated']}")
        logger.info(f"⚡ Total órdenes: {self.stats['total_orders_executed']}")
        
        success_rate = (self.stats['successful_cycles'] / 
                       (self.stats['successful_cycles'] + self.stats['failed_cycles']) * 100
                       if (self.stats['successful_cycles'] + self.stats['failed_cycles']) > 0 else 0)
        
        logger.info(f"📈 Tasa de éxito: {success_rate:.2f}%")
        
        # Generar reporte detallado
        if self.reporter:
            self.reporter.generate_report()


def main():
    """Función principal."""
    
    # Configurar el sistema
    system = TradingSystem(
        environment="local",  # "local" o "cloud"
        mode="paper"          # "paper" o "real"
    )
    
    # Ejecutar
    system.run(cycles=10, sleep_interval=30)


if __name__ == "__main__":
    main()