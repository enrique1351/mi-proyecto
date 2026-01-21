# main.py - Sistema de Trading Cuantitativo Completo

"""
Sistema de Trading Cuantitativo Universal
==========================================

Sistema modular, adaptativo y seguro para trading multi-asset.

Autor: Sistema Autogenerado
Versión: 1.0.0
Fecha: 2025-01-17
"""

import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
import io


if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Agregar path al proyecto
sys.path.insert(0, str(Path(__file__).parent))

# Imports de módulos core
from shared.core.security.credential_vault import CredentialVault
from shared.core.data.data_manager import DataManager
from shared.core.data.data_ingestion import DataIngestion
from shared.core.strategies.strategy_engine import StrategyEngine
from shared.core.strategies.adaptive_strategy_manager import AdaptiveStrategyManager
from shared.core.execution.execution_interface import ExecutionInterface, ExecutionMode
from shared.core.brokers.brokers import BrokerFactory, BrokerType, MockBroker
from shared.core.risk.risk_manager import RiskManager
from shared.core.analysis.market_regime import MarketRegimeDetector
from shared.core.ai.ai_auditor import AIAuditor
from shared.core.monitoring.system_reporter import SystemReporter
from shared.core.config.constants import ASSETS, TRADING_CONFIG


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logging(log_level: str = "INFO"):
    """Configura el sistema de logging con soporte para caracteres Unicode."""
    
    # Crear directorio de logs
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Crear manejador de archivo con encoding UTF-8
    file_handler = logging.FileHandler(
        log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Crear manejador de consola con manejo de errores de encoding
    # Para consolas sin soporte UTF-8 (cmd, PowerShell), ignorar errores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configurar el stream con manejo de errores de encoding
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            # Python 3.7+: Intentar reconfigurar con UTF-8
            sys.stdout.reconfigure(errors='replace')
        except (AttributeError, OSError):
            # AttributeError: método no disponible
            # OSError: operación no soportada en el stream
            pass
    
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[file_handler, console_handler]
    )
    
    # Silenciar logs verbosos de librerías
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ============================================================================
# QUANT TRADING SYSTEM (Clase Principal)
# ============================================================================

class QuantTradingSystem:
    """
    Sistema de Trading Cuantitativo Completo.
    
    Integra todos los módulos core:
    - Data Management
    - Strategy Engine
    - Risk Management
    - Execution
    - AI Auditor
    - Reporting
    """
    
    def __init__(
        self,
        mode: str = "paper",
        initial_capital: float = 10000.0,
        use_ai: bool = False,
        config: Optional[dict] = None
    ):
        """
        Inicializa el sistema completo.
        
        Args:
            mode: "paper" o "real"
            initial_capital: Capital inicial
            use_ai: Habilitar AI Auditor
            config: Configuración personalizada
        """
        
        logger.info("="*80)
        logger.info("🚀 INICIALIZANDO QUANT TRADING SYSTEM")
        logger.info("="*80)
        
        self.mode = mode
        self.initial_capital = initial_capital
        self.use_ai = use_ai
        self.config = config or TRADING_CONFIG
        
        # Estado
        self.running = False
        self.cycles_completed = 0
        
        # Componentes (se inicializan después)
        self.credential_vault: Optional[CredentialVault] = None
        self.data_manager: Optional[DataManager] = None
        self.data_ingestion: Optional[DataIngestion] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.adaptive_manager: Optional[AdaptiveStrategyManager] = None
        self.execution_interface: Optional[ExecutionInterface] = None
        self.risk_manager: Optional[RiskManager] = None
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.ai_auditor: Optional[AIAuditor] = None
        self.reporter: Optional[SystemReporter] = None
        
        # Inicializar
        self._initialize_system()
    
    def _initialize_system(self):
        """Inicializa todos los componentes del sistema."""
        
        try:
            # 1️⃣ Credential Vault
            logger.info("🔐 Inicializando CredentialVault...")
            self.credential_vault = CredentialVault()
            logger.info("✅ CredentialVault listo")
            
            # 2️⃣ Data Manager
            logger.info("📊 Inicializando DataManager...")
            self.data_manager = DataManager(db_path="data/db/market_data.db")
            
            # Registrar assets
            # self._register_assets()  # TODO: Implementar
            logger.info("✅ DataManager listo")
            
            # 3️⃣ Data Ingestion
            logger.info("📥 Inicializando DataIngestion...")
            self.data_ingestion = DataIngestion(self.data_manager)
            logger.info("✅ DataIngestion listo")
            
            # 4️⃣ Strategy Engine
            logger.info("🎯 Inicializando StrategyEngine...")
            assets = self.data_manager.get_assets()
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            self.strategy_engine = StrategyEngine(
                assets=assets,
                timeframes=timeframes,
                data_manager=self.data_manager
            )
            logger.info("✅ StrategyEngine listo")
            
            # 5️⃣ Market Regime Detector
            logger.info("📈 Inicializando MarketRegimeDetector...")
            self.regime_detector = MarketRegimeDetector(
                data_manager=self.data_manager
            )
            logger.info("✅ MarketRegimeDetector listo")
            
            # 6️⃣ Risk Manager
            logger.info("🛡️ Inicializando RiskManager...")
            self.risk_manager = RiskManager(
                data_manager=self.data_manager,
                initial_capital=self.initial_capital
            )
            logger.info("✅ RiskManager listo")
            
            # 7️⃣ Adaptive Strategy Manager
            logger.info("🧠 Inicializando AdaptiveStrategyManager...")
            self.adaptive_manager = AdaptiveStrategyManager(
                data_manager=self.data_manager,
                strategy_engine=self.strategy_engine,
                regime_detector=self.regime_detector,
                risk_manager=self.risk_manager
            )
            logger.info("✅ AdaptiveStrategyManager listo")
            
            # 8️⃣ Brokers & Execution
            logger.info("🏦 Inicializando Brokers...")
            brokers = self._initialize_brokers()
            
            execution_mode = ExecutionMode.REAL if self.mode == "real" else ExecutionMode.PAPER
            self.execution_interface = ExecutionInterface(
                brokers=brokers,
                mode=execution_mode
            )
            logger.info("✅ Execution Interface listo")
            
            # 9️⃣ AI Auditor
            logger.info("🤖 Inicializando AI Auditor...")
            self.ai_auditor = AIAuditor(
                data_manager=self.data_manager,
                strategy_engine=self.strategy_engine,
                adaptive_manager=self.adaptive_manager,
                use_ai=self.use_ai
            )
            logger.info("✅ AI Auditor listo")
            
            # 🔟 System Reporter
            logger.info("📋 Inicializando SystemReporter...")
            self.reporter = SystemReporter()
            logger.info("✅ SystemReporter listo")
            
            logger.info("="*80)
            logger.info("✅ SISTEMA COMPLETAMENTE INICIALIZADO")
            logger.info("="*80)
        
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            logger.exception("Traceback completo:")
            raise
    
    def _register_assets(self):
        """Registra assets en el DataManager."""
        
        logger.info("Registrando assets...")
        
        # Crypto
        crypto_assets = ASSETS.get('crypto', [])
        for asset in crypto_assets[:10]:  # Primeros 10 para empezar
            self.data_manager.register_asset(
                symbol=asset,
                asset_class='crypto',
                exchange='Binance'
            )
        
        # Stocks (si se quiere agregar)
        # stock_assets = ASSETS.get('stocks', {}).get('us', {}).get('mega_cap_tech', [])
        # for asset in stock_assets[:5]:
        #     self.data_manager.register_asset(
        #         symbol=asset,
        #         asset_class='stocks',
        #         exchange='NASDAQ'
        #     )
        
        logger.info(f"✅ {len(self.data_manager.get_assets())} assets registrados")
    
    def _initialize_brokers(self) -> list:
        """Inicializa brokers según el modo."""
        
        brokers = []
        
        if self.mode == "paper":
            # Modo paper: usar MockBroker
            mock_broker = MockBroker(
                initial_balance={'USD': self.initial_capital, 'USDT': self.initial_capital}
            )
            brokers.append(mock_broker)
            logger.info("✅ MockBroker inicializado (Paper Trading)")
        
        else:
            # Modo real: intentar inicializar brokers reales
            try:
                binance = BrokerFactory.create_broker(
                    BrokerType.BINANCE,
                    credential_vault=self.credential_vault
                )
                brokers.append(binance)
                logger.info("✅ Binance broker inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Binance: {e}")
            
            # Fallback a MockBroker si no hay brokers reales
            if not brokers:
                logger.warning("⚠️  No hay brokers reales disponibles. Usando MockBroker.")
                mock_broker = MockBroker(
                    initial_balance={'USD': self.initial_capital}
                )
                brokers.append(mock_broker)
        
        return brokers
    
    def run_single_cycle(self) -> bool:
        """
        Ejecuta un ciclo completo de trading.
        
        Returns:
            True si el ciclo fue exitoso
        """
        
        cycle_num = self.cycles_completed + 1
        
        logger.info("="*80)
        logger.info(f"🔄 CICLO #{cycle_num}")
        logger.info("="*80)
        
        try:
            # 1️⃣ Verificar kill-switch
            if self.execution_interface.killswitch.is_active():
                logger.error("🛑 KILL-SWITCH ACTIVO. Ciclo cancelado.")
                return False
            
            # 2️⃣ Ingerir datos
            logger.info(f"[{cycle_num}] 📥 Ingiriendo datos...")
            self.data_ingestion.run_ingestion_cycle()
            
            # 3️⃣ Detectar régimen de mercado
            logger.info(f"[{cycle_num}] 📊 Detectando régimen de mercado...")
            
            # Detectar para el primer asset como muestra
            main_asset = self.data_manager.get_assets()[0] if self.data_manager.get_assets() else "BTCUSDT"
            current_regime = self.regime_detector.detect(main_asset, "1h")
            
            self.strategy_engine.set_regime(current_regime)
            self.adaptive_manager.stats_layer.update_regime(current_regime)
            
            logger.info(f"📈 Régimen actual: {current_regime.value}")
            
            # 4️⃣ Generar señales
            logger.info(f"[{cycle_num}] 🎯 Generando señales...")
            raw_signals = self.strategy_engine.run_cycle()
            logger.info(f"✅ {len(raw_signals)} señales crudas generadas")
            
            # 5️⃣ Adaptar señales
            logger.info(f"[{cycle_num}] 🧠 Adaptando señales...")
            
            # Obtener capital disponible
            available_capital = {}
            for asset in self.data_manager.get_assets():
                available_capital[asset] = self.risk_manager.get_available_capital() / len(self.data_manager.get_assets())
            
            adapted_signals = self.adaptive_manager.adapt_signals(
                raw_signals,
                available_capital
            )
            logger.info(f"✅ {len(adapted_signals)} señales adaptadas")
            
            # 6️⃣ Filtrar por riesgo
            logger.info(f"[{cycle_num}] 🛡️ Filtrando señales por riesgo...")
            safe_signals = self.risk_manager.filter_signals(adapted_signals)
            
            if not safe_signals:
                logger.warning("⚠️  No hay señales seguras en este ciclo.")
                return True  # No es error, solo no hay trading
            
            logger.info(f"✅ {len(safe_signals)} señales seguras")
            
            # 7️⃣ Ejecutar órdenes
            logger.info(f"[{cycle_num}] ⚡ Ejecutando órdenes...")
            executed_orders = self.execution_interface.execute_signals(safe_signals)
            logger.info(f"✅ {len(executed_orders)} órdenes ejecutadas")
            
            # 8️⃣ Auditoría IA
            logger.info(f"[{cycle_num}] 🤖 Ejecutando auditoría IA...")
            self.ai_auditor.review_cycle(executed_orders, current_regime)
            
            # 9️⃣ Actualizar contador
            self.cycles_completed += 1
            
            logger.info(f"✅ CICLO #{cycle_num} COMPLETADO EXITOSAMENTE")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error en ciclo #{cycle_num}: {e}")
            logger.exception("Traceback completo:")
            return False
    
    def run(
        self,
        cycles: int = 10,
        sleep_interval: int = 30,
        generate_reports: bool = True
    ):
        """
        Ejecuta el sistema por N ciclos.
        
        Args:
            cycles: Número de ciclos a ejecutar
            sleep_interval: Segundos entre ciclos
            generate_reports: Generar reportes al finalizar
        """
        
        logger.info("="*80)
        logger.info("🚀 INICIANDO SISTEMA DE TRADING")
        logger.info(f"Modo: {self.mode.upper()}")
        logger.info(f"Capital inicial: ${self.initial_capital:,.2f}")
        logger.info(f"Ciclos: {cycles}")
        logger.info(f"Intervalo: {sleep_interval}s")
        logger.info("="*80)
        
        self.running = True
        successful_cycles = 0
        failed_cycles = 0
        
        try:
            for cycle in range(1, cycles + 1):
                
                # Verificar kill-switch
                should_trigger, reason = self.risk_manager.should_trigger_killswitch()
                if should_trigger:
                    logger.critical(f"🛑 ACTIVANDO KILL-SWITCH: {reason}")
                    self.execution_interface.killswitch.trigger(reason)
                    break
                
                # Ejecutar ciclo
                success = self.run_single_cycle()
                
                if success:
                    successful_cycles += 1
                else:
                    failed_cycles += 1
                
                # Pausa entre ciclos (excepto el último)
                if cycle < cycles:
                    logger.info(f"⏸️  Pausando {sleep_interval} segundos...")
                    time.sleep(sleep_interval)
            
            # Reporte final
            logger.info("="*80)
            logger.info("📊 SESIÓN FINALIZADA")
            logger.info("="*80)
            logger.info(f"✅ Ciclos exitosos: {successful_cycles}")
            logger.info(f"❌ Ciclos fallidos: {failed_cycles}")
            logger.info(f"📈 Tasa de éxito: {successful_cycles/cycles*100:.1f}%")
            
            # Generar reportes
            if generate_reports:
                logger.info("\n📋 Generando reportes finales...")
                self._generate_final_reports()
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupción manual detectada. Finalizando...")
        
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            logger.exception("Traceback completo:")
        
        finally:
            self.running = False
            self._shutdown()
    
    def _generate_final_reports(self):
        """Genera reportes finales del sistema."""
        
        try:
            # Reporte diario
            daily_report = self.reporter.generate_daily_report(
                data_manager=self.data_manager,
                adaptive_manager=self.adaptive_manager,
                risk_manager=self.risk_manager,
                ai_auditor=self.ai_auditor
            )
            
            logger.info(f"✅ Reporte diario: {daily_report}")
            
            # Exportar trades
            if self.adaptive_manager:
                trades_export = self.reporter.generate_trades_export(
                    adaptive_manager=self.adaptive_manager
                )
                
                if trades_export:
                    logger.info(f"✅ Trades exportados: {trades_export}")
            
            # Estadísticas finales
            stats = self._get_final_statistics()
            
            logger.info("\n" + "="*80)
            logger.info("📊 ESTADÍSTICAS FINALES")
            logger.info("="*80)
            
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
        
        except Exception as e:
            logger.error(f"Error generando reportes: {e}")
    
    def _get_final_statistics(self) -> dict:
        """Obtiene estadísticas finales del sistema."""
        
        stats = {
            "Ciclos completados": self.cycles_completed,
            "Capital inicial": f"${self.initial_capital:,.2f}",
        }
        
        if self.risk_manager:
            stats.update({
                "Capital actual": f"${self.risk_manager.current_capital:,.2f}",
                "PnL total": f"${self.risk_manager.current_capital - self.initial_capital:,.2f}",
                "Retorno": f"{((self.risk_manager.current_capital - self.initial_capital) / self.initial_capital * 100):.2f}%",
                "Drawdown actual": f"{self.risk_manager.current_drawdown:.2%}",
                "Posiciones abiertas": len(self.risk_manager.current_positions)
            })
        
        if self.execution_interface:
            exec_stats = self.execution_interface.get_statistics()
            stats.update({
                "Órdenes totales": exec_stats['total_orders'],
                "Órdenes exitosas": exec_stats['filled_orders'],
                "Tasa de éxito órdenes": f"{exec_stats['success_rate']:.1%}"
            })
        
        return stats
    
    def _shutdown(self):
        """Cierra el sistema de forma ordenada."""
        
        logger.info("="*80)
        logger.info("🛑 CERRANDO SISTEMA")
        logger.info("="*80)
        
        # Guardar estados
        if self.risk_manager:
            self.risk_manager._save_state()
        
        if self.adaptive_manager:
            self.adaptive_manager._save_state()
        
        logger.info("✅ Sistema cerrado correctamente")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Punto de entrada principal."""
    
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Sistema de Trading Cuantitativo Universal'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['paper', 'real'],
        default='paper',
        help='Modo de ejecución (paper o real)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Capital inicial en USD'
    )
    
    parser.add_argument(
        '--cycles',
        type=int,
        default=10,
        help='Número de ciclos a ejecutar'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Intervalo entre ciclos (segundos)'
    )
    
    parser.add_argument(
        '--ai',
        action='store_true',
        help='Habilitar AI Auditor avanzado'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nivel de logging'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(args.log_level)
    
    # Banner
    print("\n" + "="*80)
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║           QUANT TRADING SYSTEM v1.0.0                                ║
    ║           Sistema de Trading Cuantitativo Universal                  ║
    ║                                                                       ║
    ║           Multi-Asset • Multi-Strategy • Adaptive AI                 ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    print("="*80 + "\n")
    
    # Crear sistema
    system = QuantTradingSystem(
        mode=args.mode,
        initial_capital=args.capital,
        use_ai=args.ai
    )
    
    # Ejecutar
    system.run(
        cycles=args.cycles,
        sleep_interval=args.interval,
        generate_reports=True
    )


if __name__ == "__main__":
    main()
