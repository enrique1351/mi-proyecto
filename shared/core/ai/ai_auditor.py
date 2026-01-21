# shared/core/ai/ai_auditor.py

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class AlertLevel(Enum):
    """Niveles de alerta."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """Tipos de anomalías detectables."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNUSUAL_DRAWDOWN = "unusual_drawdown"
    EXCESSIVE_LOSSES = "excessive_losses"
    STRATEGY_FAILURE = "strategy_failure"
    RISK_BREACH = "risk_breach"
    DATA_QUALITY = "data_quality"
    EXECUTION_ANOMALY = "execution_anomaly"


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    """Detector de anomalías en el sistema."""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, Any] = {}
        self.anomaly_threshold = 2.0  # Desviaciones estándar
    
    def detect_performance_anomaly(
        self,
        current_metrics: Dict[str, float],
        historical_metrics: List[Dict[str, float]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Detecta anomalías en performance.
        
        Args:
            current_metrics: Métricas actuales
            historical_metrics: Historial de métricas
        
        Returns:
            Tuple (is_anomaly, description)
        """
        if not historical_metrics or len(historical_metrics) < 10:
            return False, None
        
        # Analizar win rate
        current_wr = current_metrics.get('win_rate', 0)
        historical_wr = [m.get('win_rate', 0) for m in historical_metrics]
        
        import numpy as np
        mean_wr = np.mean(historical_wr)
        std_wr = np.std(historical_wr)
        
        if std_wr > 0:
            z_score = abs(current_wr - mean_wr) / std_wr
            
            if z_score > self.anomaly_threshold:
                if current_wr < mean_wr:
                    return True, f"Win rate bajo: {current_wr:.2%} vs {mean_wr:.2%} promedio"
                else:
                    return True, f"Win rate anormalmente alto: {current_wr:.2%} (posible overfitting)"
        
        # Analizar sharpe ratio
        current_sharpe = current_metrics.get('sharpe_ratio', 0)
        historical_sharpe = [m.get('sharpe_ratio', 0) for m in historical_metrics]
        
        mean_sharpe = np.mean(historical_sharpe)
        std_sharpe = np.std(historical_sharpe)
        
        if std_sharpe > 0 and current_sharpe < mean_sharpe - 2 * std_sharpe:
            return True, f"Sharpe ratio degradado: {current_sharpe:.2f} vs {mean_sharpe:.2f}"
        
        return False, None
    
    def detect_drawdown_anomaly(
        self,
        current_drawdown: float,
        max_allowed_drawdown: float
    ) -> Tuple[bool, Optional[str]]:
        """Detecta drawdowns anormales."""
        
        # Alerta temprana al 70% del drawdown máximo
        threshold = max_allowed_drawdown * 0.7
        
        if current_drawdown >= max_allowed_drawdown:
            return True, f"Drawdown máximo alcanzado: {current_drawdown:.2%}"
        
        elif current_drawdown >= threshold:
            return True, f"Drawdown elevado: {current_drawdown:.2%} (threshold: {threshold:.2%})"
        
        return False, None
    
    def detect_execution_anomaly(
        self,
        orders: List[Dict[str, Any]],
        expected_slippage: float = 0.001
    ) -> Tuple[bool, Optional[str]]:
        """Detecta anomalías en ejecución."""
        
        if not orders:
            return False, None
        
        # Analizar slippage
        slippages = []
        for order in orders:
            if 'expected_price' in order and 'fill_price' in order:
                slippage = abs(order['fill_price'] - order['expected_price']) / order['expected_price']
                slippages.append(slippage)
        
        if slippages:
            avg_slippage = sum(slippages) / len(slippages)
            
            if avg_slippage > expected_slippage * 3:
                return True, f"Slippage excesivo: {avg_slippage:.4f} vs {expected_slippage:.4f} esperado"
        
        # Analizar órdenes rechazadas
        rejected = sum(1 for o in orders if o.get('status') == 'rejected')
        total = len(orders)
        
        if total > 0 and rejected / total > 0.2:
            return True, f"Tasa de rechazo alta: {rejected}/{total} órdenes rechazadas"
        
        return False, None


# ============================================================================
# ALERT SYSTEM
# ============================================================================

class AlertSystem:
    """Sistema de alertas."""
    
    def __init__(self, log_file: str = "data/logs/alerts.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.alerts_history: deque = deque(maxlen=100)
    
    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict] = None
    ):
        """Crea una alerta."""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'title': title,
            'message': message,
            'metadata': metadata or {}
        }
        
        # Guardar en historial
        self.alerts_history.append(alert)
        
        # Log según nivel
        if level == AlertLevel.EMERGENCY:
            logger.critical(f"🚨 EMERGENCY: {title} - {message}")
        elif level == AlertLevel.CRITICAL:
            logger.error(f"❌ CRITICAL: {title} - {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"⚠️  WARNING: {title} - {message}")
        else:
            logger.info(f"ℹ️  INFO: {title} - {message}")
        
        # Guardar en archivo
        self._log_to_file(alert)
        
        # TODO: Enviar a Telegram/Email/SMS según nivel
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._send_urgent_notification(alert)
    
    def _log_to_file(self, alert: Dict):
        """Guarda alerta en archivo."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            logger.error(f"Error guardando alerta: {e}")
    
    def _send_urgent_notification(self, alert: Dict):
        """Envía notificación urgente (placeholder)."""
        # TODO: Implementar Telegram Bot, Email, SMS
        logger.info(f"📢 Notificación urgente: {alert['title']}")
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Retorna alertas recientes."""
        return list(self.alerts_history)[-limit:]


# ============================================================================
# STRATEGY ANALYZER (IA)
# ============================================================================

class StrategyAnalyzer:
    """Analizador de estrategias con capacidad de IA."""
    
    def __init__(self, use_ai: bool = False):
        """
        Inicializa el analizador.
        
        Args:
            use_ai: Si True, usa Claude API para análisis avanzado
        """
        self.use_ai = use_ai
    
    def analyze_strategy_performance(
        self,
        strategy_name: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analiza performance de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            metrics: Métricas de performance
        
        Returns:
            Análisis con recomendaciones
        """
        analysis = {
            'strategy': strategy_name,
            'timestamp': datetime.now().isoformat(),
            'health_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Calcular health score (0-100)
        health_factors = []
        
        # Factor win rate (target: >50%)
        win_rate = metrics.get('win_rate', 0)
        win_rate_score = min(win_rate / 0.5, 1.0) * 100
        health_factors.append(win_rate_score)
        
        if win_rate < 0.4:
            analysis['issues'].append("Win rate bajo (<40%)")
            analysis['recommendations'].append("Revisar condiciones de entrada/salida")
        
        # Factor profit factor (target: >1.5)
        profit_factor = metrics.get('profit_factor', 0)
        pf_score = min(profit_factor / 1.5, 1.0) * 100
        health_factors.append(pf_score)
        
        if profit_factor < 1.0:
            analysis['issues'].append("Profit factor negativo")
            analysis['recommendations'].append("Estrategia perdiendo dinero - considerar desactivar")
        
        # Factor sharpe (target: >1.0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sharpe_score = min(max(sharpe, 0) / 1.0, 1.0) * 100
        health_factors.append(sharpe_score)
        
        if sharpe < 0.5:
            analysis['issues'].append("Sharpe ratio bajo")
            analysis['recommendations'].append("Mejorar risk-adjusted returns")
        
        # Factor max drawdown (target: <15%)
        max_dd = metrics.get('max_drawdown', 0)
        dd_score = max(0, (0.15 - max_dd) / 0.15) * 100
        health_factors.append(dd_score)
        
        if max_dd > 0.2:
            analysis['issues'].append("Drawdown excesivo (>20%)")
            analysis['recommendations'].append("Reducir tamaño de posiciones o stop loss más ajustado")
        
        # Calcular health score promedio
        analysis['health_score'] = sum(health_factors) / len(health_factors) if health_factors else 0
        
        # Clasificar estrategia
        if analysis['health_score'] > 80:
            analysis['classification'] = 'excellent'
        elif analysis['health_score'] > 60:
            analysis['classification'] = 'good'
        elif analysis['health_score'] > 40:
            analysis['classification'] = 'mediocre'
        else:
            analysis['classification'] = 'poor'
        
        # Si está habilitada IA, hacer análisis avanzado
        if self.use_ai and analysis['classification'] in ['mediocre', 'poor']:
            ai_suggestions = self._get_ai_suggestions(strategy_name, metrics)
            if ai_suggestions:
                analysis['ai_suggestions'] = ai_suggestions
        
        return analysis
    
    def _get_ai_suggestions(
        self,
        strategy_name: str,
        metrics: Dict[str, float]
    ) -> Optional[List[str]]:
        """
        Obtiene sugerencias de IA usando Claude API.
        
        Args:
            strategy_name: Nombre de la estrategia
            metrics: Métricas
        
        Returns:
            Lista de sugerencias o None
        """
        # TODO: Implementar llamada a Claude API
        # Por ahora, retornar sugerencias hardcoded
        
        suggestions = []
        
        if metrics.get('win_rate', 0) < 0.5:
            suggestions.append("Considerar agregar filtros adicionales para mejorar precisión de entradas")
            suggestions.append("Analizar si hay sesgo en horarios o días de la semana")
        
        if metrics.get('profit_factor', 0) < 1.2:
            suggestions.append("Aumentar ratio risk/reward de 1:2 a 1:3")
            suggestions.append("Implementar trailing stop para capturar más upside")
        
        if metrics.get('max_drawdown', 0) > 0.15:
            suggestions.append("Reducir tamaño de posiciones al 50% del actual")
            suggestions.append("Implementar correlación check para evitar posiciones redundantes")
        
        return suggestions if suggestions else None


# ============================================================================
# AI AUDITOR (Principal)
# ============================================================================

class AIAuditor:
    """
    Auditor inteligente del sistema.
    
    Funcionalidades:
    - Monitoreo continuo de performance
    - Detección de anomalías
    - Alertas automáticas
    - Análisis de estrategias
    - Recomendaciones de mejora
    - (Futuro) Reescritura automática de estrategias
    """
    
    def __init__(
        self,
        data_manager: Any,
        strategy_engine: Optional[Any] = None,
        adaptive_manager: Optional[Any] = None,
        use_ai: bool = False
    ):
        """
        Inicializa el AI Auditor.
        
        Args:
            data_manager: Instancia de DataManager
            strategy_engine: Instancia de StrategyEngine
            adaptive_manager: Instancia de AdaptiveStrategyManager
            use_ai: Habilitar análisis con IA (Claude API)
        """
        self.data_manager = data_manager
        self.strategy_engine = strategy_engine
        self.adaptive_manager = adaptive_manager
        self.use_ai = use_ai
        
        # Componentes
        self.anomaly_detector = AnomalyDetector()
        self.alert_system = AlertSystem()
        self.strategy_analyzer = StrategyAnalyzer(use_ai=use_ai)
        
        # Estado
        self.last_audit_time = datetime.now()
        self.audit_frequency_hours = 6  # Auditar cada 6 horas
        
        # Métricas históricas
        self.metrics_history: Dict[str, List[Dict]] = {}
        
        logger.info(f"AIAuditor inicializado (AI: {use_ai})")
    
    def review_cycle(
        self,
        executed_orders: List[Dict[str, Any]],
        current_regime: Any
    ):
        """
        Revisa un ciclo de trading completado.
        
        Args:
            executed_orders: Órdenes ejecutadas en el ciclo
            current_regime: Régimen de mercado actual
        """
        logger.info("🔍 Iniciando revisión de ciclo...")
        
        # 1️⃣ Analizar órdenes ejecutadas
        if executed_orders:
            self._analyze_executions(executed_orders)
        
        # 2️⃣ Verificar si es momento de auditoría completa
        time_since_audit = (datetime.now() - self.last_audit_time).total_seconds() / 3600
        
        if time_since_audit >= self.audit_frequency_hours:
            self.run_full_audit()
            self.last_audit_time = datetime.now()
        
        # 3️⃣ Verificar anomalías en tiempo real
        self._check_realtime_anomalies()
    
    def _analyze_executions(self, orders: List[Dict[str, Any]]):
        """Analiza ejecuciones de órdenes."""
        
        # Detectar anomalías de ejecución
        is_anomaly, description = self.anomaly_detector.detect_execution_anomaly(orders)
        
        if is_anomaly:
            self.alert_system.create_alert(
                level=AlertLevel.WARNING,
                title="Anomalía de Ejecución",
                message=description,
                metadata={'orders_count': len(orders)}
            )
    
    def _check_realtime_anomalies(self):
        """Verifica anomalías en tiempo real."""
        
        # Si hay adaptive manager, verificar métricas
        if self.adaptive_manager:
            # Obtener reporte de performance
            report = self.adaptive_manager.get_performance_report()
            
            # Verificar cada estrategia
            for strategy, metrics in report.get('strategy_metrics', {}).items():
                
                # Detectar degradación de performance
                historical = self.metrics_history.get(strategy, [])
                
                if len(historical) >= 10:
                    is_anomaly, description = self.anomaly_detector.detect_performance_anomaly(
                        metrics, historical
                    )
                    
                    if is_anomaly:
                        self.alert_system.create_alert(
                            level=AlertLevel.WARNING,
                            title=f"Anomalía en {strategy}",
                            message=description,
                            metadata={'metrics': metrics}
                        )
                
                # Guardar en historial
                self.metrics_history.setdefault(strategy, []).append(metrics)
    
    def run_full_audit(self):
        """Ejecuta auditoría completa del sistema."""
        
        logger.info("🔍 Ejecutando auditoría completa...")
        
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {},
            'overall_health': 'unknown',
            'critical_issues': [],
            'recommendations': []
        }
        
        # 1️⃣ Auditar estrategias
        if self.adaptive_manager:
            report = self.adaptive_manager.get_performance_report()
            
            strategy_health_scores = []
            
            for strategy, metrics in report.get('strategy_metrics', {}).items():
                
                # Analizar estrategia
                analysis = self.strategy_analyzer.analyze_strategy_performance(
                    strategy, metrics
                )
                
                audit_report['strategies'][strategy] = analysis
                strategy_health_scores.append(analysis['health_score'])
                
                # Si la estrategia está en mal estado
                if analysis['classification'] == 'poor':
                    audit_report['critical_issues'].append(
                        f"Estrategia {strategy} en mal estado (score: {analysis['health_score']:.0f})"
                    )
                    
                    # Crear alerta
                    self.alert_system.create_alert(
                        level=AlertLevel.CRITICAL,
                        title=f"Estrategia {strategy} degradada",
                        message=f"Health score: {analysis['health_score']:.0f}/100",
                        metadata=analysis
                    )
                    
                    # Recomendar acción
                    audit_report['recommendations'].append(
                        f"Considerar desactivar o optimizar {strategy}"
                    )
            
            # Calcular salud general
            if strategy_health_scores:
                overall_score = sum(strategy_health_scores) / len(strategy_health_scores)
                
                if overall_score > 70:
                    audit_report['overall_health'] = 'good'
                elif overall_score > 50:
                    audit_report['overall_health'] = 'fair'
                else:
                    audit_report['overall_health'] = 'poor'
                    
                    # Alerta crítica si sistema en mal estado
                    self.alert_system.create_alert(
                        level=AlertLevel.CRITICAL,
                        title="Sistema en mal estado",
                        message=f"Health score promedio: {overall_score:.0f}/100",
                        metadata=audit_report
                    )
        
        # Guardar reporte
        self._save_audit_report(audit_report)
        
        logger.info(f"✅ Auditoría completa: {audit_report['overall_health']}")
        
        return audit_report
    
    def _save_audit_report(self, report: Dict):
        """Guarda reporte de auditoría."""
        report_file = Path("data/logs/audit_reports.jsonl")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_file, 'a') as f:
                f.write(json.dumps(report) + '\n')
        except Exception as e:
            logger.error(f"Error guardando reporte: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Obtiene salud general del sistema.
        
        Returns:
            Dict con información de salud
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'components': {},
            'recent_alerts': [],
            'recommendations': []
        }
        
        # Alertas recientes
        health['recent_alerts'] = self.alert_system.get_recent_alerts(5)
        
        # Estado de componentes
        if self.data_manager:
            health['components']['data_manager'] = 'operational'
        
        if self.strategy_engine:
            engine_stats = self.strategy_engine.get_statistics()
            health['components']['strategy_engine'] = {
                'status': 'operational',
                'strategies': engine_stats.get('enabled_strategies', 0)
            }
        
        if self.adaptive_manager:
            perf_report = self.adaptive_manager.get_performance_report()
            aggregate = perf_report.get('aggregate', {})
            
            health['components']['adaptive_manager'] = {
                'status': 'operational',
                'total_trades': aggregate.get('total_trades', 0),
                'total_pnl': aggregate.get('total_pnl', 0)
            }
            
            # Determinar estado general
            if aggregate.get('total_pnl', 0) > 0:
                health['overall_status'] = 'profitable'
            else:
                health['overall_status'] = 'unprofitable'
        
        return health
    
    def suggest_strategy_improvements(
        self,
        strategy_name: str
    ) -> List[str]:
        """
        Sugiere mejoras para una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
        
        Returns:
            Lista de sugerencias
        """
        if not self.adaptive_manager:
            return ["Adaptive manager no disponible"]
        
        # Obtener métricas
        metrics = self.adaptive_manager.stats_layer.calculate_strategy_metrics(strategy_name)
        
        # Analizar
        analysis = self.strategy_analyzer.analyze_strategy_performance(strategy_name, metrics)
        
        suggestions = analysis.get('recommendations', [])
        
        if 'ai_suggestions' in analysis:
            suggestions.extend(analysis['ai_suggestions'])
        
        return suggestions
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del auditor."""
        
        # Contar alertas por nivel
        alert_counts = {level.value: 0 for level in AlertLevel}
        
        for alert in self.alert_system.alerts_history:
            level = alert['level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        return {
            'total_alerts': len(self.alert_system.alerts_history),
            'alerts_by_level': alert_counts,
            'strategies_monitored': len(self.metrics_history),
            'last_audit': self.last_audit_time.isoformat(),
            'audit_frequency_hours': self.audit_frequency_hours,
            'ai_enabled': self.use_ai
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock components
    class MockDataManager:
        pass
    
    class MockAdaptiveManager:
        def get_performance_report(self):
            return {
                'strategy_metrics': {
                    'TrendFollowing': {
                        'total_trades': 50,
                        'win_rate': 0.45,
                        'profit_factor': 1.1,
                        'sharpe_ratio': 0.8,
                        'max_drawdown': 0.12
                    },
                    'MeanReversion': {
                        'total_trades': 30,
                        'win_rate': 0.65,
                        'profit_factor': 2.0,
                        'sharpe_ratio': 1.5,
                        'max_drawdown': 0.08
                    }
                },
                'aggregate': {
                    'total_trades': 80,
                    'total_pnl': 1500
                }
            }
    
    # Crear auditor
    dm = MockDataManager()
    am = MockAdaptiveManager()
    auditor = AIAuditor(dm, adaptive_manager=am, use_ai=False)
    
    # Ejecutar auditoría
    print("\n" + "="*60)
    print("AUDITORÍA COMPLETA:")
    print("="*60)
    report = auditor.run_full_audit()
    print(json.dumps(report, indent=2, default=str))
    
    # Salud del sistema
    print("\n" + "="*60)
    print("SALUD DEL SISTEMA:")
    print("="*60)
    health = auditor.get_system_health()
    print(json.dumps(health, indent=2, default=str))
    
    # Sugerencias
    print("\n" + "="*60)
    print("SUGERENCIAS PARA TRENDFOLLOWING:")
    print("="*60)
    suggestions = auditor.suggest_strategy_improvements('TrendFollowing')
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")