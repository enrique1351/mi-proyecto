# shared/core/monitoring/system_reporter.py

import logging
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generador de reportes en múltiples formatos."""
    
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Genera reporte en formato JSON.
        
        Args:
            data: Datos del reporte
            filename: Nombre del archivo
        
        Returns:
            Path al archivo generado
        """
        filepath = self.output_dir / f"{filename}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Reporte JSON generado: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error generando reporte JSON: {e}")
            return None
    
    def generate_csv_report(
        self,
        data: pd.DataFrame,
        filename: str
    ) -> Path:
        """Genera reporte en formato CSV."""
        filepath = self.output_dir / f"{filename}.csv"
        
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Reporte CSV generado: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error generando reporte CSV: {e}")
            return None
    
    def generate_html_report(
        self,
        title: str,
        sections: List[Dict[str, Any]],
        filename: str
    ) -> Path:
        """
        Genera reporte en formato HTML.
        
        Args:
            title: Título del reporte
            sections: Lista de secciones {title, content, type}
            filename: Nombre del archivo
        
        Returns:
            Path al archivo generado
        """
        filepath = self.output_dir / f"{filename}.html"
        
        html_content = self._build_html(title, sections)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Reporte HTML generado: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error generando reporte HTML: {e}")
            return None
    
    def _build_html(self, title: str, sections: List[Dict]) -> str:
        """Construye HTML del reporte."""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            background: #f0f0f0;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .positive {{
            color: #10b981;
        }}
        .negative {{
            color: #ef4444;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Agregar secciones
        for section in sections:
            section_title = section.get('title', 'Sin título')
            section_type = section.get('type', 'text')
            content = section.get('content', '')
            
            html += f'<div class="section">\n'
            html += f'<h2>{section_title}</h2>\n'
            
            if section_type == 'table' and isinstance(content, pd.DataFrame):
                html += content.to_html(classes='table', index=False)
            
            elif section_type == 'metrics' and isinstance(content, dict):
                html += '<div class="metrics-container">\n'
                for key, value in content.items():
                    value_class = 'positive' if isinstance(value, (int, float)) and value > 0 else ''
                    html += f'''
                    <div class="metric">
                        <div class="metric-label">{key}</div>
                        <div class="metric-value {value_class}">{value}</div>
                    </div>
                    '''
                html += '</div>\n'
            
            else:
                html += f'<p>{content}</p>\n'
            
            html += '</div>\n'
        
        html += f"""
    <div class="footer">
        <p>Quant Trading System - Automated Report</p>
    </div>
</body>
</html>
"""
        
        return html


# ============================================================================
# SYSTEM REPORTER (Principal)
# ============================================================================

class SystemReporter:
    """
    Sistema de reportes completo.
    
    Genera reportes de:
    - Performance diaria/semanal/mensual
    - Estrategias individuales
    - Estado del sistema
    - Alertas y anomalías
    """
    
    def __init__(self):
        self.generator = ReportGenerator()
        
        # Logs de acciones
        self.action_log: List[Dict] = []
        
        logger.info("SystemReporter inicializado")
    
    def log_action(
        self,
        module_name: str,
        action_desc: str,
        expected: str,
        result: str
    ):
        """
        Registra una acción del sistema.
        
        Args:
            module_name: Nombre del módulo
            action_desc: Descripción de la acción
            expected: Resultado esperado
            result: Resultado real
        """
        self.action_log.append({
            'timestamp': datetime.now().isoformat(),
            'module': module_name,
            'action': action_desc,
            'expected': expected,
            'result': result
        })
    
    def generate_daily_report(
        self,
        data_manager: Any,
        adaptive_manager: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        ai_auditor: Optional[Any] = None
    ) -> Path:
        """
        Genera reporte diario del sistema.
        
        Args:
            data_manager: Instancia de DataManager
            adaptive_manager: Instancia de AdaptiveStrategyManager
            risk_manager: Instancia de RiskManager
            ai_auditor: Instancia de AIAuditor
        
        Returns:
            Path al reporte generado
        """
        logger.info("📊 Generando reporte diario...")
        
        report_data = {
            'report_type': 'daily',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'sections': {}
        }
        
        # 1️⃣ Resumen ejecutivo
        executive_summary = self._build_executive_summary(
            adaptive_manager, risk_manager
        )
        report_data['sections']['executive_summary'] = executive_summary
        
        # 2️⃣ Performance de estrategias
        if adaptive_manager:
            strategy_performance = self._build_strategy_performance(adaptive_manager)
            report_data['sections']['strategy_performance'] = strategy_performance
        
        # 3️⃣ Gestión de riesgo
        if risk_manager:
            risk_report = risk_manager.get_risk_report()
            report_data['sections']['risk_management'] = risk_report
        
        # 4️⃣ Salud del sistema
        if ai_auditor:
            system_health = ai_auditor.get_system_health()
            report_data['sections']['system_health'] = system_health
        
        # 5️⃣ Acciones registradas
        report_data['sections']['action_log'] = self.action_log[-50:]  # Últimas 50
        
        # Guardar JSON
        filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}"
        json_path = self.generator.generate_json_report(report_data, filename)
        
        # Generar HTML
        html_sections = self._prepare_html_sections(report_data)
        html_path = self.generator.generate_html_report(
            title="Daily Trading Report",
            sections=html_sections,
            filename=filename
        )
        
        logger.info(f"✅ Reporte diario generado: {json_path}")
        
        return json_path
    
    def _build_executive_summary(
        self,
        adaptive_manager: Optional[Any],
        risk_manager: Optional[Any]
    ) -> Dict[str, Any]:
        """Construye resumen ejecutivo."""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        if adaptive_manager:
            report = adaptive_manager.get_performance_report()
            aggregate = report.get('aggregate', {})
            
            summary['metrics']['total_trades'] = aggregate.get('total_trades', 0)
            summary['metrics']['total_pnl'] = aggregate.get('total_pnl', 0)
            summary['metrics']['avg_pnl_per_trade'] = aggregate.get('avg_pnl_per_trade', 0)
        
        if risk_manager:
            summary['metrics']['current_capital'] = risk_manager.current_capital
            summary['metrics']['current_drawdown'] = risk_manager.current_drawdown
            summary['metrics']['open_positions'] = len(risk_manager.current_positions)
            summary['metrics']['risk_level'] = risk_manager.risk_level.value
        
        return summary
    
    def _build_strategy_performance(
        self,
        adaptive_manager: Any
    ) -> Dict[str, Any]:
        """Construye reporte de performance por estrategia."""
        
        report = adaptive_manager.get_performance_report()
        
        performance = {
            'strategies': {},
            'top_performers': [],
            'underperformers': []
        }
        
        strategy_scores = []
        
        for strategy, metrics in report.get('strategy_metrics', {}).items():
            performance['strategies'][strategy] = metrics
            
            # Calcular score
            score = (
                metrics.get('win_rate', 0) * 0.3 +
                min(metrics.get('profit_factor', 0) / 2, 1) * 0.3 +
                min(max(metrics.get('sharpe_ratio', 0), 0), 2) / 2 * 0.4
            )
            
            strategy_scores.append((strategy, score, metrics.get('total_pnl', 0)))
        
        # Ordenar por score
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 y peores 3
        performance['top_performers'] = [
            {'strategy': s[0], 'score': s[1], 'pnl': s[2]}
            for s in strategy_scores[:3]
        ]
        
        performance['underperformers'] = [
            {'strategy': s[0], 'score': s[1], 'pnl': s[2]}
            for s in strategy_scores[-3:]
        ]
        
        return performance
    
    def _prepare_html_sections(self, report_data: Dict) -> List[Dict]:
        """Prepara secciones para reporte HTML."""
        
        sections = []
        
        # Resumen ejecutivo
        exec_summary = report_data['sections'].get('executive_summary', {})
        if exec_summary:
            sections.append({
                'title': '📊 Resumen Ejecutivo',
                'type': 'metrics',
                'content': exec_summary.get('metrics', {})
            })
        
        # Performance de estrategias
        strategy_perf = report_data['sections'].get('strategy_performance', {})
        if strategy_perf and strategy_perf.get('strategies'):
            # Convertir a DataFrame
            df = pd.DataFrame.from_dict(strategy_perf['strategies'], orient='index')
            df = df.round(2)
            
            sections.append({
                'title': '🎯 Performance por Estrategia',
                'type': 'table',
                'content': df
            })
        
        # Top performers
        if strategy_perf.get('top_performers'):
            top_df = pd.DataFrame(strategy_perf['top_performers'])
            sections.append({
                'title': '🏆 Mejores Estrategias',
                'type': 'table',
                'content': top_df
            })
        
        # Risk management
        risk = report_data['sections'].get('risk_management', {})
        if risk:
            risk_metrics = risk.get('risk_metrics', {})
            sections.append({
                'title': '🛡️ Gestión de Riesgo',
                'type': 'metrics',
                'content': risk_metrics
            })
        
        # System health
        health = report_data['sections'].get('system_health', {})
        if health:
            sections.append({
                'title': '💚 Salud del Sistema',
                'type': 'text',
                'content': f"Estado: {health.get('overall_status', 'unknown')}"
            })
        
        return sections
    
    def generate_strategy_report(
        self,
        strategy_name: str,
        adaptive_manager: Any
    ) -> Path:
        """
        Genera reporte detallado de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            adaptive_manager: Instancia de AdaptiveStrategyManager
        
        Returns:
            Path al reporte generado
        """
        logger.info(f"📊 Generando reporte para {strategy_name}...")
        
        # Obtener métricas
        metrics = adaptive_manager.stats_layer.calculate_strategy_metrics(strategy_name)
        
        report_data = {
            'strategy': strategy_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'historical_performance': []
        }
        
        # Obtener trades históricos
        for key, trades in adaptive_manager.stats_layer.historical_performance.items():
            if key.startswith(strategy_name):
                report_data['historical_performance'].extend(trades)
        
        # Análisis de trades
        if report_data['historical_performance']:
            df = pd.DataFrame(report_data['historical_performance'])
            
            # Equity curve
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Análisis por régimen
            regime_analysis = df.groupby('regime').agg({
                'success': 'mean',
                'pnl': 'sum',
                'return_pct': 'mean'
            }).to_dict()
            
            report_data['regime_analysis'] = regime_analysis
        
        # Guardar
        filename = f"strategy_report_{strategy_name}_{datetime.now().strftime('%Y%m%d')}"
        return self.generator.generate_json_report(report_data, filename)
    
    def generate_trades_export(
        self,
        adaptive_manager: Any,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Path:
        """
        Exporta trades a CSV.
        
        Args:
            adaptive_manager: Instancia de AdaptiveStrategyManager
            start_date: Fecha inicial (opcional)
            end_date: Fecha final (opcional)
        
        Returns:
            Path al archivo CSV
        """
        logger.info("📊 Exportando trades a CSV...")
        
        # Recopilar todos los trades
        all_trades = []
        
        for trades in adaptive_manager.stats_layer.historical_performance.values():
            all_trades.extend(trades)
        
        if not all_trades:
            logger.warning("No hay trades para exportar")
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(all_trades)
        
        # Convertir timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filtrar por fecha si se especifica
        if start_date:
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Exportar
        filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.generator.generate_csv_report(df, filename)
    
    def generate_report(self):
        """Genera reporte final de funcionamiento (legacy method)."""
        
        logger.info("="*60)
        logger.info("📊 REPORTE FINAL DEL SISTEMA")
        logger.info("="*60)
        
        if not self.action_log:
            logger.info("No hay acciones registradas")
            return
        
        # Agrupar por módulo
        by_module = defaultdict(list)
        
        for action in self.action_log:
            by_module[action['module']].append(action)
        
        # Mostrar por módulo
        for module, actions in by_module.items():
            logger.info(f"\n{module}:")
            logger.info(f"  Total acciones: {len(actions)}")
            
            # Últimas 3 acciones
            for action in actions[-3:]:
                logger.info(f"  - {action['action']}")
                logger.info(f"    Resultado: {action['result']}")
        
        # Resumen
        logger.info("\n" + "="*60)
        logger.info(f"Total módulos: {len(by_module)}")
        logger.info(f"Total acciones: {len(self.action_log)}")
        logger.info("="*60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del reporter."""
        
        by_module = defaultdict(int)
        
        for action in self.action_log:
            by_module[action['module']] += 1
        
        return {
            'total_actions_logged': len(self.action_log),
            'modules_tracked': len(by_module),
            'actions_by_module': dict(by_module),
            'reports_directory': str(self.generator.output_dir)
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear reporter
    reporter = SystemReporter()
    
    # Log de acciones
    reporter.log_action(
        module_name="DataManager",
        action_desc="Inicialización",
        expected="DataManager listo",
        result="✅ Inicializado correctamente"
    )
    
    reporter.log_action(
        module_name="StrategyEngine",
        action_desc="Generación de señales",
        expected="Señales generadas",
        result="✅ 5 señales generadas"
    )
    
    reporter.log_action(
        module_name="ExecutionInterface",
        action_desc="Ejecución de órdenes",
        expected="Órdenes ejecutadas",
        result="✅ 3 órdenes ejecutadas"
    )
    
    # Generar reporte
    print("\n" + "="*60)
    print("GENERANDO REPORTE:")
    print("="*60)
    reporter.generate_report()
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS:")
    print("="*60)
    stats = reporter.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Ejemplo de reporte HTML
    print("\n" + "="*60)
    print("GENERANDO REPORTE HTML DE EJEMPLO:")
    print("="*60)
    
    sections = [
        {
            'title': 'Resumen Ejecutivo',
            'type': 'metrics',
            'content': {
                'Total Trades': 150,
                'Total PnL': 2500.50,
                'Win Rate': '65%',
                'Sharpe Ratio': 1.85
            }
        },
        {
            'title': 'Performance por Estrategia',
            'type': 'table',
            'content': pd.DataFrame({
                'Strategy': ['TrendFollowing', 'MeanReversion', 'Breakout'],
                'Trades': [50, 60, 40],
                'Win Rate': [0.60, 0.70, 0.55],
                'PnL': [1000, 1200, 300]
            })
        }
    ]
    
    html_path = reporter.generator.generate_html_report(
        title="Trading System Report - Example",
        sections=sections,
        filename="example_report"
    )
    
    print(f"✅ Reporte HTML generado en: {html_path}")