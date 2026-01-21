# Sistema de Trading Cuantitativo Universal

Sistema modular, adaptativo y seguro para trading multi-asset con inteligencia artificial.

## 🚀 Características

- **Multi-Asset**: Soporte para criptomonedas, acciones, forex, commodities, bonos, opciones y ETFs
- **Estrategias Adaptativas**: Sistema de IA que evoluciona y adapta estrategias según condiciones del mercado
- **Gestión de Riesgo**: Sistema robusto de gestión de riesgo con kill-switch automático
- **Arquitectura Modular**: Componentes independientes y reutilizables
- **Seguridad**: Protección contra robo, suplantación y manipulación
- **Backtesting**: Motor de backtesting integrado para validación de estrategias

## 📋 Requisitos Previos

- Python 3.12 o superior
- Docker y Docker Compose (opcional, para despliegue en contenedores)
- TA-Lib (para análisis técnico)

## 🔧 Instalación

### Instalación Local

1. **Clonar el repositorio**:
```bash
git clone https://github.com/enrique1351/mi-proyecto.git
cd mi-proyecto
```

2. **Crear y activar entorno virtual**:
```bash
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/Mac:
source .venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

### Instalación con Docker

```bash
docker-compose up -d
```

## 🎯 Uso

### Modo Paper Trading (Simulación)

```bash
python main.py --mode paper --capital 10000 --cycles 10
```

### Modo Real Trading

```bash
python main.py --mode real --capital 1000 --cycles 100 --ai
```

### Opciones Disponibles

- `--mode`: Modo de ejecución (`paper` o `real`)
- `--capital`: Capital inicial en USD
- `--cycles`: Número de ciclos a ejecutar
- `--interval`: Intervalo entre ciclos en segundos
- `--ai`: Habilitar AI Auditor avanzado
- `--log-level`: Nivel de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## 🏗️ Arquitectura

El sistema está organizado en módulos core:

```
shared/core/
├── ai/                 # AI Auditor
├── analysis/           # Análisis de mercado y detección de régimen
├── backtesting/        # Motor de backtesting
├── brokers/            # Interfaces con brokers
├── config/             # Configuración y constantes
├── data/               # Gestión de datos y database
├── execution/          # Ejecución de órdenes
├── monitoring/         # Monitoreo y reportes
├── risk/               # Gestión de riesgo
├── security/           # Seguridad y credenciales
└── strategies/         # Motor de estrategias
```

## 🧪 Tests

Ejecutar los tests unitarios:

```bash
pytest tests/ -v
```

Ejecutar tests con cobertura:

```bash
pytest tests/ --cov=shared --cov-report=html
```

## 📊 Funcionalidades Principales

### 1. Data Management
- Ingesta de datos en tiempo real
- Almacenamiento eficiente en SQLite
- Soporte para múltiples timeframes

### 2. Strategy Engine
- Sistema de señales multi-estrategia
- Adaptación automática según régimen de mercado
- Backtesting integrado

### 3. Risk Management
- Gestión de capital
- Control de drawdown
- Kill-switch automático
- Position sizing dinámico

### 4. Execution Interface
- Ejecución de órdenes
- Soporte para múltiples brokers
- Modo paper trading

### 5. AI Auditor
- Auditoría automática de trades
- Detección de anomalías
- Recomendaciones de mejora

## 🔐 Seguridad

El sistema implementa múltiples capas de seguridad:

- Almacenamiento seguro de credenciales con encriptación
- Kill-switch automático ante pérdidas excesivas
- Validación de todas las órdenes antes de ejecutar
- Monitoreo continuo de anomalías

## 📈 Monitoreo

El sistema genera reportes automáticos:

- Reportes diarios de performance
- Exportación de trades
- Estadísticas de ejecución
- Métricas de riesgo

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código privado. Todos los derechos reservados.

## ⚠️ Disclaimer

Este software es para fines educativos y de investigación. El trading conlleva riesgos significativos de pérdida de capital. Use bajo su propio riesgo.

## 📧 Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio.

---

**Versión**: 1.0.0  
**Última actualización**: 2026-01-21
