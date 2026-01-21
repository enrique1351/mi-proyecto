# 🚀 Quant Trading System v1.0.0

Sistema de Trading Cuantitativo Universal - Modular, Adaptativo y Seguro

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Módulos](#-módulos)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

---

## 🎯 Descripción

Sistema de trading cuantitativo completamente modular que soporta:

- **Multi-Asset**: Crypto, Acciones, Forex, Commodities, Bonos, Opciones, ETFs
- **Multi-Broker**: Binance, Coinbase, Interactive Brokers, Alpaca, etc.
- **Multi-Strategy**: Trend Following, Mean Reversion, Breakout, Momentum, y más
- **Adaptive AI**: Sistema que aprende y evoluciona automáticamente
- **Risk Management**: Gestión de riesgo avanzada con kill-switch
- **Security First**: Encriptación AES-256, anomaly detection

---

## ✨ Características

### Core Features

✅ **Trading Automatizado**
- Ejecución automática 24/7
- Paper trading y real trading
- Multi-timeframe analysis
- Smart order routing

✅ **Gestión de Riesgo**
- Position sizing dinámico (Kelly, Volatility-based, Risk Parity)
- Stop-loss y take-profit automáticos
- Drawdown protection
- Correlation risk management
- Kill-switch automático

✅ **Inteligencia Artificial**
- Detección automática de régimen de mercado
- Adaptación de estrategias según performance
- AI Auditor supervisando 24/7
- Anomaly detection
- Strategy optimization

✅ **Data Management**
- SQLite + Pandas (optimizado para time-series)
- Múltiples data providers (Binance, Yahoo Finance, etc.)
- Cache inteligente en memoria
- Historical data storage

✅ **Seguridad**
- Credenciales encriptadas (AES-256)
- Hardware fingerprint
- API firewall
- Rate limiting
- Integrity monitoring

✅ **Reporting**
- Reportes automáticos (JSON, CSV, HTML)
- Dashboard de métricas
- Equity curve visualization
- Strategy performance analytics

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                     QUANT TRADING SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Data Layer   │  │ Strategy     │  │ Execution    │     │
│  │              │  │ Layer        │  │ Layer        │     │
│  │ - Ingestion  │  │ - Engine     │  │ - Brokers    │     │
│  │ - Manager    │  │ - Adaptive   │  │ - Interface  │     │
│  │ - Storage    │  │ - Regime     │  │ - Risk Mgmt  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ AI Layer     │  │ Security     │  │ Monitoring   │     │
│  │              │  │ Layer        │  │ Layer        │     │
│  │ - Auditor    │  │ - Vault      │  │ - Reporter   │     │
│  │ - Analyzer   │  │ - KillSwitch │  │ - Alerts     │     │
│  │ - Optimizer  │  │ - Firewall   │  │ - Logs       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Instalación

### Requisitos Previos

- Python 3.10 o superior
- pip
- Git
- (Opcional) TA-Lib para indicadores técnicos

### Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/quant-system.git
cd quant-system

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Copiar configuración
cp .env.example .env

# Editar .env con tus credenciales
nano .env  # o tu editor preferido
```

### Instalación de TA-Lib (Opcional pero Recomendado)

**macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**Ubuntu/Debian:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

**Windows:**
- Descargar binarios desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Instalar: `pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl`

---

## ⚙️ Configuración

### 1. Configurar Credenciales

Editar `.env` con tus API keys:

```bash
# Exchanges
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret

# Security
VAULT_SECRET=un-secret-muy-fuerte-y-aleatorio
```

### 2. Configurar Assets

Editar `shared/core/constants.py` para seleccionar assets:

```python
# Ejemplo: solo crypto majors
ASSETS = {
    "crypto": {
        "spot": {
            "majors": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        }
    }
}
```

### 3. Configurar Parámetros de Trading

En `shared/core/constants.py`:

```python
TRADING_CONFIG = {
    "initial_capital": 10000,
    "max_drawdown": 0.20,  # 20%
    "risk_per_trade": 0.02,  # 2%
    "max_open_positions": 10
}
```

---

## 🚀 Uso

### Modo Paper Trading (Simulación)

```bash
# Básico: 10 ciclos con $10,000
python main.py --mode paper --capital 10000 --cycles 10

# Con intervalo personalizado (60 segundos)
python main.py --mode paper --cycles 50 --interval 60

# Con AI habilitado
python main.py --mode paper --ai --cycles 100

# Con logging detallado
python main.py --mode paper --log-level DEBUG
```

### Modo Real Trading

⚠️ **ADVERTENCIA**: Asegúrate de haber testeado extensivamente en paper mode primero.

```bash
# Con capital limitado para empezar
python main.py --mode real --capital 1000 --cycles 10

# Producción
python main.py --mode real --capital 10000 --cycles 1000 --interval 300
```

### Opciones de Línea de Comandos

```
--mode          Modo de ejecución: 'paper' o 'real' (default: paper)
--capital       Capital inicial en USD (default: 10000)
--cycles        Número de ciclos a ejecutar (default: 10)
--interval      Segundos entre ciclos (default: 30)
--ai            Habilitar AI Auditor avanzado
--log-level     Nivel de logging: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

---

## 📚 Módulos

### 1. Data Layer

**data_manager.py**
- Almacenamiento SQLite + cache Pandas
- Gestión de datos OHLCV multi-asset
- Indicadores técnicos
- Performance tracking

**data_ingestion.py**
- Multi-source data providers
- Binance, Yahoo Finance, Mock
- Rate limiting
- Auto-selection de providers

### 2. Strategy Layer

**strategy_engine.py**
- Sistema modular de estrategias
- TrendFollowing, MeanReversion, Breakout, Momentum
- Registry pattern
- Multi-timeframe

**adaptive_strategy_manager.py**
- Adaptación automática de estrategias
- Statistics layer
- Confidence adjustment
- Regime-based selection

### 3. Execution Layer

**execution_interface.py**
- Multi-broker abstraction
- Order types: Market, Limit, Stop-Loss, Take-Profit
- Position tracking
- Slippage modeling

**brokers.py**
- Binance, Coinbase, MockBroker
- Factory pattern
- Unified API

### 4. Risk Management

**risk_manager.py**
- Position sizing (Kelly, Volatility, Risk Parity)
- Stop-loss/take-profit automation
- Drawdown protection
- Correlation risk
- Kill-switch

### 5. Analysis

**market_regime.py**
- 8 regímenes identificables
- ADX, Hurst Exponent
- Market structure detection
- Volatility regimes

### 6. AI Layer

**ai_auditor.py**
- Anomaly detection
- Performance monitoring
- Alert system (4 levels)
- Strategy health scores
- Auto-optimization

### 7. Security

**credential_vault.py**
- AES-256 encryption
- Hardware fingerprint
- Key rotation
- Secure storage

### 8. Monitoring

**system_reporter.py**
- JSON, CSV, HTML reports
- Equity curve visualization
- Performance analytics
- Daily/weekly/monthly reports

---

## 🧪 Testing

```bash
# Ejecutar tests unitarios
pytest tests/unit/

# Tests de integración
pytest tests/integration/

# Con coverage
pytest --cov=shared tests/

# Test específico
pytest tests/unit/test_strategy_engine.py
```

---

## 🌐 Deployment

### Local 24/7

```bash
# Usando screen o tmux
screen -S trading
python main.py --mode real --cycles 99999 --interval 300
# Ctrl+A D para detach
```

### Cloud (AWS, GCP, DigitalOcean)

Ver `docs/DEPLOYMENT.md` para guías detalladas de deployment en cloud.

---

## 🗺️ Roadmap

### ✅ v1.0.0 (Actual)
- [x] Sistema core completo
- [x] Multi-asset support
- [x] Risk management
- [x] AI Auditor básico

### 🔜 v1.1.0 (Q2 2025)
- [ ] Integración con Claude API para strategy generation
- [ ] Backtesting engine completo
- [ ] Walk-forward analysis
- [ ] Dashboard web interactivo

### 🔮 v2.0.0 (Q3 2025)
- [ ] Machine Learning models (LSTM, XGBoost)
- [ ] Reinforcement Learning
- [ ] Multi-exchange arbitrage
- [ ] High-frequency trading module

---

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crear branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

---

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

---

## ⚠️ Disclaimer

**IMPORTANTE**: Este sistema es para fines educativos y de investigación. El trading conlleva riesgos significativos. Nunca operes con dinero que no puedas permitirte perder. Los resultados pasados no garantizan resultados futuros.

**NO nos hacemos responsables de pérdidas financieras derivadas del uso de este sistema.**

---

## 📞 Soporte

- 📧 Email: support@quantsystem.com
- 💬 Discord: [Únete a la comunidad](https://discord.gg/quantsystem)
- 📖 Docs: [docs.quantsystem.com](https://docs.quantsystem.com)

---

## 🙏 Agradecimientos

- Anthropic (Claude AI)
- Comunidad de trading cuantitativo
- Contributors y testers

---

**Made with ❤️ by the Quant Trading System Team**