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
- [Deployment](#-deployment)
- [Notificaciones](#-notificaciones)
- [Machine Learning](#-machine-learning)
- [Seguridad](#-seguridad)
- [Testing](#-testing)
- [Roadmap](#-roadmap)

---

## 🎯 Descripción

Sistema de trading cuantitativo completamente modular que soporta:

- **Multi-Asset**: Crypto, Acciones, Forex, Commodities, Bonos, Opciones, ETFs
- **Multi-Broker**: Binance, Alpaca, Oanda, Interactive Brokers, Coinbase, etc.
- **Multi-Strategy**: Trend Following, Mean Reversion, Breakout, Momentum, y más
- **Adaptive AI**: Sistema que aprende y evoluciona automáticamente
- **Risk Management**: Gestión de riesgo avanzada con kill-switch
- **Security First**: Encriptación AES-256, anomaly detection
- **24/7 Operation**: Desplegable en VPS, Docker, Cloud

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
- Time series prediction (LSTM, ARIMA)
- NLP para análisis de noticias (BERT)

✅ **Data Management**
- SQLite + Pandas (optimizado para time-series)
- Múltiples data providers (Binance, Yahoo Finance, Alpaca, etc.)
- Cache inteligente en memoria
- Historical data storage

✅ **Notificaciones en Tiempo Real**
- Telegram Bot API
- Email (SMTP)
- SMS (Twilio)
- Push notifications (Pushbullet)

✅ **Seguridad**
- Credenciales encriptadas (AES-256)
- Azure KeyVault / AWS Secrets Manager
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
│  │ - ML Models  │  │ - KillSwitch │  │ - Alerts     │     │
│  │ - NLP        │  │ - Firewall   │  │ - Logs       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Trading      │  │ Notifications│  │ ML           │     │
│  │ - Alpaca     │  │ - Telegram   │  │ - LSTM       │     │
│  │ - Oanda      │  │ - Email      │  │ - ARIMA      │     │
│  │ - Broker Mgr │  │ - SMS/Push   │  │ - NLP/BERT   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Estructura de Directorios

```
quant-system/
├── shared/core/           # Core system modules
│   ├── ai/               # AI auditor
│   ├── analysis/         # Market regime detection
│   ├── brokers/          # Broker integrations (Binance, etc.)
│   ├── config/           # Configuration and constants
│   ├── data/             # Data management
│   ├── execution/        # Order execution
│   ├── monitoring/       # System monitoring
│   ├── risk/             # Risk management
│   ├── security/         # Security and credentials
│   └── strategies/       # Strategy engine
├── trading/              # NEW: Trading module
│   ├── alpaca_broker.py  # Alpaca integration
│   ├── forex_broker.py   # Oanda/Forex integration
│   └── broker_manager.py # Unified broker management
├── notifications/        # NEW: Notifications module
│   ├── telegram_notifier.py
│   ├── email_notifier.py
│   ├── sms_notifier.py
│   ├── pushbullet_notifier.py
│   └── notification_manager.py
├── machine_learning/     # NEW: ML module
│   ├── time_series_models.py  # LSTM, ARIMA
│   └── nlp_analyzer.py        # BERT, sentiment analysis
├── data_processing/      # NEW: Data processing module
├── data/                 # Data storage
│   ├── db/              # Databases
│   ├── logs/            # Log files
│   └── cache/           # Cache
├── tests/               # Unit tests
├── docker/              # Docker configurations
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.prod.yml  # Production deployment
└── README.md           # This file
```

---

## 📦 Instalación

### Requisitos Previos

- Python 3.10 o superior
- pip
- Git
- (Opcional) Docker y Docker Compose
- (Opcional) TA-Lib para indicadores técnicos

### Instalación Local

```bash
# 1. Clonar repositorio
git clone https://github.com/enrique1351/mi-proyecto.git
cd mi-proyecto

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Copiar configuración
cp .env.example .env

# 6. Editar .env con tus credenciales
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

### 1. Configurar Credenciales (.env)

```bash
# ============================================================================
# CRYPTO EXCHANGES
# ============================================================================
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# ============================================================================
# STOCK BROKERS
# ============================================================================
# Alpaca
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ============================================================================
# FOREX BROKERS
# ============================================================================
# Oanda
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_oanda_account_id_here
OANDA_PRACTICE=true

# ============================================================================
# NOTIFICATIONS
# ============================================================================
# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Twilio (SMS)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1234567890

# Pushbullet
PUSHBULLET_TOKEN=your_pushbullet_token

# ============================================================================
# SECURITY
# ============================================================================
VAULT_SECRET=CHANGE-THIS-TO-A-STRONG-RANDOM-SECRET-KEY

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
ENVIRONMENT=local
LOG_LEVEL=INFO
TRADING_MODE=paper
INITIAL_CAPITAL=10000
```

### 2. Configurar Assets

Editar `shared/core/config/constants.py` para seleccionar assets:

```python
ASSETS = {
    "crypto": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "stocks": ["AAPL", "GOOGL", "MSFT"],
    "forex": ["EUR_USD", "GBP_USD"]
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
```

---

## 🐳 Deployment

### Docker (Local)

```bash
# Build imagen
docker build -t trading-bot .

# Run container
docker run -d \
  --name trading-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  trading-bot
```

### Docker Compose (Producción)

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f trading-bot

# Stop services
docker-compose -f docker-compose.prod.yml down
```

### VPS Deployment (DigitalOcean/AWS)

```bash
# 1. Crear VPS (Ubuntu 22.04 LTS)
# 2. Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Instalar Docker Compose
sudo apt-get install docker-compose-plugin

# 4. Clonar repositorio
git clone https://github.com/enrique1351/mi-proyecto.git
cd mi-proyecto

# 5. Configurar .env
cp .env.example .env
nano .env

# 6. Deploy
docker compose -f docker-compose.prod.yml up -d

# 7. Configurar systemd para auto-start
sudo systemctl enable docker
```

### Crear Ejecutable con PyInstaller

```bash
# Instalar PyInstaller
pip install pyinstaller

# Crear ejecutable
pyinstaller --onefile \
  --add-data "shared:shared" \
  --add-data "trading:trading" \
  --add-data "notifications:notifications" \
  --add-data "machine_learning:machine_learning" \
  --hidden-import=numpy \
  --hidden-import=pandas \
  main.py

# Ejecutable estará en dist/main.exe (Windows) o dist/main (Linux/Mac)
```

---

## 🔔 Notificaciones

El sistema soporta múltiples canales de notificación:

### Telegram Bot

1. Crear bot con @BotFather en Telegram
2. Obtener bot token
3. Obtener chat ID: enviar mensaje al bot y visitar:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
4. Configurar en .env:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

### Email (SMTP)

Configurar en .env con tu proveedor SMTP (Gmail, Outlook, etc.)

### SMS (Twilio)

1. Crear cuenta en Twilio
2. Obtener credenciales
3. Configurar en .env

### Pushbullet

1. Crear cuenta en Pushbullet
2. Obtener API token
3. Configurar en .env

---

## 🤖 Machine Learning

### Time Series Prediction

El sistema incluye modelos LSTM y ARIMA para predicción de precios:

```python
from machine_learning.time_series_models import LSTMPredictor

# Crear predictor
predictor = LSTMPredictor(sequence_length=60, units=50)

# Entrenar
predictor.train(X_train, y_train, epochs=50)

# Predecir
predictions = predictor.predict(X_test)
```

### NLP para News Analysis

Análisis de sentimiento de noticias con BERT:

```python
from machine_learning.nlp_analyzer import NewsAnalyzer

# Crear analizador
analyzer = NewsAnalyzer()

# Analizar sentimiento
sentiment = analyzer.analyze_sentiment("Bitcoin reaches new all-time high")

# Obtener sentimiento de mercado
market_sentiment = analyzer.get_market_sentiment(news_list)
```

---

## 🔐 Seguridad

### Credential Management

El sistema usa `CredentialVault` para almacenar credenciales encriptadas:

```python
from shared.core.security.credential_vault import CredentialVault

vault = CredentialVault()
vault.set_credential('binance', 'api_key', 'your_key')
api_key = vault.get_credential('binance', 'api_key')
```

### Azure KeyVault / AWS Secrets Manager

Para producción, se recomienda usar servicios cloud:

```bash
# Azure KeyVault
AZURE_KEYVAULT_URI=https://your-vault.vault.azure.net

# AWS Secrets Manager
AWS_SECRETS_REGION=us-east-1
AWS_SECRET_NAME=trading-bot-secrets
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shared --cov=trading --cov=notifications --cov=machine_learning

# Run specific test file
pytest tests/unit/test_notifications.py
```

---

## 📊 Módulos

### Core Modules (shared/core/)

- **Data Management**: Ingesta y almacenamiento de datos
- **Strategy Engine**: Motor de estrategias adaptativas
- **Risk Management**: Gestión de riesgo y posiciones
- **Execution**: Ejecución de órdenes multi-broker
- **AI Auditor**: Auditoría inteligente con IA

### Trading Module (trading/)

- **Alpaca Broker**: Integración con Alpaca para acciones
- **Oanda Broker**: Integración con Oanda para forex
- **Broker Manager**: Gestión unificada de brokers

### Notifications Module (notifications/)

- **Telegram**: Notificaciones vía Telegram Bot
- **Email**: Notificaciones vía SMTP
- **SMS**: Notificaciones vía Twilio
- **Pushbullet**: Push notifications

### Machine Learning Module (machine_learning/)

- **Time Series Models**: LSTM, ARIMA para predicción
- **NLP Analyzer**: BERT para análisis de sentimiento

---

## 🗺️ Roadmap

### Fase 1: ✅ Completada
- [x] Modularización del código
- [x] Integración Alpaca
- [x] Integración Oanda/Forex
- [x] Sistema de notificaciones (Telegram, Email, SMS, Pushbullet)
- [x] Modelos ML (LSTM, ARIMA, BERT)
- [x] Dockerización
- [x] Documentación completa

### Fase 2: En Progreso
- [ ] Interactive Brokers integration
- [ ] Coinbase Advanced Trade integration
- [ ] Dashboard web (Streamlit/Dash)
- [ ] Backtesting mejorado
- [ ] Paper trading simulator avanzado

### Fase 3: Futuro
- [ ] Options trading
- [ ] Multi-account management
- [ ] Advanced ML models (Transformers, Reinforcement Learning)
- [ ] Cloud-native deployment (Kubernetes)
- [ ] API REST para control remoto

---

## 📝 Licencia

MIT License

---

## 👥 Contribución

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contacto

- GitHub: [@enrique1351](https://github.com/enrique1351)
- Email: Contact via GitHub

---

## ⚠️ Disclaimer

**TRADING DISCLAIMER**: This software is for educational purposes only. Trading cryptocurrencies, stocks, forex, and other financial instruments involves substantial risk of loss. Past performance is not indicative of future results. Use at your own risk.

---

## 🙏 Acknowledgments

- CCXT for crypto exchange integration
- Alpaca Markets for stock trading API
- Oanda for forex trading API
- Hugging Face for NLP models
- OpenAI/Anthropic for AI capabilities

---

**Made with ❤️ for algorithmic traders**
