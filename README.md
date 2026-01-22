# Sistema de Trading Cuantitativo Universal

[![CI/CD Pipeline](https://github.com/enrique1351/mi-proyecto/actions/workflows/ci.yml/badge.svg)](https://github.com/enrique1351/mi-proyecto/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Sistema de trading cuantitativo modular, robusto y escalable con soporte para múltiples brokers, activos y estrategias.

## 🚀 Características Principales

### Brokers Soportados
- **Alpaca** - Acciones y ETFs de EE.UU.
- **CCXT** - 100+ exchanges de criptomonedas (Binance, Kraken, etc.)
- **Interactive Brokers** - Acciones, opciones, futuros, forex (stub para implementación futura)
- **Mock Broker** - Simulación para pruebas

### Notificaciones en Tiempo Real
- **Telegram Bot** - Alertas instantáneas de trades, errores y rendimiento
- **Twilio SMS** - Notificaciones críticas vía SMS
- **Notificaciones Unificadas** - Manager centralizado para múltiples canales

### Bases de Datos
- **SQLite** - Base de datos ligera por defecto
- **PostgreSQL** - Base de datos empresarial para producción
- **MongoDB** - Almacenamiento NoSQL para datos no estructurados

### Machine Learning
- **Price Predictor** - Predicción de precios usando Random Forest y Gradient Boosting
- **Trend Predictor** - Clasificación de tendencias (UP/DOWN)
- **Model Trainer** - Sistema unificado de entrenamiento y gestión de modelos

### Automatización
- Scripts de configuración para **Raspberry Pi**
- Scripts de despliegue para **VPS**
- Soporte para **Docker** y **Docker Compose**

## 📋 Requisitos

- Python 3.9+
- pip
- virtualenv (recomendado)

## 🛠️ Instalación

### Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/enrique1351/mi-proyecto.git
cd mi-proyecto

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar credenciales
cp .env.example .env
# Editar .env con tus credenciales
```

### Instalación en Raspberry Pi

```bash
chmod +x scripts/setup_raspberry_pi.sh
./scripts/setup_raspberry_pi.sh
```

### Instalación en VPS

```bash
chmod +x scripts/setup_vps.sh
./scripts/setup_vps.sh
```

## 🔧 Configuración

### Archivo .env

Crear un archivo `.env` basado en `.env.example`:

```bash
# Broker Credentials
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+0987654321

# Database
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=your_password
MONGO_USER=trading_user
MONGO_PASSWORD=your_password
```

## 🚦 Uso

### Ejecución Básica

```bash
# Modo paper trading (simulación)
python main.py --mode paper

# Modo real (trading real)
python main.py --mode real

# Con IA habilitada
python main.py --mode paper --use-ai
```

### Ejemplo de Código

```python
from shared.core.brokers.brokers import BrokerFactory, BrokerType
from shared.core.notifications.notification_manager import NotificationManager
from shared.core.ml.model_trainer import ModelTrainer

# Inicializar broker
broker = BrokerFactory.create_broker(BrokerType.ALPACA, paper_trading=True)
broker.connect()

# Obtener balance
balance = broker.get_balance()
print(f"Balance: {balance}")

# Configurar notificaciones
notifier = NotificationManager()
notifier.setup_telegram()
notifier.notify_trade('AAPL', 'BUY', 10, 150.00)

# Entrenar modelos ML
trainer = ModelTrainer()
data = get_market_data()  # Tu función para obtener datos
metrics = trainer.train_all_models(data)
```

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=shared --cov-report=html

# Solo pruebas unitarias
pytest tests/unit/ -v

# Solo pruebas de integración
pytest tests/integration/ -v
```

## 📊 Estructura del Proyecto

```
mi-proyecto/
├── main.py                          # Punto de entrada principal
├── requirements.txt                 # Dependencias Python
├── .env.example                     # Plantilla de configuración
├── shared/
│   └── core/
│       ├── brokers/                 # Integraciones de brokers
│       │   ├── alpaca_broker.py
│       │   ├── ccxt_broker.py
│       │   ├── ib_broker.py
│       │   └── brokers.py
│       ├── notifications/           # Sistema de notificaciones
│       │   ├── telegram_notifier.py
│       │   ├── twilio_notifier.py
│       │   └── notification_manager.py
│       ├── data/                    # Gestión de datos
│       │   ├── data_manager.py
│       │   ├── postgres_manager.py
│       │   └── mongo_manager.py
│       ├── ml/                      # Machine Learning
│       │   ├── price_predictor.py
│       │   ├── trend_predictor.py
│       │   └── model_trainer.py
│       ├── strategies/              # Estrategias de trading
│       ├── execution/               # Motor de ejecución
│       ├── risk/                    # Gestión de riesgo
│       └── monitoring/              # Monitoreo y reportes
├── scripts/                         # Scripts de automatización
│   ├── setup_raspberry_pi.sh
│   └── setup_vps.sh
├── tests/                           # Pruebas
│   ├── unit/
│   └── integration/
└── data/                            # Datos, logs y modelos
    ├── db/
    ├── logs/
    └── models/
```

## 🔐 Seguridad

- ✅ Credenciales almacenadas de forma segura usando CredentialVault
- ✅ Variables de entorno para información sensible
- ✅ Sin credenciales en el código fuente
- ✅ Conexiones encriptadas a APIs
- ✅ Rate limiting para evitar baneos
- ✅ Logs de auditoría

## 📈 Roadmap

- [x] Integración de brokers (Alpaca, CCXT, IB)
- [x] Sistema de notificaciones (Telegram, Twilio)
- [x] Soporte multi-base de datos (PostgreSQL, MongoDB)
- [x] Modelos de Machine Learning
- [x] Scripts de automatización
- [x] CI/CD con GitHub Actions
- [ ] Dashboard web con FastAPI
- [ ] Backtesting avanzado
- [ ] Optimización de estrategias con RL
- [ ] Integración con más brokers

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 📧 Contacto

Para preguntas o soporte, abre un issue en GitHub.

## ⚠️ Disclaimer

Este software es solo para fines educativos. El trading de acciones, criptomonedas y otros instrumentos financieros conlleva riesgos. Usa este sistema bajo tu propia responsabilidad.
