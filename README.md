# Advanced Trading Bot - Modular Architecture

Sistema de trading cuantitativo avanzado con soporte multi-activo, machine learning e integraciones con múltiples brokers.

## 🏗️ Arquitectura Modular

El proyecto está organizado en módulos especializados para facilitar el mantenimiento y la escalabilidad:

```
mi-proyecto/
├── data_processing/          # Procesamiento de datos
│   ├── external_apis/        # APIs externas (Alpaca, Forex Factory)
│   ├── news/                 # Agregación de noticias y análisis
│   └── scrapers/             # Web scraping de datos financieros
│
├── machine_learning/         # Machine Learning
│   ├── models/               # Modelos (LSTM, BERT, RNN)
│   ├── training/             # Pipeline de entrenamiento
│   ├── prediction/           # Motor de predicciones
│   └── utils/                # Utilidades ML
│
├── trading/                  # Trading
│   ├── brokers/              # Integraciones con brokers
│   ├── strategies/           # Estrategias de trading
│   └── execution/            # Ejecución de órdenes
│
├── shared/                   # Código compartido (legacy)
│   └── core/                 # Módulos core originales
│
├── data/                     # Datos y logs
├── tests/                    # Tests unitarios
└── main.py                   # Punto de entrada principal
```

## 🚀 Características Principales

### 1. Procesamiento de Datos Avanzado

#### APIs Externas
- **Alpaca API**: Integración con acciones y bonos estadounidenses
- **Forex Factory**: Datos macroeconómicos y calendario económico
- **CCXT**: Soporte para 100+ exchanges de criptomonedas

#### Agregación de Noticias
- Múltiples fuentes de noticias financieras
- Análisis de sentimiento con NLP
- Detección de eventos que mueven el mercado

#### Web Scraping
- Indicadores económicos
- Sentimiento de mercado desde redes sociales
- Datos de precio cuando las APIs no están disponibles

### 2. Machine Learning

#### Modelos de Time Series
- **LSTM/RNN**: Predicción de precios y tendencias
- Análisis de secuencias temporales
- Forecasting multi-horizonte

#### Modelos NLP
- **BERT**: Análisis de sentimiento en noticias
- Procesamiento de texto financiero
- Clasificación de eventos de mercado

#### Pipeline de Entrenamiento
- Preparación automatizada de datos
- Búsqueda de hiperparámetros
- Validación cruzada
- Evaluación de modelos

#### Motor de Predicciones
- Predicciones en tiempo real
- Cache inteligente
- Predicciones ensemble
- Análisis de volatilidad y tendencias

### 3. Trading Multi-Activo

#### Clases de Activos Soportadas
- 🪙 **Criptomonedas**: Bitcoin, Ethereum, 100+ altcoins
- 📈 **Acciones**: Mercado estadounidense via Alpaca
- 💰 **Bonos**: Bonos del tesoro y corporativos
- 🌍 **Forex**: Pares de divisas principales
- 🛢️ **Commodities**: Oro, petróleo, metales
- 📊 **ETFs**: Fondos cotizados
- 🎯 **Opciones y Futuros**: Derivados

#### Brokers Soportados
- **Alpaca**: Acciones, bonos, ETFs (papel y real)
- **Huobi**: Criptomonedas y futuros
- **Binance**: Criptomonedas (via CCXT)
- **TD Ameritrade**: Acciones y opciones (preparado)
- **Interactive Brokers**: Multi-activo (preparado)

#### Estrategias de Trading
- **Trend Following**: Seguimiento de tendencias con MAs
- **Mean Reversion**: Reversión a la media con Bollinger Bands
- **ML Enhanced**: Estrategias potenciadas con ML
- **Multi-estrategia**: Combinación de señales

## 📦 Instalación

### Requisitos Previos
- Python 3.9 o superior
- pip (gestor de paquetes)
- (Opcional) GPU con CUDA para entrenamiento ML acelerado

### Instalación Básica

```bash
# Clonar el repositorio
git clone <your-repository-url>
cd mi-proyecto

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de APIs

Crear un archivo `.env` en la raíz del proyecto:

```env
# Alpaca
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_PAPER=true

# Huobi
HUOBI_API_KEY=your_huobi_key
HUOBI_SECRET_KEY=your_huobi_secret

# Otras APIs
FOREX_FACTORY_API_KEY=optional
```

## 🎯 Uso

### Ejemplo Básico

```python
from data_processing.external_apis.api_integrations import AlpacaAPI, APIManager
from machine_learning.prediction.prediction_engine import PredictionEngine
from trading.strategies.trading_strategies import StrategyManager, TrendFollowingStrategy

# Inicializar componentes
api_manager = APIManager()
prediction_engine = PredictionEngine()
strategy_manager = StrategyManager()

# Configurar API de Alpaca
alpaca = AlpacaAPI(api_key="your_key", secret_key="your_secret")
api_manager.register_api("alpaca", alpaca)

# Registrar estrategia
trend_strategy = TrendFollowingStrategy()
strategy_manager.register_strategy(trend_strategy)
strategy_manager.activate_strategy("TrendFollowing")

# Obtener datos y generar señales
# ... (ver ejemplos completos en cada módulo)
```

### Ejecutar el Sistema Principal

```bash
python main.py --mode paper --capital 10000
```

### Entrenar Modelos ML

```python
from machine_learning.models.ml_models import LSTMModel
from machine_learning.training.model_training import ModelTrainer, DataPreparator

# Preparar datos
preparator = DataPreparator()
X, y = preparator.prepare_timeseries_data(price_data)

# Entrenar modelo
lstm = LSTMModel(input_dim=10)
lstm.build(hidden_units=64, num_layers=2)

trainer = ModelTrainer()
history = trainer.train_model(lstm, X_train, y_train)
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=. --cov-report=html

# Tests específicos
pytest tests/unit/test_data_manager.py
```

## 📊 Módulos Principales

### Data Processing
- `api_integrations.py`: Integraciones con APIs externas
- `news_aggregator.py`: Agregación y análisis de noticias
- `web_scraper.py`: Web scraping genérico

### Machine Learning
- `ml_models.py`: Modelos LSTM y BERT
- `model_training.py`: Pipeline de entrenamiento
- `prediction_engine.py`: Motor de predicciones

### Trading
- `broker_integrations.py`: Integraciones con brokers
- `trading_strategies.py`: Estrategias de trading

## 🔒 Seguridad

- Credenciales almacenadas en variables de entorno
- No se commitean secretos al repositorio
- Validación de entrada en todas las APIs
- Rate limiting en requests a APIs externas

## 🚧 Roadmap

- [ ] Integración completa con TD Ameritrade
- [ ] Soporte para más exchanges crypto
- [ ] Dashboard web interactivo
- [ ] Backtesting avanzado con datos históricos
- [ ] Optimización automática de hiperparámetros
- [ ] Alertas en tiempo real (Telegram, email)
- [ ] API REST para control remoto

## 📝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es privado. Todos los derechos reservados.

## 📞 Contacto

Para preguntas o soporte, contacta al equipo de desarrollo.

## ⚠️ Disclaimer

Este software es para fines educativos y de investigación. El trading implica riesgos significativos. 
No nos hacemos responsables de pérdidas financieras. Usa bajo tu propio riesgo.

---

**Versión**: 2.0.0  
**Última actualización**: 2026-01-21
