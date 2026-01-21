# Migration Guide - Sistema Modular

## Resumen de Cambios

El proyecto ha sido reorganizado en una arquitectura modular para mejorar la mantenibilidad, escalabilidad y claridad del código.

## Nueva Estructura

### Módulos Principales

1. **data_processing/** - Todo lo relacionado con obtención de datos
   - `external_apis/` - Integraciones con APIs externas (Alpaca, Forex Factory)
   - `news/` - Agregación y análisis de noticias
   - `scrapers/` - Web scraping

2. **machine_learning/** - Todo lo relacionado con ML/AI
   - `models/` - Modelos LSTM, BERT, RNN
   - `training/` - Pipeline de entrenamiento
   - `prediction/` - Motor de predicciones
   - `utils/` - Utilidades ML

3. **trading/** - Todo lo relacionado con trading
   - `brokers/` - Integraciones con brokers
   - `strategies/` - Estrategias de trading
   - `execution/` - Ejecución de órdenes

## Migración de Código Existente

### Desde shared/core/ a la Nueva Estructura

| Archivo Original | Nueva Ubicación | Estado |
|-----------------|-----------------|--------|
| `shared/core/data/data_ingestion.py` | `data_processing/external_apis/` | ✅ Extendido |
| `shared/core/brokers/brokers.py` | `trading/brokers/broker_integrations.py` | ✅ Mejorado |
| `shared/core/strategies/` | `trading/strategies/` | ✅ Extendido |
| - | `data_processing/news/` | ✅ Nuevo |
| - | `data_processing/scrapers/` | ✅ Nuevo |
| - | `machine_learning/models/` | ✅ Nuevo |
| - | `machine_learning/training/` | ✅ Nuevo |
| - | `machine_learning/prediction/` | ✅ Nuevo |

### Actualizaciones de Imports

#### Antes:
```python
from shared.core.brokers.brokers import BrokerFactory, BrokerType
from shared.core.data.data_ingestion import DataIngestion
from shared.core.strategies.strategy_engine import StrategyEngine
```

#### Ahora:
```python
# Brokers
from trading.brokers.broker_integrations import BrokerManager, AlpacaBroker, HuobiBroker

# Datos
from data_processing.external_apis.api_integrations import APIManager, AlpacaAPI
from data_processing.news.news_aggregator import NewsAggregator

# Machine Learning
from machine_learning.models.ml_models import LSTMModel, BERTSentimentModel
from machine_learning.prediction.prediction_engine import PredictionEngine

# Estrategias
from trading.strategies.trading_strategies import StrategyManager, TrendFollowingStrategy
```

## Nuevas Funcionalidades

### 1. APIs Externas Adicionales

```python
from data_processing.external_apis.api_integrations import AlpacaAPI, ForexFactoryAPI

# Alpaca para acciones/bonos
alpaca = AlpacaAPI(api_key="your_key", secret_key="your_secret")
alpaca.connect()

# Forex Factory para datos macro
forex = ForexFactoryAPI()
forex.connect()
calendar = forex.get_economic_calendar(start_date, end_date)
```

### 2. Análisis de Noticias

```python
from data_processing.news.news_aggregator import NewsAggregator, SentimentAnalyzer

# Agregar noticias
aggregator = NewsAggregator()
articles = aggregator.fetch_all_news(keywords=["Bitcoin", "stocks"])

# Análisis de sentimiento
analyzer = SentimentAnalyzer()
articles = analyzer.add_sentiment_to_articles(articles)
```

### 3. Machine Learning

```python
from machine_learning.models.ml_models import LSTMModel
from machine_learning.training.model_training import ModelTrainer, DataPreparator
from machine_learning.prediction.prediction_engine import PredictionEngine

# Preparar datos
preparator = DataPreparator()
X, y = preparator.prepare_timeseries_data(df)

# Entrenar modelo
lstm = LSTMModel(input_dim=10)
lstm.build()
trainer = ModelTrainer()
trainer.train_model(lstm, X_train, y_train)

# Hacer predicciones
engine = PredictionEngine()
engine.register_model("lstm", lstm)
prediction = engine.predict_price("BTCUSDT", "lstm", data)
```

### 4. Estrategias Mejoradas

```python
from trading.strategies.trading_strategies import (
    StrategyManager, 
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MLEnhancedStrategy
)

manager = StrategyManager()

# Registrar múltiples estrategias
manager.register_strategy(TrendFollowingStrategy())
manager.register_strategy(MeanReversionStrategy())
manager.register_strategy(MLEnhancedStrategy(prediction_engine))

# Activar y generar señales
manager.activate_strategy("TrendFollowing")
signals = manager.generate_signals(data, symbol="BTCUSDT")
```

## Archivos Pendientes por Migrar

Los archivos en `pendientes por organizar/` son versiones anteriores o borradores. 
La mayoría de su funcionalidad ha sido incorporada en la nueva estructura:

- `main.py` (viejo) → El `main.py` principal ya existe
- `bootstrap.py` → Funcionalidad integrada en módulos principales
- `strategy_optimizer.py` → Ver `machine_learning/training/`
- `verify_brokers.py` → Ver `trading/brokers/broker_integrations.py`
- Archivos `.txt` → Documentación/notas históricas

## Pasos para Actualizar Código Existente

1. **Actualizar imports** según la tabla de migración
2. **Usar nuevas APIs** cuando estén disponibles (ej: `BrokerManager` en lugar de `BrokerFactory`)
3. **Integrar ML** si es relevante para tu funcionalidad
4. **Probar** el código migrado
5. **Actualizar tests** para reflejar los nuevos imports

## Compatibilidad hacia Atrás

- Los módulos en `shared/core/` **siguen disponibles** para compatibilidad
- Se recomienda migrar gradualmente a la nueva estructura
- Los módulos legacy eventualmente serán deprecados

## Beneficios de la Nueva Estructura

✅ **Separación de responsabilidades clara**  
✅ **Más fácil de mantener y escalar**  
✅ **Mejor organización del código**  
✅ **Facilita el trabajo en equipo**  
✅ **Testing más simple**  
✅ **Documentación más clara**  

## Ejemplo Completo

Ver `integration_example.py` para un ejemplo completo de cómo usar todos los módulos juntos.

## Soporte

Para preguntas sobre la migración, consulta:
- README.md - Documentación general
- Código de ejemplo en cada módulo (`if __name__ == "__main__"`)
- integration_example.py - Ejemplo de integración completa

---

**Última actualización**: 2026-01-21
