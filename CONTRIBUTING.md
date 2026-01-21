# Contributing Guide

## Cómo Contribuir

¡Gracias por tu interés en contribuir al Sistema de Trading Cuantitativo! Este documento te guiará en el proceso.

## 🚀 Configuración del Entorno de Desarrollo

### 1. Fork y Clonar

```bash
git clone https://github.com/tu-usuario/mi-proyecto.git
cd mi-proyecto
```

### 2. Crear Entorno Virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales de desarrollo
```

## 📝 Guía de Código

### Estilo de Código

- Seguir PEP 8 para Python
- Usar type hints cuando sea posible
- Máximo 100 caracteres por línea
- Usar docstrings para funciones y clases

### Ejemplo de Docstring

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calcula el Sharpe Ratio de una serie de retornos.
    
    Args:
        returns: Serie de pandas con los retornos
        risk_free_rate: Tasa libre de riesgo anualizada (default: 0.02)
    
    Returns:
        float: Sharpe Ratio anualizado
    
    Raises:
        ValueError: Si la serie está vacía o tiene desviación estándar cero
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
    pass
```

### Nombres de Variables

- Variables: `snake_case`
- Constantes: `UPPER_SNAKE_CASE`
- Clases: `PascalCase`
- Funciones: `snake_case`

## 🧪 Tests

### Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/unit/test_data_manager.py -v

# Con cobertura
pytest tests/ --cov=shared --cov-report=html
```

### Escribir Tests

```python
import unittest
from shared.core.data.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.dm = DataManager(":memory:")  # Base de datos en memoria para tests
    
    def test_save_ohlcv(self):
        # Arrange
        data = create_sample_data()
        
        # Act
        result = self.dm.save_ohlcv("BTCUSDT", "1h", data)
        
        # Assert
        self.assertTrue(result)
```

## 🔀 Proceso de Pull Request

### 1. Crear Rama

```bash
git checkout -b feature/nombre-descriptivo
# o
git checkout -b fix/descripcion-del-bug
```

### 2. Hacer Commits

```bash
git add .
git commit -m "Descripción clara del cambio"
```

**Formato de Commits:**

- `feat: ` - Nueva funcionalidad
- `fix: ` - Corrección de bug
- `docs: ` - Cambios en documentación
- `style: ` - Formateo, sin cambios en código
- `refactor: ` - Refactorización de código
- `test: ` - Agregar o modificar tests
- `chore: ` - Mantenimiento general

**Ejemplos:**
```
feat: Add support for Kraken exchange
fix: Correct Sharpe ratio calculation
docs: Update installation instructions
refactor: Simplify risk manager logic
```

### 3. Push y PR

```bash
git push origin feature/nombre-descriptivo
```

Luego crear Pull Request en GitHub con:
- Título descriptivo
- Descripción de cambios
- Referencias a issues relacionados
- Screenshots si aplica

## 📋 Checklist del PR

Antes de enviar tu PR, verifica:

- [ ] El código sigue las guías de estilo
- [ ] Se agregaron/actualizaron tests
- [ ] Todos los tests pasan
- [ ] Se actualizó la documentación
- [ ] No hay credenciales hardcodeadas
- [ ] Los commits tienen mensajes descriptivos
- [ ] El código está comentado donde es necesario

## 🐛 Reportar Bugs

Al reportar un bug, incluye:

1. **Descripción**: ¿Qué pasó?
2. **Reproducción**: Pasos para reproducir el error
3. **Comportamiento esperado**: ¿Qué debería pasar?
4. **Entorno**: OS, versión de Python, versiones de dependencias
5. **Logs**: Mensajes de error o logs relevantes
6. **Screenshots**: Si aplica

## 💡 Sugerir Mejoras

Para sugerir nuevas funcionalidades:

1. **Descripción**: ¿Qué quieres agregar?
2. **Motivación**: ¿Por qué es útil?
3. **Alternativas**: ¿Consideraste otras opciones?
4. **Referencias**: Links a recursos relevantes

## 🏗️ Estructura del Proyecto

```
mi-proyecto/
├── shared/
│   └── core/
│       ├── ai/              # Inteligencia artificial
│       ├── analysis/        # Análisis de mercado
│       ├── backtesting/     # Motor de backtesting
│       ├── brokers/         # Interfaces con brokers
│       ├── config/          # Configuración
│       ├── data/            # Gestión de datos
│       ├── execution/       # Ejecución de órdenes
│       ├── monitoring/      # Monitoreo
│       ├── risk/            # Gestión de riesgo
│       ├── security/        # Seguridad
│       └── strategies/      # Estrategias de trading
├── tests/
│   └── unit/               # Tests unitarios
├── docs/                   # Documentación
├── main.py                 # Punto de entrada
└── requirements.txt        # Dependencias
```

## 🔐 Seguridad

- **NUNCA** commits credenciales reales
- Usa `.env` para secrets locales
- Revisa que `.gitignore` excluye archivos sensibles
- Reporta vulnerabilidades en privado

## 📚 Recursos

- [PEP 8 - Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)

## 💬 Comunicación

- Issues de GitHub para bugs y features
- Discussions para preguntas generales
- Pull Requests para cambios de código

## 📄 Licencia

Al contribuir, aceptas que tus contribuciones se licencien bajo la misma licencia del proyecto.

---

¡Gracias por contribuir! 🎉
