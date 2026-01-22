# ============================================================================
# ARCHIVO 1: Dockerfile
# Ruta: quant_system/docker/Dockerfile
# ============================================================================

# Imagen base con Python 3.11
FROM python:3.11-slim

# Metadata
LABEL maintainer="quant-system@example.com"
LABEL description="Sistema de Trading Cuantitativo"
LABEL version="2.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/db data/logs data/reports data/backtest data/trades

# Usuario no-root para seguridad
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Puerto (si se usa API/dashboard)
EXPOSE 8000

# Comando por defecto
CMD ["python", "main.py"]


# ============================================================================
# ARCHIVO 2: docker-compose.yml
# Ruta: quant_system/docker-compose.yml
# ============================================================================

version: '3.8'

services:
  # Servicio principal de trading
  trading-system:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: quant-trading-system
    restart: unless-stopped
    
    # Variables de entorno
    environment:
      - MODE=paper  # paper | live
      - LOG_LEVEL=INFO
      - TZ=UTC
      
      # Database
      - DB_PATH=/app/data/db/trading.db
      
      # API Keys (usar secrets en producción)
      - BINANCE_API_KEY=${BINANCE_API_KEY:-}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET:-}
      - ALPACA_API_KEY=${ALPACA_API_KEY:-}
      - ALPACA_API_SECRET=${ALPACA_API_SECRET:-}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-}
    
    # Volúmenes persistentes
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./shared:/app/shared
      - ./main.py:/app/main.py
    
    # Red
    networks:
      - trading-network
    
    # Recursos
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
    # Dependencias
    depends_on:
      - redis
      - postgres
  
  
  # Redis para cache y message queue
  redis:
    image: redis:7-alpine
    container_name: quant-redis
    restart: unless-stopped
    
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-changeme}
    
    volumes:
      - redis-data:/data
    
    networks:
      - trading-network
    
    ports:
      - "6379:6379"
    
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
  
  
  # PostgreSQL para data warehouse (opcional)
  postgres:
    image: postgres:15-alpine
    container_name: quant-postgres
    restart: unless-stopped
    
    environment:
      - POSTGRES_DB=trading
      - POSTGRES_USER=trader
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-changeme}
      - PGDATA=/var/lib/postgresql/data/pgdata
    
    volumes:
      - postgres-data:/var/lib/postgresql/data
    
    networks:
      - trading-network
    
    ports:
      - "5432:5432"
    
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader"]
      interval: 10s
      timeout: 3s
      retries: 3
  
  
  # Prometheus para métricas (opcional)
  prometheus:
    image: prom/prometheus:latest
    container_name: quant-prometheus
    restart: unless-stopped
    
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    
    networks:
      - trading-network
    
    ports:
      - "9090:9090"
  
  
  # Grafana para visualización (opcional)
  grafana:
    image: grafana/grafana:latest
    container_name: quant-grafana
    restart: unless-stopped
    
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    
    volumes:
      - grafana-data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./docker/grafana/datasources:/etc/grafana/provisioning/datasources
    
    networks:
      - trading-network
    
    ports:
      - "3000:3000"
    
    depends_on:
      - prometheus


# Volúmenes
volumes:
  redis-data:
    driver: local
  postgres-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local


# Red
networks:
  trading-network:
    driver: bridge


# ============================================================================
# ARCHIVO 3: .dockerignore
# Ruta: quant_system/.dockerignore
# ============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Tests
.pytest_cache/
.coverage
htmlcov/
*.cover

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data y logs (usar volúmenes)
data/
logs/
*.log

# Git
.git/
.gitignore

# Docker
.dockerignore
Dockerfile
docker-compose*.yml

# Docs
docs/
*.md
README.md

# Otros
.env
.DS_Store
*.bak


# ============================================================================
# ARCHIVO 4: requirements.txt
# Ruta: quant_system/requirements.txt
# ============================================================================

# Core
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1

# Data & APIs
requests==2.31.0
aiohttp==3.8.5
websockets==11.0.3
ccxt==4.0.0
yfinance==0.2.28

# Database
SQLAlchemy==2.0.19
psycopg2-binary==2.9.7
redis==4.6.0

# Technical Analysis
ta-lib==0.4.28
pandas-ta==0.3.14b0

# ML & AI
scikit-learn==1.3.0
tensorflow==2.13.0
torch==2.0.1

# Backtesting & Optimization
backtrader==1.9.78.123
optuna==3.3.0
cvxpy==1.3.2

# Visualization
matplotlib==3.7.2
plotly==5.16.1
seaborn==0.12.2

# Monitoring & Logging
prometheus-client==0.17.1
python-json-logger==2.0.7

# Security
cryptography==41.0.3
pycryptodome==3.18.0

# Utils
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
schedule==1.2.0

# Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.1


# ============================================================================
# ARCHIVO 5: .env.example
# Ruta: quant_system/.env.example
# ============================================================================

# Trading Mode
MODE=paper  # paper | live

# Logging
LOG_LEVEL=INFO

# Database
DB_PATH=data/db/trading.db

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=changeme

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=trading
POSTGRES_USER=trader
POSTGRES_PASSWORD=changeme

# API Keys - Binance
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# API Keys - Alpaca
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret

# API Keys - Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# API Keys - Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# Risk Management
MAX_POSITION_SIZE=0.1
MAX_DRAWDOWN=0.2
MAX_POSITIONS=5

# Grafana
GRAFANA_PASSWORD=admin


# ============================================================================
# ARCHIVO 6: docker-compose.dev.yml
# Ruta: quant_system/docker-compose.dev.yml
# Configuración para desarrollo
# ============================================================================

version: '3.8'

services:
  trading-system:
    build:
      context: .
      dockerfile: docker/Dockerfile
      target: development
    
    volumes:
      # Mount todo para hot-reload
      - .:/app
    
    environment:
      - MODE=paper
      - LOG_LEVEL=DEBUG
      - PYTHONPATH=/app
    
    command: python main.py --dev
    
    ports:
      - "8000:8000"  # API
      - "5678:5678"  # Debugger


# ============================================================================
# ARCHIVO 7: Makefile
# Ruta: quant_system/Makefile
# Comandos útiles
# ============================================================================

.PHONY: help build up down logs test clean

help:
	@echo "Comandos disponibles:"
	@echo "  make build     - Construir imágenes Docker"
	@echo "  make up        - Iniciar servicios"
	@echo "  make down      - Detener servicios"
	@echo "  make logs      - Ver logs"
	@echo "  make test      - Ejecutar tests"
	@echo "  make clean     - Limpiar containers y volúmenes"
	@echo "  make shell     - Shell interactivo en container"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f trading-system

test:
	docker-compose run --rm trading-system pytest tests/ -v

clean:
	docker-compose down -v
	docker system prune -f

shell:
	docker-compose exec trading-system /bin/bash

dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

restart:
	docker-compose restart trading-system

status:
	docker-compose ps