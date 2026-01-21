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