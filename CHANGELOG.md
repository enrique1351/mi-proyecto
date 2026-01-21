# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README.md with installation and usage instructions
- CONTRIBUTING.md guide for contributors
- Security documentation (docs/SECURITY.md)
- Proper .gitignore file to exclude build artifacts and virtual environments
- Documentation folder for planning and reference materials
- Enhanced code comments in main.py for better clarity

### Changed
- Updated all configuration files to use correct extensions (removed .txt suffix)
  - requirements.txt.txt → requirements.txt
  - Makefile.txt → Makefile
  - docker-compose.yml.txt → docker-compose.yml
  - docker-compose.dev.yml.txt → docker-compose.dev.yml
  - .dockerignore.txt → .dockerignore
  - .env.example.txt → .env.example
- Updated dependency versions in requirements.txt with version ranges for better compatibility
- Improved import statements in main.py (removed duplicate sys import)
- Organized planning documentation into docs/ folder

### Removed
- Removed __pycache__ directories from git tracking
- Removed .venv directory from git tracking
- Cleaned up temporary files (temp_method.txt, por crear.txt)
- Removed outdated "pendientes por organizar" folder
- Removed planning note files from root directory

### Fixed
- Fixed duplicate import of sys module in main.py
- Corrected file extensions for all configuration files
- Ensured all Python modules compile without syntax errors

### Security
- Verified no hardcoded credentials in source code
- All API keys and secrets loaded from environment variables or encrypted vault
- Implemented proper credential management with AES-256 encryption
- Added security best practices documentation

## [1.0.0] - 2026-01-21

### Added
- Initial release of Quantitative Trading System
- Multi-asset support (crypto, stocks, forex, commodities)
- Adaptive strategy engine with AI integration
- Comprehensive risk management system
- Broker integrations (Binance, Coinbase, mock broker)
- Data management with SQLite storage
- Market regime detection
- Kill-switch for emergency situations
- Backtesting engine
- Monitoring and reporting system
- Docker support with docker-compose
- Unit tests for core components

### Features
- Paper trading mode for simulation
- Real trading mode for live execution
- Configurable via command-line arguments
- Logging system with rotating files
- Prometheus metrics integration (optional)
- Grafana dashboards (optional)
- PostgreSQL support (optional)
- Redis caching support (optional)

---

**Note**: This project is under active development. Features and APIs may change.
