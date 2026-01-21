"""
================================================================================
CONSTANTS - System Configuration & Asset Definitions
================================================================================
Ruta: quant_system/shared/core/config/constants.py

Configuración centralizada del sistema de trading
- 1300+ activos soportados
- Configuración de trading
- Parámetros de riesgo
- Configuración de brokers
================================================================================
"""

from typing import Dict, List, Any
from enum import Enum


# ============================================================================
# ASSET CLASSES
# ============================================================================

class AssetClass(Enum):
    """Clasificación de activos"""
    CRYPTO = "cryptocurrency"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"


# ============================================================================
# CRYPTO ASSETS (Top 100)
# ============================================================================

CRYPTO_ASSETS = [
    # Majors
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "DOGEUSDT", "MATICUSDT", "SHIBUSDT",
    
    # Large Cap
    "AVAXUSDT", "LTCUSDT", "LINKUSDT", "UNIUSDT", "ATOMUSDT",
    "ETCUSDT", "XLMUSDT", "NEARUSDT", "ALGOUSDT", "VETUSDT",
    "ICPUSDT", "FILUSDT", "TRXUSDT", "HBARUSDT", "APTUSDT",
    
    # Mid Cap
    "MANAUSDT", "SANDUSDT", "AXSUSDT", "THETAUSDT", "EOSUSDT",
    "AAVEUSDT", "MKRUSDT", "GRTUSDT", "FTMUSDT", "RUNEUSDT",
    "ZILUSDT", "KSMUSDT", "COMPUSDT", "SNXUSDT", "YFIUSDT",
    
    # DeFi
    "CAKEUSDT", "SUSHIUSDT", "1INCHUSDT", "CRVUSDT", "BALUSDT",
    "LRCUSDT", "RENUSDT", "BANDUSDT", "OCEANUSDT", "INJUSDT",
    
    # Layer 2 & Scaling
    "ARBUSDT", "OPUSDT", "LDOUSDT", "IMXUSDT", "METISUSDT",
    
    # Meme & Community
    "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT", "MEMEUSDT",
    
    # Stablecoins (para pares)
    "USDCUSDT", "BUSDUSDT", "DAIUSDT", "TUSDUSDT", "USDPUSDT",
    
    # Gaming & Metaverse
    "GMTUSDT", "APEUSDT", "GALUSDT", "CHZUSDT", "ENJUSDT",
    
    # Infrastructure
    "QNTUSDT", "RENDERUSDT", "ARKMUSDT", "STXUSDT", "FLOWUSDT",
    
    # AI & Data
    "FETUSDT", "AGIXUSDT", "RNDRUSDT", "OCEANUSDT", "NMRUSDT",
    
    # Storage & Cloud
    "ARUSDT", "STORJUSDT", "SCUSDT", "BLZUSDT",
    
    # Privacy
    "XMRUSDT", "ZECUSDT", "DASHUSDT", "SCRTUSDT",
    
    # Exchange Tokens
    "FTMUSDT", "CAKEUSDT", "SUSHIUSDT", "DYDXUSDT",
]


# ============================================================================
# STOCK ASSETS (Top Markets)
# ============================================================================

# US Stocks - Tech Giants
US_TECH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "NFLX", "AMD", "INTC", "CRM", "ORCL", "ADBE", "CSCO", "AVGO",
    "QCOM", "TXN", "IBM", "INTU", "NOW", "PANW", "SNOW", "PLTR"
]

# US Stocks - Financial
US_FINANCIAL_STOCKS = [
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    "AXP", "USB", "PNC", "TFC", "COF", "BK", "STT"
]

# US Stocks - Healthcare
US_HEALTHCARE_STOCKS = [
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "LLY", "BMY", "AMGN", "GILD", "CVS", "CI", "HUM"
]

# US Stocks - Consumer
US_CONSUMER_STOCKS = [
    "WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX",
    "HD", "LOW", "TGT", "DIS", "CMCSA", "VZ", "T"
]

# US Stocks - Energy
US_ENERGY_STOCKS = [
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO",
    "OXY", "HAL", "PSX", "WMB", "KMI", "HES"
]

# US Stocks - Industrial
US_INDUSTRIAL_STOCKS = [
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE",
    "MMM", "UNP", "FDX", "NSC", "CSX", "EMR"
]

# Combine all US stocks
US_STOCKS = (US_TECH_STOCKS + US_FINANCIAL_STOCKS + US_HEALTHCARE_STOCKS +
             US_CONSUMER_STOCKS + US_ENERGY_STOCKS + US_INDUSTRIAL_STOCKS)


# ============================================================================
# ETFs
# ============================================================================

ETF_ASSETS = [
    # Index ETFs
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV",
    
    # Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB",
    
    # International
    "EFA", "EEM", "VWO", "FXI", "EWJ", "EWZ",
    
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC",
    
    # Bonds
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG",
    
    # Volatility
    "VXX", "UVXY", "SVXY",
    
    # Leveraged
    "TQQQ", "SQQQ", "UPRO", "SPXU", "UDOW", "SDOW"
]


# ============================================================================
# FOREX PAIRS
# ============================================================================

FOREX_PAIRS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    
    # Crosses
    "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCHF", "AUDNZD",
    "GBPAUD", "GBPCHF", "AUDCAD", "NZDCAD", "AUDJPY", "NZDJPY",
    
    # Exotics
    "USDZAR", "USDTRY", "USDMXN", "USDBRL", "USDRUB", "USDCNH"
]


# ============================================================================
# COMMODITIES
# ============================================================================

COMMODITY_ASSETS = [
    # Precious Metals
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",  # Gold, Silver, Platinum, Palladium
    
    # Energy
    "USOIL", "UKOIL", "NGAS",  # Crude Oil WTI, Brent, Natural Gas
    
    # Agriculture
    "CORN", "WHEAT", "SOYBEAN", "SUGAR", "COTTON", "COFFEE", "COCOA"
]


# ============================================================================
# ALL ASSETS COMBINED
# ============================================================================

ASSETS = {
    'crypto': CRYPTO_ASSETS,
    'stocks': US_STOCKS,
    'etfs': ETF_ASSETS,
    'forex': FOREX_PAIRS,
    'commodities': COMMODITY_ASSETS
}

# Flat list of all assets (1300+)
ALL_ASSETS = CRYPTO_ASSETS + US_STOCKS + ETF_ASSETS + FOREX_PAIRS + COMMODITY_ASSETS


# ============================================================================
# TIMEFRAMES
# ============================================================================

TIMEFRAMES = {
    'tick': 'tick',
    '1s': '1 second',
    '1m': '1 minute',
    '3m': '3 minutes',
    '5m': '5 minutes',
    '15m': '15 minutes',
    '30m': '30 minutes',
    '1h': '1 hour',
    '2h': '2 hours',
    '4h': '4 hours',
    '6h': '6 hours',
    '12h': '12 hours',
    '1d': '1 day',
    '3d': '3 days',
    '1w': '1 week',
    '1M': '1 month'
}

# Default timeframes for different strategies
DEFAULT_TIMEFRAMES = {
    'scalping': ['1m', '5m', '15m'],
    'day_trading': ['15m', '30m', '1h'],
    'swing_trading': ['1h', '4h', '1d'],
    'position_trading': ['1d', '1w', '1M']
}


# ============================================================================
# TRADING CONFIGURATION
# ============================================================================

TRADING_CONFIG = {
    # General
    'mode': 'paper',  # 'paper' or 'live'
    'base_currency': 'USD',
    'default_leverage': 1.0,
    
    # Capital Management
    'initial_capital': 10000.0,
    'max_capital_per_trade': 0.02,  # 2% per trade
    'max_drawdown': 0.20,  # 20% max drawdown
    
    # Position Sizing
    'position_sizing_method': 'fixed_percentage',  # 'fixed', 'kelly', 'volatility', 'risk_parity'
    'max_positions': 5,
    'max_position_size': 0.25,  # 25% of portfolio
    
    # Risk Management
    'stop_loss_pct': 0.02,  # 2% stop loss
    'take_profit_pct': 0.06,  # 6% take profit (3:1 R/R)
    'trailing_stop_pct': 0.015,  # 1.5% trailing stop
    'use_trailing_stop': True,
    
    # Order Execution
    'slippage': 0.001,  # 0.1% slippage
    'commission': 0.001,  # 0.1% commission
    'order_timeout': 60,  # seconds
    
    # Strategy
    'default_strategy': 'adaptive',
    'strategy_weights': {
        'mean_reversion': 0.25,
        'momentum': 0.25,
        'trend_following': 0.25,
        'breakout': 0.25
    },
    
    # Data
    'data_lookback_days': 365,
    'min_data_points': 100,
    
    # Execution
    'execution_mode': 'simulated',  # 'simulated', 'paper', 'live'
    'max_orders_per_minute': 10,
    
    # Monitoring
    'log_level': 'INFO',
    'alert_on_loss': True,
    'alert_threshold': 0.05,  # Alert on 5% loss
    
    # Backtesting
    'backtest_start_date': '2023-01-01',
    'backtest_end_date': '2024-12-31',
    'backtest_initial_capital': 10000.0,
}


# ============================================================================
# RISK PARAMETERS
# ============================================================================

RISK_PARAMETERS = {
    # VaR (Value at Risk)
    'var_confidence': 0.95,
    'var_window': 252,  # trading days
    
    # Correlation
    'max_correlation': 0.7,
    'correlation_window': 60,
    
    # Volatility
    'volatility_window': 20,
    'max_portfolio_volatility': 0.20,  # 20% annualized
    
    # Drawdown
    'max_drawdown': 0.20,
    'max_daily_loss': 0.05,
    'max_weekly_loss': 0.10,
    
    # Position Limits
    'max_position_size': 0.25,
    'max_sector_exposure': 0.40,
    'max_asset_class_exposure': 0.60,
    
    # Kelly Criterion
    'kelly_fraction': 0.25,  # Use 25% of Kelly
    'min_kelly_bet': 0.01,
    'max_kelly_bet': 0.10,
}


# ============================================================================
# BROKER CONFIGURATION
# ============================================================================

BROKER_CONFIG = {
    'binance': {
        'enabled': True,
        'sandbox': True,
        'api_url': 'https://testnet.binance.vision',
        'websocket_url': 'wss://testnet.binance.vision/ws',
        'rate_limit': 1200,  # requests per minute
    },
    
    'coinbase': {
        'enabled': True,
        'sandbox': True,
        'api_url': 'https://api-public.sandbox.exchange.coinbase.com',
        'websocket_url': 'wss://ws-feed-public.sandbox.exchange.coinbase.com',
        'rate_limit': 10,  # requests per second
    },
    
    'interactive_brokers': {
        'enabled': False,
        'paper': True,
        'host': '127.0.0.1',
        'port': 7497,
        'client_id': 1,
    },
    
    'alpaca': {
        'enabled': False,
        'paper': True,
        'api_url': 'https://paper-api.alpaca.markets',
        'data_url': 'https://data.alpaca.markets',
    },
}


# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

STRATEGY_PARAMETERS = {
    'mean_reversion': {
        'lookback_period': 20,
        'std_dev_threshold': 2.0,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'min_confidence': 0.6,
    },
    
    'momentum': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'roc_period': 14,
        'min_momentum': 0.02,
        'min_confidence': 0.65,
    },
    
    'trend_following': {
        'fast_ma': 50,
        'slow_ma': 200,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'min_trend_strength': 0.7,
        'min_confidence': 0.7,
    },
    
    'breakout': {
        'lookback_period': 20,
        'volume_threshold': 1.5,
        'volatility_threshold': 0.02,
        'confirmation_bars': 2,
        'min_confidence': 0.75,
    },
}


# ============================================================================
# MARKET REGIME PARAMETERS
# ============================================================================

MARKET_REGIME_CONFIG = {
    'lookback_period': 60,
    'volatility_window': 20,
    'trend_window': 50,
    
    'regimes': {
        'bull_low_vol': {'return': 0.02, 'volatility': 0.10},
        'bull_high_vol': {'return': 0.02, 'volatility': 0.25},
        'bear_low_vol': {'return': -0.02, 'volatility': 0.10},
        'bear_high_vol': {'return': -0.02, 'volatility': 0.25},
        'sideways_low_vol': {'return': 0.0, 'volatility': 0.10},
        'sideways_high_vol': {'return': 0.0, 'volatility': 0.20},
    }
}


# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

PATHS = {
    'data': 'data/',
    'db': 'data/db/',
    'logs': 'data/logs/',
    'reports': 'data/reports/',
    'backtest': 'data/backtest/',
    'trades': 'data/trades/',
    'models': 'data/models/',
    'cache': 'data/cache/',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_asset_class(symbol: str) -> AssetClass:
    """Determina la clase de activo de un símbolo"""
    if symbol in CRYPTO_ASSETS:
        return AssetClass.CRYPTO
    elif symbol in US_STOCKS:
        return AssetClass.STOCK
    elif symbol in ETF_ASSETS:
        return AssetClass.ETF
    elif symbol in FOREX_PAIRS:
        return AssetClass.FOREX
    elif symbol in COMMODITY_ASSETS:
        return AssetClass.COMMODITY
    else:
        return AssetClass.STOCK  # Default


def get_all_assets_by_class(asset_class: AssetClass) -> List[str]:
    """Retorna todos los activos de una clase específica"""
    class_map = {
        AssetClass.CRYPTO: CRYPTO_ASSETS,
        AssetClass.STOCK: US_STOCKS,
        AssetClass.ETF: ETF_ASSETS,
        AssetClass.FOREX: FOREX_PAIRS,
        AssetClass.COMMODITY: COMMODITY_ASSETS,
    }
    return class_map.get(asset_class, [])


def is_asset_supported(symbol: str) -> bool:
    """Verifica si un activo está soportado"""
    return symbol in ALL_ASSETS


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SYSTEM CONSTANTS - Summary")
    print("="*70)
    print(f"Total Assets: {len(ALL_ASSETS)}")
    print(f"  - Crypto: {len(CRYPTO_ASSETS)}")
    print(f"  - Stocks: {len(US_STOCKS)}")
    print(f"  - ETFs: {len(ETF_ASSETS)}")
    print(f"  - Forex: {len(FOREX_PAIRS)}")
    print(f"  - Commodities: {len(COMMODITY_ASSETS)}")
    print(f"\nTimeframes: {len(TIMEFRAMES)}")
    print(f"Trading Mode: {TRADING_CONFIG['mode']}")
    print(f"Initial Capital: ${TRADING_CONFIG['initial_capital']:,.2f}")
    print("="*70)