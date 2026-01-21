"""
Integration Example
===================

This script demonstrates how to use the new modular components together.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_data_processing():
    """Demonstrate data processing modules"""
    logger.info("=== Data Processing Demo ===")
    
    # API Integrations
    from data_processing.external_apis.api_integrations import APIManager, AlpacaAPI, ForexFactoryAPI
    
    manager = APIManager()
    
    # Register Alpaca
    alpaca = AlpacaAPI(api_key="demo", secret_key="demo")
    manager.register_api("alpaca", alpaca)
    
    # Register Forex Factory
    forex = ForexFactoryAPI()
    manager.register_api("forex_factory", forex)
    
    logger.info(f"Registered APIs: {manager.list_apis()}")
    
    # News Aggregation
    from data_processing.news.news_aggregator import NewsAggregator, MockNewsSource, SentimentAnalyzer
    
    news_agg = NewsAggregator()
    news_agg.add_source(MockNewsSource())
    
    articles = news_agg.fetch_all_news(keywords=["Bitcoin", "stocks"], limit_per_source=3)
    
    # Add sentiment
    sentiment = SentimentAnalyzer()
    articles = sentiment.add_sentiment_to_articles(articles)
    
    logger.info(f"Fetched {len(articles)} articles with sentiment analysis")
    
    # Web Scraping
    from data_processing.scrapers.web_scraper import ScraperManager, EconomicIndicatorScraper
    
    scraper_mgr = ScraperManager()
    scraper_mgr.register_scraper(EconomicIndicatorScraper())
    
    econ_data = scraper_mgr.get_scraper("EconomicIndicator").scrape("GDP", "US")
    logger.info(f"Scraped economic data: {len(econ_data)} records")


def demo_machine_learning():
    """Demonstrate machine learning modules"""
    logger.info("\n=== Machine Learning Demo ===")
    
    # Create mock data
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='1H')
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Data Preparation
    from machine_learning.training.model_training import DataPreparator, ModelTrainer
    
    preparator = DataPreparator()
    X, y = preparator.prepare_timeseries_data(mock_data, sequence_length=60)
    logger.info(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_data(X, y)
    
    # Train LSTM Model
    from machine_learning.models.ml_models import LSTMModel
    
    lstm = LSTMModel(input_dim=X.shape[2], output_dim=1)
    lstm.build(hidden_units=32, num_layers=2)
    
    trainer = ModelTrainer()
    history = trainer.train_model(lstm, X_train, y_train, X_val, y_val, epochs=5)
    logger.info(f"Model trained in {history.get('training_time', 0):.2f}s")
    
    # Prediction Engine
    from machine_learning.prediction.prediction_engine import PredictionEngine
    
    pred_engine = PredictionEngine()
    pred_engine.register_model("lstm", lstm)
    
    # Make predictions
    price_pred = pred_engine.predict_price("BTCUSDT", "lstm", mock_data.tail(100))
    logger.info(f"Price prediction: ${price_pred.get('predicted_price', 0):.2f}")
    
    trend_pred = pred_engine.predict_trend("BTCUSDT", mock_data.tail(100))
    logger.info(f"Trend: {trend_pred.get('trend', 'unknown')}")
    
    vol_pred = pred_engine.predict_volatility("BTCUSDT", mock_data.tail(100))
    logger.info(f"Volatility: {vol_pred.get('volatility_level', 'unknown')}")


def demo_trading():
    """Demonstrate trading modules"""
    logger.info("\n=== Trading Demo ===")
    
    # Broker Integration
    from trading.brokers.broker_integrations import (
        BrokerManager, AlpacaBroker, HuobiBroker, 
        OrderSide, OrderType
    )
    
    broker_mgr = BrokerManager()
    
    # Register brokers
    alpaca = AlpacaBroker(paper_trading=True)
    alpaca.connect(api_key="demo", secret_key="demo")
    broker_mgr.register_broker(alpaca)
    
    huobi = HuobiBroker()
    huobi.connect(api_key="demo", secret_key="demo")
    broker_mgr.register_broker(huobi)
    
    logger.info(f"Registered brokers: {broker_mgr.list_brokers()}")
    
    # Get balances
    balances = broker_mgr.get_all_balances()
    logger.info(f"Account balances: {balances}")
    
    # Place order
    order = alpaca.place_order("AAPL", OrderSide.BUY, OrderType.MARKET, 10)
    logger.info(f"Order placed: {order.get('order_id', 'N/A')}")
    
    # Trading Strategies
    from trading.strategies.trading_strategies import (
        StrategyManager, TrendFollowingStrategy, 
        MeanReversionStrategy, MLEnhancedStrategy
    )
    
    strategy_mgr = StrategyManager()
    
    # Register strategies
    trend = TrendFollowingStrategy(fast_period=10, slow_period=30)
    mean_rev = MeanReversionStrategy(window=20)
    ml_strat = MLEnhancedStrategy()
    
    strategy_mgr.register_strategy(trend)
    strategy_mgr.register_strategy(mean_rev)
    strategy_mgr.register_strategy(ml_strat)
    
    # Activate strategies
    strategy_mgr.activate_strategy("TrendFollowing")
    strategy_mgr.activate_strategy("MeanReversion")
    
    # Generate mock data for signals
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
    })
    
    # Generate signals
    signals = strategy_mgr.generate_signals(mock_data, symbol="BTCUSDT")
    
    logger.info(f"Generated {len(signals)} trading signals")
    for signal in signals:
        logger.info(f"  Signal: {signal.action.upper()} - "
                   f"Strength: {signal.strength:.2f}, "
                   f"Confidence: {signal.confidence:.2f}")


def demo_integration():
    """Demonstrate full integration"""
    logger.info("\n=== Full Integration Demo ===")
    
    # 1. Fetch market data
    logger.info("Step 1: Fetching market data...")
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 101,
        'low': np.random.randn(200).cumsum() + 99,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.uniform(1000, 10000, 200)
    })
    
    # 2. Get news sentiment
    logger.info("Step 2: Analyzing news sentiment...")
    from data_processing.news.news_aggregator import NewsAggregator, MockNewsSource
    
    news_agg = NewsAggregator()
    news_agg.add_source(MockNewsSource())
    articles = news_agg.fetch_all_news(keywords=["market"], limit_per_source=2)
    
    # 3. Make ML predictions
    logger.info("Step 3: Making ML predictions...")
    from machine_learning.prediction.prediction_engine import PredictionEngine
    
    pred_engine = PredictionEngine()
    trend = pred_engine.predict_trend("AAPL", market_data)
    
    # 4. Generate trading signals
    logger.info("Step 4: Generating trading signals...")
    from trading.strategies.trading_strategies import StrategyManager, TrendFollowingStrategy
    
    strategy_mgr = StrategyManager()
    strategy = TrendFollowingStrategy()
    strategy_mgr.register_strategy(strategy)
    strategy_mgr.activate_strategy("TrendFollowing")
    
    signals = strategy_mgr.generate_signals(market_data, symbol="AAPL")
    
    # 5. Execute trades (paper trading)
    logger.info("Step 5: Executing trades...")
    from trading.brokers.broker_integrations import AlpacaBroker, OrderSide, OrderType
    
    broker = AlpacaBroker(paper_trading=True)
    broker.connect(api_key="demo", secret_key="demo")
    
    if signals:
        signal = signals[0]
        if signal.action == 'buy' and signal.strength > 0.5:
            order = broker.place_order(
                "AAPL", 
                OrderSide.BUY, 
                OrderType.MARKET, 
                10
            )
            logger.info(f"Order executed: {order.get('order_id', 'N/A')}")
    
    logger.info("Integration demo completed!")


def main():
    """Main entry point"""
    print("=" * 70)
    print("ADVANCED TRADING BOT - INTEGRATION EXAMPLE")
    print("=" * 70)
    
    try:
        demo_data_processing()
        demo_machine_learning()
        demo_trading()
        demo_integration()
        
        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
