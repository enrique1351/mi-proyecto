"""
Example Usage of Trading Bot Modules
=====================================

This file demonstrates how to use the newly implemented modules:
- Notifications
- Trading (Brokers)
- Machine Learning

NOTE: This is for demonstration purposes. Make sure to configure
your .env file with proper credentials before running.
"""

import os
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("TRADING BOT - Module Examples")
print("="*80)

# ============================================================================
# 1. NOTIFICATIONS EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("1. NOTIFICATIONS MODULE")
print("="*80)

from notifications.notification_manager import (
    NotificationManager,
    NotificationLevel,
    NotificationChannel
)

# Initialize notification manager
# In production, pass actual credentials from .env
notification_config = {
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'smtp': {
        'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'port': int(os.getenv('SMTP_PORT', 587)),
        'user': os.getenv('SMTP_USER'),
        'password': os.getenv('SMTP_PASSWORD'),
        'from_email': os.getenv('SMTP_USER'),
        'to_email': os.getenv('SMTP_USER')
    }
}

notifier = NotificationManager(notification_config)

print("\n📱 Notification channels available:")
for channel, active in notifier.health_check().items():
    status = "✅ Active" if active else "❌ Inactive"
    print(f"  - {channel.value}: {status}")

# Example: Send a test notification
if any(notifier.health_check().values()):
    print("\n📨 Sending test notification...")
    # Uncomment to actually send:
    # notifier.send(
    #     "🚀 Trading bot started successfully!",
    #     level=NotificationLevel.INFO,
    #     title="Bot Started"
    # )
    print("  (Disabled in example - uncomment to actually send)")

# Example: Send trade alert
trade_data = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'quantity': 0.01,
    'price': 45000.00
}

print("\n📊 Example trade alert:")
print(f"  Symbol: {trade_data['symbol']}")
print(f"  Side: {trade_data['side']}")
print(f"  Quantity: {trade_data['quantity']}")
print(f"  Price: ${trade_data['price']:,.2f}")
# Uncomment to send:
# notifier.send_trade_alert(trade_data)

# ============================================================================
# 2. TRADING MODULE (BROKERS) EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("2. TRADING MODULE (BROKERS)")
print("="*80)

from trading.broker_manager import BrokerManager, BrokerType

# Initialize broker manager
# In production, load credentials from .env
broker_config = {
    'alpaca': {
        'api_key': os.getenv('ALPACA_API_KEY', 'demo_key'),
        'api_secret': os.getenv('ALPACA_API_SECRET', 'demo_secret'),
        'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    },
    'oanda': {
        'api_key': os.getenv('OANDA_API_KEY', 'demo_key'),
        'account_id': os.getenv('OANDA_ACCOUNT_ID', 'demo_account'),
        'practice': True
    }
}

print("\n🏦 Initializing broker manager...")
print("  Note: Using demo credentials (will fail auth, that's OK for demo)")

# Uncomment if you have real credentials:
# broker_manager = BrokerManager(broker_config)
# 
# print("\n📊 Available brokers:")
# for broker_type in broker_manager.brokers.keys():
#     print(f"  - {broker_type.value}")
# 
# # Get account info
# print("\n💰 Account information:")
# accounts = broker_manager.get_all_accounts()
# for broker_type, account in accounts.items():
#     print(f"\n  {broker_type.value}:")
#     for key, value in account.items():
#         print(f"    {key}: {value}")

print("  (Broker initialization disabled - requires real credentials)")

# Example: Place an order
print("\n📝 Example order placement:")
print("  broker_manager.place_order(")
print("      broker_type=BrokerType.ALPACA,")
print("      symbol='AAPL',")
print("      side='buy',")
print("      quantity=10,")
print("      order_type='market'")
print("  )")

# ============================================================================
# 3. MACHINE LEARNING EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("3. MACHINE LEARNING MODULE")
print("="*80)

print("\n🧠 Time Series Models")
print("-" * 40)

# LSTM Example
print("\n📈 LSTM Predictor:")
print("  from machine_learning.time_series_models import LSTMPredictor")
print("  ")
print("  # Initialize")
print("  predictor = LSTMPredictor(sequence_length=60, units=50)")
print("  ")
print("  # Prepare data")
print("  X, y = predictor.prepare_data(price_data)")
print("  ")
print("  # Train")
print("  predictor.train(X, y, epochs=50)")
print("  ")
print("  # Predict")
print("  predictions = predictor.predict(X_test)")

# ARIMA Example
print("\n📊 ARIMA Predictor:")
print("  from machine_learning.time_series_models import ARIMAPredictor")
print("  ")
print("  # Initialize with order (p, d, q)")
print("  predictor = ARIMAPredictor(order=(5, 1, 0))")
print("  ")
print("  # Train")
print("  predictor.train(price_series)")
print("  ")
print("  # Forecast")
print("  forecast = predictor.predict(steps=10)")

print("\n🤖 NLP Analyzer")
print("-" * 40)

print("\n📰 News Sentiment Analysis:")
print("  from machine_learning.nlp_analyzer import NewsAnalyzer")
print("  ")
print("  # Initialize")
print("  analyzer = NewsAnalyzer()")
print("  ")
print("  # Analyze sentiment")
print("  sentiment = analyzer.analyze_sentiment(")
print("      'Bitcoin reaches new all-time high'")
print("  )")
print("  ")
print("  # Get market sentiment from multiple news")
print("  market_sentiment = analyzer.get_market_sentiment(news_list)")
print("  print(market_sentiment['overall_sentiment'])  # 'positive', 'negative', 'neutral'")

# ============================================================================
# 4. INTEGRATION EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("4. FULL INTEGRATION EXAMPLE")
print("="*80)

print("""
# Complete trading bot workflow integrating all modules:

from notifications.notification_manager import NotificationManager, NotificationLevel
from trading.broker_manager import BrokerManager, BrokerType
from machine_learning.nlp_analyzer import NewsAnalyzer
from machine_learning.time_series_models import LSTMPredictor

# 1. Initialize systems
notifier = NotificationManager(notification_config)
broker_manager = BrokerManager(broker_config)
news_analyzer = NewsAnalyzer()
price_predictor = LSTMPredictor()

# 2. Get news sentiment
news = news_aggregator.fetch_crypto_news(limit=20)
sentiment = news_analyzer.get_market_sentiment(news)

# 3. Get price prediction
prediction = price_predictor.predict(recent_prices)

# 4. Make trading decision
if sentiment['overall_sentiment'] == 'positive' and prediction > current_price:
    # Place order
    order = broker_manager.place_order(
        broker_type=BrokerType.BINANCE,
        symbol='BTCUSDT',
        side='buy',
        quantity=0.01
    )
    
    # Send notification
    notifier.send_trade_alert(order)

# 5. Monitor and report
daily_summary = {
    'total_trades': 5,
    'pnl': 125.50,
    'win_rate': 0.6
}
notifier.send_daily_summary(daily_summary)
""")

print("\n" + "="*80)
print("For more information, see:")
print("  - README.md for full documentation")
print("  - RESUMEN_PROYECTO.txt for detailed Spanish summary")
print("="*80)
