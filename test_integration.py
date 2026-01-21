"""
Quick Integration Test
======================

Test that all new modules can be imported and basic functionality works.
This test doesn't require external dependencies.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("INTEGRATION TEST - New Modules")
print("=" * 80)

# Test 1: Notification Manager
print("\n[TEST 1] Notification Manager Import")
try:
    from notifications.notification_manager import (
        NotificationManager,
        NotificationLevel,
        NotificationChannel
    )
    print("✅ PASS: NotificationManager imported successfully")
    print(f"   - NotificationLevel enum: {list(NotificationLevel)}")
    print(f"   - NotificationChannel enum: {list(NotificationChannel)}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 2: Telegram Notifier
print("\n[TEST 2] Telegram Notifier Import")
try:
    from notifications.telegram_notifier import TelegramNotifier
    print("✅ PASS: TelegramNotifier imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 3: Email Notifier
print("\n[TEST 3] Email Notifier Import")
try:
    from notifications.email_notifier import EmailNotifier
    print("✅ PASS: EmailNotifier imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 4: Broker Manager
print("\n[TEST 4] Broker Manager Import")
try:
    from trading.broker_manager import BrokerManager, BrokerType
    print("✅ PASS: BrokerManager imported successfully")
    print(f"   - BrokerType enum: {list(BrokerType)}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 5: Alpaca Broker
print("\n[TEST 5] Alpaca Broker Import")
try:
    from trading.alpaca_broker import AlpacaBroker
    print("✅ PASS: AlpacaBroker imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 6: Forex Broker
print("\n[TEST 6] Forex Broker Import")
try:
    from trading.forex_broker import OandaBroker
    print("✅ PASS: OandaBroker imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 7: ML Time Series (will fail without deps, that's OK)
print("\n[TEST 7] ML Time Series Import")
try:
    from machine_learning.time_series_models import LSTMPredictor, ARIMAPredictor
    print("✅ PASS: Time Series models imported successfully")
except ImportError as e:
    print(f"⚠️  SKIP: {e} (expected without ML dependencies)")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 8: ML NLP (will fail without deps, that's OK)
print("\n[TEST 8] ML NLP Import")
try:
    from machine_learning.nlp_analyzer import NewsAnalyzer
    print("✅ PASS: NLP analyzer imported successfully")
except ImportError as e:
    print(f"⚠️  SKIP: {e} (expected without ML dependencies)")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 9: NotificationManager initialization
print("\n[TEST 9] NotificationManager Initialization")
try:
    config = {}  # Empty config
    manager = NotificationManager(config)
    print("✅ PASS: NotificationManager initialized with empty config")
    print(f"   - Channels active: {sum(manager.health_check().values())}/4")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 10: NotificationLevel enum
print("\n[TEST 10] NotificationLevel Values")
try:
    levels = [
        NotificationLevel.INFO,
        NotificationLevel.WARNING,
        NotificationLevel.ERROR,
        NotificationLevel.CRITICAL
    ]
    print(f"✅ PASS: All notification levels accessible")
    for level in levels:
        print(f"   - {level.name}: {level.value}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 11: BrokerType enum
print("\n[TEST 11] BrokerType Values")
try:
    brokers = [
        BrokerType.BINANCE,
        BrokerType.ALPACA,
        BrokerType.OANDA,
        BrokerType.INTERACTIVE_BROKERS
    ]
    print(f"✅ PASS: All broker types accessible")
    for broker in brokers:
        print(f"   - {broker.name}: {broker.value}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ All core imports successful")
print("✅ Enums and classes properly defined")
print("✅ NotificationManager can be instantiated")
print("⚠️  ML modules require additional dependencies (expected)")
print("\n🎉 Integration test PASSED!")
print("=" * 80)
