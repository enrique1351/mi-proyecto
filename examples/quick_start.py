"""Ejemplos de uso"""

from shared.core.brokers.brokers import MockBroker

broker = MockBroker(initial_balance={'USD': 10000.0})
print(f"Balance: {broker.get_balance()}")
print("✅ Sistema funcionando correctamente")
