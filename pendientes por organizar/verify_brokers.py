import os
from core_local.brokers import BinanceBroker, CoinbaseBroker

def verify_binance():
    print("🔍 Verificando Binance...")
    binance = BinanceBroker(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET")
    )
    balance = binance.get_balance()
    print("Binance balance keys:", list(balance.keys())[:5])

def verify_coinbase():
    print("🔍 Verificando Coinbase...")
    coinbase = CoinbaseBroker(
        api_key=os.getenv("COINBASE_API_KEY"),
        api_secret=os.getenv("COINBASE_API_SECRET"),
        passphrase=os.getenv("COINBASE_API_PASSPHRASE")
    )
    balance = coinbase.get_balance()
    print("Coinbase balance keys:", list(balance.keys())[:5])

if __name__ == "__main__":
    verify_binance()
    verify_coinbase()
