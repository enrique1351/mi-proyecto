"""
Tests for broker integrations
"""

import pytest
import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared.core.brokers.brokers import BrokerFactory, BrokerType, MockBroker


class TestMockBroker:
    """Tests for MockBroker"""
    
    def test_initialization(self):
        """Test broker initialization"""
        broker = MockBroker()
        assert broker.connected == True
        assert broker.name == "MockBroker"
    
    def test_get_balance(self):
        """Test balance retrieval"""
        broker = MockBroker(initial_balance={'USD': 10000.0})
        balance = broker.get_balance()
        assert 'USD' in balance
        assert balance['USD'] == 10000.0
    
    def test_get_current_price(self):
        """Test price retrieval"""
        broker = MockBroker()
        price = broker.get_current_price('BTCUSDT')
        assert price > 0
    
    def test_place_order(self):
        """Test order placement"""
        broker = MockBroker(initial_balance={'USDT': 10000.0})
        
        order = broker.place_order(
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.1,
            order_type='MARKET'
        )
        
        assert order is not None
        assert 'order_id' in order
        assert order['status'] == 'FILLED'
    
    def test_cancel_order(self):
        """Test order cancellation"""
        broker = MockBroker()
        
        # Place order first
        order = broker.place_order(
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.1
        )
        
        # Cancel it
        result = broker.cancel_order(order['order_id'])
        assert result == True
    
    def test_get_order_status(self):
        """Test order status retrieval"""
        broker = MockBroker()
        
        order = broker.place_order(
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.1
        )
        
        status = broker.get_order_status(order['order_id'])
        assert status is not None
        assert 'status' in status


class TestBrokerFactory:
    """Tests for BrokerFactory"""
    
    def test_create_mock_broker(self):
        """Test creating mock broker via factory"""
        broker = BrokerFactory.create_broker(BrokerType.MOCK)
        assert isinstance(broker, MockBroker)
        assert broker.connected == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
