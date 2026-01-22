"""
Tests for notification modules
"""

import pytest
import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared.core.notifications.notification_manager import NotificationManager, NotificationLevel


class TestNotificationManager:
    """Tests for NotificationManager"""
    
    def test_initialization(self):
        """Test notification manager initialization"""
        manager = NotificationManager()
        assert manager is not None
        assert manager.enabled_channels['telegram'] == False
        assert manager.enabled_channels['twilio'] == False
    
    def test_statistics(self):
        """Test getting statistics"""
        manager = NotificationManager()
        stats = manager.get_statistics()
        
        assert 'total_notifications' in stats
        assert 'by_level' in stats
        assert 'enabled_channels' in stats
    
    def test_notification_levels(self):
        """Test notification level enum"""
        assert NotificationLevel.DEBUG.value == 'debug'
        assert NotificationLevel.INFO.value == 'info'
        assert NotificationLevel.WARNING.value == 'warning'
        assert NotificationLevel.ERROR.value == 'error'
        assert NotificationLevel.CRITICAL.value == 'critical'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
