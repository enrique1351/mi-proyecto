"""
Notification Manager
Unified interface for multi-channel notifications
"""

import logging
from typing import Optional, List, Any
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationManager:
    """
    Unified notification manager supporting multiple channels
    """
    
    def __init__(self, credential_vault: Optional[Any] = None):
        """
        Initialize notification manager
        
        Args:
            credential_vault: Vault with credentials for all notifiers
        """
        self.credential_vault = credential_vault
        
        # Notifiers
        self.telegram = None
        self.twilio = None
        
        # Configuration
        self.enabled_channels = {
            'telegram': False,
            'twilio': False
        }
        
        # Statistics
        self.total_notifications = 0
        self.notifications_by_level = {level: 0 for level in NotificationLevel}
        
        logger.info("NotificationManager initialized")
    
    def setup_telegram(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Setup Telegram notifier
        
        Args:
            bot_token: Telegram bot token (optional if using credential_vault)
            chat_id: Telegram chat ID (optional if using credential_vault)
        
        Returns:
            True if setup successful
        """
        try:
            from .telegram_notifier import TelegramNotifier
            
            self.telegram = TelegramNotifier(
                bot_token=bot_token,
                chat_id=chat_id,
                credential_vault=self.credential_vault
            )
            
            if self.telegram.connect():
                self.enabled_channels['telegram'] = True
                logger.info("✅ Telegram notifier enabled")
                return True
            else:
                logger.warning("⚠️  Telegram notifier failed to connect")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️  Telegram not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to setup Telegram: {e}")
            return False
    
    def setup_twilio(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None
    ) -> bool:
        """
        Setup Twilio notifier
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: Sender phone number
            to_number: Recipient phone number
        
        Returns:
            True if setup successful
        """
        try:
            from .twilio_notifier import TwilioNotifier
            
            self.twilio = TwilioNotifier(
                account_sid=account_sid,
                auth_token=auth_token,
                from_number=from_number,
                to_number=to_number,
                credential_vault=self.credential_vault
            )
            
            if self.twilio.connect():
                self.enabled_channels['twilio'] = True
                logger.info("✅ Twilio notifier enabled")
                return True
            else:
                logger.warning("⚠️  Twilio notifier failed to connect")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️  Twilio not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to setup Twilio: {e}")
            return False
    
    def notify(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        Send notification through specified channels
        
        Args:
            message: Notification message
            level: Notification severity level
            channels: List of channels to use (default: all enabled)
            **kwargs: Additional parameters for specific notifiers
        
        Returns:
            True if at least one notification sent successfully
        """
        self.total_notifications += 1
        self.notifications_by_level[level] += 1
        
        # Determine which channels to use
        if channels is None:
            channels = [ch for ch, enabled in self.enabled_channels.items() if enabled]
        
        success = False
        
        # Send via Telegram
        if 'telegram' in channels and self.telegram:
            try:
                if self.telegram.send_message(message):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
        
        # Send via Twilio (only for WARNING, ERROR, CRITICAL)
        if 'twilio' in channels and self.twilio:
            if level in [NotificationLevel.WARNING, NotificationLevel.ERROR, NotificationLevel.CRITICAL]:
                try:
                    if self.twilio.send_sms(message):
                        success = True
                except Exception as e:
                    logger.error(f"Failed to send Twilio notification: {e}")
        
        return success
    
    def notify_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str = "EXECUTED",
        channels: Optional[List[str]] = None
    ) -> bool:
        """Send trade execution notification"""
        success = False
        
        if channels is None:
            channels = [ch for ch, enabled in self.enabled_channels.items() if enabled]
        
        # Send via Telegram
        if 'telegram' in channels and self.telegram:
            try:
                if self.telegram.send_trade_alert(symbol, side, quantity, price, status):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send Telegram trade alert: {e}")
        
        # Send via Twilio
        if 'twilio' in channels and self.twilio:
            try:
                if self.twilio.send_trade_alert(symbol, side, quantity, price):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send Twilio trade alert: {e}")
        
        return success
    
    def notify_error(
        self,
        error_message: str,
        severity: str = "ERROR",
        channels: Optional[List[str]] = None
    ) -> bool:
        """Send error notification"""
        level = NotificationLevel.ERROR if severity == "ERROR" else NotificationLevel.CRITICAL
        
        success = False
        
        if channels is None:
            channels = [ch for ch, enabled in self.enabled_channels.items() if enabled]
        
        # Send via Telegram
        if 'telegram' in channels and self.telegram:
            try:
                if self.telegram.send_error_alert(error_message, severity):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send Telegram error alert: {e}")
        
        # Send via Twilio (for critical errors)
        if 'twilio' in channels and self.twilio and severity == "CRITICAL":
            try:
                if self.twilio.send_critical_alert(error_message):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send Twilio error alert: {e}")
        
        return success
    
    def notify_performance(
        self,
        total_pnl: float,
        daily_pnl: float,
        win_rate: float,
        total_trades: int,
        channels: Optional[List[str]] = None
    ) -> bool:
        """Send performance update notification"""
        success = False
        
        if channels is None:
            channels = ['telegram']  # Performance updates only via Telegram
        
        if 'telegram' in channels and self.telegram:
            try:
                if self.telegram.send_performance_update(
                    total_pnl, daily_pnl, win_rate, total_trades
                ):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send performance update: {e}")
        
        return success
    
    def notify_system_status(
        self,
        status: str,
        uptime: Optional[str] = None,
        active_strategies: int = 0,
        channels: Optional[List[str]] = None
    ) -> bool:
        """Send system status notification"""
        success = False
        
        if channels is None:
            channels = [ch for ch, enabled in self.enabled_channels.items() if enabled]
        
        if 'telegram' in channels and self.telegram:
            try:
                if self.telegram.send_system_status(status, uptime, active_strategies):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send system status: {e}")
        
        # Critical status changes via SMS
        if status in ["STOPPED", "ERROR", "CRASHED"] and 'twilio' in channels and self.twilio:
            try:
                if self.twilio.send_system_shutdown_alert(status):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send system shutdown alert: {e}")
        
        return success
    
    def get_statistics(self) -> dict:
        """Get notification statistics"""
        stats = {
            'total_notifications': self.total_notifications,
            'by_level': {level.value: count for level, count in self.notifications_by_level.items()},
            'enabled_channels': self.enabled_channels
        }
        
        if self.telegram:
            stats['telegram'] = self.telegram.get_statistics()
        
        if self.twilio:
            stats['twilio'] = self.twilio.get_statistics()
        
        return stats
