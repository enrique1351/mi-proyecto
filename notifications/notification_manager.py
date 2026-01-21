"""
Notification Manager
====================

Centralized notification system for sending alerts via multiple channels.
"""

import logging
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels."""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SMS = "sms"
    PUSHBULLET = "pushbullet"


class NotificationManager:
    """
    Manages notifications across multiple channels.
    
    Supports:
    - Telegram Bot API
    - Email (SMTP)
    - SMS (Twilio)
    - Pushbullet
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize notification manager.
        
        Args:
            config: Configuration dict with credentials for each channel
        """
        self.config = config or {}
        self.channels: Dict[NotificationChannel, bool] = {}
        
        # Initialize channels
        self._initialize_telegram()
        self._initialize_email()
        self._initialize_sms()
        self._initialize_pushbullet()
        
        logger.info(f"NotificationManager initialized with {len([c for c in self.channels.values() if c])} active channels")
    
    def _initialize_telegram(self) -> bool:
        """Initialize Telegram bot."""
        try:
            bot_token = self.config.get('telegram_bot_token')
            chat_id = self.config.get('telegram_chat_id')
            
            if bot_token and chat_id:
                from .telegram_notifier import TelegramNotifier
                self.telegram = TelegramNotifier(bot_token, chat_id)
                self.channels[NotificationChannel.TELEGRAM] = True
                logger.info("✅ Telegram notifications enabled")
                return True
            else:
                self.channels[NotificationChannel.TELEGRAM] = False
                logger.warning("⚠️ Telegram credentials not found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram: {e}")
            self.channels[NotificationChannel.TELEGRAM] = False
            return False
    
    def _initialize_email(self) -> bool:
        """Initialize email notifications."""
        try:
            smtp_config = self.config.get('smtp')
            
            if smtp_config:
                from .email_notifier import EmailNotifier
                self.email = EmailNotifier(smtp_config)
                self.channels[NotificationChannel.EMAIL] = True
                logger.info("✅ Email notifications enabled")
                return True
            else:
                self.channels[NotificationChannel.EMAIL] = False
                logger.debug("Email credentials not found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Email: {e}")
            self.channels[NotificationChannel.EMAIL] = False
            return False
    
    def _initialize_sms(self) -> bool:
        """Initialize SMS notifications via Twilio."""
        try:
            twilio_config = self.config.get('twilio')
            
            if twilio_config:
                from .sms_notifier import SMSNotifier
                self.sms = SMSNotifier(twilio_config)
                self.channels[NotificationChannel.SMS] = True
                logger.info("✅ SMS notifications enabled")
                return True
            else:
                self.channels[NotificationChannel.SMS] = False
                logger.debug("Twilio credentials not found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize SMS: {e}")
            self.channels[NotificationChannel.SMS] = False
            return False
    
    def _initialize_pushbullet(self) -> bool:
        """Initialize Pushbullet notifications."""
        try:
            pushbullet_token = self.config.get('pushbullet_token')
            
            if pushbullet_token:
                from .pushbullet_notifier import PushbulletNotifier
                self.pushbullet = PushbulletNotifier(pushbullet_token)
                self.channels[NotificationChannel.PUSHBULLET] = True
                logger.info("✅ Pushbullet notifications enabled")
                return True
            else:
                self.channels[NotificationChannel.PUSHBULLET] = False
                logger.debug("Pushbullet credentials not found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pushbullet: {e}")
            self.channels[NotificationChannel.PUSHBULLET] = False
            return False
    
    def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        title: Optional[str] = None
    ) -> Dict[NotificationChannel, bool]:
        """
        Send notification to one or more channels.
        
        Args:
            message: Notification message
            level: Priority level
            channels: List of channels to use (None = all active channels)
            title: Optional title for the notification
            
        Returns:
            Dict mapping each channel to success status
        """
        results = {}
        
        # Use all active channels if not specified
        if channels is None:
            channels = [ch for ch, active in self.channels.items() if active]
        
        # Add timestamp and level to message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level.value.upper()}]\n{message}"
        
        # Send to each channel
        for channel in channels:
            try:
                if not self.channels.get(channel, False):
                    logger.warning(f"Channel {channel.value} not active, skipping")
                    results[channel] = False
                    continue
                
                success = self._send_to_channel(channel, formatted_message, title, level)
                results[channel] = success
                
                if success:
                    logger.debug(f"✅ Sent to {channel.value}")
                else:
                    logger.warning(f"⚠️ Failed to send to {channel.value}")
                    
            except Exception as e:
                logger.error(f"❌ Error sending to {channel.value}: {e}")
                results[channel] = False
        
        return results
    
    def _send_to_channel(
        self,
        channel: NotificationChannel,
        message: str,
        title: Optional[str],
        level: NotificationLevel
    ) -> bool:
        """Send to a specific channel."""
        try:
            if channel == NotificationChannel.TELEGRAM:
                return self.telegram.send(message, level)
            elif channel == NotificationChannel.EMAIL:
                return self.email.send(message, title or "Trading Bot Alert", level)
            elif channel == NotificationChannel.SMS:
                # Only send SMS for WARNING and above
                if level in [NotificationLevel.WARNING, NotificationLevel.ERROR, NotificationLevel.CRITICAL]:
                    return self.sms.send(message)
                return True  # Skip INFO messages for SMS
            elif channel == NotificationChannel.PUSHBULLET:
                return self.pushbullet.send(title or "Trading Bot", message, level)
            else:
                logger.error(f"Unknown channel: {channel}")
                return False
        except Exception as e:
            logger.error(f"Error in _send_to_channel for {channel.value}: {e}")
            return False
    
    def send_trade_alert(self, trade_data: Dict) -> bool:
        """
        Send trade execution alert.
        
        Args:
            trade_data: Dict with trade information
            
        Returns:
            True if sent successfully to at least one channel
        """
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'UNKNOWN')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        
        message = f"""
🔔 Trade Executed

Symbol: {symbol}
Side: {side}
Quantity: {quantity}
Price: ${price:,.2f}
Total: ${quantity * price:,.2f}
"""
        
        results = self.send(
            message,
            level=NotificationLevel.INFO,
            title="Trade Executed"
        )
        
        return any(results.values())
    
    def send_error_alert(self, error_message: str) -> bool:
        """
        Send error alert to all critical channels.
        
        Args:
            error_message: Error description
            
        Returns:
            True if sent successfully to at least one channel
        """
        message = f"⚠️ ERROR: {error_message}"
        
        results = self.send(
            message,
            level=NotificationLevel.ERROR,
            title="Trading Bot Error"
        )
        
        return any(results.values())
    
    def send_daily_summary(self, summary_data: Dict) -> bool:
        """
        Send daily performance summary.
        
        Args:
            summary_data: Dict with daily statistics
            
        Returns:
            True if sent successfully to at least one channel
        """
        message = f"""
📊 Daily Summary

Trades: {summary_data.get('total_trades', 0)}
PnL: ${summary_data.get('pnl', 0):,.2f}
Win Rate: {summary_data.get('win_rate', 0):.1%}
Largest Win: ${summary_data.get('largest_win', 0):,.2f}
Largest Loss: ${summary_data.get('largest_loss', 0):,.2f}
"""
        
        results = self.send(
            message,
            level=NotificationLevel.INFO,
            title="Daily Summary"
        )
        
        return any(results.values())
    
    def health_check(self) -> Dict[NotificationChannel, bool]:
        """
        Check health of all channels.
        
        Returns:
            Dict mapping each channel to health status
        """
        return self.channels.copy()
