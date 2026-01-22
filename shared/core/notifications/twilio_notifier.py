"""
Twilio Notifier
SMS alerts via Twilio API
"""

import logging
from typing import Optional, Any
from datetime import datetime

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class TwilioNotifier:
    """
    Twilio SMS notification service for critical trading alerts
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None,
        credential_vault: Optional[Any] = None
    ):
        """
        Initialize Twilio notifier
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: Twilio phone number (sender)
            to_number: Recipient phone number
            credential_vault: Vault with credentials
        """
        if not TWILIO_AVAILABLE:
            raise ImportError(
                "Twilio not installed. Install with: pip install twilio"
            )
        
        # Get credentials
        if credential_vault:
            self.account_sid = credential_vault.get_credential("twilio_account_sid")
            self.auth_token = credential_vault.get_credential("twilio_auth_token")
            self.from_number = credential_vault.get_credential("twilio_from_number")
            self.to_number = credential_vault.get_credential("twilio_to_number")
        else:
            import os
            self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
            self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
            self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
            self.to_number = to_number or os.getenv("TWILIO_TO_NUMBER")
        
        if not all([self.account_sid, self.auth_token, self.from_number, self.to_number]):
            raise ValueError("All Twilio credentials are required")
        
        self.client: Optional[Client] = None
        self.connected = False
        self.messages_sent = 0
        self.failed_messages = 0
        
    def connect(self) -> bool:
        """Connect to Twilio API"""
        try:
            self.client = Client(self.account_sid, self.auth_token)
            # Test connection by fetching account info
            account = self.client.api.accounts(self.account_sid).fetch()
            self.connected = True
            logger.info(f"✅ Connected to Twilio (Account: {account.friendly_name})")
            return True
            
        except TwilioRestException as e:
            logger.error(f"❌ Failed to connect to Twilio: {e}")
            self.connected = False
            return False
    
    def send_sms(self, message: str, priority: bool = False) -> bool:
        """
        Send SMS message
        
        Args:
            message: Message text (max 160 chars for single SMS)
            priority: If True, send immediately even if rate limited
        
        Returns:
            True if sent successfully
        """
        if not self.connected:
            logger.error("Not connected to Twilio")
            return False
        
        try:
            # Truncate message if too long
            if len(message) > 160:
                message = message[:157] + "..."
            
            msg = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            
            self.messages_sent += 1
            logger.info(f"✅ SMS sent (SID: {msg.sid})")
            return True
            
        except TwilioRestException as e:
            logger.error(f"❌ Failed to send SMS: {e}")
            self.failed_messages += 1
            return False
    
    def send_critical_alert(self, alert_message: str) -> bool:
        """Send critical alert via SMS"""
        message = f"🚨 CRITICAL: {alert_message}"
        return self.send_sms(message, priority=True)
    
    def send_trade_alert(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> bool:
        """Send trade execution alert"""
        message = (
            f"Trade: {side} {quantity} {symbol} @ ${price:,.2f} "
            f"[{datetime.now().strftime('%H:%M')}]"
        )
        return self.send_sms(message)
    
    def send_error_alert(self, error_message: str) -> bool:
        """Send error alert"""
        message = f"⚠️ ERROR: {error_message[:140]}"
        return self.send_sms(message, priority=True)
    
    def send_stop_loss_alert(
        self,
        symbol: str,
        loss_amount: float,
        loss_percentage: float
    ) -> bool:
        """Send stop loss triggered alert"""
        message = (
            f"🛑 STOP LOSS: {symbol} -${loss_amount:,.2f} "
            f"({loss_percentage:.1f}%)"
        )
        return self.send_sms(message, priority=True)
    
    def send_margin_call_alert(self, margin_level: float) -> bool:
        """Send margin call alert"""
        message = f"⚠️ MARGIN CALL: Level at {margin_level:.1f}% - Action required!"
        return self.send_sms(message, priority=True)
    
    def send_system_shutdown_alert(self, reason: str = "Unknown") -> bool:
        """Send system shutdown alert"""
        message = f"🔴 System Shutdown: {reason}"
        return self.send_sms(message, priority=True)
    
    def get_statistics(self) -> dict:
        """Get notification statistics"""
        return {
            'connected': self.connected,
            'messages_sent': self.messages_sent,
            'failed_messages': self.failed_messages,
            'success_rate': (
                self.messages_sent / (self.messages_sent + self.failed_messages)
                if (self.messages_sent + self.failed_messages) > 0
                else 0.0
            )
        }
