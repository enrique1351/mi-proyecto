"""
SMS Notifier
============

Send notifications via SMS using Twilio.
"""

import logging
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class SMSNotifier:
    """
    Send SMS notifications via Twilio.
    
    Requires Twilio account and credentials.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SMS notifier.
        
        Args:
            config: Twilio configuration dict with:
                - account_sid: Twilio account SID
                - auth_token: Twilio auth token
                - from_number: Twilio phone number
                - to_number: Recipient phone number
        """
        self.account_sid = config.get('account_sid')
        self.auth_token = config.get('auth_token')
        self.from_number = config.get('from_number')
        self.to_number = config.get('to_number')
        
        if not all([self.account_sid, self.auth_token, self.from_number, self.to_number]):
            raise ValueError("Twilio configuration incomplete")
        
        # Import Twilio client
        try:
            from twilio.rest import Client
            self.client = Client(self.account_sid, self.auth_token)
            logger.info("Twilio client initialized")
        except ImportError:
            logger.error("Twilio package not installed. Install with: pip install twilio")
            raise
    
    def send(self, message: str) -> bool:
        """
        Send SMS message.
        
        Args:
            message: SMS text (max 160 chars recommended)
            
        Returns:
            True if sent successfully
        """
        try:
            # Truncate long messages
            if len(message) > 160:
                message = message[:157] + "..."
            
            # Send SMS
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            
            logger.debug(f"SMS sent successfully. SID: {message_obj.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
