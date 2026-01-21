"""
Telegram Notifier
=================

Send notifications via Telegram Bot API.
"""

import logging
import requests
from typing import Optional
from .notification_manager import NotificationLevel


logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Send notifications via Telegram Bot API.
    
    Requires:
    - Bot token (from @BotFather)
    - Chat ID (user or group)
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Verify bot connection
        if not self._verify_connection():
            logger.warning("⚠️ Could not verify Telegram bot connection")
    
    def _verify_connection(self) -> bool:
        """Verify bot token is valid."""
        try:
            response = requests.get(
                f"{self.base_url}/getMe",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram connection verification failed: {e}")
            return False
    
    def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        parse_mode: str = "HTML"
    ) -> bool:
        """
        Send message via Telegram.
        
        Args:
            message: Message text
            level: Notification level (affects formatting)
            parse_mode: Telegram parse mode (HTML, Markdown, None)
            
        Returns:
            True if sent successfully
        """
        try:
            # Add emoji based on level
            emoji_map = {
                NotificationLevel.INFO: "ℹ️",
                NotificationLevel.WARNING: "⚠️",
                NotificationLevel.ERROR: "❌",
                NotificationLevel.CRITICAL: "🚨"
            }
            
            emoji = emoji_map.get(level, "")
            formatted_message = f"{emoji} {message}"
            
            # Send message
            payload = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_document(self, file_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a document file via Telegram.
        
        Args:
            file_path: Path to file to send
            caption: Optional caption for the file
            
        Returns:
            True if sent successfully
        """
        try:
            with open(file_path, 'rb') as file:
                payload = {
                    'chat_id': self.chat_id
                }
                
                if caption:
                    payload['caption'] = caption
                
                files = {'document': file}
                
                response = requests.post(
                    f"{self.base_url}/sendDocument",
                    data=payload,
                    files=files,
                    timeout=30
                )
                
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Failed to send Telegram document: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a photo via Telegram.
        
        Args:
            photo_path: Path to photo file
            caption: Optional caption for the photo
            
        Returns:
            True if sent successfully
        """
        try:
            with open(photo_path, 'rb') as photo:
                payload = {
                    'chat_id': self.chat_id
                }
                
                if caption:
                    payload['caption'] = caption
                
                files = {'photo': photo}
                
                response = requests.post(
                    f"{self.base_url}/sendPhoto",
                    data=payload,
                    files=files,
                    timeout=30
                )
                
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False
