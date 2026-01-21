"""
Pushbullet Notifier
===================

Send push notifications via Pushbullet.
"""

import logging
import requests
from .notification_manager import NotificationLevel


logger = logging.getLogger(__name__)


class PushbulletNotifier:
    """
    Send push notifications via Pushbullet.
    
    Requires Pushbullet API token.
    """
    
    def __init__(self, api_token: str):
        """
        Initialize Pushbullet notifier.
        
        Args:
            api_token: Pushbullet API access token
        """
        self.api_token = api_token
        self.base_url = "https://api.pushbullet.com/v2"
        self.headers = {
            'Access-Token': api_token,
            'Content-Type': 'application/json'
        }
        
        # Verify token
        if not self._verify_token():
            logger.warning("⚠️ Could not verify Pushbullet token")
    
    def _verify_token(self) -> bool:
        """Verify API token is valid."""
        try:
            response = requests.get(
                f"{self.base_url}/users/me",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Pushbullet token verification failed: {e}")
            return False
    
    def send(
        self,
        title: str,
        body: str,
        level: NotificationLevel = NotificationLevel.INFO
    ) -> bool:
        """
        Send push notification.
        
        Args:
            title: Notification title
            body: Notification body
            level: Notification level
            
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
            formatted_title = f"{emoji} {title}"
            
            # Send push
            payload = {
                'type': 'note',
                'title': formatted_title,
                'body': body
            }
            
            response = requests.post(
                f"{self.base_url}/pushes",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug("Pushbullet notification sent successfully")
                return True
            else:
                logger.error(f"Pushbullet API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Pushbullet notification: {e}")
            return False
    
    def send_link(self, title: str, url: str) -> bool:
        """
        Send a link notification.
        
        Args:
            title: Link title
            url: URL to share
            
        Returns:
            True if sent successfully
        """
        try:
            payload = {
                'type': 'link',
                'title': title,
                'url': url
            }
            
            response = requests.post(
                f"{self.base_url}/pushes",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send Pushbullet link: {e}")
            return False
