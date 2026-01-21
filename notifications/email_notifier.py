"""
Email Notifier
==============

Send notifications via SMTP email.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict
from .notification_manager import NotificationLevel


logger = logging.getLogger(__name__)


class EmailNotifier:
    """
    Send notifications via SMTP email.
    
    Supports Gmail, Outlook, and custom SMTP servers.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize email notifier.
        
        Args:
            config: SMTP configuration dict with:
                - server: SMTP server address
                - port: SMTP port
                - user: Email username
                - password: Email password or app password
                - from_email: Sender email address
                - to_email: Recipient email address
        """
        self.server = config.get('server', 'smtp.gmail.com')
        self.port = config.get('port', 587)
        self.user = config.get('user')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.user)
        self.to_email = config.get('to_email')
        
        if not all([self.user, self.password, self.to_email]):
            raise ValueError("Email configuration incomplete")
    
    def send(
        self,
        message: str,
        subject: str = "Trading Bot Notification",
        level: NotificationLevel = NotificationLevel.INFO
    ) -> bool:
        """
        Send email notification.
        
        Args:
            message: Email body
            subject: Email subject
            level: Notification level
            
        Returns:
            True if sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{level.value.upper()}] {subject}"
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            
            # Create HTML and plain text versions
            text_part = MIMEText(message, 'plain')
            html_part = MIMEText(self._format_html(message, level), 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            
            logger.debug("Email sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _format_html(self, message: str, level: NotificationLevel) -> str:
        """Format message as HTML."""
        color_map = {
            NotificationLevel.INFO: "#17a2b8",
            NotificationLevel.WARNING: "#ffc107",
            NotificationLevel.ERROR: "#dc3545",
            NotificationLevel.CRITICAL: "#721c24"
        }
        
        color = color_map.get(level, "#17a2b8")
        
        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="padding: 20px; border-left: 4px solid {color};">
                    <h2 style="color: {color};">Trading Bot Notification</h2>
                    <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
{message}
                    </pre>
                </div>
            </body>
        </html>
        """
        
        return html
