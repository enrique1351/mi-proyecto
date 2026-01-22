"""
Telegram Notifier
Real-time alerts via Telegram Bot API
"""

import logging
from typing import Optional, List, Any
from datetime import datetime

try:
    from telegram import Bot
    from telegram.error import TelegramError
    import asyncio
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification service for trading alerts
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        credential_vault: Optional[Any] = None
    ):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Target chat ID for notifications
            credential_vault: Vault with credentials
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot not installed. Install with: "
                "pip install python-telegram-bot"
            )
        
        # Get credentials
        if credential_vault:
            self.bot_token = credential_vault.get_credential("telegram_bot_token")
            self.chat_id = credential_vault.get_credential("telegram_chat_id")
        else:
            import os
            self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
            self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            raise ValueError("Telegram bot_token and chat_id are required")
        
        self.bot: Optional[Bot] = None
        self.connected = False
        self.messages_sent = 0
        self.failed_messages = 0
        
    async def _connect_async(self) -> bool:
        """Connect to Telegram API (async)"""
        try:
            self.bot = Bot(token=self.bot_token)
            # Test connection
            await self.bot.get_me()
            self.connected = True
            logger.info("✅ Connected to Telegram")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Telegram: {e}")
            self.connected = False
            return False
    
    def connect(self) -> bool:
        """Connect to Telegram API (sync wrapper)"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._connect_async())
    
    async def _send_message_async(
        self,
        message: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """Send message (async)"""
        if not self.connected:
            logger.error("Not connected to Telegram")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            self.messages_sent += 1
            logger.debug(f"✅ Telegram message sent: {message[:50]}...")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")
            self.failed_messages += 1
            return False
    
    def send_message(
        self,
        message: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """Send message (sync wrapper)"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._send_message_async(message, parse_mode, disable_notification)
        )
    
    def send_trade_alert(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str = "EXECUTED"
    ) -> bool:
        """Send trade execution alert"""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        
        message = (
            f"{emoji} <b>Trade Alert</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Quantity:</b> {quantity}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>Status:</b> {status}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return self.send_message(message)
    
    def send_error_alert(self, error_message: str, severity: str = "ERROR") -> bool:
        """Send error alert"""
        emoji = "⚠️" if severity == "WARNING" else "🚨"
        
        message = (
            f"{emoji} <b>{severity}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{error_message}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return self.send_message(message, disable_notification=(severity == "WARNING"))
    
    def send_performance_update(
        self,
        total_pnl: float,
        daily_pnl: float,
        win_rate: float,
        total_trades: int
    ) -> bool:
        """Send performance summary"""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        message = (
            f"{pnl_emoji} <b>Performance Update</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"<b>Total P&L:</b> ${total_pnl:,.2f}\n"
            f"<b>Daily P&L:</b> ${daily_pnl:,.2f}\n"
            f"<b>Win Rate:</b> {win_rate:.1f}%\n"
            f"<b>Total Trades:</b> {total_trades}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return self.send_message(message, disable_notification=True)
    
    def send_system_status(
        self,
        status: str,
        uptime: str = None,
        active_strategies: int = 0
    ) -> bool:
        """Send system status update"""
        emoji = "🟢" if status == "RUNNING" else "🔴"
        
        message = (
            f"{emoji} <b>System Status</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"<b>Status:</b> {status}\n"
        )
        
        if uptime:
            message += f"<b>Uptime:</b> {uptime}\n"
        
        message += (
            f"<b>Active Strategies:</b> {active_strategies}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return self.send_message(message)
    
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
