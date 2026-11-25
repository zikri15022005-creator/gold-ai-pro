"""
autonomous_scanner.py - 24/7 Autonomous Background Market Scanner for Gold AI Pro.
Continuously scans XAUUSD across timeframes and strategies without requiring user clicks.
Automatically emits notifications, logs sniper signals, and triggers execution if enabled.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("AutonomousScanner")
WIB_TZ = timezone(timedelta(hours=7))

class AutonomousScanner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AutonomousScanner, cls).__new__(cls)
            cls._instance._init_scanner()
        return cls._instance

    def _init_scanner(self):
        self.is_running = True
        self.scan_interval_sec = 15
        self.min_quality_score = 80
        self.max_spread_allowed = 0.85
        self.last_signal_time = 0
        self.last_signal_action = "WAIT"
        self.latest_result: Dict[str, Any] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Starts the autonomous background scanning thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self.is_running = True
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()
            logger.info("Autonomous Scanner started successfully.")

    def stop(self):
        """Stops the autonomous background scanning thread."""
        with self._lock:
            self.is_running = False

    def _scan_loop(self):
        # Lazy import to avoid circular dependency
        from gold_ai_pro import GoldAIPro
        from mt5_engine import MT5Engine

        ai = GoldAIPro()
        mt5_eng = MT5Engine()

        while self.is_running:
            try:
                result = ai.analyze_market()
                self.latest_result = result

                action = result.get("action", "WAIT")
                quality = result.get("quality_score", 0)
                spread = result.get("spot_quote", {}).get("spread", 0.30)
                is_sniper = result.get("is_sniper_valid", False)

                now = time.time()
                # Check for new high-conviction sniper signal
                if is_sniper and action in ["BUY", "SELL"]:
                    # Deduplicate: Only alert if 5 mins passed or action flipped
                    if (now - self.last_signal_time > 300) or (action != self.last_signal_action):
                        self.last_signal_time = now
                        self.last_signal_action = action

                        alert_entry = {
                            "time": datetime.now(WIB_TZ).strftime("%H:%M:%S WIB"),
                            "action": action,
                            "price": result.get("entry_price"),
                            "strategy": result.get("strategy"),
                            "quality": quality,
                            "sl": result.get("sl_price"),
                            "tp1": result.get("tp1_price"),
                            "message": f"SNIPER SIGNAL {action} @ {result.get('entry_price')} (Kualitas: {quality}/100)"
                        }
                        self.alert_history.insert(0, alert_entry)
                        if len(self.alert_history) > 30:
                            self.alert_history.pop()

                        # If MT5 auto-trading is active, execute order
                        if mt5_eng.auto_trading_enabled and not mt5_eng.kill_switch_active:
                            lot = result.get("recommended_lot", 0.01)
                            sl = result.get("sl_price", 0.0)
                            tp = result.get("tp1_price", 0.0)
                            mt5_eng.execute_market_order(action, lot, sl, tp, comment="GoldAIPro Auto")

            except Exception as e:
                logger.warning(f"Scanner exception: {e}")

            time.sleep(self.scan_interval_sec)

    def get_scanner_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "scan_interval_sec": self.scan_interval_sec,
            "min_quality_score": self.min_quality_score,
            "max_spread_allowed": self.max_spread_allowed,
            "recent_alerts": self.alert_history[:10],
            "latest_signal": self.latest_result.get("action", "WAIT"),
            "latest_price": self.latest_result.get("current_price", 0.0)
        }
