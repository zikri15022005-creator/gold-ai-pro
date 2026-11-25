"""
app.py - GOLD AI PRO Master Web Server & Real-Time Trading Terminal.
Integrates:
- MetaTrader 5 Terminal Bridge (DEMO & LIVE safety guards)
- Real-Time Autonomous Market Scanner
- Multi-Timeframe & SMC Decision Engine
- Backtesting & Monte Carlo API
- SQLite Performance Journal
- Cloudflare Public Tunnel for instant mobile & remote sharing
"""

import os
import sys
import io
import socket
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, jsonify, request, send_file
import qrcode

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from gold_ai_pro import GoldAIPro
from autonomous_scanner import AutonomousScanner
from mt5_engine import MT5Engine
from performance_db import PerformanceDatabase
from backtest_engine import BacktestEngine
from tunnel_manager import get_public_url, start_tunnel

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

ai = GoldAIPro()
scanner = AutonomousScanner()
mt5_eng = MT5Engine()
db = PerformanceDatabase()

# Automatically start autonomous scanner in background
if not scanner.is_running:
    try:
        scanner.start()
    except Exception as e:
        app.logger.warning(f"Scanner auto-start: {e}")

WIB_TZ = timezone(timedelta(hours=7))

def get_local_ip() -> str:
    """Get the primary local IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
PORT = int(os.environ.get("PORT", 5000))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/signal")
def api_signal():
    """Returns comprehensive multi-timeframe analysis and signal."""
    strategy_override = request.args.get("strategy")
    if strategy_override == "AUTO" or not strategy_override:
        strategy_override = None

    try:
        result = ai.analyze_market(strategy_override=strategy_override)
        result["local_ip"] = LOCAL_IP
        result["public_url"] = get_public_url()
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Signal API error: {e}")
        return jsonify({
            "action": "WAIT",
            "error": str(e),
            "system_name": "GOLD AI PRO",
            "current_price": 0.0,
            "reasons": [f"Terjadi sinkronisasi data: {e}"]
        }), 500

@app.route("/api/scanner")
def api_scanner():
    """Returns autonomous scanner status and recent alerts."""
    return jsonify(scanner.get_scanner_status())

@app.route("/api/scanner/toggle", methods=["POST"])
def api_scanner_toggle():
    """Toggles autonomous background scanner ON/OFF."""
    data = request.get_json() or {}
    enable = data.get("enable", not scanner.is_running)
    if enable:
        scanner.start()
    else:
        scanner.stop()
    return jsonify({"success": True, "is_running": scanner.is_running})

@app.route("/api/mt5/status")
def api_mt5_status():
    """Returns MT5 connection status, account balance, and open positions."""
    acc = mt5_eng.get_account_info()
    positions = mt5_eng.get_open_positions()
    specs = mt5_eng.get_symbol_specs()
    return jsonify({
        "account": acc,
        "open_positions": positions,
        "symbol_specs": specs,
        "auto_trading_enabled": mt5_eng.auto_trading_enabled,
        "trading_mode": mt5_eng.trading_mode,
        "kill_switch_active": mt5_eng.kill_switch_active
    })

@app.route("/api/mt5/execute", methods=["POST"])
def api_mt5_execute():
    """Execute order via MT5 bridge (Strictly DEMO by default)."""
    data = request.get_json() or {}
    action = data.get("action", "").upper()
    volume = float(data.get("volume", 0.01))
    sl = float(data.get("sl", 0.0))
    tp = float(data.get("tp", 0.0))
    comment = data.get("comment", "GoldAIPro Manual")

    if action not in ["BUY", "SELL"]:
        return jsonify({"success": False, "message": "Aksi harus BUY atau SELL"}), 400

    res = mt5_eng.execute_market_order(action, volume, sl, tp, comment=comment)
    return jsonify(res)

@app.route("/api/mt5/close", methods=["POST"])
def api_mt5_close():
    """Close position by ticket."""
    data = request.get_json() or {}
    ticket = data.get("ticket")
    if not ticket:
        return jsonify({"success": False, "message": "Ticket required"}), 400
    res = mt5_eng.close_position(int(ticket))
    return jsonify(res)

@app.route("/api/mt5/mode", methods=["POST"])
def api_mt5_set_mode():
    """Toggle Auto-Trading mode."""
    data = request.get_json() or {}
    auto_trade = data.get("auto_trade", False)
    mode = data.get("mode", "DEMO") # DEMO or LIVE
    mt5_eng.auto_trading_enabled = bool(auto_trade)
    mt5_eng.trading_mode = mode
    return jsonify({
        "success": True,
        "auto_trading_enabled": mt5_eng.auto_trading_enabled,
        "trading_mode": mt5_eng.trading_mode
    })

@app.route("/api/trade/close", methods=["POST"])
def api_trade_close():
    """Manually close the currently locked active trade and reset scanner to look for next entry."""
    data = ai.mtf_engine.get_live_spot_quote()
    curr_price = float(data.get("price", 0.0))
    res = db.close_active_trade_manually(curr_price)
    return jsonify(res)

@app.route("/api/performance")
def api_performance():
    """Returns historical trade journal summary and recent signals."""
    summary = db.get_performance_summary()
    recent = db.get_recent_signals(limit=15)
    return jsonify({"summary": summary, "recent_signals": recent})

@app.route("/api/backtest")
def api_backtest():
    """Executes backtest for requested strategy on M15 candles."""
    strat = request.args.get("strategy", "INTRADAY")
    m15_df = ai.mtf_engine.fetch_timeframe_candles("M15", num_bars=150)
    result = BacktestEngine.run_strategy_backtest(m15_df, strategy=strat)
    return jsonify(result)

@app.route("/api/network-info")
def api_network():
    pub = get_public_url()
    return jsonify({
        "local_ip": LOCAL_IP,
        "port": PORT,
        "local_url": f"http://{LOCAL_IP}:{PORT}",
        "public_url": pub,
        "share_url": pub or f"http://{LOCAL_IP}:{PORT}",
        "wifi_guide": f"Buka http://{LOCAL_IP}:{PORT} jika terhubung ke Wi-Fi rumah.",
        "data_guide": f"Buka {pub} untuk akses bebas dari mana saja (HP kuota data/siapapun)." if pub else "Mengaktifkan link publik..."
    })

@app.route("/api/qrcode")
def api_qrcode():
    url = get_public_url() or f"http://{LOCAL_IP}:{PORT}"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    start_tunnel(PORT)
    scanner.start() # Start autonomous scanner background thread
    local_url = f"http://{LOCAL_IP}:{PORT}"
    pub_url = get_public_url()

    print("=================================================================")
    print("   [+] GOLD AI PRO — ULTIMATE AI TRADING SYSTEM AKTIF")
    print("=================================================================")
    print(f" [1] Buka di Laptop           : http://localhost:{PORT}")
    print(f" [2] Buka di HP (Wi-Fi Rumah) : {local_url}")
    if pub_url:
        print(f" [3] LINK PUBLIK (SHARE)      : {pub_url}")
        print("     (BISA DIBAGIKAN KE SIAPAPUN / KUOTA DATA HP DARI MANA SAJA)")
    print("=================================================================")

    app.run(host="0.0.0.0", port=PORT, debug=False)
