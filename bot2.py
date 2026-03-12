"""
alpaca_live_bot_verbose.py

Verbose Alpaca crypto bot with the SAME strategy logic as your original bot:
- Long only
- One local position per symbol
- Python-managed stop / targets / trailing stop
- Paper trading by default
- Extra diagnostics so you can see if it is connected and what it is doing
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("alpaca_live_bot")


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class BotConfig:
    paper: bool = True

    symbols: tuple[str, ...] = (
        "AVAX/USD",
        "ETH/USD",
        "SOL/USD",
        # "ALGO/USD",
        # "ATOM/USD",
        # "NEAR/USD",
    )

    risk_pct: float = 0.0025
    max_cash_per_trade_pct: float = 0.25
    strong_body_pct: float = 0.65
    adx_min: float = 20.0
    rsi_long_max: float = 65.0
    pullback_lookback: int = 10
    time_stop_bars: int = 96
    max_stop_pct: float = 0.03
    partial_r: float = 1.0
    final_r: float = 2.0
    atr_trail_mult: float = 1.5
    atr_stop_cap: float = 2.0
    atr_wick_pad: float = 0.1

    sleep_seconds: int = 30
    lookback_hours: int = 40
    min_notional_usd: float = 10.0
    min_bars_required: int = 250

    log_no_entry_reasons: bool = True
    log_hold_updates: bool = True
    log_cycle_header: bool = True


@dataclass
class PositionState:
    entry: float
    stop: float
    target_1: float
    target_2: float
    risk_price: float
    qty: float
    partial_done: bool
    entry_bar_time: pd.Timestamp


@dataclass
class SymbolState:
    last_bar_time: Optional[pd.Timestamp] = None
    position: Optional[PositionState] = None


CONFIG = BotConfig()


# ============================================================
# HELPERS
# ============================================================

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def normalize_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "").replace("-", "").upper()


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fmt(value) -> str:
    return str(value) if value is not None else "N/A"


# ============================================================
# CLIENTS
# ============================================================

API_KEY = "PKNKZ2KHJG6J5GBSB3XQIOV372"
API_SECRET = "CzRam4m8KToFnFKKQeZ4pp886XU9CgAwH5mnNSwthgN6"

trade_client = TradingClient(API_KEY, API_SECRET, paper=CONFIG.paper)
data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

STATE: Dict[str, SymbolState] = {symbol: SymbolState() for symbol in CONFIG.symbols}


# ============================================================
# ACCOUNT / PORTFOLIO VISIBILITY
# ============================================================

def log_account_snapshot() -> None:
    account = trade_client.get_account()

    logger.info("========== ACCOUNT SNAPSHOT ==========")
    logger.info("Account ID: %s", fmt(getattr(account, "id", None)))
    logger.info("Account Number: %s", fmt(getattr(account, "account_number", None)))
    logger.info("Status: %s", fmt(getattr(account, "status", None)))
    logger.info("Currency: %s", fmt(getattr(account, "currency", None)))
    logger.info("Cash: %s", fmt(getattr(account, "cash", None)))
    logger.info("Equity: %s", fmt(getattr(account, "equity", None)))
    logger.info("Portfolio Value: %s", fmt(getattr(account, "portfolio_value", None)))
    logger.info("Buying Power: %s", fmt(getattr(account, "buying_power", None)))
    logger.info("Long Market Value: %s", fmt(getattr(account, "long_market_value", None)))
    logger.info("Short Market Value: %s", fmt(getattr(account, "short_market_value", None)))
    logger.info("Trading Blocked: %s", fmt(getattr(account, "trading_blocked", None)))
    logger.info("Transfers Blocked: %s", fmt(getattr(account, "transfers_blocked", None)))
    logger.info("Account Blocked: %s", fmt(getattr(account, "account_blocked", None)))
    logger.info("Pattern Day Trader: %s", fmt(getattr(account, "pattern_day_trader", None)))
    logger.info("Paper Mode Config: %s", CONFIG.paper)
    logger.info("======================================")


def get_broker_positions_map() -> Dict[str, object]:
    positions = trade_client.get_all_positions()
    out = {}
    for p in positions:
        symbol = getattr(p, "symbol", "")
        out[normalize_symbol(symbol)] = p
    return out


def log_open_positions() -> None:
    positions = trade_client.get_all_positions()

    logger.info("========== OPEN POSITIONS ==========")
    if not positions:
        logger.info("No open broker positions found.")
        logger.info("====================================")
        return

    for p in positions:
        logger.info(
            "Position | symbol=%s qty=%s side=%s market_value=%s avg_entry=%s unrealized_pl=%s unrealized_plpc=%s",
            fmt(getattr(p, "symbol", None)),
            fmt(getattr(p, "qty", None)),
            fmt(getattr(p, "side", None)),
            fmt(getattr(p, "market_value", None)),
            fmt(getattr(p, "avg_entry_price", None)),
            fmt(getattr(p, "unrealized_pl", None)),
            fmt(getattr(p, "unrealized_plpc", None)),
        )

    logger.info("====================================")


def sync_state_with_broker_positions() -> None:
    try:
        broker_positions = get_broker_positions_map()

        for symbol in CONFIG.symbols:
            norm_symbol = normalize_symbol(symbol)
            local_has_position = STATE[symbol].position is not None
            broker_has_position = norm_symbol in broker_positions

            if local_has_position and not broker_has_position:
                logger.warning(
                    "RECONCILE | symbol=%s local state had a position, but broker shows none. Clearing local state.",
                    symbol,
                )
                STATE[symbol].position = None

            elif (not local_has_position) and broker_has_position:
                broker_pos = broker_positions[norm_symbol]
                logger.warning(
                    "RECONCILE | symbol=%s broker has a live position but local state is empty | qty=%s avg_entry=%s unrealized_pl=%s",
                    symbol,
                    fmt(getattr(broker_pos, "qty", None)),
                    fmt(getattr(broker_pos, "avg_entry_price", None)),
                    fmt(getattr(broker_pos, "unrealized_pl", None)),
                )

    except Exception as exc:
        logger.exception("Failed to reconcile broker positions: %s", exc)


def log_symbol_broker_status(symbol: str) -> None:
    try:
        broker_positions = get_broker_positions_map()
        broker_pos = broker_positions.get(normalize_symbol(symbol))

        if broker_pos is None:
            logger.info("BROKER | symbol=%s no broker-side open position", symbol)
        else:
            logger.info(
                "BROKER | symbol=%s qty=%s avg_entry=%s market_value=%s unrealized_pl=%s",
                symbol,
                fmt(getattr(broker_pos, "qty", None)),
                fmt(getattr(broker_pos, "avg_entry_price", None)),
                fmt(getattr(broker_pos, "market_value", None)),
                fmt(getattr(broker_pos, "unrealized_pl", None)),
            )
    except Exception as exc:
        logger.exception("Failed to log broker status for %s: %s", symbol, exc)


# ============================================================
# DATA
# ============================================================

def get_5m_bars(symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=CONFIG.lookback_hours)

    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
    )

    df = data_client.get_crypto_bars(request).df
    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex):
        if "symbol" in df.index.names:
            df = df.xs(symbol, level="symbol")
        else:
            for level in range(len(df.index.names)):
                try:
                    df = df.xs(symbol, level=level)
                    break
                except Exception:
                    continue

    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    hl = out["High"] - out["Low"]
    hc = (out["High"] - out["Close"].shift(1)).abs()
    lc = (out["Low"] - out["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    out["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    up_move = out["High"].diff()
    down_move = -out["Low"].diff()

    plus_dm_raw = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm_raw, index=out.index)
    minus_dm = pd.Series(minus_dm_raw, index=out.index)

    atr_smma = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_smma
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_smma
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    out["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return out


# ============================================================
# STRATEGY HELPERS
# ============================================================

def is_strong_bull_candle(row: pd.Series) -> bool:
    candle_range = row["High"] - row["Low"]
    if candle_range <= 0:
        return False

    body_ratio = abs(row["Close"] - row["Open"]) / candle_range
    return body_ratio >= CONFIG.strong_body_pct and row["Close"] > row["Open"]


def get_account_equity_and_cash() -> tuple[float, float]:
    account = trade_client.get_account()
    return safe_float(getattr(account, "equity", 0.0)), safe_float(getattr(account, "cash", 0.0))


def calc_notional(entry: float, stop: float) -> float:
    equity, cash = get_account_equity_and_cash()

    stop_pct = abs(entry - stop) / entry
    if stop_pct <= 0 or stop_pct > CONFIG.max_stop_pct:
        return 0.0

    risk_usd = equity * CONFIG.risk_pct
    by_risk = risk_usd / stop_pct
    by_cash = cash * CONFIG.max_cash_per_trade_pct
    notional = min(by_risk, by_cash)

    return max(0.0, round(notional, 2))


def diagnose_entry(symbol: str, df: pd.DataFrame) -> tuple[bool, str, dict]:
    row = df.iloc[-1]

    needed = ["ATR", "RSI", "ADX", "MA50", "MA200"]
    if any(pd.isna(row[c]) for c in needed):
        return False, "Indicators not ready yet", {}

    bull = (row["Close"] > row["MA200"]) and (row["MA50"] > row["MA200"])
    if not bull:
        return False, "Trend filter failed (not above MA200 or MA50 <= MA200)", {}

    if row["ADX"] < CONFIG.adx_min:
        return False, f"ADX too low ({row['ADX']:.2f} < {CONFIG.adx_min})", {}

    if row["RSI"] > CONFIG.rsi_long_max:
        return False, f"RSI too high ({row['RSI']:.2f} > {CONFIG.rsi_long_max})", {}

    if not is_strong_bull_candle(row):
        return False, "Current candle is not a strong bullish candle", {}

    atr = float(row["ATR"])
    if atr <= 0:
        return False, "ATR invalid", {}

    win = df.iloc[-CONFIG.pullback_lookback:]
    if not (win["Low"] <= win["MA50"] + atr).any():
        return False, "No recent pullback near MA50 + ATR zone", {}

    ma50_distance_atr = (row["Close"] - row["MA50"]) / atr
    if ma50_distance_atr > 4:
        return False, f"Too extended from MA50 ({ma50_distance_atr:.2f} ATR)", {}

    entry = float(row["Close"])
    stop = max(
        float(row["Low"] - CONFIG.atr_wick_pad * atr),
        float(row["Close"] - CONFIG.atr_stop_cap * atr),
    )

    if stop >= entry:
        return False, "Computed stop is not below entry", {}

    notional = calc_notional(entry, stop)
    if notional < CONFIG.min_notional_usd:
        return False, f"Notional too small ({notional:.2f} < {CONFIG.min_notional_usd:.2f})", {}

    payload = {
        "entry": entry,
        "stop": stop,
        "notional": notional,
        "atr": atr,
        "risk_price": entry - stop,
    }
    return True, "Entry conditions passed", payload


# ============================================================
# ORDER EXECUTION
# ============================================================

def submit_buy(symbol: str, notional: float):
    order = MarketOrderRequest(
        symbol=symbol,
        notional=notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )
    return trade_client.submit_order(order_data=order)


def submit_sell(symbol: str, qty: float):
    qty = round(float(qty), 6)
    if qty <= 0:
        return None

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
    )
    return trade_client.submit_order(order_data=order)


# ============================================================
# ENTRY
# ============================================================

def maybe_enter(symbol: str, df: pd.DataFrame) -> None:
    state = STATE[symbol]

    allowed, reason, payload = diagnose_entry(symbol, df)

    if not allowed:
        if CONFIG.log_no_entry_reasons:
            logger.info("NO_ENTRY | symbol=%s reason=%s", symbol, reason)
        return

    entry = payload["entry"]
    stop = payload["stop"]
    notional = payload["notional"]
    risk_price = payload["risk_price"]

    order = submit_buy(symbol, notional)
    qty = notional / entry

    state.position = PositionState(
        entry=entry,
        stop=stop,
        target_1=entry + CONFIG.partial_r * risk_price,
        target_2=entry + CONFIG.final_r * risk_price,
        risk_price=risk_price,
        qty=qty,
        partial_done=False,
        entry_bar_time=df.index[-1],
    )

    logger.info(
        "BUY | symbol=%s order_id=%s entry=%.4f stop=%.4f target_1=%.4f target_2=%.4f notional=%.2f qty=%.6f",
        symbol,
        fmt(getattr(order, "id", None)),
        entry,
        stop,
        state.position.target_1,
        state.position.target_2,
        notional,
        qty,
    )

    log_symbol_broker_status(symbol)


# ============================================================
# EXIT
# ============================================================

def maybe_exit(symbol: str, df: pd.DataFrame) -> None:
    state = STATE[symbol]
    pos = state.position
    if pos is None:
        return

    row = df.iloc[-1]
    low = float(row["Low"])
    high = float(row["High"])
    close = float(row["Close"])

    entry = pos.entry
    stop = pos.stop
    t1 = pos.target_1
    t2 = pos.target_2
    qty = pos.qty
    risk_price = pos.risk_price

    if low <= stop:
        order = submit_sell(symbol, qty)
        logger.info(
            "SELL | symbol=%s reason=STOP order_id=%s qty=%.6f stop=%.4f close=%.4f",
            symbol,
            fmt(getattr(order, "id", None)),
            qty,
            stop,
            close,
        )
        state.position = None
        log_symbol_broker_status(symbol)
        return

    if high >= t2:
        order = submit_sell(symbol, qty)
        logger.info(
            "SELL | symbol=%s reason=TARGET2 order_id=%s qty=%.6f target_2=%.4f close=%.4f",
            symbol,
            fmt(getattr(order, "id", None)),
            qty,
            t2,
            close,
        )
        state.position = None
        log_symbol_broker_status(symbol)
        return

    if not pos.partial_done and high >= t1:
        half = qty / 2.0
        order = submit_sell(symbol, half)
        pos.qty = qty - half
        pos.partial_done = True
        pos.stop = entry

        logger.info(
            "SELL | symbol=%s reason=TARGET1 order_id=%s sold_qty=%.6f remaining_qty=%.6f new_stop=%.4f",
            symbol,
            fmt(getattr(order, "id", None)),
            half,
            pos.qty,
            pos.stop,
        )

    if pos.partial_done:
        trail = close - CONFIG.atr_trail_mult * float(row["ATR"])
        if trail > pos.stop:
            old_stop = pos.stop
            pos.stop = trail
            logger.info(
                "TRAIL | symbol=%s old_stop=%.4f new_stop=%.4f close=%.4f",
                symbol,
                old_stop,
                pos.stop,
                close,
            )

    bars_since_entry = len(df[df.index >= pos.entry_bar_time]) - 1
    if bars_since_entry > CONFIG.time_stop_bars and close < entry + 0.3 * risk_price:
        order = submit_sell(symbol, pos.qty)
        logger.info(
            "SELL | symbol=%s reason=TIMEOUT order_id=%s qty=%.6f bars_since_entry=%d close=%.4f",
            symbol,
            fmt(getattr(order, "id", None)),
            pos.qty,
            bars_since_entry,
            close,
        )
        state.position = None
        log_symbol_broker_status(symbol)
        return

    if CONFIG.log_hold_updates and state.position is not None:
        logger.info(
            "HOLD | symbol=%s qty=%.6f entry=%.4f stop=%.4f t1=%.4f t2=%.4f close=%.4f partial_done=%s",
            symbol,
            pos.qty,
            pos.entry,
            pos.stop,
            pos.target_1,
            pos.target_2,
            close,
            pos.partial_done,
        )


# ============================================================
# PROCESSING
# ============================================================

def log_bar_snapshot(symbol: str, df: pd.DataFrame) -> None:
    row = df.iloc[-1]
    logger.info(
        "BAR | symbol=%s time=%s open=%.4f high=%.4f low=%.4f close=%.4f volume=%.4f ma50=%.4f ma200=%.4f atr=%.4f rsi=%.2f adx=%.2f",
        symbol,
        df.index[-1],
        float(row["Open"]),
        float(row["High"]),
        float(row["Low"]),
        float(row["Close"]),
        float(row["Volume"]),
        float(row["MA50"]) if pd.notna(row["MA50"]) else float("nan"),
        float(row["MA200"]) if pd.notna(row["MA200"]) else float("nan"),
        float(row["ATR"]) if pd.notna(row["ATR"]) else float("nan"),
        float(row["RSI"]) if pd.notna(row["RSI"]) else float("nan"),
        float(row["ADX"]) if pd.notna(row["ADX"]) else float("nan"),
    )


def process_symbol(symbol: str) -> None:
    state = STATE[symbol]

    df = get_5m_bars(symbol)
    if df.empty:
        logger.warning("No data returned for %s", symbol)
        return

    if len(df) < CONFIG.min_bars_required:
        logger.info(
            "Skipping %s: insufficient bars (%d/%d)",
            symbol,
            len(df),
            CONFIG.min_bars_required,
        )
        return

    df = add_indicators(df)
    last_bar_time = df.index[-1]

    if state.last_bar_time == last_bar_time:
        return

    state.last_bar_time = last_bar_time

    logger.info("--------------------------------------------------")
    log_bar_snapshot(symbol, df)

    if state.position is None:
        logger.info("LOCAL | symbol=%s no local in-memory position", symbol)
    else:
        logger.info(
            "LOCAL | symbol=%s qty=%.6f entry=%.4f stop=%.4f t1=%.4f t2=%.4f partial_done=%s",
            symbol,
            state.position.qty,
            state.position.entry,
            state.position.stop,
            state.position.target_1,
            state.position.target_2,
            state.position.partial_done,
        )

    log_symbol_broker_status(symbol)

    if state.position is not None:
        maybe_exit(symbol, df)

    if state.position is None:
        maybe_enter(symbol, df)


# ============================================================
# MAIN LOOP
# ============================================================

def run() -> None:
    logger.info("Starting Alpaca crypto bot")
    logger.info("Paper mode: %s", CONFIG.paper)
    logger.info("Symbols: %s", ", ".join(CONFIG.symbols))
    logger.info("Sleep seconds: %s", CONFIG.sleep_seconds)
    logger.info("Lookback hours: %s", CONFIG.lookback_hours)

    log_account_snapshot()
    log_open_positions()
    sync_state_with_broker_positions()

    cycle = 0

    while True:
        try:
            cycle += 1
            if CONFIG.log_cycle_header:
                logger.info("========== CYCLE %d | UTC %s ==========", cycle, datetime.now(timezone.utc))

            sync_state_with_broker_positions()

            for symbol in CONFIG.symbols:
                process_symbol(symbol)

            time.sleep(CONFIG.sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break

        except Exception as exc:
            logger.exception("Unhandled error in main loop: %s", exc)
            time.sleep(10)


if __name__ == "__main__":
    run()
