# alpaca_live_bot.py
# Simple live crypto bot for Alpaca
# Based on your backtest logic, simplified for real trading:
# - long only
# - one position per symbol
# - Python-managed stop/targets
# - paper trading by default

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# =========================
# CONFIG
# =========================
PAPER = True

# Requested: AVAX, ALGO, ETH, ATOM, NEAR, SOLANA
# Alpaca currently lists AVAX, ETH, SOL; enable those by default.
SYMBOLS = [
    "AVAX/USD",
    "ETH/USD",
    "SOL/USD",
    # "ALGO/USD",
    # "ATOM/USD",
    # "NEAR/USD",
]

RISK_PCT = 0.0025
MAX_CASH_PER_TRADE_PCT = 0.25
STRONG_BODY_PCT = 0.65
ADX_MIN = 20
RSI_LONG_MAX = 70
PULLBACK_LOOKBACK = 10
TIME_STOP_BARS = 96
MAX_STOP_PCT = 0.03
PARTIAL_R = 0.5
FINAL_R = 2.0
ATR_TRAIL_MULT = 1.5
ATR_STOP_CAP = 2.0
ATR_WICK_PAD = 0.1

SLEEP_SECONDS = 30
LOOKBACK_HOURS = 40

API_KEY = "PKNKZ2KHJG6J5GBSB3XQIOV372"
API_SECRET = "CzRam4m8KToFnFKKQeZ4pp886XU9CgAwH5mnNSwthgN6"

trade_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

STATE = {
    symbol: {
        "last_bar_time": None,
        "position": None,
    }
    for symbol in SYMBOLS
}


# =========================
# DATA
# =========================
def get_5m_bars(symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)

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
                    pass

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


# =========================
# INDICATORS
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm_raw = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm_raw, index=df.index)
    minus_dm = pd.Series(minus_dm_raw, index=df.index)

    atr_smma = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_smma
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_smma
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return df


# =========================
# STRATEGY
# =========================
def is_strong_bull_candle(row) -> bool:
    rng = row["High"] - row["Low"]
    if rng <= 0:
        return False
    body = abs(row["Close"] - row["Open"]) / rng
    return body >= STRONG_BODY_PCT and row["Close"] > row["Open"]


def calc_notional(entry: float, stop: float) -> float:
    account = trade_client.get_account()
    equity = float(account.equity)
    cash = float(account.cash)

    stop_pct = abs(entry - stop) / entry
    if stop_pct <= 0 or stop_pct > MAX_STOP_PCT:
        return 0.0

    risk_usd = equity * RISK_PCT
    by_risk = risk_usd / stop_pct
    by_cash = cash * MAX_CASH_PER_TRADE_PCT
    notional = min(by_risk, by_cash)

    return max(0.0, round(notional, 2))


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


def maybe_enter(symbol: str, df: pd.DataFrame):
    row = df.iloc[-1]

    needed = ["ATR", "RSI", "ADX", "MA50", "MA200"]
    if any(pd.isna(row[c]) for c in needed):
        return

    bull = (row["Close"] > row["MA200"]) and (row["MA50"] > row["MA200"])
    if not bull:
        return

    if row["ADX"] < ADX_MIN:
        return

    if row["RSI"] > RSI_LONG_MAX:
        return

    if not is_strong_bull_candle(row):
        return

    atr = row["ATR"]
    if atr <= 0:
        return

    win = df.iloc[-PULLBACK_LOOKBACK:]
    if not (win["Low"] <= win["MA50"] + atr).any():
        return

    if (row["Close"] - row["MA50"]) / atr > 4:
        return

    entry = float(row["Close"])
    stop = max(row["Low"] - ATR_WICK_PAD * atr, row["Close"] - ATR_STOP_CAP * atr)
    if stop >= entry:
        return

    notional = calc_notional(entry, stop)
    if notional < 10:
        return

    submit_buy(symbol, notional)

    qty = notional / entry
    risk_price = entry - stop

    STATE[symbol]["position"] = {
        "entry": entry,
        "stop": float(stop),
        "target_1": entry + PARTIAL_R * risk_price,
        "target_2": entry + FINAL_R * risk_price,
        "risk_price": risk_price,
        "qty": qty,
        "partial_done": False,
        "entry_bar_time": row.name,
    }

    print(f"{datetime.now()} BUY  {symbol}  entry={entry:.4f} stop={stop:.4f} notional=${notional:.2f}")


def maybe_exit(symbol: str, df: pd.DataFrame):
    pos = STATE[symbol]["position"]
    if pos is None:
        return

    row = df.iloc[-1]
    low = float(row["Low"])
    high = float(row["High"])
    close = float(row["Close"])

    entry = pos["entry"]
    stop = pos["stop"]
    t1 = pos["target_1"]
    t2 = pos["target_2"]
    qty = pos["qty"]
    risk_price = pos["risk_price"]

    # Stop first
    if low <= stop:
        submit_sell(symbol, qty)
        print(f"{datetime.now()} SELL {symbol}  reason=STOP qty={qty:.6f}")
        STATE[symbol]["position"] = None
        return

    # Final target
    if high >= t2:
        submit_sell(symbol, qty)
        print(f"{datetime.now()} SELL {symbol}  reason=TARGET2 qty={qty:.6f}")
        STATE[symbol]["position"] = None
        return

    # Partial target
    if not pos["partial_done"] and high >= t1:
        half = qty / 2
        submit_sell(symbol, half)
        pos["qty"] = qty - half
        pos["partial_done"] = True
        pos["stop"] = entry
        print(f"{datetime.now()} SELL {symbol}  reason=TARGET1 qty={half:.6f}")

    # Trail after partial
    if pos["partial_done"]:
        trail = close - ATR_TRAIL_MULT * float(row["ATR"])
        if trail > pos["stop"]:
            pos["stop"] = trail

    # Time stop
    bars_since_entry = len(df[df.index >= pos["entry_bar_time"]]) - 1
    if bars_since_entry > TIME_STOP_BARS and close < entry + 0.3 * risk_price:
        submit_sell(symbol, pos["qty"])
        print(f"{datetime.now()} SELL {symbol}  reason=TIMEOUT qty={pos['qty']:.6f}")
        STATE[symbol]["position"] = None


# =========================
# LOOP
# =========================
def run():
    print("Starting Alpaca crypto bot...")
    print("Paper mode:", PAPER)
    print("Symbols:", ", ".join(SYMBOLS))

    while True:
        try:
            for symbol in SYMBOLS:
                df = get_5m_bars(symbol)
                if df.empty or len(df) < 250:
                    continue

                df = add_indicators(df)
                last_bar_time = df.index[-1]

                if STATE[symbol]["last_bar_time"] == last_bar_time:
                    continue

                STATE[symbol]["last_bar_time"] = last_bar_time

                if STATE[symbol]["position"] is not None:
                    maybe_exit(symbol, df)

                if STATE[symbol]["position"] is None:
                    maybe_enter(symbol, df)

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print(f"{datetime.now()} ERROR: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run()
