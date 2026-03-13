import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN


@dataclass
class StrategyConfig:
    # Risk management
    risk_pct: float = 0.01           # 1% of equity risked per trade
    max_trade_pct_equity: float = 0.25  # max notional as % of equity (not cash)
    # Candle quality
    strong_body_pct: float = 0.3
    # Trend / momentum filters
    adx_min: float = 15.0
    rsi_long_max: float = 60.0
    # Pullback
    pullback_lookback: int = 10
    # Exit
    time_stop_bars: int = 96
    time_stop_min_r: float = 0.5     # exit if below 0.5R after timeout (was 0.3, too forgiving)
    # R-multiple targets
    partial_r: float = 1.0
    final_r: float = 3.0
    # ATR stop construction
    atr_stop_mult: float = 2.0       # default stop: entry - 2*ATR (raised from 1.5 cap)
    atr_wick_pad: float = 0.1        # extra buffer below candle low
    # ATR trailing (Chandelier-style)
    atr_trail_mult: float = 2.5      # raised from 1.5; applied to highest close since entry
    # Extension filter
    max_extension_atr: float = 3.0   # max distance from MA50 in ATR units (tightened from 4)
    # Guards
    max_stop_pct: float = 0.08       # raised from 0.03; let ATR determine the real stop
    min_notional_usd: float = 10.0
    min_bars_required: int = 250


@dataclass
class PositionState:
    entry: float
    stop: float
    target_1: float
    target_2: float
    risk_price: float                # entry - stop, positive scalar
    qty: float
    partial_done: bool
    entry_bar_time: pd.Timestamp
    highest_close_since_entry: float = 0.0   # NEW: Chandelier trailing basis


class Bot7Strategy:
    """Core logic for Bot 7 – corrected math, stop logic, and exit management."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    # ------------------------------------------------------------------
    # INDICATORS
    # ------------------------------------------------------------------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["MA50"]  = x["Close"].rolling(50).mean()
        x["MA200"] = x["Close"].rolling(200).mean()

        tr = pd.concat([
            x["High"] - x["Low"],
            (x["High"] - x["Close"].shift()).abs(),
            (x["Low"]  - x["Close"].shift()).abs(),
        ], axis=1).max(axis=1)

        x["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

        # RSI
        delta = x["Close"].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        x["RSI"] = 100 - (100 / (1 + gain / loss))

        # ADX  (Wilder SMMA via ewm alpha=1/14)
        up_move   =  x["High"].diff()
        down_move = -x["Low"].diff()

        plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr_smma  = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di   = 100 * pd.Series(plus_dm,  index=x.index).ewm(alpha=1/14, adjust=False).mean() / atr_smma
        minus_di  = 100 * pd.Series(minus_dm, index=x.index).ewm(alpha=1/14, adjust=False).mean() / atr_smma
        dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        x["ADX"]  = dx.ewm(alpha=1 / 14, adjust=False).mean()

        return x

    # ------------------------------------------------------------------
    # CANDLE QUALITY
    # ------------------------------------------------------------------
    def is_strong_bull_candle(self, row: pd.Series) -> bool:
        candle_range = row["High"] - row["Low"]
        if candle_range <= 0:
            return False
        body_ratio = abs(row["Close"] - row["Open"]) / candle_range
        return body_ratio >= self.config.strong_body_pct and row["Close"] > row["Open"]

    # ------------------------------------------------------------------
    # POSITION SIZING  (fixed-fractional, properly capped)
    # ------------------------------------------------------------------
    def calc_notional(self, entry: float, stop: float, equity: float, cash: float) -> float:
        """
        Fixed-fractional sizing:
            risk_usd  = equity * risk_pct
            notional  = risk_usd / stop_pct
        Caps:
            1. notional <= equity * max_trade_pct_equity   (portfolio heat cap)
            2. notional <= cash                            (liquidity guard)
        The old code capped against cash * 25% which could shrink valid sizes
        when cash was temporarily low. We now cap against equity first, then cash.
        """
        stop_pct = abs(entry - stop) / entry
        if stop_pct <= 0 or stop_pct > self.config.max_stop_pct:
            return 0.0

        risk_usd  = equity * self.config.risk_pct
        by_risk   = risk_usd / stop_pct                            # correct sizing
        equity_cap = equity * self.config.max_trade_pct_equity     # portfolio heat
        notional  = min(by_risk, equity_cap)                       # apply heat cap
        notional  = min(notional, cash)                            # enforce cash available
        return max(0.0, round(notional, 2))

    # ------------------------------------------------------------------
    # ENTRY DIAGNOSIS
    # ------------------------------------------------------------------
    def diagnose_entry(
        self, df: pd.DataFrame, equity: float, cash: float
    ) -> Tuple[bool, str, Optional[dict]]:

        if len(df) < self.config.min_bars_required:
            return False, "Not enough bars", None

        row    = df.iloc[-1]
        needed = ["ATR", "RSI", "ADX", "MA50", "MA200"]
        if any(pd.isna(row[c]) for c in needed):
            return False, "Indicators not ready", None

        # --- Trend filter ---
        if not (row["Close"] > row["MA200"] and row["MA50"] > row["MA200"]):
            return False, "Trend filter failed", None

        # --- Momentum / strength ---
        if row["ADX"] < self.config.adx_min:
            return False, f"ADX too low ({row['ADX']:.2f})", None

        if row["RSI"] > self.config.rsi_long_max:
            return False, f"RSI too high ({row['RSI']:.2f})", None

        if not self.is_strong_bull_candle(row):
            return False, "Not strong bullish candle", None

        atr = float(row["ATR"])
        if atr <= 0:
            return False, "ATR invalid", None

        # --- Pullback near MA50 (FIXED: price must actually touch ± 1 ATR band) ---
        # Old code: win["Low"] <= win["MA50"] + atr  (always true in uptrend – bug)
        # New code: Low must be within [MA50 - ATR, MA50 + ATR] in recent window
        win         = df.iloc[-self.config.pullback_lookback:]
        lower_band  = win["MA50"] - atr
        upper_band  = win["MA50"] + atr
        touched_band = ((win["Low"] >= lower_band) & (win["Low"] <= upper_band)).any()
        if not touched_band:
            return False, "No recent pullback to MA50 ± 1 ATR", None

        # --- Extension cap (tightened from 4 to max_extension_atr) ---
        extension_atr = (row["Close"] - row["MA50"]) / atr
        if extension_atr > self.config.max_extension_atr:
            return False, f"Too extended from MA50 ({extension_atr:.2f} ATR)", None

        # --- Stop placement (FIXED: use min, not max) ---
        # We want the LOWER of:
        #   a) just below the candle low (structural stop)
        #   b) entry - atr_stop_mult * ATR (volatility stop)
        # Taking max() as before would push stop UP (closer to entry),
        # shrinking risk_price and inflating position size dangerously.
        entry         = float(row["Close"])
        stop_structural = float(row["Low"]) - self.config.atr_wick_pad * atr
        stop_atr        = entry - self.config.atr_stop_mult * atr
        stop            = min(stop_structural, stop_atr)   # more conservative (lower)

        if stop >= entry:
            return False, "Computed stop is not below entry", None

        # --- Sizing ---
        notional = self.calc_notional(entry, stop, equity, cash)
        if notional < self.config.min_notional_usd:
            return False, f"Notional too small (${notional:.2f})", None

        # --- R-multiple targets (FIXED: now computed here and passed out) ---
        risk_price = entry - stop               # positive
        target_1   = entry + self.config.partial_r * risk_price   # +1R
        target_2   = entry + self.config.final_r  * risk_price    # +3R

        payload = {
            "entry":      entry,
            "stop":       stop,
            "target_1":   target_1,
            "target_2":   target_2,
            "notional":   notional,
            "risk_price": risk_price,
        }
        return True, "Entry conditions passed", payload

    # ------------------------------------------------------------------
    # EXIT MANAGEMENT
    # ------------------------------------------------------------------
    def check_exit(
        self, df: pd.DataFrame, pos: PositionState
    ) -> List[Tuple[str, float, str]]:
        """
        Returns list of actions: [("SELL_PCT", fraction_of_position, reason)]
        Updates pos in-place for trail stop and partial flag.
        """
        row  = df.iloc[-1]
        low  = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])
        atr   = float(row["ATR"])

        actions: List[Tuple[str, float, str]] = []

        # 1. HARD STOP
        if low <= pos.stop:
            actions.append(("SELL_PCT", 1.0, "STOP"))
            return actions

        # 2. FINAL TARGET (full exit at +3R)
        if high >= pos.target_2:
            actions.append(("SELL_PCT", 1.0, "TARGET2"))
            return actions

        # 3. PARTIAL TARGET (50% exit at +1R, move stop to breakeven)
        if not pos.partial_done and high >= pos.target_1:
            actions.append(("SELL_PCT", 0.5, "TARGET1"))
            pos.partial_done = True
            pos.stop = pos.entry                             # breakeven stop
            pos.highest_close_since_entry = close            # reset Chandelier basis

        # 4. TIME STOP (FIXED: threshold raised from 0.3R to time_stop_min_r = 0.5R)
        bars_since_entry = len(df[df.index >= pos.entry_bar_time]) - 1
        if (bars_since_entry > self.config.time_stop_bars
                and close < pos.entry + self.config.time_stop_min_r * pos.risk_price):
            actions.append(("SELL_PCT", 1.0, "TIMEOUT"))
            return actions

        # 5. CHANDELIER TRAILING STOP (FIXED: track highest close, not current close)
        # Old code trailed from current close, which drops the stop on pullback candles.
        # Chandelier anchors to the highest close seen since entry (or since partial).
        if pos.partial_done and atr > 0:
            pos.highest_close_since_entry = max(
                pos.highest_close_since_entry, close
            )
            new_trail = pos.highest_close_since_entry - self.config.atr_trail_mult * atr
            if new_trail > pos.stop:
                pos.stop = new_trail

        return actions
