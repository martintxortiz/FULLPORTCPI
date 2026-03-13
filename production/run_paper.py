import os
import time
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
from decimal import Decimal, ROUND_DOWN

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from strategy import StrategyConfig, Bot7Strategy, PositionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("alpaca_bot")

class LiveTrader:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.trade = TradingClient(api_key, api_secret, paper=paper)
        self.data = CryptoHistoricalDataClient(api_key, api_secret)
        self.config = StrategyConfig()
        self.bot = Bot7Strategy(self.config)
        self.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        self.state = {s: {"last_bar": None, "pos": None} for s in self.symbols}
        self.rules_cache = {}

    def get_symbol_rules(self, symbol: str):
        if symbol in self.rules_cache: return self.rules_cache[symbol]
        try:
            asset = self.trade.get_asset(symbol)
            min_qty = float(getattr(asset, "min_order_size", 0.0))
            step = float(getattr(asset, "min_trade_increment", 0.0))
            if min_qty > 0 and step > 0:
                self.rules_cache[symbol] = (min_qty, step)
                return self.rules_cache[symbol]
        except Exception:
            pass
        fallback = {"BTC/USD": (0.0001, 0.0001), "ETH/USD": (0.001, 0.001), "SOL/USD": (0.01, 0.01), "AVAX/USD": (0.01, 0.01)}
        self.rules_cache[symbol] = fallback.get(symbol, (0.000001, 0.000001))
        return self.rules_cache[symbol]

    def normalize_symbol(self, s: str): return s.replace("/", "").replace("-", "").upper()

    def get_broker_positions(self):
        out = {}
        try:
            for p in self.trade.get_all_positions(): out[self.normalize_symbol(getattr(p, "symbol", ""))] = p
        except Exception as e: log.warning("Could not fetch broker positions: %s", e)
        return out

    def sync_state(self):
        bps = self.get_broker_positions()
        for sym in self.symbols:
            ns = self.normalize_symbol(sym)
            local = self.state[sym]["pos"]
            brkr = bps.get(ns)
            if local and not brkr:
                self.state[sym]["pos"] = None
            elif local and brkr:
                local.qty = float(getattr(brkr, "qty", local.qty))
                avg = float(getattr(brkr, "avg_entry_price", local.entry))
                if avg > 0: local.entry = avg

    def get_bars(self, symbol: str):
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=40)
        req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start, end=end)
        df = self.data.get_crypto_bars(req).df
        if df.empty: return df
        if isinstance(df.index, pd.MultiIndex):
            if "symbol" in df.index.names: df = df.xs(symbol, level="symbol")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
        return df[~df.index.duplicated(keep="last")]

    def execute_buy(self, symbol, notional):
        return self.trade.submit_order(MarketOrderRequest(symbol=symbol, notional=notional, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))

    def execute_sell(self, symbol, qty):
        step = self.get_symbol_rules(symbol)[1]
        v = Decimal(str(qty))
        s = Decimal(str(step))
        q = float((v / s).quantize(Decimal("1"), rounding=ROUND_DOWN) * s)
        if q <= 0: return None
        return self.trade.submit_order(MarketOrderRequest(symbol=symbol, qty=q, side=OrderSide.SELL, time_in_force=TimeInForce.GTC))

    def wait_fill(self, oid, sym):
        dl = time.time() + 15
        while time.time() < dl:
            o = self.trade.get_order_by_id(oid)
            if str(getattr(o, "status", "")).split(".")[-1].lower() in {"filled", "partially_filled", "canceled", "expired", "rejected"}: return o
            time.sleep(1)
        try: self.trade.cancel_order_by_id(oid)
        except Exception: pass
        return self.trade.get_order_by_id(oid)

    def process_symbol(self, sym):
        df = self.get_bars(sym)
        if df.empty or len(df) < self.config.min_bars_required: return
        df = self.bot.add_indicators(df)
        cb = df.index[-1]
        if self.state[sym]["last_bar"] == cb: return
        self.state[sym]["last_bar"] = cb

        self.sync_state()
        pos = self.state[sym]["pos"]

        if pos:
            actions = self.bot.check_exit(df, pos)
            for act, pct, rsn in actions:
                if act == "SELL_PCT":
                    qty = pos.qty * pct
                    o = self.execute_sell(sym, qty)
                    if o:
                        self.wait_fill(str(o.id), sym)
                        log.info(f"SOLD {qty} of {sym} reason: {rsn}")
            self.sync_state()

        if not self.state[sym]["pos"] and not self.get_broker_positions().get(self.normalize_symbol(sym)):
            acc = self.trade.get_account()
            eq, ca = float(acc.equity), float(acc.cash)
            ok, rsn, pay = self.bot.diagnose_entry(df, eq, ca)
            if ok:
                log.info(f"BUY SIGNAL {sym} notional={pay['notional']}")
                o = self.execute_buy(sym, pay["notional"])
                self.wait_fill(str(o.id), sym)
                self.sync_state()
                bp = self.get_broker_positions().get(self.normalize_symbol(sym))
                if bp:
                    qty, ent = float(getattr(bp, "qty", 0)), float(getattr(bp, "avg_entry_price", 0))
                    if qty > 0 and ent > 0:
                        rp = ent - pay["stop"]
                        self.state[sym]["pos"] = PositionState(
                            entry=ent, stop=pay["stop"],
                            target_1=ent + self.config.partial_r * rp,
                            target_2=ent + self.config.final_r * rp,
                            risk_price=rp, qty=qty, partial_done=False, entry_bar_time=df.index[-1]
                        )
                        log.info(f"ENTERED {sym} @ {ent} stop={pay['stop']}")

    def run(self):
        log.info("Starting production paper trader...")
        while True:
            try:
                for sym in self.symbols: self.process_symbol(sym)
                time.sleep(30)
            except Exception as e:
                log.exception(f"Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    LiveTrader("PKNKZ2KHJG6J5GBSB3XQIOV372", "CzRam4m8KToFnFKKQeZ4pp886XU9CgAwH5mnNSwthgN6").run()
