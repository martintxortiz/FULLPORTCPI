import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# CAPITAL MANAGEMENT SETTINGS  —  edit these to match your account
# ══════════════════════════════════════════════════════════════════════════════
STARTING_CAPITAL = 100   # USD starting equity
LEVERAGE         = 10       # 1:10 leverage ceiling
MAX_RISK_PCT     = 0.01     # 1% of current equity risked per trade
LIQUIDATION_PCT  = 0.0      # halt trading if equity reaches 0


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(symbol, n_periods=52560, seed=42):
    """1 year of 5-min OHLC — 12 candles/hr x 24 hr x 365 days = 52,560 bars"""
    np.random.seed(seed + hash(symbol) % 1000)
    params = {
        "SOLANA": {"price": 180,  "volatility": 0.035, "trend": 0.00025},
        "AVAX":   {"price": 38,   "volatility": 0.038, "trend": 0.0003 },
        "NEAR":   {"price": 6.5,  "volatility": 0.040, "trend": 0.00028},
        "MATIC":  {"price": 0.95, "volatility": 0.036, "trend": 0.00022},
        "ALGO":   {"price": 0.35, "volatility": 0.034, "trend": 0.00020},
        "ATOM":   {"price": 11.5, "volatility": 0.037, "trend": 0.00024},
        "DOT":    {"price": 7.8,  "volatility": 0.036, "trend": 0.00026},
        "ADA":    {"price": 0.92, "volatility": 0.033, "trend": 0.00021},
        "LINK":   {"price": 22,   "volatility": 0.035, "trend": 0.00023},
        "UNI":    {"price": 13,   "volatility": 0.039, "trend": 0.00027},
        "APT":    {"price": 12,   "volatility": 0.042, "trend": 0.00031},
        "SUI":    {"price": 3.5,  "volatility": 0.041, "trend": 0.00029},
    }
    p = params[symbol]
    base_price, vol, trend = p["price"], p["volatility"], p["trend"]
    prices = [base_price]
    for _ in range(n_periods):
        change = trend * base_price + np.random.normal(0, vol * base_price)
        mr = -0.001 * (prices[-1] - base_price)
        prices.append(max(prices[-1] + change + mr, base_price * 0.1))
    data = []
    start_time = datetime.now() - timedelta(days=365)
    for i in range(n_periods):
        close = prices[i + 1]; open_p = prices[i]
        rs = abs(np.random.normal(0, vol * abs(close)))
        if close > open_p:
            high = close  + np.random.uniform(0, rs * 0.3)
            low  = open_p - np.random.uniform(0, rs * 0.3)
        else:
            high = open_p + np.random.uniform(0, rs * 0.3)
            low  = close  - np.random.uniform(0, rs * 0.3)
        high = max(high, open_p, close)
        low  = min(low,  open_p, close)
        data.append({"Open": open_p, "High": high, "Low": low, "Close": close})
    df = pd.DataFrame(data)
    df.index = pd.date_range(start=start_time, periods=n_periods, freq="5min")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS  —  ATR-14, RSI-14, ADX-14, MA50, MA200
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift(1)).abs()
    lc  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0)
    tr14     = tr.rolling(14).sum()
    plus_di  = 100 * plus_dm.rolling(14).sum()  / tr14
    minus_di = 100 * minus_dm.rolling(14).sum() / tr14
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE QUALITY  —  65% body threshold
# ══════════════════════════════════════════════════════════════════════════════
def is_strong_candle(row, min_body_pct=0.65):
    rng = row["High"] - row["Low"]
    if rng == 0:
        return False, None
    if abs(row["Close"] - row["Open"]) / rng >= min_body_pct:
        return True, ("bull" if row["Close"] > row["Open"] else "bear")
    return False, None


# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZER
# ══════════════════════════════════════════════════════════════════════════════
def size_position(equity, entry, stop):
    """
    Risk MAX_RISK_PCT of equity per trade.
    Notional is capped at equity x LEVERAGE (hard margin ceiling).
    Returns (notional_usd, actual_risk_usd, effective_leverage).
    """
    if equity <= LIQUIDATION_PCT:
        return 0.0, 0.0, 0.0
    stop_pct = abs(entry - stop) / entry
    if stop_pct == 0:
        return 0.0, 0.0, 0.0
    risk_usd     = equity * MAX_RISK_PCT
    notional     = risk_usd / stop_pct
    max_notional = equity * LEVERAGE
    notional     = min(notional, max_notional)
    actual_risk  = notional * stop_pct
    eff_leverage = notional / equity
    return notional, actual_risk, eff_leverage


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST  —  V2 with leverage and dollar P&L tracking
# ══════════════════════════════════════════════════════════════════════════════
def backtest(df, symbol, starting_equity):
    """
    Entry filters
    -------------
    1.  Both MA50 and price must confirm trend regime
    2.  ADX >= 20  —  trending markets only, no chop
    3.  RSI <= 65 for longs  /  RSI >= 35 for shorts  —  avoid exhausted entries
    4.  Real pullback: price touched MA50 band within last 10 bars
    5.  Extension cap: price within 4 ATR of MA50
    6.  ATR-based stop  —  1.5x ATR floored at candle extreme
    7.  Hard stop cap: skip if stop > 3% of price (liquidation guard)

    Exit rules
    ----------
    8.  Partial TP at +1.5R  ->  stop moved to breakeven (free trade)
    9.  Trailing stop at 1.5x ATR after breakeven  (ratchets with price)
    10. Full TP at +3.0R  ->  avg winning trade ~ 2.25R
    11. Time-stop at 96 bars (8 h) if price has not moved 0.3R

    Position sizing
    ---------------
    12. Risk = 1% of current equity per trade  (compounds with equity)
    13. Notional capped at 10x equity  (leverage ceiling)
    14. Margin call: trading halts if equity reaches 0
    """
    trades   = []
    position = None
    equity   = starting_equity
    LOOKBACK = 10

    for i in range(250, len(df)):
        row = df.iloc[i]
        if pd.isna(row["ATR"]) or pd.isna(row["RSI"]) or pd.isna(row["ADX"]):
            continue
        if equity <= LIQUIDATION_PCT:
            print(f"  MARGIN CALL on {symbol} at bar {i}  equity = ${equity:.2f}")
            break

        # ── MANAGE OPEN POSITION ─────────────────────────────────────────────
        if position is not None:
            e      = position["entry"]
            sl     = position["stop"]
            t1     = position["target1"]
            t2     = position["target2"]
            risk_p = position["risk_price"]
            d      = position["direction"]
            notl   = position["notional"]
            a_risk = position["actual_risk"]

            def _close(exit_px, R_raw, reason):
                nonlocal equity
                if d == "long":
                    pnl = notl * (exit_px - e) / e
                else:
                    pnl = notl * (e - exit_px) / e
                equity = max(equity + pnl, 0.0)
                return {
                    "symbol":        symbol,
                    "entry_time":    position["entry_time"],
                    "exit_time":     row.name,
                    "direction":     d,
                    "entry":         round(e, 6),
                    "stop":          round(sl, 6),
                    "target":        round(t2, 6),
                    "exit":          round(exit_px, 6),
                    "R":             round(R_raw, 4),
                    "pnl_usd":       round(pnl, 4),
                    "notional_usd":  round(notl, 2),
                    "risk_usd":      round(a_risk, 4),
                    "leverage_used": round(position["eff_leverage"], 2),
                    "equity_after":  round(equity, 4),
                    "exit_reason":   reason,
                }

            if d == "long":
                hit_sl = row["Low"]  <= sl
                hit_t1 = (not position["p1"]) and row["High"] >= t1
                hit_t2 = row["High"] >= t2
                stale  = (i - position["bar"]) > 96 and row["Close"] < e + 0.3 * risk_p

                if hit_sl:
                    trades.append(_close(sl, (sl - e) / risk_p, "stop"))
                    position = None; continue
                if hit_t2:
                    R = (1.5 + 3.0) / 2 if position["p1"] else 3.0
                    trades.append(_close(t2, R, "target"))
                    position = None; continue
                if hit_t1:
                    position["p1"]   = True
                    position["stop"] = e                    # move to breakeven
                if position["p1"]:
                    trail = row["Close"] - 1.5 * row["ATR"]
                    if trail > position["stop"]:
                        position["stop"] = trail            # ratchet up only
                if stale:
                    trades.append(_close(row["Close"], (row["Close"] - e) / risk_p, "timeout"))
                    position = None; continue

            else:  # short
                hit_sl = row["High"] >= sl
                hit_t1 = (not position["p1"]) and row["Low"] <= t1
                hit_t2 = row["Low"]  <= t2
                stale  = (i - position["bar"]) > 96 and row["Close"] > e - 0.3 * risk_p

                if hit_sl:
                    trades.append(_close(sl, (e - sl) / risk_p, "stop"))
                    position = None; continue
                if hit_t2:
                    R = (1.5 + 3.0) / 2 if position["p1"] else 3.0
                    trades.append(_close(t2, R, "target"))
                    position = None; continue
                if hit_t1:
                    position["p1"]   = True
                    position["stop"] = e                    # move to breakeven
                if position["p1"]:
                    trail = row["Close"] + 1.5 * row["ATR"]
                    if trail < position["stop"]:
                        position["stop"] = trail            # ratchet down only
                if stale:
                    trades.append(_close(row["Close"], (e - row["Close"]) / risk_p, "timeout"))
                    position = None; continue

        # ── ENTRY LOGIC ──────────────────────────────────────────────────────
        if position is None:
            if pd.isna(row["MA50"]) or pd.isna(row["MA200"]):
                continue

            bull = (row["Close"] > row["MA200"]) and (row["MA50"] > row["MA200"])
            bear = (row["Close"] < row["MA200"]) and (row["MA50"] < row["MA200"])

            if row["ADX"] < 20:
                continue

            strong, cdir = is_strong_candle(row)
            if not strong:
                continue

            atr = row["ATR"]
            win = df.iloc[i - LOOKBACK:i]

            # ── LONG ─────────────────────────────────────────────────────────
            if bull and cdir == "bull":
                if row["RSI"] > 65:
                    continue
                if not (win["Low"] <= win["MA50"] + atr).any():
                    continue
                if (row["Close"] - row["MA50"]) / atr > 4:
                    continue
                entry  = row["Close"]
                sl     = max(row["Low"] - 0.1 * atr, entry - 2.0 * atr)
                risk_p = entry - sl
                if risk_p <= 0 or risk_p / entry > 0.03:
                    continue
                notl, a_risk, eff_lev = size_position(equity, entry, sl)
                if notl == 0:
                    continue
                position = {
                    "direction":  "long",  "entry": entry, "stop": sl,
                    "risk_price": risk_p,  "target1": entry + 1.5 * risk_p,
                    "target2":    entry + 3.0 * risk_p,
                    "notional":   notl,    "actual_risk": a_risk,
                    "eff_leverage": eff_lev,
                    "entry_time": row.name, "bar": i, "p1": False,
                }

            # ── SHORT ────────────────────────────────────────────────────────
            elif bear and cdir == "bear":
                if row["RSI"] < 35:
                    continue
                if not (win["High"] >= win["MA50"] - atr).any():
                    continue
                if (row["MA50"] - row["Close"]) / atr > 4:
                    continue
                entry  = row["Close"]
                sl     = min(row["High"] + 0.1 * atr, entry + 2.0 * atr)
                risk_p = sl - entry
                if risk_p <= 0 or risk_p / entry > 0.03:
                    continue
                notl, a_risk, eff_lev = size_position(equity, entry, sl)
                if notl == 0:
                    continue
                position = {
                    "direction":  "short", "entry": entry, "stop": sl,
                    "risk_price": risk_p,  "target1": entry - 1.5 * risk_p,
                    "target2":    entry - 3.0 * risk_p,
                    "notional":   notl,    "actual_risk": a_risk,
                    "eff_leverage": eff_lev,
                    "entry_time": row.name, "bar": i, "p1": False,
                }

    return pd.DataFrame(trades), equity


# ══════════════════════════════════════════════════════════════════════════════
# METRICS  —  R-based + dollar-based + Calmar
# ══════════════════════════════════════════════════════════════════════════════
def calc_metrics(tdf, starting_equity, final_equity):
    if len(tdf) == 0:
        return {}
    wins       = (tdf["R"] > 0).sum()
    gp         = tdf.loc[tdf["R"] > 0, "R"].sum()
    gl         = tdf.loc[tdf["R"] < 0, "R"].abs().sum()
    cum_r      = tdf["R"].cumsum()
    max_dd_r   = (cum_r - cum_r.expanding().max()).min()
    cum_eq     = tdf["equity_after"]
    max_dd_usd = (cum_eq - cum_eq.expanding().max()).min()
    max_dd_pct = max_dd_usd / starting_equity * 100
    total_pnl  = final_equity - starting_equity
    roi_pct    = total_pnl   / starting_equity * 100
    calmar     = abs(total_pnl / max_dd_usd) if max_dd_usd != 0 else 0
    return dict(
        total_trades   = len(tdf),
        win_rate       = wins / len(tdf) * 100,
        avg_r          = tdf["R"].mean(),
        net_r          = tdf["R"].sum(),
        profit_factor  = gp / gl if gl > 0 else 0,
        max_dd_r       = max_dd_r,
        max_dd_usd     = max_dd_usd,
        max_dd_pct     = max_dd_pct,
        final_equity   = final_equity,
        total_pnl_usd  = total_pnl,
        roi_pct        = roi_pct,
        calmar         = calmar,
        avg_notional   = tdf["notional_usd"].mean(),
        avg_leverage   = tdf["leverage_used"].mean(),
        long_count     = (tdf["direction"] == "long").sum(),
        short_count    = (tdf["direction"] == "short").sum(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
assets = [
    "SOLANA", "AVAX", "NEAR", "MATIC", "ALGO", "ATOM",
    "DOT",    "ADA",  "LINK", "UNI",   "APT",  "SUI",
]

results    = {}
all_trades = []

print("=" * 80)
print("EXTENDED BACKTEST V2  --  1:10 LEVERAGE  |  1% MAX RISK PER TRADE")
print("=" * 80)
print(f"  Starting capital  : ${STARTING_CAPITAL:,.0f}")
print(f"  Leverage          : {LEVERAGE}x")
print(f"  Max risk / trade  : {MAX_RISK_PCT*100:.1f}% of current equity")
print(f"  Dataset           : 1 year of 5-min candles (52,560 bars / asset)")
print()
print("Processing...\n")

for symbol in assets:
    print(f"{symbol:.<20}", end=" ", flush=True)
    df = generate_synthetic_data(symbol)
    df = add_indicators(df)
    tdf, feq = backtest(df, symbol, STARTING_CAPITAL)
    m = calc_metrics(tdf, STARTING_CAPITAL, feq)
    results[symbol] = m
    if len(tdf) > 0:
        all_trades.append(tdf)
    print(f"{m['total_trades']:>5} trades | Net R: {m['net_r']:>8.2f} | "
          f"ROI: {m['roi_pct']:>7.2f}% | Final eq: ${m['final_equity']:>10,.2f}")

combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY  --  1 YEAR BACKTEST V2  (1:10 LEVERAGE)")
print("=" * 80 + "\n")

rows = []
for name, m in results.items():
    rows.append({
        "Asset":   name,
        "Trades":  m["total_trades"],
        "Win %":   f"{m['win_rate']:.1f}",
        "Avg R":   f"{m['avg_r']:.3f}",
        "Net R":   f"{m['net_r']:.1f}",
        "PF":      f"{m['profit_factor']:.2f}",
        "ROI %":   f"{m['roi_pct']:.2f}",
        "Final $": f"${m['final_equity']:,.0f}",
        "MaxDD %": f"{m['max_dd_pct']:.1f}%",
        "Calmar":  f"{m['calmar']:.2f}",
        "Avg Lev": f"{m['avg_leverage']:.1f}x",
    })
print(pd.DataFrame(rows).to_string(index=False))

total_trades = sum(m["total_trades"] for m in results.values())
print("\n" + "=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80)
print(f"  Total trades             : {total_trades:,}")
print(f"  Average win rate         : {np.mean([m['win_rate']      for m in results.values()]):.1f}%")
print(f"  Average profit factor    : {np.mean([m['profit_factor'] for m in results.values()]):.2f}")
print(f"  Average ROI (1 year)     : {np.mean([m['roi_pct']       for m in results.values()]):.2f}%")
print(f"  Average max drawdown     : {np.mean([m['max_dd_pct']    for m in results.values()]):.1f}% of starting capital")
print(f"  Average Calmar ratio     : {np.mean([m['calmar']        for m in results.values()]):.2f}")
print(f"  Avg trades per asset     : {total_trades / len(assets):.0f}")

print("\n" + "=" * 80)
print("TOP 5 PERFORMERS BY ROI")
print("=" * 80)
sorted_r = sorted(results.items(), key=lambda x: x[1]["roi_pct"], reverse=True)
for rank, (name, m) in enumerate(sorted_r[:5], 1):
    print(f"  {rank}. {name:<8} ROI: {m['roi_pct']:>8.2f}%  |  Final: ${m['final_equity']:>10,.2f}  |  "
          f"WR: {m['win_rate']:.1f}%  |  PF: {m['profit_factor']:.2f}  |  "
          f"MaxDD: {m['max_dd_pct']:.1f}%  |  Calmar: {m['calmar']:.2f}")

print("\n" + "=" * 80)
print("SAMPLE TRADE LOG  --  first 15 trades")
print("=" * 80 + "\n")
if len(combined_trades) > 0:
    cols = ["symbol","entry_time","direction","entry","stop","exit",
            "R","pnl_usd","notional_usd","risk_usd","leverage_used","equity_after","exit_reason"]
    print(combined_trades[cols].head(15).to_string(index=False))
    print(f"\n... showing 15 of {len(combined_trades):,} total trades")
    combined_trades.to_csv("crypto_v2_leveraged_trades.csv", index=False)
    print("Full trade log saved: crypto_v2_leveraged_trades.csv")

    print("\n" + "=" * 80)
    print("EXIT REASON BREAKDOWN")
    print("=" * 80)
    er = combined_trades.groupby("exit_reason").agg(
        count     = ("R",       "count"),
        win_rate  = ("R",       lambda x: f"{(x>0).mean()*100:.1f}%"),
        avg_r     = ("R",       "mean"),
        total_pnl = ("pnl_usd", "sum"),
    )
    print(er.to_string())

print("\n" + "=" * 80)
print("MONTHLY BREAKDOWN  --  BEST PERFORMER")
print("=" * 80)
best_asset  = sorted_r[0][0]
best_trades = [t for t in all_trades if t.iloc[0]["symbol"] == best_asset][0].copy()
best_trades["month"] = pd.to_datetime(best_trades["entry_time"]).dt.to_period("M")
monthly = best_trades.groupby("month").agg(
    Trades  = ("R",       "count"),
    Net_R   = ("R",       "sum"),
    Avg_R   = ("R",       "mean"),
    PnL_USD = ("pnl_usd", "sum"),
)
print(f"\n{best_asset} -- Monthly Performance:\n")
print(monthly.to_string())
print("\n" + "=" * 80)
