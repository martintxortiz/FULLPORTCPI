"""
v5.py  (fixed)
==============
Fixes applied vs original v5:
  1. Trading fees     : FEE_RATE = 0.0004 (0.04% taker) deducted at every fill
  2. Partial exit     : 50% genuinely closed and PnL booked at T1 with its own
                        trade record; remaining 50% managed to T2/stop/timeout
  3. Indicators       : RSI, ATR, ADX all use Wilder's SMMA (EWM alpha=1/14)
                        instead of simple rolling mean
  4. ADX DM bug fixed : +DM and -DM computed simultaneously with np.where to
                        avoid sequencing artefact when up_move == down_move
  5. Entry price      : filled at NEXT bar's Open (pending-order approach)
                        so the signal candle's close is never traded on itself
"""

import os
import glob
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STARTING_CAPITAL  = 10_000
LEVERAGE          = 10
MAX_RISK_PCT      = 0.0025    # 0.25 % of current equity risked per trade
MAX_DRAWDOWN_PCT  = 0.10      # 10 % peak-to-trough → hard FAIL
STRONG_BODY_PCT   = 0.65
ADX_MIN           = 20
RSI_LONG_MAX      = 65
RSI_SHORT_MIN     = 35
PULLBACK_LOOKBACK = 10
TIME_STOP_BARS    = 96        # 8 h on 5-min bars
MAX_STOP_PCT      = 0.03      # skip if stop distance > 3 % of entry
PARTIAL_R         = 1.5
FINAL_R           = 3.0
ATR_TRAIL_MULT    = 1.5
ATR_STOP_CAP      = 2.0
ATR_WICK_PAD      = 0.1
FEE_RATE          = 0.0004   # FIX 1 — 0.04 % taker per side (Binance)

ASSET_MAP: dict[str, str] = {
    'SOLUSDT':   'SOLANA',
    'AVAXUSDT':  'AVAX',
    'NEARUSDT':  'NEAR',
    'MATICUSDT': 'MATIC',
    'ALGOUSDT':  'ALGO',
    'ATOMUSDT':  'ATOM',
    'DOTUSDT':   'DOT',
    'ADAUSDT':   'ADA',
    'LINKUSDT':  'LINK',
    'UNIUSDT':   'UNI',
    'APTUSDT':   'APT',
    'SUIUSDT':   'SUI',
}

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.dirname(__file__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(binance_symbol: str) -> pd.DataFrame:
    """Load a pre-downloaded CSV.gz file for a given Binance symbol."""
    pattern = os.path.join(DATA_DIR, f'{binance_symbol}_5m.csv.gz')
    matches = glob.glob(pattern)
    if not matches:
        return pd.DataFrame()
    df = pd.read_csv(matches[0], index_col='datetime', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df


def discover_assets() -> list[tuple[str, str]]:
    """Return list of (binance_symbol, friendly_name) with existing data files."""
    available = []
    for bsym, friendly in ASSET_MAP.items():
        pattern = os.path.join(DATA_DIR, f'{bsym}_5m.csv.gz')
        if glob.glob(pattern):
            available.append((bsym, friendly))
        else:
            print(f'  [SKIP] No data file for {bsym} — run fetch_data.py first.')
    return available


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS  — FIX 3 & 4: all use Wilder's SMMA via EWM(alpha=1/14)
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # ── ATR (Wilder's SMMA) ──────────────────────────────────────────────────
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()   # FIX 3

    # ── RSI (Wilder's SMMA) ──────────────────────────────────────────────────
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()   # FIX 3
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── ADX (Wilder's SMMA + simultaneous DM) ────────────────────────────────
    up_move   = df['High'].diff()
    down_move = -df['Low'].diff()
    # FIX 4 — compute both sides at once with np.where (no sequencing bug)
    plus_dm_raw  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm_raw,  index=df.index)
    minus_dm = pd.Series(minus_dm_raw, index=df.index)

    atr_smma = tr.ewm(alpha=1/14, adjust=False).mean()   # FIX 3
    plus_di  = 100 * plus_dm.ewm(alpha=1/14,  adjust=False).mean() / atr_smma
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smma
    dx       = 100 * (plus_di - minus_di).abs() /                      (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()   # FIX 3

    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_strong_candle(row, min_body_pct: float = STRONG_BODY_PCT):
    rng = row['High'] - row['Low']
    if rng == 0:
        return False, None
    body = abs(row['Close'] - row['Open']) / rng
    if body >= min_body_pct:
        return True, ('bull' if row['Close'] > row['Open'] else 'bear')
    return False, None


def size_position(equity: float, entry: float, stop: float):
    stop_pct = abs(entry - stop) / entry
    if equity <= 0 or stop_pct <= 0:
        return 0.0, 0.0, 0.0, 0.0
    risk_usd    = equity * MAX_RISK_PCT
    notional    = min(risk_usd / stop_pct, equity * LEVERAGE)
    actual_risk = notional * stop_pct
    lev_used    = notional / equity
    return notional, actual_risk, lev_used, stop_pct


def mark_to_market(closed_equity: float, position: dict, price: float) -> float:
    if position is None:
        return closed_equity
    if position['direction'] == 'long':
        pnl = position['notional_usd'] * (price - position['entry']) / position['entry']
    else:
        pnl = position['notional_usd'] * (position['entry'] - price) / position['entry']
    return closed_equity + pnl


def build_trade(symbol, pos, exit_time, exit_price, r, pnl, equity_after,
                reason, notional_override=None) -> dict:
    notl = notional_override if notional_override is not None else pos['notional_usd']
    return {
        'symbol':        symbol,
        'entry_time':    pos['entry_time'],
        'exit_time':     exit_time,
        'direction':     pos['direction'],
        'entry':         round(pos['entry'],        6),
        'stop':          round(pos['stop'],         6),
        'target_1':      round(pos['target_1'],     6),
        'target_2':      round(pos['target_2'],     6),
        'exit':          round(exit_price,            6),
        'R':             round(r,                     4),
        'pnl_usd':       round(pnl,                  4),
        'notional_usd':  round(notl,                 2),
        'risk_usd':      round(pos['risk_usd'],     4),
        'stop_pct':      round(pos['stop_pct'] * 100, 4),
        'leverage_used': round(pos['leverage_used'],4),
        'equity_after':  round(equity_after,          4),
        'exit_reason':   reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def backtest_asset(df: pd.DataFrame, symbol: str,
                   starting_capital: float = STARTING_CAPITAL):
    trades       = []
    equity_marks = []
    position     = None
    pending      = None   # FIX 5 — pending entry fills at next bar's Open

    closed_equity  = starting_capital
    peak_equity    = starting_capital
    worst_drawdown = 0.0
    failed         = False
    fail_reason    = None
    fail_time      = None
    halt_equity    = None

    def update_dd(ts, eq_val, mark_type):
        nonlocal peak_equity, worst_drawdown, failed, fail_reason, fail_time, halt_equity
        peak_equity = max(peak_equity, eq_val)
        dd = (eq_val / peak_equity - 1.0) if peak_equity > 0 else -1.0
        worst_drawdown = min(worst_drawdown, dd)
        equity_marks.append({
            'symbol':       symbol,
            'time':         ts,
            'equity':       round(eq_val,       4),
            'peak_equity':  round(peak_equity,  4),
            'drawdown_pct': round(dd * 100,     4),
            'mark_type':    mark_type,
        })
        if not failed and dd <= -MAX_DRAWDOWN_PCT:
            failed      = True
            fail_reason = 'max_drawdown_breach'
            fail_time   = ts
            halt_equity = eq_val

    update_dd(df.index[0], closed_equity, 'start')

    for i in range(250, len(df)):
        row = df.iloc[i]

        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or            pd.isna(row['ADX']) or pd.isna(row['MA50']) or pd.isna(row['MA200']):
            continue

        # ── FIX 5: Fill pending entry at this bar's Open ────────────────────
        if pending is not None and position is None:
            entry    = row['Open']
            raw_stop = pending['raw_stop']
            direction = pending['direction']
            rp       = abs(entry - raw_stop)
            stop_pct = rp / entry if entry > 0 else 1.0

            if rp > 0 and stop_pct <= MAX_STOP_PCT:
                notl, ar, el, sp = size_position(closed_equity, entry, raw_stop)
                if notl > 0:
                    entry_fee      = notl * FEE_RATE   # FIX 1 — entry fee
                    closed_equity -= entry_fee
                    position = {
                        'direction':       direction,
                        'entry':           entry,
                        'stop':            raw_stop,
                        'risk_price':      rp,
                        'target_1':        entry + PARTIAL_R * rp if direction == 'long'
                                            else entry - PARTIAL_R * rp,
                        'target_2':        entry + FINAL_R   * rp if direction == 'long'
                                            else entry - FINAL_R   * rp,
                        'notional_usd':    notl,
                        'risk_usd':        ar,
                        'leverage_used':   el,
                        'stop_pct':        sp,
                        'entry_time':      row.name,
                        'entry_bar':       i,
                        'partial_done':    False,
                    }
            pending = None

        # ── Mark-to-market + drawdown check ─────────────────────────────────
        if position is not None:
            mtm = mark_to_market(closed_equity, position, row['Close'])
            update_dd(row.name, mtm, 'mtm')

            if failed:   # force-close at current bar close
                e, rp_pos = position['entry'], position['risk_price']
                notl      = position['notional_usd']
                if position['direction'] == 'long':
                    pnl = notl * (row['Close'] - e) / e
                    r   = (row['Close'] - e) / rp_pos
                else:
                    pnl = notl * (e - row['Close']) / e
                    r   = (e - row['Close']) / rp_pos
                pnl          -= notl * FEE_RATE   # FIX 1 — exit fee
                closed_equity = max(closed_equity + pnl, 0.0)
                trades.append(build_trade(symbol, position, row.name,
                                          row['Close'], r, pnl, closed_equity, 'dd_halt'))
                update_dd(row.name, closed_equity, 'closed_dd_halt')
                position = None
                break

        # ── Manage open position ─────────────────────────────────────────────
        if position is not None:
            e    = position['entry']
            s    = position['stop']
            t1   = position['target_1']
            t2   = position['target_2']
            rp   = position['risk_price']
            notl = position['notional_usd']
            d    = position['direction']
            xt   = None

            if d == 'long':
                if row['Low'] <= s:
                    exit_fee = notl * FEE_RATE   # FIX 1
                    pnl      = notl * (s - e) / e - exit_fee
                    r        = (s - e) / rp
                    xt       = build_trade(symbol, position, row.name, s, r, pnl, 0.0, 'stop')

                elif row['High'] >= t2:
                    exit_fee = notl * FEE_RATE   # FIX 1
                    pnl      = notl * (t2 - e) / e - exit_fee
                    r        = (t2 - e) / rp
                    xt       = build_trade(symbol, position, row.name, t2, r, pnl, 0.0, 'target')

                else:
                    # ── FIX 2: Genuine partial exit at T1 ─────────────────
                    if not position['partial_done'] and row['High'] >= t1:
                        half_notl    = notl / 2
                        partial_fee  = half_notl * FEE_RATE   # FIX 1
                        partial_pnl  = half_notl * (t1 - e) / e - partial_fee
                        closed_equity = max(closed_equity + partial_pnl, 0.0)
                        partial_r    = (t1 - e) / rp
                        pt = build_trade(symbol, position, row.name, t1,
                                         partial_r, partial_pnl, closed_equity,
                                         'partial_target', notional_override=half_notl)
                        trades.append(pt)
                        update_dd(row.name, closed_equity, 'partial_closed')
                        position['partial_done'] = True
                        position['notional_usd'] = half_notl   # FIX 2 — halve remaining
                        position['stop']         = e           # move to break-even
                        notl = half_notl

                    if position['partial_done']:
                        trail = row['Close'] - ATR_TRAIL_MULT * row['ATR']
                        if trail > position['stop']:
                            position['stop'] = trail

                    if (i - position['entry_bar']) > TIME_STOP_BARS and                             row['Close'] < e + 0.3 * rp:
                        exit_fee = notl * FEE_RATE   # FIX 1
                        pnl      = notl * (row['Close'] - e) / e - exit_fee
                        r        = (row['Close'] - e) / rp
                        xt       = build_trade(symbol, position, row.name,
                                               row['Close'], r, pnl, 0.0, 'timeout')

            else:  # short
                if row['High'] >= s:
                    exit_fee = notl * FEE_RATE   # FIX 1
                    pnl      = notl * (e - s) / e - exit_fee
                    r        = (e - s) / rp
                    xt       = build_trade(symbol, position, row.name, s, r, pnl, 0.0, 'stop')

                elif row['Low'] <= t2:
                    exit_fee = notl * FEE_RATE   # FIX 1
                    pnl      = notl * (e - t2) / e - exit_fee
                    r        = (e - t2) / rp
                    xt       = build_trade(symbol, position, row.name, t2, r, pnl, 0.0, 'target')

                else:
                    # ── FIX 2: Genuine partial exit at T1 ─────────────────
                    if not position['partial_done'] and row['Low'] <= t1:
                        half_notl    = notl / 2
                        partial_fee  = half_notl * FEE_RATE   # FIX 1
                        partial_pnl  = half_notl * (e - t1) / e - partial_fee
                        closed_equity = max(closed_equity + partial_pnl, 0.0)
                        partial_r    = (e - t1) / rp
                        pt = build_trade(symbol, position, row.name, t1,
                                         partial_r, partial_pnl, closed_equity,
                                         'partial_target', notional_override=half_notl)
                        trades.append(pt)
                        update_dd(row.name, closed_equity, 'partial_closed')
                        position['partial_done'] = True
                        position['notional_usd'] = half_notl   # FIX 2 — halve remaining
                        position['stop']         = e           # move to break-even
                        notl = half_notl

                    if position['partial_done']:
                        trail = row['Close'] + ATR_TRAIL_MULT * row['ATR']
                        if trail < position['stop']:
                            position['stop'] = trail

                    if (i - position['entry_bar']) > TIME_STOP_BARS and                             row['Close'] > e - 0.3 * rp:
                        exit_fee = notl * FEE_RATE   # FIX 1
                        pnl      = notl * (e - row['Close']) / e - exit_fee
                        r        = (e - row['Close']) / rp
                        xt       = build_trade(symbol, position, row.name,
                                               row['Close'], r, pnl, 0.0, 'timeout')

            if xt is not None:
                closed_equity = max(closed_equity + xt['pnl_usd'], 0.0)
                xt['equity_after'] = round(closed_equity, 4)
                trades.append(xt)
                update_dd(row.name, closed_equity, 'closed_trade')
                position = None
                if failed:
                    break

        if failed:
            break

        # ── Entry signal → queue pending order (filled next bar) ─────────────
        if position is None and pending is None:
            bull = (row['Close'] > row['MA200']) and (row['MA50'] > row['MA200'])
            bear = (row['Close'] < row['MA200']) and (row['MA50'] < row['MA200'])

            if row['ADX'] < ADX_MIN:
                continue
            strong, cdir = is_strong_candle(row)
            if not strong:
                continue

            atr = row['ATR']
            win = df.iloc[i - PULLBACK_LOOKBACK:i]

            if bull and cdir == 'bull':
                if row['RSI'] > RSI_LONG_MAX:
                    continue
                if not (win['Low'] <= win['MA50'] + atr).any():
                    continue
                if (row['Close'] - row['MA50']) / atr > 4:
                    continue
                raw_stop = max(row['Low'] - ATR_WICK_PAD * atr,
                               row['Close'] - ATR_STOP_CAP * atr)
                pending = {'direction': 'long', 'raw_stop': raw_stop}

            elif bear and cdir == 'bear':
                if row['RSI'] < RSI_SHORT_MIN:
                    continue
                if not (win['High'] >= win['MA50'] - atr).any():
                    continue
                if (row['MA50'] - row['Close']) / atr > 4:
                    continue
                raw_stop = min(row['High'] + ATR_WICK_PAD * atr,
                               row['Close'] + ATR_STOP_CAP * atr)
                pending = {'direction': 'short', 'raw_stop': raw_stop}

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_marks)
    run_info  = {
        'symbol':           symbol,
        'final_equity':     closed_equity,
        'peak_equity':      peak_equity,
        'max_drawdown_pct': worst_drawdown * 100,
        'failed':           failed,
        'pass_status':      'FAILED' if failed else 'PASSED',
        'fail_reason':      fail_reason,
        'fail_time':        str(fail_time) if fail_time else None,
        'halt_equity':      halt_equity,
        'total_trades':     len(trades_df),
    }
    return trades_df, equity_df, run_info


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def calc_metrics(trades_df: pd.DataFrame, run_info: dict,
                 starting_capital: float = STARTING_CAPITAL) -> dict:
    roi  = (run_info['final_equity'] / starting_capital - 1) * 100
    mdd  = abs(run_info['max_drawdown_pct'])
    base = {
        **run_info,
        'win_rate': 0.0, 'avg_r': 0.0, 'net_r': 0.0,
        'profit_factor': 0.0, 'roi_pct': roi, 'calmar': 0.0,
        'avg_notional_usd': 0.0, 'avg_leverage_used': 0.0,
        'long_count': 0, 'short_count': 0,
    }
    if len(trades_df) == 0:
        return base

    wins = (trades_df['R'] > 0).sum()
    gp   = trades_df.loc[trades_df['R'] > 0, 'R'].sum()
    gl   = trades_df.loc[trades_df['R'] < 0, 'R'].abs().sum()

    return {
        **run_info,
        'win_rate':          wins / len(trades_df) * 100,
        'avg_r':             trades_df['R'].mean(),
        'net_r':             trades_df['R'].sum(),
        'profit_factor':     gp / gl if gl > 0 else 0.0,
        'roi_pct':           roi,
        'calmar':            roi / mdd if mdd > 0 else 0.0,
        'avg_notional_usd':  trades_df['notional_usd'].mean(),
        'avg_leverage_used': trades_df['leverage_used'].mean(),
        'long_count':        (trades_df['direction'] == 'long').sum(),
        'short_count':       (trades_df['direction'] == 'short').sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    results    = []
    all_trades = []
    all_equity = []

    print('=' * 90)
    print('PROFESSIONAL BACKTEST  v5 (fixed)  —  Binance local data  —  5-min OHLCV')
    print('=' * 90)
    print(f'Starting capital   : ${STARTING_CAPITAL:,.0f}')
    print(f'Leverage cap       : {LEVERAGE}x')
    print(f'Risk per trade     : {MAX_RISK_PCT * 100:.2f}% of current equity')
    print(f'Max drawdown limit : {MAX_DRAWDOWN_PCT * 100:.0f}%  →  hard FAIL')
    print(f'Fee rate           : {FEE_RATE * 100:.2f}% taker per side  (round-trip {FEE_RATE*2*100:.2f}%)')
    print(f'Data directory     : {os.path.abspath(DATA_DIR)}')
    print()

    assets = discover_assets()
    if not assets:
        print('\nERROR: No data files found.')
        print(f'Run  python fetch_data.py  from {os.path.dirname(DATA_DIR)} first.\n')
        return

    print(f'Found {len(assets)} asset(s) with local data.\n')
    print('Processing...\n')

    for bsym, friendly in assets:
        df_raw = load_data(bsym)
        if df_raw.empty:
            print(f'{friendly:.<20} No data — skipping.')
            continue

        df = add_indicators(df_raw)
        n_bars     = len(df)
        date_range = (df.index.min().strftime('%Y-%m-%d'),
                      df.index.max().strftime('%Y-%m-%d'))

        print(f'{friendly:.<20} {n_bars:>8,} bars  ({date_range[0]} → {date_range[1]})',
              end='  ', flush=True)

        trades_df, equity_df, run_info = backtest_asset(df, friendly)
        metrics = calc_metrics(trades_df, run_info)
        results.append(metrics)

        if len(trades_df) > 0:
            all_trades.append(trades_df)
        if len(equity_df) > 0:
            all_equity.append(equity_df)

        print(f"{metrics['pass_status']:<6} | Trades: {metrics['total_trades']:>4} | "
              f"ROI: {metrics['roi_pct']:>8.2f}% | "
              f"MaxDD: {metrics['max_drawdown_pct']:>7.2f}% | "
              f"Final: ${metrics['final_equity']:>10,.2f}")

    if not results:
        print('No results to display.')
        return

    summary_df  = pd.DataFrame(results).sort_values(
        ['pass_status', 'roi_pct'], ascending=[True, False])
    trades_all  = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_all  = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()

    summary_path = os.path.join(OUTPUT_DIR, 'v5_summary.csv')
    trades_path  = os.path.join(OUTPUT_DIR, 'v5_trades.csv')
    equity_path  = os.path.join(OUTPUT_DIR, 'v5_equity_marks.csv')

    summary_df.to_csv(summary_path, index=False)
    if not trades_all.empty:
        trades_all.to_csv(trades_path, index=False)
    if not equity_all.empty:
        equity_all.to_csv(equity_path, index=False)

    display_cols = [
        'symbol', 'pass_status', 'total_trades', 'win_rate',
        'net_r', 'profit_factor', 'roi_pct', 'max_drawdown_pct',
        'calmar', 'final_equity',
    ]
    print('\n' + '=' * 90)
    print('SUMMARY — ALL ASSETS')
    print('=' * 90)
    print(summary_df[display_cols].to_string(index=False))

    passed = summary_df[summary_df['pass_status'] == 'PASSED'].sort_values(
        'roi_pct', ascending=False)
    failed = summary_df[summary_df['pass_status'] == 'FAILED']

    print('\n' + '=' * 90)
    print(f'PASS / FAIL  →  Passed: {len(passed)}   Failed: {len(failed)}')
    print('=' * 90)

    print('\nTOP PERFORMERS — PASSED ONLY')
    if len(passed) == 0:
        print('  None.')
    else:
        print(passed[display_cols].head(5).to_string(index=False))

    print('\nFAILED — EXCLUDED FROM RANKING')
    if len(failed) == 0:
        print('  None.')
    else:
        print(failed[['symbol', 'roi_pct', 'max_drawdown_pct',
                       'fail_reason', 'fail_time', 'final_equity']].to_string(index=False))

    print('\nEXIT REASON BREAKDOWN')
    if not trades_all.empty:
        er = trades_all.groupby('exit_reason').agg(
            count     = ('R',       'count'),
            win_rate  = ('R',       lambda x: f'{(x > 0).mean() * 100:.1f}%'),
            avg_r     = ('R',       'mean'),
            total_pnl = ('pnl_usd', 'sum'),
        )
        print(er.to_string())
    else:
        print('  No trades executed.')

    print('\nFiles saved:')
    for p in [summary_path, trades_path, equity_path]:
        if os.path.exists(p):
            print(f'  {p}')


if __name__ == '__main__':
    main()
