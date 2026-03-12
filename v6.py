"""
v6.py
=====
Multi-regime backtester for Binance 5m OHLCV data.

Expected input layout:
    data/
      BTCUSDT/
        BTCUSDT_bull_2020_q4_2021_q1_5m.csv.gz
        BTCUSDT_ftx_crash_2022_5m.csv.gz
      ETHUSDT/
        ETHUSDT_bear_2022_pre_ftx_5m.csv.gz
        ETHUSDT_etf_breakout_2024_q1_5m.csv.gz

What this changes vs v5:
- Recursively discovers regime files inside symbol subfolders
- Backtests each symbol/regime file separately
- Saves summary, trades, equity marks, and regime-level aggregate stats
- Lets you filter by symbol and/or regime from CLI
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STARTING_CAPITAL  = 10_000
LEVERAGE          = 1
MAX_RISK_PCT      = 0.0025    # 0.25 % of current equity risked per trade
MAX_DRAWDOWN_PCT  = 0.10      # 10 % peak-to-trough → hard FAIL
STRONG_BODY_PCT   = 0.15
ADX_MIN           = 20
RSI_LONG_MAX      = 70
RSI_SHORT_MIN     = 20
PULLBACK_LOOKBACK = 10
TIME_STOP_BARS    = 96        # 8 h on 5-min bars
MAX_STOP_PCT      = 0.03      # skip if stop distance > 3 % of entry
PARTIAL_R         = 0.5
FINAL_R           = 2.0
ATR_TRAIL_MULT    = 1.5
ATR_STOP_CAP      = 2.0
ATR_WICK_PAD      = 0.1
FEE_RATE          = 0.0    # 0.04 % taker per side


ASSET_MAP: dict[str, str] = {
    'BTCUSDT':  'BTC',
    'ETHUSDT':  'ETH',
    'SOLUSDT':  'SOLANA',
    'AVAXUSDT': 'AVAX',
    'NEARUSDT': 'NEAR',
    'MATICUSDT':'MATIC',
    'ALGOUSDT': 'ALGO',
    'ATOMUSDT': 'ATOM',
    'DOTUSDT':  'DOT',
    'ADAUSDT':  'ADA',
    'LINKUSDT': 'LINK',
    'UNIUSDT':  'UNI',
    'APTUSDT':  'APT',
    'SUIUSDT':  'SUI',
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.dirname(__file__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def discover_datasets(symbol_filters: list[str] | None = None,
                      regime_filters: list[str] | None = None) -> list[dict]:
    """
    Discover files like:
        data/BTCUSDT/BTCUSDT_ftx_crash_2022_5m.csv.gz

    Returns:
        [
            {
                'binance_symbol': 'BTCUSDT',
                'friendly': 'BTC',
                'regime': 'ftx_crash_2022',
                'path': '/abs/.../BTCUSDT_ftx_crash_2022_5m.csv.gz',
                'file': 'BTCUSDT_ftx_crash_2022_5m.csv.gz',
            },
            ...
        ]
    """
    root = Path(DATA_DIR)
    if not root.exists():
        return []

    symbol_filter_set = {s.upper() for s in symbol_filters} if symbol_filters else None
    regime_filter_set = set(regime_filters) if regime_filters else None

    datasets = []
    pattern = re.compile(r'^([A-Z0-9]+)_(.*)_5m\.csv\.gz$')

    for path in root.rglob('*_5m.csv.gz'):
        if not path.is_file():
            continue

        m = pattern.match(path.name)
        if not m:
            continue

        binance_symbol = m.group(1)
        regime = m.group(2)

        if symbol_filter_set and binance_symbol not in symbol_filter_set:
            continue
        if regime_filter_set and regime not in regime_filter_set:
            continue

        datasets.append({
            'binance_symbol': binance_symbol,
            'friendly': ASSET_MAP.get(binance_symbol, binance_symbol),
            'regime': regime,
            'path': str(path),
            'file': path.name,
        })

    datasets.sort(key=lambda x: (x['binance_symbol'], x['regime']))
    return datasets


def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col='datetime', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame, use_sma: bool = False) -> pd.DataFrame:
    df = df.copy()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    if use_sma:
        # v4 uses SMA (rolling mean)
        df['ATR'] = tr.rolling(14).mean()
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        plus_dm  = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0.0)
        minus_dm = minus_dm.where(minus_dm > plus_dm,  0.0)
        tr14     = tr.rolling(14).sum()
        plus_di  = 100 * plus_dm.rolling(14).sum()  / tr14
        minus_di = 100 * minus_dm.rolling(14).sum() / tr14
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.rolling(14).mean()
    else:
        # Standard v6 uses EWM (Wilder's)
        df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        up_move   = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm_raw  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm  = pd.Series(plus_dm_raw, index=df.index)
        minus_dm = pd.Series(minus_dm_raw, index=df.index)

        atr_smma = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smma
        minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smma
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

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


def build_trade(binance_symbol, friendly, regime, pos, exit_time, exit_price, r, pnl,
                equity_after, reason, notional_override=None) -> dict:
    notl = notional_override if notional_override is not None else pos['notional_usd']
    return {
        'binance_symbol': binance_symbol,
        'friendly': friendly,
        'regime': regime,
        'entry_time': pos['entry_time'],
        'exit_time': exit_time,
        'direction': pos['direction'],
        'entry': round(pos['entry'], 6),
        'stop': round(pos['stop'], 6),
        'target_1': round(pos['target_1'], 6),
        'target_2': round(pos['target_2'], 6),
        'exit': round(exit_price, 6),
        'R': round(r, 4),
        'pnl_usd': round(pnl, 4),
        'notional_usd': round(notl, 2),
        'risk_usd': round(pos['risk_usd'], 4),
        'stop_pct': round(pos['stop_pct'] * 100, 4),
        'leverage_used': round(pos['leverage_used'], 4),
        'equity_after': round(equity_after, 4),
        'exit_reason': reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def backtest_dataset(df: pd.DataFrame,
                     binance_symbol: str,
                     friendly: str,
                     regime: str,
                     starting_capital: float = STARTING_CAPITAL,
                     fee_rate: float = FEE_RATE,
                     instant_entry: bool = False):
    trades = []
    equity_marks = []
    position = None
    pending = None

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
            'binance_symbol': binance_symbol,
            'friendly': friendly,
            'regime': regime,
            'time': ts,
            'equity': round(eq_val, 4),
            'peak_equity': round(peak_equity, 4),
            'drawdown_pct': round(dd * 100, 4),
            'mark_type': mark_type,
        })
        if not failed and dd <= -MAX_DRAWDOWN_PCT:
            failed = True
            fail_reason = 'max_drawdown_breach'
            fail_time = ts
            halt_equity = eq_val

    update_dd(df.index[0], closed_equity, 'start')

    for i in range(250, len(df)):
        row = df.iloc[i]

        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or pd.isna(row['ADX']) or pd.isna(row['MA50']) or pd.isna(row['MA200']):
            continue

        if pending is not None and position is None:
            # v4 entries happen at signal bar Close
            # v6 entries happen at next bar Open (more realistic)
            entry = row['Close'] if instant_entry else row['Open']
            raw_stop = pending['raw_stop']
            direction = pending['direction']
            rp = abs(entry - raw_stop)
            stop_pct = rp / entry if entry > 0 else 1.0

            if rp > 0 and stop_pct <= MAX_STOP_PCT:
                notl, ar, el, sp = size_position(closed_equity, entry, raw_stop)
                if notl > 0:
                    entry_fee = notl * fee_rate
                    closed_equity -= entry_fee
                    position = {
                        'direction': direction,
                        'entry': entry,
                        'stop': raw_stop,
                        'risk_price': rp,
                        'target_1': entry + PARTIAL_R * rp if direction == 'long' else entry - PARTIAL_R * rp,
                        'target_2': entry + FINAL_R * rp if direction == 'long' else entry - FINAL_R * rp,
                        'notional_usd': notl,
                        'risk_usd': ar,
                        'leverage_used': el,
                        'stop_pct': sp,
                        'entry_time': row.name,
                        'entry_bar': i,
                        'partial_done': False,
                    }
            pending = None

        if position is not None:
            mtm = mark_to_market(closed_equity, position, row['Close'])
            update_dd(row.name, mtm, 'mtm')

            if failed:
                e, rp_pos = position['entry'], position['risk_price']
                notl = position['notional_usd']
                if position['direction'] == 'long':
                    pnl = notl * (row['Close'] - e) / e
                    r = (row['Close'] - e) / rp_pos
                else:
                    pnl = notl * (e - row['Close']) / e
                    r = (e - row['Close']) / rp_pos
                pnl -= notl * fee_rate
                closed_equity = max(closed_equity + pnl, 0.0)
                trades.append(build_trade(
                    binance_symbol, friendly, regime, position, row.name,
                    row['Close'], r, pnl, closed_equity, 'dd_halt'
                ))
                update_dd(row.name, closed_equity, 'closed_dd_halt')
                position = None
                break

        if position is not None:
            e = position['entry']
            s = position['stop']
            t1 = position['target_1']
            t2 = position['target_2']
            rp = position['risk_price']
            notl = position['notional_usd']
            d = position['direction']
            xt = None

            if d == 'long':
                if row['Low'] <= s:
                    exit_fee = notl * fee_rate
                    pnl = notl * (s - e) / e - exit_fee
                    r = (s - e) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, s, r, pnl, 0.0, 'stop')

                elif row['High'] >= t2:
                    exit_fee = notl * fee_rate
                    pnl = notl * (t2 - e) / e - exit_fee
                    r = (t2 - e) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, t2, r, pnl, 0.0, 'target')

                else:
                    if not position['partial_done'] and row['High'] >= t1:
                        half_notl = notl / 2
                        partial_fee = half_notl * fee_rate
                        partial_pnl = half_notl * (t1 - e) / e - partial_fee
                        closed_equity = max(closed_equity + partial_pnl, 0.0)
                        partial_r = (t1 - e) / rp
                        pt = build_trade(
                            binance_symbol, friendly, regime, position, row.name, t1,
                            partial_r, partial_pnl, closed_equity, 'partial_target',
                            notional_override=half_notl
                        )
                        trades.append(pt)
                        update_dd(row.name, closed_equity, 'partial_closed')
                        position['partial_done'] = True
                        position['notional_usd'] = half_notl
                        position['stop'] = e
                        notl = half_notl

                    if position['partial_done']:
                        trail = row['Close'] - ATR_TRAIL_MULT * row['ATR']
                        if trail > position['stop']:
                            position['stop'] = trail

                    if (i - position['entry_bar']) > TIME_STOP_BARS and row['Close'] < e + 0.3 * rp:
                        exit_fee = notl * fee_rate
                        pnl = notl * (row['Close'] - e) / e - exit_fee
                        r = (row['Close'] - e) / rp
                        xt = build_trade(binance_symbol, friendly, regime, position, row.name,
                                         row['Close'], r, pnl, 0.0, 'timeout')

            else:
                if row['High'] >= s:
                    exit_fee = notl * fee_rate
                    pnl = notl * (e - s) / e - exit_fee
                    r = (e - s) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, s, r, pnl, 0.0, 'stop')

                elif row['Low'] <= t2:
                    exit_fee = notl * fee_rate
                    pnl = notl * (e - t2) / e - exit_fee
                    r = (e - t2) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, t2, r, pnl, 0.0, 'target')

                else:
                    if not position['partial_done'] and row['Low'] <= t1:
                        half_notl = notl / 2
                        partial_fee = half_notl * fee_rate
                        partial_pnl = half_notl * (e - t1) / e - partial_fee
                        closed_equity = max(closed_equity + partial_pnl, 0.0)
                        partial_r = (e - t1) / rp
                        pt = build_trade(
                            binance_symbol, friendly, regime, position, row.name, t1,
                            partial_r, partial_pnl, closed_equity, 'partial_target',
                            notional_override=half_notl
                        )
                        trades.append(pt)
                        update_dd(row.name, closed_equity, 'partial_closed')
                        position['partial_done'] = True
                        position['notional_usd'] = half_notl
                        position['stop'] = e
                        notl = half_notl

                    if position['partial_done']:
                        trail = row['Close'] + ATR_TRAIL_MULT * row['ATR']
                        if trail < position['stop']:
                            position['stop'] = trail

                    if (i - position['entry_bar']) > TIME_STOP_BARS and row['Close'] > e - 0.3 * rp:
                        exit_fee = notl * fee_rate
                        pnl = notl * (e - row['Close']) / e - exit_fee
                        r = (e - row['Close']) / rp
                        xt = build_trade(binance_symbol, friendly, regime, position, row.name,
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
    run_info = {
        'binance_symbol': binance_symbol,
        'friendly': friendly,
        'regime': regime,
        'final_equity': closed_equity,
        'peak_equity': peak_equity,
        'max_drawdown_pct': worst_drawdown * 100,
        'failed': failed,
        'pass_status': 'FAILED' if failed else 'PASSED',
        'fail_reason': fail_reason,
        'fail_time': str(fail_time) if fail_time else None,
        'halt_equity': halt_equity,
        'total_trades': len(trades_df),
    }
    return trades_df, equity_df, run_info


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def calc_metrics(trades_df: pd.DataFrame, run_info: dict,
                 starting_capital: float = STARTING_CAPITAL) -> dict:
    roi = (run_info['final_equity'] / starting_capital - 1) * 100
    mdd = abs(run_info['max_drawdown_pct'])

    base = {
        **run_info,
        'win_rate': 0.0,
        'avg_r': 0.0,
        'net_r': 0.0,
        'profit_factor': 0.0,
        'roi_pct': roi,
        'calmar': 0.0,
        'avg_notional_usd': 0.0,
        'avg_leverage_used': 0.0,
        'long_count': 0,
        'short_count': 0,
    }
    if len(trades_df) == 0:
        return base

    wins = (trades_df['R'] > 0).sum()
    gp = trades_df.loc[trades_df['R'] > 0, 'R'].sum()
    gl = trades_df.loc[trades_df['R'] < 0, 'R'].abs().sum()

    return {
        **run_info,
        'win_rate': wins / len(trades_df) * 100,
        'avg_r': trades_df['R'].mean(),
        'net_r': trades_df['R'].sum(),
        'profit_factor': gp / gl if gl > 0 else 0.0,
        'roi_pct': roi,
        'calmar': roi / mdd if mdd > 0 else 0.0,
        'avg_notional_usd': trades_df['notional_usd'].mean(),
        'avg_leverage_used': trades_df['leverage_used'].mean(),
        'long_count': (trades_df['direction'] == 'long').sum(),
        'short_count': (trades_df['direction'] == 'short').sum(),
    }


def build_regime_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    out = summary_df.groupby('regime', as_index=False).agg(
        datasets=('regime', 'size'),
        passed=('pass_status', lambda x: int((x == 'PASSED').sum())),
        failed=('pass_status', lambda x: int((x == 'FAILED').sum())),
        avg_roi_pct=('roi_pct', 'mean'),
        median_roi_pct=('roi_pct', 'median'),
        avg_max_drawdown_pct=('max_drawdown_pct', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        total_trades=('total_trades', 'sum'),
    )
    return out.sort_values(['avg_roi_pct', 'regime'], ascending=[False, True])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(symbol_filters: list[str] | None = None,
         regime_filters: list[str] | None = None,
         v4_compat: bool = False,
         no_fees: bool = False,
         sma_indicators: bool = False,
         instant_entry: bool = False):
    
    # Resolve conflicting flags
    if v4_compat:
        no_fees = True
        sma_indicators = True
        instant_entry = True

    final_fee_rate = 0.0 if no_fees else FEE_RATE

    results = []
    all_trades = []
    all_equity = []

    print('=' * 100)
    print('PROFESSIONAL BACKTEST v6 — MULTI-REGIME — Binance local data — 5-min OHLCV')
    print('=' * 100)
    if v4_compat:
        print('*** RUNNING IN v4 COMPATIBILITY MODE (Optimistic Settings) ***')
    print(f'Starting capital   : ${STARTING_CAPITAL:,.0f}')
    print(f'Leverage cap       : {LEVERAGE}x')
    print(f'Risk per trade     : {MAX_RISK_PCT * 100:.2f}% of current equity')
    print(f'Max drawdown limit : {MAX_DRAWDOWN_PCT * 100:.0f}% -> hard FAIL')
    print(f'Fee rate           : {final_fee_rate * 100:.2f}% taker per side')
    print(f'Indicators         : {"SMA (v4 style)" if sma_indicators else "EWM (Wilder style)"}')
    print(f'Entry Timing       : {"Instant (Bar Close)" if instant_entry else "Next Bar Open"}')
    print(f'Data directory     : {os.path.abspath(DATA_DIR)}')
    if symbol_filters:
        print(f'Symbol filter      : {", ".join(symbol_filters)}')
    if regime_filters:
        print(f'Regime filter      : {", ".join(regime_filters)}')
    print()

    datasets = discover_datasets(symbol_filters=symbol_filters, regime_filters=regime_filters)
    if not datasets:
        print('ERROR: No multi-regime data files found.')
        print('Expected files like:')
        print('  data/BTCUSDT/BTCUSDT_ftx_crash_2022_5m.csv.gz')
        return

    symbols_found = sorted({d['binance_symbol'] for d in datasets})
    regimes_found = sorted({d['regime'] for d in datasets})

    print(f'Found {len(datasets)} dataset(s) across {len(symbols_found)} symbol(s) and {len(regimes_found)} regime(s).')
    print('Processing...\\n')

    for idx, ds in enumerate(datasets, start=1):
        bsym = ds['binance_symbol']
        friendly = ds['friendly']
        regime = ds['regime']
        path = ds['path']

        df_raw = load_data_from_path(path)
        if df_raw.empty:
            print(f'[{idx:>3}/{len(datasets)}] {friendly} [{regime}] ... No data — skipping.')
            continue

        df = add_indicators(df_raw, use_sma=sma_indicators)
        n_bars = len(df)
        date_range = (
            df.index.min().strftime('%Y-%m-%d'),
            df.index.max().strftime('%Y-%m-%d')
        )

        label = f'{friendly} [{regime}]'
        print(f'[{idx:>3}/{len(datasets)}] {label:.<42} {n_bars:>8,} bars  ({date_range[0]} -> {date_range[1]})',
              end='  ', flush=True)

        trades_df, equity_df, run_info = backtest_dataset(
            df, bsym, friendly, regime, 
            fee_rate=final_fee_rate, 
            instant_entry=instant_entry
        )
        metrics = calc_metrics(trades_df, run_info)
        metrics['source_file'] = ds['file']
        results.append(metrics)

        if not trades_df.empty:
            trades_df['source_file'] = ds['file']
            all_trades.append(trades_df)

        if not equity_df.empty:
            equity_df['source_file'] = ds['file']
            all_equity.append(equity_df)

        print(f"{metrics['pass_status']:<6} | Trades: {metrics['total_trades']:>4} | "
              f"ROI: {metrics['roi_pct']:>8.2f}% | "
              f"MaxDD: {metrics['max_drawdown_pct']:>7.2f}% | "
              f"Final: ${metrics['final_equity']:>10,.2f}")

    if not results:
        print('\\nNo results to display.')
        return

    summary_df = pd.DataFrame(results).sort_values(
        ['pass_status', 'binance_symbol', 'roi_pct'],
        ascending=[True, True, False]
    )
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_all = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    regime_summary_df = build_regime_summary(summary_df)

    summary_path = os.path.join(OUTPUT_DIR, 'v6_summary.csv')
    trades_path = os.path.join(OUTPUT_DIR, 'v6_trades.csv')
    equity_path = os.path.join(OUTPUT_DIR, 'v6_equity_marks.csv')
    regime_summary_path = os.path.join(OUTPUT_DIR, 'v6_regime_summary.csv')

    summary_df.to_csv(summary_path, index=False)
    if not trades_all.empty:
        trades_all.to_csv(trades_path, index=False)
    if not equity_all.empty:
        equity_all.to_csv(equity_path, index=False)
    if not regime_summary_df.empty:
        regime_summary_df.to_csv(regime_summary_path, index=False)

    display_cols = [
        'binance_symbol', 'regime', 'pass_status', 'total_trades',
        'win_rate', 'net_r', 'profit_factor', 'roi_pct',
        'max_drawdown_pct', 'calmar', 'final_equity',
    ]

    print('\\n' + '=' * 100)
    print('SUMMARY — ALL DATASETS')
    print('=' * 100)
    print(summary_df[display_cols].to_string(index=False))

    passed = summary_df[summary_df['pass_status'] == 'PASSED'].sort_values(
        'roi_pct', ascending=False
    )
    failed = summary_df[summary_df['pass_status'] == 'FAILED']

    print('\\n' + '=' * 100)
    print(f'PASS / FAIL -> Passed: {len(passed)}   Failed: {len(failed)}')
    print('=' * 100)

    print('\\nTOP DATASETS — PASSED ONLY')
    if len(passed) == 0:
        print('  None.')
    else:
        print(passed[display_cols].head(10).to_string(index=False))

    print('\\nFAILED — EXCLUDED FROM RANKING')
    if len(failed) == 0:
        print('  None.')
    else:
        print(failed[['binance_symbol', 'regime', 'roi_pct', 'max_drawdown_pct',
                      'fail_reason', 'fail_time', 'final_equity']].to_string(index=False))

    print('\\nREGIME SUMMARY')
    if regime_summary_df.empty:
        print('  None.')
    else:
        print(regime_summary_df.to_string(index=False))

    print('\\nEXIT REASON BREAKDOWN')
    if not trades_all.empty:
        er = trades_all.groupby('exit_reason').agg(
            count=('R', 'count'),
            win_rate=('R', lambda x: f'{(x > 0).mean() * 100:.1f}%'),
            avg_r=('R', 'mean'),
            total_pnl=('pnl_usd', 'sum'),
        )
        print(er.to_string())
    else:
        print('  No trades executed.')

    print('\\nFiles saved:')
    for p in [summary_path, trades_path, equity_path, regime_summary_path]:
        if os.path.exists(p):
            print(f'  {p}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-regime Binance OHLCV backtester')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Optional symbol filter, e.g. BTCUSDT ETHUSDT')
    parser.add_argument('--regimes', nargs='+', default=None,
                        help='Optional regime filter, e.g. bull_2020_q4_2021_q1 ftx_crash_2022')
    parser.add_argument('--v4-compat', action='store_true', help='Match v4 settings (no fees, SMA indicators, instant entry)')
    parser.add_argument('--no-fees', action='store_true', help='Disable trade fees')
    parser.add_argument('--sma-indicators', action='store_true', help='Use SMA instead of Wilder/EWM')
    parser.add_argument('--instant-entry', action='store_true', help='Enter at signal bar close instead of next bar open')

    args = parser.parse_args()

    main(
        symbol_filters=args.symbols,
        regime_filters=args.regimes,
        v4_compat=args.v4_compat,
        no_fees=args.no_fees,
        sma_indicators=args.sma_indicators,
        instant_entry=args.instant_entry,
    )
