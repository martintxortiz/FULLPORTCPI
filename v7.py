"""
v7_optimizer.py
===============
Parameter optimizer for the v6 multi-regime Binance 5m backtester.

Features
--------
- Recursively discovers data/*/*_5m.csv.gz
- Refactors the strategy into tunable params
- Uses position-level metrics (partial exits merged into one position)
- Optimizes with Optuna
- Optional train/validation split inside each dataset
- Saves:
    - optuna_trials.csv
    - best_params.json
    - best_validation_summary.csv
    - best_train_summary.csv

Install
-------
pip install optuna pandas numpy

Examples
--------
python v7_optimizer.py --trials 300
python v7_optimizer.py --symbols BTCUSDT ETHUSDT --trials 400
python v7_optimizer.py --symbols BTCUSDT --regimes ftx_crash_2022 --trials 200
python v7_optimizer.py --train-frac 0.7 --trials 300 --no-pruner
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import optuna


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL = 10_000
LEVERAGE = 1

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
# PARAMS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    MAX_RISK_PCT: float = 0.0025
    MAX_DRAWDOWN_PCT: float = 0.10
    STRONG_BODY_PCT: float = 0.35
    ADX_MIN: int = 20
    RSI_LONG_MAX: int = 65
    RSI_SHORT_MIN: int = 35
    PULLBACK_LOOKBACK: int = 10
    TIME_STOP_BARS: int = 96
    MAX_STOP_PCT: float = 0.03
    PARTIAL_R: float = 1.0
    FINAL_R: float = 2.0
    ATR_TRAIL_MULT: float = 1.5
    ATR_STOP_CAP: float = 2.0
    ATR_WICK_PAD: float = 0.1
    FEE_RATE: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DATA DISCOVERY / LOADING
# ══════════════════════════════════════════════════════════════════════════════

def discover_datasets(symbol_filters: list[str] | None = None,
                      regime_filters: list[str] | None = None) -> list[dict]:
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


def split_train_valid(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 600:
        return df.copy(), df.iloc[0:0].copy()

    split_idx = int(len(df) * train_frac)
    split_idx = max(300, min(split_idx, len(df) - 200))
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame, use_sma: bool = False) -> pd.DataFrame:
    df = df.copy()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    if use_sma:
        df['ATR'] = tr.rolling(14).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * plus_dm.rolling(14).sum() / tr14
        minus_di = 100 * minus_dm.rolling(14).sum() / tr14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.rolling(14).mean()
    else:
        df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm_raw = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm_raw, index=df.index)
        minus_dm = pd.Series(minus_dm_raw, index=df.index)

        atr_smma = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smma
        minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smma
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_strong_candle(row, min_body_pct: float):
    rng = row['High'] - row['Low']
    if rng == 0:
        return False, None
    body = abs(row['Close'] - row['Open']) / rng
    if body >= min_body_pct:
        return True, ('bull' if row['Close'] > row['Open'] else 'bear')
    return False, None


def size_position(equity: float, entry: float, stop: float, params: Params):
    stop_pct = abs(entry - stop) / entry
    if equity <= 0 or stop_pct <= 0:
        return 0.0, 0.0, 0.0, 0.0
    risk_usd = equity * params.MAX_RISK_PCT
    notional = min(risk_usd / stop_pct, equity * LEVERAGE)
    actual_risk = notional * stop_pct
    lev_used = notional / equity if equity > 0 else 0.0
    return notional, actual_risk, lev_used, stop_pct


def mark_to_market(closed_equity: float, position: dict | None, price: float) -> float:
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
        'R': round(r, 6),
        'pnl_usd': round(pnl, 6),
        'notional_usd': round(notl, 2),
        'risk_usd': round(pos['risk_usd'], 6),
        'stop_pct': round(pos['stop_pct'] * 100, 6),
        'leverage_used': round(pos['leverage_used'], 6),
        'equity_after': round(equity_after, 6),
        'exit_reason': reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def backtest_dataset(df: pd.DataFrame,
                     binance_symbol: str,
                     friendly: str,
                     regime: str,
                     params: Params,
                     starting_capital: float = STARTING_CAPITAL,
                     instant_entry: bool = False):
    trades = []
    equity_marks = []
    position = None
    pending = None

    closed_equity = starting_capital
    peak_equity = starting_capital
    worst_drawdown = 0.0
    failed = False
    fail_reason = None
    fail_time = None
    halt_equity = None

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), {
            'binance_symbol': binance_symbol,
            'friendly': friendly,
            'regime': regime,
            'final_equity': starting_capital,
            'peak_equity': starting_capital,
            'max_drawdown_pct': 0.0,
            'failed': False,
            'pass_status': 'PASSED',
            'fail_reason': None,
            'fail_time': None,
            'halt_equity': None,
            'total_trades': 0,
        }

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
            'equity': round(eq_val, 6),
            'peak_equity': round(peak_equity, 6),
            'drawdown_pct': round(dd * 100, 6),
            'mark_type': mark_type,
        })
        if not failed and dd <= -params.MAX_DRAWDOWN_PCT:
            failed = True
            fail_reason = 'max_drawdown_breach'
            fail_time = ts
            halt_equity = eq_val

    update_dd(df.index[0], closed_equity, 'start')

    start_i = max(250, params.PULLBACK_LOOKBACK + 5)

    for i in range(start_i, len(df)):
        row = df.iloc[i]

        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or pd.isna(row['ADX']) or pd.isna(row['MA50']) or pd.isna(row['MA200']):
            continue

        if pending is not None and position is None:
            entry = row['Close'] if instant_entry else row['Open']
            raw_stop = pending['raw_stop']
            direction = pending['direction']
            rp = abs(entry - raw_stop)
            stop_pct = rp / entry if entry > 0 else 1.0

            if rp > 0 and stop_pct <= params.MAX_STOP_PCT:
                notl, ar, el, sp = size_position(closed_equity, entry, raw_stop, params)
                if notl > 0:
                    entry_fee = notl * params.FEE_RATE
                    closed_equity -= entry_fee
                    position = {
                        'direction': direction,
                        'entry': entry,
                        'stop': raw_stop,
                        'risk_price': rp,
                        'target_1': entry + params.PARTIAL_R * rp if direction == 'long' else entry - params.PARTIAL_R * rp,
                        'target_2': entry + params.FINAL_R * rp if direction == 'long' else entry - params.FINAL_R * rp,
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
                pnl -= notl * params.FEE_RATE
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
                    exit_fee = notl * params.FEE_RATE
                    pnl = notl * (s - e) / e - exit_fee
                    r = (s - e) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, s, r, pnl, 0.0, 'stop')

                elif row['High'] >= t2:
                    exit_fee = notl * params.FEE_RATE
                    pnl = notl * (t2 - e) / e - exit_fee
                    r = (t2 - e) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, t2, r, pnl, 0.0, 'target')

                else:
                    if (not position['partial_done']) and (row['High'] >= t1):
                        half_notl = notl / 2
                        partial_fee = half_notl * params.FEE_RATE
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
                        trail = row['Close'] - params.ATR_TRAIL_MULT * row['ATR']
                        if trail > position['stop']:
                            position['stop'] = trail

                    if (i - position['entry_bar']) > params.TIME_STOP_BARS and row['Close'] < e + 0.3 * rp:
                        exit_fee = notl * params.FEE_RATE
                        pnl = notl * (row['Close'] - e) / e - exit_fee
                        r = (row['Close'] - e) / rp
                        xt = build_trade(binance_symbol, friendly, regime, position, row.name,
                                         row['Close'], r, pnl, 0.0, 'timeout')

            else:
                if row['High'] >= s:
                    exit_fee = notl * params.FEE_RATE
                    pnl = notl * (e - s) / e - exit_fee
                    r = (e - s) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, s, r, pnl, 0.0, 'stop')

                elif row['Low'] <= t2:
                    exit_fee = notl * params.FEE_RATE
                    pnl = notl * (e - t2) / e - exit_fee
                    r = (e - t2) / rp
                    xt = build_trade(binance_symbol, friendly, regime, position, row.name, t2, r, pnl, 0.0, 'target')

                else:
                    if (not position['partial_done']) and (row['Low'] <= t1):
                        half_notl = notl / 2
                        partial_fee = half_notl * params.FEE_RATE
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
                        trail = row['Close'] + params.ATR_TRAIL_MULT * row['ATR']
                        if trail < position['stop']:
                            position['stop'] = trail

                    if (i - position['entry_bar']) > params.TIME_STOP_BARS and row['Close'] > e - 0.3 * rp:
                        exit_fee = notl * params.FEE_RATE
                        pnl = notl * (e - row['Close']) / e - exit_fee
                        r = (e - row['Close']) / rp
                        xt = build_trade(binance_symbol, friendly, regime, position, row.name,
                                         row['Close'], r, pnl, 0.0, 'timeout')

            if xt is not None:
                closed_equity = max(closed_equity + xt['pnl_usd'], 0.0)
                xt['equity_after'] = round(closed_equity, 6)
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

            if row['ADX'] < params.ADX_MIN:
                continue

            strong, cdir = is_strong_candle(row, params.STRONG_BODY_PCT)
            if not strong:
                continue

            atr = row['ATR']
            lb = params.PULLBACK_LOOKBACK
            win = df.iloc[i - lb:i]

            if bull and cdir == 'bull':
                if row['RSI'] > params.RSI_LONG_MAX:
                    continue
                if not (win['Low'] <= win['MA50'] + atr).any():
                    continue
                if (row['Close'] - row['MA50']) / atr > 4:
                    continue
                raw_stop = max(row['Low'] - params.ATR_WICK_PAD * atr,
                               row['Close'] - params.ATR_STOP_CAP * atr)
                pending = {'direction': 'long', 'raw_stop': raw_stop}

            elif bear and cdir == 'bear':
                if row['RSI'] < params.RSI_SHORT_MIN:
                    continue
                if not (win['High'] >= win['MA50'] - atr).any():
                    continue
                if (row['MA50'] - row['Close']) / atr > 4:
                    continue
                raw_stop = min(row['High'] + params.ATR_WICK_PAD * atr,
                               row['Close'] + params.ATR_STOP_CAP * atr)
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
        'fail_time': str(fail_time) if fail_time is not None else None,
        'halt_equity': halt_equity,
        'total_trades': len(trades_df),
    }
    return trades_df, equity_df, run_info


# ══════════════════════════════════════════════════════════════════════════════
# POSITION-LEVEL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_positions(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[
            'binance_symbol', 'friendly', 'regime', 'entry_time', 'exit_time',
            'direction', 'entry', 'pnl_usd', 'risk_usd', 'legs', 'R_position', 'win'
        ])

    key_cols = ['binance_symbol', 'friendly', 'regime', 'entry_time', 'direction', 'entry']
    grp = trades_df.groupby(key_cols, dropna=False, as_index=False)

    out = grp.agg(
        exit_time=('exit_time', 'max'),
        pnl_usd=('pnl_usd', 'sum'),
        risk_usd=('risk_usd', 'first'),
        legs=('R', 'count'),
        last_equity=('equity_after', 'last'),
    )

    out['R_position'] = np.where(out['risk_usd'] > 0, out['pnl_usd'] / out['risk_usd'], 0.0)
    out['win'] = out['R_position'] > 0
    return out


def calc_position_metrics(pos_df: pd.DataFrame, run_info: dict,
                          starting_capital: float = STARTING_CAPITAL) -> dict:
    roi = (run_info['final_equity'] / starting_capital - 1.0) * 100.0
    mdd = abs(run_info['max_drawdown_pct'])

    base = {
        **run_info,
        'positions': 0,
        'expectancy_r': -999.0,
        'win_rate': 0.0,
        'avg_win_r': 0.0,
        'avg_loss_r': 0.0,
        'avg_r': 0.0,
        'net_r': 0.0,
        'profit_factor_r': 0.0,
        'roi_pct': roi,
        'calmar': roi / mdd if mdd > 0 else 0.0,
    }

    if pos_df.empty:
        return base

    wins = pos_df.loc[pos_df['R_position'] > 0, 'R_position']
    losses = pos_df.loc[pos_df['R_position'] < 0, 'R_position'].abs()

    win_rate = len(wins) / len(pos_df)
    loss_rate = 1.0 - win_rate
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = win_rate * avg_win - loss_rate * avg_loss
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(losses.sum()) if len(losses) else 0.0

    return {
        **run_info,
        'positions': int(len(pos_df)),
        'expectancy_r': expectancy,
        'win_rate': win_rate * 100.0,
        'avg_win_r': avg_win,
        'avg_loss_r': avg_loss,
        'avg_r': float(pos_df['R_position'].mean()),
        'net_r': float(pos_df['R_position'].sum()),
        'profit_factor_r': gp / gl if gl > 0 else 0.0,
        'roi_pct': roi,
        'calmar': roi / mdd if mdd > 0 else 0.0,
    }


def score_metrics(m: dict, min_positions: int = 25) -> float:
    score = 0.0
    score += 100.0 * m['expectancy_r']
    score += 0.25 * m['roi_pct']
    score += 8.0 * min(m['profit_factor_r'], 5.0)
    score += 0.05 * m['win_rate']
    score -= 1.50 * abs(m['max_drawdown_pct'])

    if m['positions'] < min_positions:
        score -= (min_positions - m['positions']) * 20.0
    if m['failed']:
        score -= 5000.0
    if m['roi_pct'] < 0:
        score += m['roi_pct'] * 2.0

    return float(score)


# ══════════════════════════════════════════════════════════════════════════════
# PARAM SEARCH SPACE
# ══════════════════════════════════════════════════════════════════════════════

def sample_params(trial: optuna.trial.Trial, fee_rate: float, max_dd_pct: float) -> Params:
    partial_r = trial.suggest_float("PARTIAL_R", 0.5, 2.0, step=0.1)
    final_r_low = max(1.0, partial_r)
    final_r = trial.suggest_float("FINAL_R", final_r_low, 4.5, step=0.1)

    return Params(
        MAX_RISK_PCT=trial.suggest_float("MAX_RISK_PCT", 0.001, 0.01, log=True),
        MAX_DRAWDOWN_PCT=max_dd_pct,
        STRONG_BODY_PCT=trial.suggest_float("STRONG_BODY_PCT", 0.20, 0.70, step=0.01),
        ADX_MIN=trial.suggest_int("ADX_MIN", 10, 40),
        RSI_LONG_MAX=trial.suggest_int("RSI_LONG_MAX", 50, 80),
        RSI_SHORT_MIN=trial.suggest_int("RSI_SHORT_MIN", 20, 50),
        PULLBACK_LOOKBACK=trial.suggest_int("PULLBACK_LOOKBACK", 3, 30),
        TIME_STOP_BARS=trial.suggest_int("TIME_STOP_BARS", 12, 180),
        MAX_STOP_PCT=trial.suggest_float("MAX_STOP_PCT", 0.005, 0.05, step=0.001),
        PARTIAL_R=partial_r,
        FINAL_R=final_r,
        ATR_TRAIL_MULT=trial.suggest_float("ATR_TRAIL_MULT", 0.5, 4.0, step=0.1),
        ATR_STOP_CAP=trial.suggest_float("ATR_STOP_CAP", 0.5, 4.0, step=0.1),
        ATR_WICK_PAD=trial.suggest_float("ATR_WICK_PAD", 0.0, 0.5, step=0.01),
        FEE_RATE=fee_rate,
    )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_one(df: pd.DataFrame, ds: dict, params: Params, instant_entry: bool) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    trades_df, equity_df, run_info = backtest_dataset(
        df=df,
        binance_symbol=ds['binance_symbol'],
        friendly=ds['friendly'],
        regime=ds['regime'],
        params=params,
        starting_capital=STARTING_CAPITAL,
        instant_entry=instant_entry,
    )
    pos_df = aggregate_positions(trades_df)
    metrics = calc_position_metrics(pos_df, run_info, starting_capital=STARTING_CAPITAL)
    return metrics, trades_df, pos_df


def summarize_metric_list(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {
            'datasets': 0,
            'passed': 0,
            'failed': 0,
            'median_score': -999999.0,
            'median_expectancy_r': -999.0,
            'median_roi_pct': -999.0,
            'median_mdd_pct': 999.0,
            'total_positions': 0,
            'pass_rate_pct': 0.0,
        }

    scores = [score_metrics(m) for m in metrics_list]
    return {
        'datasets': len(metrics_list),
        'passed': int(sum(1 for m in metrics_list if not m['failed'])),
        'failed': int(sum(1 for m in metrics_list if m['failed'])),
        'median_score': float(np.median(scores)),
        'median_expectancy_r': float(np.median([m['expectancy_r'] for m in metrics_list])),
        'median_roi_pct': float(np.median([m['roi_pct'] for m in metrics_list])),
        'median_mdd_pct': float(np.median([abs(m['max_drawdown_pct']) for m in metrics_list])),
        'total_positions': int(sum(m['positions'] for m in metrics_list)),
        'pass_rate_pct': float(np.mean([0 if m['failed'] else 1 for m in metrics_list]) * 100.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class Optimizer:
    def __init__(self,
                 datasets: list[dict],
                 train_frac: float,
                 instant_entry: bool,
                 use_sma: bool,
                 fee_rate: float,
                 max_dd_pct: float):
        self.datasets = datasets
        self.train_frac = train_frac
        self.instant_entry = instant_entry
        self.use_sma = use_sma
        self.fee_rate = fee_rate
        self.max_dd_pct = max_dd_pct

        self.cached = []
        for ds in datasets:
            raw = load_data_from_path(ds['path'])
            ind = add_indicators(raw, use_sma=use_sma)
            train_df, valid_df = split_train_valid(ind, train_frac=train_frac)
            self.cached.append({
                'meta': ds,
                'all_df': ind,
                'train_df': train_df,
                'valid_df': valid_df,
            })

    def objective(self, trial: optuna.trial.Trial):
        params = sample_params(trial, fee_rate=self.fee_rate, max_dd_pct=self.max_dd_pct)

        train_metrics = []
        valid_metrics = []

        for idx, item in enumerate(self.cached):
            ds = item['meta']
            train_df = item['train_df']
            valid_df = item['valid_df']

            m_train, _, _ = evaluate_one(train_df, ds, params, self.instant_entry)
            train_metrics.append(m_train)

            trial.report(float(np.median([score_metrics(x) for x in train_metrics])), step=idx + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if len(valid_df) > 300:
                m_valid, _, _ = evaluate_one(valid_df, ds, params, self.instant_entry)
                valid_metrics.append(m_valid)

        train_summary = summarize_metric_list(train_metrics)
        valid_summary = summarize_metric_list(valid_metrics) if valid_metrics else None

        trial.set_user_attr("train_summary", train_summary)
        if valid_summary is not None:
            trial.set_user_attr("valid_summary", valid_summary)

        # Optimize validation if available, otherwise train
        if valid_summary is not None and valid_summary['datasets'] > 0:
            return valid_summary['median_score']
        return train_summary['median_score']

    def evaluate_best(self, params: Params):
        train_rows = []
        valid_rows = []

        for item in self.cached:
            ds = item['meta']
            train_df = item['train_df']
            valid_df = item['valid_df']

            m_train, _, _ = evaluate_one(train_df, ds, params, self.instant_entry)
            train_rows.append({
                'split': 'train',
                'source_file': ds['file'],
                **m_train,
            })

            if len(valid_df) > 300:
                m_valid, _, _ = evaluate_one(valid_df, ds, params, self.instant_entry)
                valid_rows.append({
                    'split': 'valid',
                    'source_file': ds['file'],
                    **m_valid,
                })

        return pd.DataFrame(train_rows), pd.DataFrame(valid_rows)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Optuna optimizer for multi-regime Binance OHLCV strategy')
    parser.add_argument('--symbols', nargs='+', default=None, help='Optional symbol filter, e.g. BTCUSDT ETHUSDT')
    parser.add_argument('--regimes', nargs='+', default=None, help='Optional regime filter')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--train-frac', type=float, default=0.7, help='Train fraction per dataset')
    parser.add_argument('--study-name', type=str, default='v7_optimizer', help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None, help='Optional Optuna storage URL, e.g. sqlite:///optuna.db')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampler')
    parser.add_argument('--no-pruner', action='store_true', help='Disable MedianPruner')
    parser.add_argument('--instant-entry', action='store_true', help='Use signal bar close entry')
    parser.add_argument('--sma-indicators', action='store_true', help='Use SMA indicator mode')
    parser.add_argument('--fee-rate', type=float, default=0.0, help='Fee per side, e.g. 0.0004')
    parser.add_argument('--max-dd-pct', type=float, default=0.10, help='Hard drawdown fail threshold, e.g. 0.10')
    args = parser.parse_args()

    datasets = discover_datasets(symbol_filters=args.symbols, regime_filters=args.regimes)
    if not datasets:
        print('ERROR: No datasets found in data/*/*_5m.csv.gz')
        return

    print('=' * 100)
    print('V7 OPTIMIZER — MULTI-REGIME PARAM SEARCH')
    print('=' * 100)
    print(f'Datasets       : {len(datasets)}')
    print(f'Trials         : {args.trials}')
    print(f'Train fraction : {args.train_frac:.2f}')
    print(f'Fee rate       : {args.fee_rate}')
    print(f'Max DD fail    : {args.max_dd_pct:.2%}')
    print(f'Indicators     : {"SMA" if args.sma_indicators else "Wilder/EWM"}')
    print(f'Entry timing   : {"Instant close" if args.instant_entry else "Next bar open"}')
    print()

    optimizer = Optimizer(
        datasets=datasets,
        train_frac=args.train_frac,
        instant_entry=args.instant_entry,
        use_sma=args.sma_indicators,
        fee_rate=args.fee_rate,
        max_dd_pct=args.max_dd_pct,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.NopPruner() if args.no_pruner else optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=max(1, len(datasets) // 3),
        interval_steps=1,
        n_min_trials=5,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True if args.storage else False,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(optimizer.objective, n_trials=args.trials, show_progress_bar=False)

    best_params = Params(**study.best_params, FEE_RATE=args.fee_rate, MAX_DRAWDOWN_PCT=args.max_dd_pct)
    best_train_df, best_valid_df = optimizer.evaluate_best(best_params)

    trials_rows = []
    for t in study.trials:
        row = {
            'number': t.number,
            'state': str(t.state),
            'value': t.value,
        }
        for k, v in t.params.items():
            row[k] = v

        train_summary = t.user_attrs.get('train_summary', {})
        valid_summary = t.user_attrs.get('valid_summary', {})
        for k, v in train_summary.items():
            row[f'train_{k}'] = v
        for k, v in valid_summary.items():
            row[f'valid_{k}'] = v
        trials_rows.append(row)

    trials_df = pd.DataFrame(trials_rows).sort_values(['value', 'number'], ascending=[False, True])

    trials_path = os.path.join(OUTPUT_DIR, 'optuna_trials.csv')
    best_params_path = os.path.join(OUTPUT_DIR, 'best_params.json')
    best_train_path = os.path.join(OUTPUT_DIR, 'best_train_summary.csv')
    best_valid_path = os.path.join(OUTPUT_DIR, 'best_validation_summary.csv')

    trials_df.to_csv(trials_path, index=False)
    best_train_df.to_csv(best_train_path, index=False)
    if not best_valid_df.empty:
        best_valid_df.to_csv(best_valid_path, index=False)

    payload = {
        'study_name': args.study_name,
        'best_value': study.best_value,
        'best_params': asdict(best_params),
        'best_train_median_score': float(np.median([
            score_metrics(r) for r in best_train_df.to_dict('records')
        ])) if not best_train_df.empty else None,
        'best_valid_median_score': float(np.median([
            score_metrics(r) for r in best_valid_df.to_dict('records')
        ])) if not best_valid_df.empty else None,
    }

    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('=' * 100)
    print('BEST TRIAL')
    print('=' * 100)
    print(f'Best value: {study.best_value:.6f}')
    print(json.dumps(study.best_params, indent=2))

    if not best_train_df.empty:
        print('\nTRAIN SUMMARY')
        cols = [
            'binance_symbol', 'regime', 'pass_status', 'positions',
            'expectancy_r', 'win_rate', 'net_r', 'profit_factor_r',
            'roi_pct', 'max_drawdown_pct', 'final_equity'
        ]
        print(best_train_df[cols].to_string(index=False))

    if not best_valid_df.empty:
        print('\nVALIDATION SUMMARY')
        cols = [
            'binance_symbol', 'regime', 'pass_status', 'positions',
            'expectancy_r', 'win_rate', 'net_r', 'profit_factor_r',
            'roi_pct', 'max_drawdown_pct', 'final_equity'
        ]
        print(best_valid_df[cols].to_string(index=False))

    print('\nFiles saved:')
    print(f'  {trials_path}')
    print(f'  {best_params_path}')
    print(f'  {best_train_path}')
    if not best_valid_df.empty:
        print(f'  {best_valid_path}')


if __name__ == '__main__':
    main()
