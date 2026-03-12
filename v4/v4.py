"""
backtest_v3_real_data.py
========================
Identical strategy logic to backtest_v3_professional.py but
uses REAL historical 5-minute OHLCV data fetched via yfinance.

Requirements:
    pip install yfinance pandas numpy

yfinance limitation: 5-min data is capped at 60 days per request.
This script fetches in 59-day chunks to cover a full year.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STARTING_CAPITAL = 10_000
LEVERAGE = 10
MAX_RISK_PCT = 0.0025          # 0.25% risk per trade
MAX_DRAWDOWN_PCT = 0.10        # 10% peak-to-trough hard limit
STRONG_BODY_PCT = 0.65
ADX_MIN = 20
RSI_LONG_MAX = 65
RSI_SHORT_MIN = 35
PULLBACK_LOOKBACK = 10
TIME_STOP_BARS = 96
MAX_STOP_PCT = 0.03
PARTIAL_R = 1.5
FINAL_R = 3.0
ATR_TRAIL_MULT = 1.5
ATR_STOP_CAP = 2.0
ATR_WICK_PAD = 0.1

# Yahoo Finance tickers  →  internal symbol names
ASSET_MAP = {
    'SOL-USD':  'SOLANA',
    'AVAX-USD': 'AVAX',
    'NEAR-USD': 'NEAR',
    'MATIC-USD':'MATIC',
    'ALGO-USD': 'ALGO',
    'ATOM-USD': 'ATOM',
    'DOT-USD':  'DOT',
    'ADA-USD':  'ADA',
    'LINK-USD': 'LINK',
    'UNI-USD':  'UNI',
    'APT-USD':  'APT',
    'SUI-USD':  'SUI',
}


# ══════════════════════════════════════════════════════════════════════════════
# REAL DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════
def fetch_real_data(ticker, days_back=365, interval='5m', chunk_days=59):
    """
    Fetch real historical OHLCV data from Yahoo Finance.
    yfinance allows max 60 days per request for 5m data,
    so we fetch in chunks and concatenate.
    Returns a clean DataFrame indexed by UTC datetime.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)
    frames = []

    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)
        print(f'    Fetching {ticker}  {chunk_start.strftime("%Y-%m-%d")} → {chunk_end.strftime("%Y-%m-%d")} ...', end=' ')
        try:
            raw = yf.download(
                tickers=ticker,
                start=chunk_start.strftime('%Y-%m-%d'),
                end=chunk_end.strftime('%Y-%m-%d'),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if len(raw) > 0:
                # Flatten multi-level column if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                frames.append(raw)
                print(f'{len(raw)} bars')
            else:
                print('0 bars (no data returned)')
        except Exception as e:
            print(f'ERROR: {e}')
        chunk_start = chunk_end + timedelta(minutes=5)
        time.sleep(0.5)   # polite rate limiting

    if not frames:
        print(f'  WARNING: No data available for {ticker}')
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = 'datetime'

    print(f'  → Total: {len(df):,} bars for {ticker}')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
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
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def is_strong_candle(row, min_body_pct=STRONG_BODY_PCT):
    candle_range = row['High'] - row['Low']
    if candle_range == 0:
        return False, None
    body_pct = abs(row['Close'] - row['Open']) / candle_range
    if body_pct >= min_body_pct:
        return True, 'bull' if row['Close'] > row['Open'] else 'bear'
    return False, None


def size_position(equity, entry, stop):
    stop_pct = abs(entry - stop) / entry
    if equity <= 0 or stop_pct <= 0:
        return 0.0, 0.0, 0.0, 0.0
    target_risk_usd = equity * MAX_RISK_PCT
    notional = min(target_risk_usd / stop_pct, equity * LEVERAGE)
    actual_risk_usd = notional * stop_pct
    leverage_used = notional / equity
    return notional, actual_risk_usd, leverage_used, stop_pct


def mark_to_market(closed_equity, position, mark_price):
    if position is None:
        return closed_equity
    if position['direction'] == 'long':
        pnl = position['notional_usd'] * (mark_price - position['entry']) / position['entry']
    else:
        pnl = position['notional_usd'] * (position['entry'] - mark_price) / position['entry']
    return closed_equity + pnl


def build_trade(symbol, position, exit_time, exit_price, r_result, pnl_usd, equity_after, exit_reason):
    return {
        'symbol':        symbol,
        'entry_time':    position['entry_time'],
        'exit_time':     exit_time,
        'direction':     position['direction'],
        'entry':         round(position['entry'], 6),
        'stop':          round(position['stop'], 6),
        'target_1':      round(position['target_1'], 6),
        'target_2':      round(position['target_2'], 6),
        'exit':          round(exit_price, 6),
        'R':             round(r_result, 4),
        'pnl_usd':       round(pnl_usd, 4),
        'notional_usd':  round(position['notional_usd'], 2),
        'risk_usd':      round(position['risk_usd'], 4),
        'stop_pct':      round(position['stop_pct'] * 100, 4),
        'leverage_used': round(position['leverage_used'], 4),
        'equity_after':  round(equity_after, 4),
        'exit_reason':   exit_reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def backtest_asset(df, symbol, starting_capital=STARTING_CAPITAL):
    trades       = []
    equity_marks = []
    position     = None

    closed_equity  = starting_capital
    peak_equity    = starting_capital
    worst_drawdown = 0.0
    failed         = False
    fail_reason    = None
    fail_time      = None
    halt_equity    = None

    def update_dd(ts, eq_value, mark_type):
        nonlocal peak_equity, worst_drawdown, failed, fail_reason, fail_time, halt_equity
        peak_equity = max(peak_equity, eq_value)
        dd = eq_value / peak_equity - 1.0 if peak_equity > 0 else -1.0
        worst_drawdown = min(worst_drawdown, dd)
        equity_marks.append({
            'symbol': symbol, 'time': ts,
            'equity': round(eq_value, 4),
            'peak_equity': round(peak_equity, 4),
            'drawdown_pct': round(dd * 100, 4),
            'mark_type': mark_type,
        })
        if not failed and dd <= -MAX_DRAWDOWN_PCT:
            failed     = True
            fail_reason = 'max_drawdown_breach'
            fail_time  = ts
            halt_equity = eq_value

    update_dd(df.index[0], closed_equity, 'start')

    for i in range(250, len(df)):
        row = df.iloc[i]

        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or pd.isna(row['ADX']) or \
           pd.isna(row['MA50']) or pd.isna(row['MA200']):
            continue

        # Mark-to-market check every bar for open position
        if position is not None:
            mtm = mark_to_market(closed_equity, position, row['Close'])
            update_dd(row.name, mtm, 'mtm')

            if failed:
                # Force close at current bar close
                e = position['entry']; rp = position['risk_price']
                notl = position['notional_usd']
                if position['direction'] == 'long':
                    pnl = notl * (row['Close'] - e) / e
                    r   = (row['Close'] - e) / rp
                else:
                    pnl = notl * (e - row['Close']) / e
                    r   = (e - row['Close']) / rp
                closed_equity = max(closed_equity + pnl, 0.0)
                trades.append(build_trade(symbol, position, row.name, row['Close'], r, pnl, closed_equity, 'dd_halt'))
                update_dd(row.name, closed_equity, 'closed_dd_halt')
                position = None
                break

        # Manage open position
        if position is not None:
            e    = position['entry']
            s    = position['stop']
            t1   = position['target_1']
            t2   = position['target_2']
            rp   = position['risk_price']
            notl = position['notional_usd']
            d    = position['direction']
            exit_trade = None

            if d == 'long':
                if row['Low'] <= s:
                    pnl = notl * (s - e) / e
                    r   = (s - e) / rp
                    exit_trade = build_trade(symbol, position, row.name, s, r, pnl, 0.0, 'stop')
                elif row['High'] >= t2:
                    pnl = notl * (t2 - e) / e
                    r   = (PARTIAL_R + FINAL_R) / 2 if position['partial_done'] else FINAL_R
                    exit_trade = build_trade(symbol, position, row.name, t2, r, pnl, 0.0, 'target')
                else:
                    if not position['partial_done'] and row['High'] >= t1:
                        position['partial_done'] = True
                        position['stop'] = e
                    if position['partial_done']:
                        trail = row['Close'] - ATR_TRAIL_MULT * row['ATR']
                        if trail > position['stop']:
                            position['stop'] = trail
                    if (i - position['entry_bar']) > TIME_STOP_BARS and row['Close'] < e + 0.3 * rp:
                        pnl = notl * (row['Close'] - e) / e
                        r   = (row['Close'] - e) / rp
                        exit_trade = build_trade(symbol, position, row.name, row['Close'], r, pnl, 0.0, 'timeout')
            else:
                if row['High'] >= s:
                    pnl = notl * (e - s) / e
                    r   = (e - s) / rp
                    exit_trade = build_trade(symbol, position, row.name, s, r, pnl, 0.0, 'stop')
                elif row['Low'] <= t2:
                    pnl = notl * (e - t2) / e
                    r   = (PARTIAL_R + FINAL_R) / 2 if position['partial_done'] else FINAL_R
                    exit_trade = build_trade(symbol, position, row.name, t2, r, pnl, 0.0, 'target')
                else:
                    if not position['partial_done'] and row['Low'] <= t1:
                        position['partial_done'] = True
                        position['stop'] = e
                    if position['partial_done']:
                        trail = row['Close'] + ATR_TRAIL_MULT * row['ATR']
                        if trail < position['stop']:
                            position['stop'] = trail
                    if (i - position['entry_bar']) > TIME_STOP_BARS and row['Close'] > e - 0.3 * rp:
                        pnl = notl * (e - row['Close']) / e
                        r   = (e - row['Close']) / rp
                        exit_trade = build_trade(symbol, position, row.name, row['Close'], r, pnl, 0.0, 'timeout')

            if exit_trade is not None:
                closed_equity = max(closed_equity + exit_trade['pnl_usd'], 0.0)
                exit_trade['equity_after'] = round(closed_equity, 4)
                trades.append(exit_trade)
                update_dd(row.name, closed_equity, 'closed_trade')
                position = None
                if failed:
                    break

        if failed:
            break

        # Entry logic
        if position is None:
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
                entry  = row['Close']
                stop   = max(row['Low'] - ATR_WICK_PAD * atr, entry - ATR_STOP_CAP * atr)
                rp     = entry - stop
                if rp <= 0 or rp / entry > MAX_STOP_PCT:
                    continue
                notl, ar, el, sp = size_position(closed_equity, entry, stop)
                if notl <= 0:
                    continue
                position = {
                    'direction': 'long', 'entry': entry, 'stop': stop, 'risk_price': rp,
                    'target_1': entry + PARTIAL_R * rp, 'target_2': entry + FINAL_R * rp,
                    'notional_usd': notl, 'risk_usd': ar, 'leverage_used': el, 'stop_pct': sp,
                    'entry_time': row.name, 'entry_bar': i, 'partial_done': False,
                }

            elif bear and cdir == 'bear':
                if row['RSI'] < RSI_SHORT_MIN:
                    continue
                if not (win['High'] >= win['MA50'] - atr).any():
                    continue
                if (row['MA50'] - row['Close']) / atr > 4:
                    continue
                entry  = row['Close']
                stop   = min(row['High'] + ATR_WICK_PAD * atr, entry + ATR_STOP_CAP * atr)
                rp     = stop - entry
                if rp <= 0 or rp / entry > MAX_STOP_PCT:
                    continue
                notl, ar, el, sp = size_position(closed_equity, entry, stop)
                if notl <= 0:
                    continue
                position = {
                    'direction': 'short', 'entry': entry, 'stop': stop, 'risk_price': rp,
                    'target_1': entry - PARTIAL_R * rp, 'target_2': entry - FINAL_R * rp,
                    'notional_usd': notl, 'risk_usd': ar, 'leverage_used': el, 'stop_pct': sp,
                    'entry_time': row.name, 'entry_bar': i, 'partial_done': False,
                }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_marks)
    run_info  = {
        'symbol':          symbol,
        'final_equity':    closed_equity,
        'peak_equity':     peak_equity,
        'max_drawdown_pct': worst_drawdown * 100,
        'failed':          failed,
        'pass_status':     'FAILED' if failed else 'PASSED',
        'fail_reason':     fail_reason,
        'fail_time':       str(fail_time),
        'halt_equity':     halt_equity,
        'total_trades':    len(trades_df),
    }
    return trades_df, equity_df, run_info


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def calc_metrics(trades_df, run_info, starting_capital=STARTING_CAPITAL):
    base = {
        **run_info,
        'win_rate': 0.0, 'avg_r': 0.0, 'net_r': 0.0,
        'profit_factor': 0.0,
        'roi_pct': (run_info['final_equity'] / starting_capital - 1) * 100,
        'calmar': 0.0, 'avg_notional_usd': 0.0, 'avg_leverage_used': 0.0,
        'long_count': 0, 'short_count': 0,
    }
    if len(trades_df) == 0:
        return base

    wins = (trades_df['R'] > 0).sum()
    gp   = trades_df.loc[trades_df['R'] > 0, 'R'].sum()
    gl   = trades_df.loc[trades_df['R'] < 0, 'R'].abs().sum()
    roi  = (run_info['final_equity'] / starting_capital - 1) * 100
    mdd  = abs(run_info['max_drawdown_pct'])

    return {
        **run_info,
        'win_rate':         wins / len(trades_df) * 100,
        'avg_r':            trades_df['R'].mean(),
        'net_r':            trades_df['R'].sum(),
        'profit_factor':    gp / gl if gl > 0 else 0.0,
        'roi_pct':          roi,
        'calmar':           roi / mdd if mdd > 0 else 0.0,
        'avg_notional_usd': trades_df['notional_usd'].mean(),
        'avg_leverage_used':trades_df['leverage_used'].mean(),
        'long_count':       (trades_df['direction'] == 'long').sum(),
        'short_count':      (trades_df['direction'] == 'short').sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    results    = []
    all_trades = []
    all_equity = []

    print('=' * 90)
    print('PROFESSIONAL BACKTEST  —  REAL MARKET DATA via yfinance  —  5-min OHLCV')
    print('=' * 90)
    print(f'Starting capital     : ${STARTING_CAPITAL:,.0f}')
    print(f'Leverage cap         : {LEVERAGE}x')
    print(f'Risk per trade       : {MAX_RISK_PCT * 100:.2f}% of current equity')
    print(f'Max drawdown         : {MAX_DRAWDOWN_PCT * 100:.0f}% peak-to-trough  →  hard FAIL')
    print(f'Data                 : real 5-min OHLCV  ~60 days per request, chunked to 1 year')
    print()

    for ticker, symbol in ASSET_MAP.items():
        print(f'\n{"─" * 70}')
        print(f'  {symbol}  ({ticker})')
        print(f'{"─" * 70}')
        df = fetch_real_data(ticker, days_back=365, interval='5m')

        if df.empty:
            print(f'  SKIP: no data returned for {ticker}')
            continue

        df = add_indicators(df)
        trades_df, equity_df, run_info = backtest_asset(df, symbol)
        metrics = calc_metrics(trades_df, run_info)
        results.append(metrics)

        if len(trades_df) > 0:
            all_trades.append(trades_df)
        if len(equity_df) > 0:
            all_equity.append(equity_df)

        print(f'  {metrics["pass_status"]:<6}  |  Trades: {metrics["total_trades"]:>4}  |  '
              f'ROI: {metrics["roi_pct"]:>8.2f}%  |  MaxDD: {metrics["max_drawdown_pct"]:>7.2f}%  |  '
              f'Final: ${metrics["final_equity"]:>10,.2f}')

    summary_df  = pd.DataFrame(results).sort_values(['pass_status', 'roi_pct'], ascending=[True, False])
    trades_df_a = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df_a = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()

    summary_df.to_csv('real_backtest_summary.csv', index=False)
    if not trades_df_a.empty:
        trades_df_a.to_csv('real_backtest_trades.csv', index=False)
    if not equity_df_a.empty:
        equity_df_a.to_csv('real_backtest_equity_marks.csv', index=False)

    print('\n' + '=' * 90)
    print('SUMMARY — ALL ASSETS')
    print('=' * 90)
    cols = ['symbol','pass_status','total_trades','win_rate','net_r','profit_factor',
            'roi_pct','max_drawdown_pct','calmar','final_equity']
    print(summary_df[cols].to_string(index=False))

    passed = summary_df[summary_df['pass_status'] == 'PASSED'].sort_values('roi_pct', ascending=False)
    failed = summary_df[summary_df['pass_status'] == 'FAILED']

    print('\n' + '=' * 90)
    print(f'PASSED  {len(passed)} assets  /  FAILED  {len(failed)} assets')
    print('=' * 90)

    print('\nTOP PERFORMERS — PASSED ONLY')
    if len(passed) == 0:
        print('  None.')
    else:
        print(passed[cols].head(5).to_string(index=False))

    print('\nFAILED — EXCLUDED FROM RANKING')
    if len(failed) == 0:
        print('  None.')
    else:
        print(failed[['symbol','roi_pct','max_drawdown_pct','fail_time','final_equity']].to_string(index=False))

    print('\nEXIT REASON BREAKDOWN')
    if not trades_df_a.empty:
        er = trades_df_a.groupby('exit_reason').agg(
            count       = ('R', 'count'),
            win_rate    = ('R', lambda x: f'{(x > 0).mean()*100:.1f}%'),
            avg_r       = ('R', 'mean'),
            total_pnl   = ('pnl_usd', 'sum'),
        )
        print(er.to_string())

    print('\nFiles saved:')
    print('  real_backtest_summary.csv')
    print('  real_backtest_trades.csv')
    print('  real_backtest_equity_marks.csv')


if __name__ == '__main__':
    main()
