import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STARTING_CAPITAL = 10_000
LEVERAGE = 10
MAX_RISK_PCT = 0.0025          # 0.25% risk per trade
MAX_DRAWDOWN_PCT = 0.10        # 10% max peak-to-trough drawdown per asset
STRONG_BODY_PCT = 0.65
ADX_MIN = 20
RSI_LONG_MAX = 65
RSI_SHORT_MIN = 35
PULLBACK_LOOKBACK = 10
TIME_STOP_BARS = 96            # 8 hours on 5-min data
MAX_STOP_PCT = 0.03            # skip trades with >3% stop distance
PARTIAL_R = 1.5
FINAL_R = 3.0
ATR_TRAIL_MULT = 1.5
ATR_STOP_CAP = 2.0
ATR_WICK_PAD = 0.1
N_PERIODS = 52_560
SEED = 42

ASSETS = [
    'SOLANA', 'AVAX', 'NEAR', 'MATIC', 'ALGO', 'ATOM',
    'DOT', 'ADA', 'LINK', 'UNI', 'APT', 'SUI'
]

PARAMS = {
    'SOLANA': {'price': 180, 'volatility': 0.035, 'trend': 0.00025},
    'AVAX':   {'price': 38,  'volatility': 0.038, 'trend': 0.00030},
    'NEAR':   {'price': 6.5, 'volatility': 0.040, 'trend': 0.00028},
    'MATIC':  {'price': 0.95,'volatility': 0.036, 'trend': 0.00022},
    'ALGO':   {'price': 0.35,'volatility': 0.034, 'trend': 0.00020},
    'ATOM':   {'price': 11.5,'volatility': 0.037, 'trend': 0.00024},
    'DOT':    {'price': 7.8, 'volatility': 0.036, 'trend': 0.00026},
    'ADA':    {'price': 0.92,'volatility': 0.033, 'trend': 0.00021},
    'LINK':   {'price': 22,  'volatility': 0.035, 'trend': 0.00023},
    'UNI':    {'price': 13,  'volatility': 0.039, 'trend': 0.00027},
    'APT':    {'price': 12,  'volatility': 0.042, 'trend': 0.00031},
    'SUI':    {'price': 3.5, 'volatility': 0.041, 'trend': 0.00029},
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(symbol, n_periods=N_PERIODS, seed=SEED):
    np.random.seed(seed + hash(symbol) % 1000)
    p = PARAMS[symbol]
    base_price = p['price']
    vol = p['volatility']
    trend = p['trend']

    prices = [base_price]
    for _ in range(n_periods):
        change = trend * base_price + np.random.normal(0, vol * base_price)
        mean_reversion = -0.001 * (prices[-1] - base_price)
        new_price = max(prices[-1] + change + mean_reversion, base_price * 0.1)
        prices.append(new_price)

    data = []
    start_time = datetime.now() - timedelta(days=365)
    for i in range(n_periods):
        close = prices[i + 1]
        open_price = prices[i]
        range_size = abs(np.random.normal(0, vol * abs(close)))
        if close > open_price:
            high = close + np.random.uniform(0, range_size * 0.3)
            low = open_price - np.random.uniform(0, range_size * 0.3)
        else:
            high = open_price + np.random.uniform(0, range_size * 0.3)
            low = close - np.random.uniform(0, range_size * 0.3)
        data.append({
            'Open': open_price,
            'High': max(high, open_price, close),
            'Low': min(low, open_price, close),
            'Close': close,
        })

    df = pd.DataFrame(data)
    df.index = pd.date_range(start=start_time, periods=n_periods, freq='5min')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / tr14
    minus_di = 100 * minus_dm.rolling(14).sum() / tr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
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
    uncapped_notional = target_risk_usd / stop_pct
    max_notional = equity * LEVERAGE
    notional = min(uncapped_notional, max_notional)
    actual_risk_usd = notional * stop_pct
    leverage_used = notional / equity if equity > 0 else 0.0
    return notional, actual_risk_usd, leverage_used, stop_pct


def mark_to_market_equity(closed_equity, position, mark_price):
    if position is None:
        return closed_equity
    if position['direction'] == 'long':
        unrealized = position['notional_usd'] * (mark_price - position['entry']) / position['entry']
    else:
        unrealized = position['notional_usd'] * (position['entry'] - mark_price) / position['entry']
    return closed_equity + unrealized


def build_trade_record(symbol, position, exit_time, exit_price, r_result, pnl_usd, equity_after, exit_reason):
    return {
        'symbol': symbol,
        'entry_time': position['entry_time'],
        'exit_time': exit_time,
        'direction': position['direction'],
        'entry': round(position['entry'], 6),
        'stop': round(position['stop'], 6),
        'target_1': round(position['target_1'], 6),
        'target_2': round(position['target_2'], 6),
        'exit': round(exit_price, 6),
        'R': round(r_result, 4),
        'pnl_usd': round(pnl_usd, 4),
        'notional_usd': round(position['notional_usd'], 2),
        'risk_usd': round(position['risk_usd'], 4),
        'stop_pct': round(position['stop_pct'] * 100, 4),
        'leverage_used': round(position['leverage_used'], 4),
        'equity_after': round(equity_after, 4),
        'exit_reason': exit_reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def backtest_asset(df, symbol, starting_capital=STARTING_CAPITAL):
    trades = []
    equity_marks = []
    position = None

    closed_equity = starting_capital
    peak_equity = starting_capital
    worst_drawdown = 0.0
    failed = False
    fail_reason = None
    fail_time = None
    halt_equity = None

    def update_drawdown(mark_time, equity_value, mark_type):
        nonlocal peak_equity, worst_drawdown, failed, fail_reason, fail_time, halt_equity
        peak_equity = max(peak_equity, equity_value)
        dd = equity_value / peak_equity - 1.0 if peak_equity > 0 else -1.0
        worst_drawdown = min(worst_drawdown, dd)
        equity_marks.append({
            'symbol': symbol,
            'time': mark_time,
            'equity': round(equity_value, 4),
            'peak_equity': round(peak_equity, 4),
            'drawdown_pct': round(dd * 100, 4),
            'mark_type': mark_type,
        })
        if dd <= -MAX_DRAWDOWN_PCT and not failed:
            failed = True
            fail_reason = 'max_drawdown_breach'
            fail_time = mark_time
            halt_equity = equity_value
        return dd

    first_time = df.index[0]
    update_drawdown(first_time, closed_equity, 'start')

    for i in range(250, len(df)):
        row = df.iloc[i]
        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or pd.isna(row['ADX']) or pd.isna(row['MA50']) or pd.isna(row['MA200']):
            continue

        # Mark-to-market risk control for open position before any new entry decisions.
        if position is not None:
            mtm_equity = mark_to_market_equity(closed_equity, position, row['Close'])
            update_drawdown(row.name, mtm_equity, 'mtm')
            if failed:
                if position['direction'] == 'long':
                    pnl_usd = position['notional_usd'] * (row['Close'] - position['entry']) / position['entry']
                    r_result = (row['Close'] - position['entry']) / position['risk_price']
                else:
                    pnl_usd = position['notional_usd'] * (position['entry'] - row['Close']) / position['entry']
                    r_result = (position['entry'] - row['Close']) / position['risk_price']
                closed_equity = max(closed_equity + pnl_usd, 0.0)
                trades.append(build_trade_record(symbol, position, row.name, row['Close'], r_result, pnl_usd, closed_equity, 'dd_halt'))
                update_drawdown(row.name, closed_equity, 'closed_after_dd_halt')
                position = None
                break

        # Manage open position.
        if position is not None:
            exit_trade = None
            e = position['entry']
            s = position['stop']
            t1 = position['target_1']
            t2 = position['target_2']
            rp = position['risk_price']
            notional = position['notional_usd']

            if position['direction'] == 'long':
                if row['Low'] <= s:
                    exit_price = s
                    pnl_usd = notional * (exit_price - e) / e
                    r_result = (exit_price - e) / rp
                    exit_trade = build_trade_record(symbol, position, row.name, exit_price, r_result, pnl_usd, 0.0, 'stop')
                elif row['High'] >= t2:
                    exit_price = t2
                    pnl_usd = notional * (exit_price - e) / e
                    r_result = (PARTIAL_R + FINAL_R) / 2 if position['partial_done'] else FINAL_R
                    exit_trade = build_trade_record(symbol, position, row.name, exit_price, r_result, pnl_usd, 0.0, 'target')
                else:
                    if (not position['partial_done']) and row['High'] >= t1:
                        position['partial_done'] = True
                        position['stop'] = e
                    if position['partial_done']:
                        trail = row['Close'] - ATR_TRAIL_MULT * row['ATR']
                        if trail > position['stop']:
                            position['stop'] = trail
                    if (i - position['entry_bar']) > TIME_STOP_BARS and row['Close'] < e + 0.3 * rp:
                        exit_price = row['Close']
                        pnl_usd = notional * (exit_price - e) / e
                        r_result = (exit_price - e) / rp
                        exit_trade = build_trade_record(symbol, position, row.name, exit_price, r_result, pnl_usd, 0.0, 'timeout')
            else:
                if row['High'] >= s:
                    exit_price = s
                    pnl_usd = notional * (e - exit_price) / e
                    r_result = (e - exit_price) / rp
                    exit_trade = build_trade_record(symbol, position, row.name, exit_price, r_result, pnl_usd, 0.0, 'stop')
                elif row['Low'] <= t2:
                    exit_price = t2
                    pnl_usd = notional * (e - exit_price) / e
                    r_result = (PARTIAL_R + FINAL_R) / 2 if position['partial_done'] else FINAL_R
                    exit_trade = build_trade_record(symbol, position, row.name, exit_price, r_result, pnl_usd, 0.0, 'target')
                else:
                    if (not position['partial_done']) and row['Low'] <= t1:
                        position['partial_done'] = True
                        position['stop'] = e
                    if position['partial_done']:
                        trail = row['Close'] + ATR_TRAIL_MULT * row['ATR']
                        if trail < position['stop']:
                            position['stop'] = trail
                    if (i - position['entry_bar']) > TIME_STOP_BARS and row['Close'] > e - 0.3 * rp:
                        exit_price = row['Close']
                        pnl_usd = notional * (e - exit_price) / e
                        r_result = (e - exit_price) / rp
                        exit_trade = build_trade_record(symbol, position, row.name, exit_price, r_result, pnl_usd, 0.0, 'timeout')

            if exit_trade is not None:
                closed_equity = max(closed_equity + exit_trade['pnl_usd'], 0.0)
                exit_trade['equity_after'] = round(closed_equity, 4)
                trades.append(exit_trade)
                update_drawdown(row.name, closed_equity, 'closed_trade')
                position = None
                if failed:
                    break

        if failed:
            break

        # Entry logic.
        if position is None:
            bullish_regime = (row['Close'] > row['MA200']) and (row['MA50'] > row['MA200'])
            bearish_regime = (row['Close'] < row['MA200']) and (row['MA50'] < row['MA200'])
            if row['ADX'] < ADX_MIN:
                continue

            strong, candle_dir = is_strong_candle(row)
            if not strong:
                continue

            atr = row['ATR']
            lookback_window = df.iloc[i - PULLBACK_LOOKBACK:i]

            if bullish_regime and candle_dir == 'bull':
                if row['RSI'] > RSI_LONG_MAX:
                    continue
                if not (lookback_window['Low'] <= lookback_window['MA50'] + atr).any():
                    continue
                if (row['Close'] - row['MA50']) / atr > 4:
                    continue

                entry = row['Close']
                stop = max(row['Low'] - ATR_WICK_PAD * atr, entry - ATR_STOP_CAP * atr)
                risk_price = entry - stop
                if risk_price <= 0 or risk_price / entry > MAX_STOP_PCT:
                    continue

                notional_usd, risk_usd, leverage_used, stop_pct = size_position(closed_equity, entry, stop)
                if notional_usd <= 0:
                    continue

                position = {
                    'direction': 'long',
                    'entry': entry,
                    'stop': stop,
                    'risk_price': risk_price,
                    'target_1': entry + PARTIAL_R * risk_price,
                    'target_2': entry + FINAL_R * risk_price,
                    'notional_usd': notional_usd,
                    'risk_usd': risk_usd,
                    'leverage_used': leverage_used,
                    'stop_pct': stop_pct,
                    'entry_time': row.name,
                    'entry_bar': i,
                    'partial_done': False,
                }

            elif bearish_regime and candle_dir == 'bear':
                if row['RSI'] < RSI_SHORT_MIN:
                    continue
                if not (lookback_window['High'] >= lookback_window['MA50'] - atr).any():
                    continue
                if (row['MA50'] - row['Close']) / atr > 4:
                    continue

                entry = row['Close']
                stop = min(row['High'] + ATR_WICK_PAD * atr, entry + ATR_STOP_CAP * atr)
                risk_price = stop - entry
                if risk_price <= 0 or risk_price / entry > MAX_STOP_PCT:
                    continue

                notional_usd, risk_usd, leverage_used, stop_pct = size_position(closed_equity, entry, stop)
                if notional_usd <= 0:
                    continue

                position = {
                    'direction': 'short',
                    'entry': entry,
                    'stop': stop,
                    'risk_price': risk_price,
                    'target_1': entry - PARTIAL_R * risk_price,
                    'target_2': entry - FINAL_R * risk_price,
                    'notional_usd': notional_usd,
                    'risk_usd': risk_usd,
                    'leverage_used': leverage_used,
                    'stop_pct': stop_pct,
                    'entry_time': row.name,
                    'entry_bar': i,
                    'partial_done': False,
                }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_marks)

    result = {
        'symbol': symbol,
        'final_equity': closed_equity,
        'peak_equity': peak_equity,
        'max_drawdown_pct': worst_drawdown * 100,
        'failed': failed,
        'pass_status': 'FAILED' if failed else 'PASSED',
        'fail_reason': fail_reason,
        'fail_time': fail_time,
        'halt_equity': halt_equity,
        'total_trades': len(trades_df),
    }
    return trades_df, equity_df, result


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def calc_metrics(trades_df, run_info, starting_capital=STARTING_CAPITAL):
    if len(trades_df) == 0:
        return {
            **run_info,
            'win_rate': 0.0,
            'avg_r': 0.0,
            'net_r': 0.0,
            'profit_factor': 0.0,
            'roi_pct': (run_info['final_equity'] / starting_capital - 1) * 100,
            'calmar': 0.0,
            'avg_notional_usd': 0.0,
            'avg_leverage_used': 0.0,
            'long_count': 0,
            'short_count': 0,
        }

    wins = (trades_df['R'] > 0).sum()
    gross_profit = trades_df.loc[trades_df['R'] > 0, 'R'].sum()
    gross_loss = trades_df.loc[trades_df['R'] < 0, 'R'].abs().sum()
    roi_pct = (run_info['final_equity'] / starting_capital - 1) * 100
    max_dd_abs = abs(run_info['max_drawdown_pct'])
    calmar = roi_pct / max_dd_abs if max_dd_abs > 0 else 0.0

    return {
        **run_info,
        'win_rate': wins / len(trades_df) * 100,
        'avg_r': trades_df['R'].mean(),
        'net_r': trades_df['R'].sum(),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0.0,
        'roi_pct': roi_pct,
        'calmar': calmar,
        'avg_notional_usd': trades_df['notional_usd'].mean(),
        'avg_leverage_used': trades_df['leverage_used'].mean(),
        'long_count': (trades_df['direction'] == 'long').sum(),
        'short_count': (trades_df['direction'] == 'short').sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    results = []
    all_trades = []
    all_equity = []

    print('=' * 90)
    print('PROFESSIONAL BACKTEST — CORRECTED RISK, DRAWDOWN, AND PASS/FAIL LOGIC')
    print('=' * 90)
    print(f'Starting capital      : ${STARTING_CAPITAL:,.0f}')
    print(f'Leverage cap          : {LEVERAGE}x')
    print(f'Risk per trade        : {MAX_RISK_PCT * 100:.2f}% of current equity')
    print(f'Max drawdown allowed  : {MAX_DRAWDOWN_PCT * 100:.0f}% peak-to-trough')
    print(f'Fail policy           : Hard FAIL once drawdown limit is breached')
    print(f'Data                  : 1 year of 5-minute synthetic OHLC per asset')
    print()
    print('Processing...')
    print()

    for symbol in ASSETS:
        print(f'{symbol:.<18}', end=' ', flush=True)
        df = add_indicators(generate_synthetic_data(symbol))
        trades_df, equity_df, run_info = backtest_asset(df, symbol)
        metrics = calc_metrics(trades_df, run_info)
        results.append(metrics)
        if len(trades_df) > 0:
            all_trades.append(trades_df)
        if len(equity_df) > 0:
            all_equity.append(equity_df)
        print(f"{metrics['pass_status']:<6} | Trades: {metrics['total_trades']:>4} | ROI: {metrics['roi_pct']:>8.2f}% | "
              f"MaxDD: {metrics['max_drawdown_pct']:>7.2f}% | Final: ${metrics['final_equity']:>10,.2f}")

    summary_df = pd.DataFrame(results).sort_values(['pass_status', 'roi_pct'], ascending=[True, False])
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()

    summary_cols = [
        'symbol', 'pass_status', 'failed', 'fail_reason', 'fail_time', 'total_trades',
        'win_rate', 'avg_r', 'net_r', 'profit_factor', 'roi_pct', 'max_drawdown_pct',
        'calmar', 'final_equity', 'peak_equity', 'avg_notional_usd', 'avg_leverage_used',
        'long_count', 'short_count'
    ]
    summary_df = summary_df[summary_cols]

    summary_df.to_csv('professional_backtest_summary.csv', index=False)
    if len(trades_df) > 0:
        trades_df.to_csv('professional_backtest_trades.csv', index=False)
    if len(equity_df) > 0:
        equity_df.to_csv('professional_backtest_equity_marks.csv', index=False)

    print('\n' + '=' * 90)
    print('SUMMARY — ALL ASSETS')
    print('=' * 90)
    print(summary_df.to_string(index=False))

    passed_df = summary_df[summary_df['pass_status'] == 'PASSED'].sort_values('roi_pct', ascending=False)
    failed_df = summary_df[summary_df['pass_status'] == 'FAILED'].sort_values('max_drawdown_pct')

    print('\n' + '=' * 90)
    print('PASS / FAIL')
    print('=' * 90)
    print(f'Passed assets : {len(passed_df)}')
    print(f'Failed assets : {len(failed_df)}')

    print('\n' + '=' * 90)
    print('TOP PERFORMERS — PASSED ASSETS ONLY')
    print('=' * 90)
    if len(passed_df) == 0:
        print('No assets passed the 10% max drawdown rule.')
    else:
        print(passed_df.head(5).to_string(index=False))

    print('\n' + '=' * 90)
    print('FAILED ASSETS — EXCLUDED FROM RANKING')
    print('=' * 90)
    if len(failed_df) == 0:
        print('No failed assets.')
    else:
        fail_view = failed_df[['symbol', 'roi_pct', 'max_drawdown_pct', 'fail_reason', 'fail_time', 'final_equity']]
        print(fail_view.to_string(index=False))

    print('\n' + '=' * 90)
    print('EXIT REASON BREAKDOWN')
    print('=' * 90)
    if len(trades_df) > 0:
        exit_stats = trades_df.groupby('exit_reason').agg(
            count=('R', 'count'),
            win_rate=('R', lambda x: f"{(x > 0).mean() * 100:.1f}%"),
            avg_r=('R', 'mean'),
            total_pnl_usd=('pnl_usd', 'sum')
        )
        print(exit_stats.to_string())
    else:
        print('No trades executed.')

    print('\nFiles saved:')
    print(' - professional_backtest_summary.csv')
    print(' - professional_backtest_trades.csv')
    print(' - professional_backtest_equity_marks.csv')


if __name__ == '__main__':
    main()
