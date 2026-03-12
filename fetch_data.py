import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


BASE_URL = "https://api.binance.com"
ENDPOINT = "/api/v3/klines"
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000
LIMIT = 1000
SLEEP_BETWEEN = 0.12
REQUEST_TIMEOUT = 20
MAX_RETRIES = 4

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT",
    "MATICUSDT", "ALGOUSDT", "ATOMUSDT", "DOTUSDT", "ADAUSDT",
    "LINKUSDT", "UNIUSDT", "APTUSDT", "SUIUSDT"
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]
KEEP_COLS = ["open_time", "open", "high", "low", "close", "volume"]

PRESET_REGIMES = [
    ("bull_2020_q4_2021_q1", "2020-10-01", "2021-04-15"),
    ("midcycle_correction_2021", "2021-04-16", "2021-07-31"),
    ("late_bull_2021_ath", "2021-08-01", "2021-11-15"),
    ("bear_2022_pre_ftx", "2022-04-01", "2022-10-31"),
    ("ftx_crash_2022", "2022-11-01", "2022-12-15"),
    ("recovery_2023_h1", "2023-01-01", "2023-06-30"),
    ("range_2023_h2", "2023-07-01", "2023-12-31"),
    ("etf_breakout_2024_q1", "2024-01-01", "2024-03-31"),
    ("post_ath_pullback_2024_q2", "2024-04-01", "2024-06-30"),
    ("current_times", "2025-04-01", "2026-02-28"),
]

log = logging.getLogger("fetch_data")


def setup_logging(verbosity: int, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    else:
        levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        level = levels[min(verbosity, len(levels) - 1)]

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def normalize_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=COLUMNS)[KEEP_COLS].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.rename(columns={
        "open_time": "datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }, inplace=True)
    df.set_index("datetime", inplace=True)
    df = df.astype({
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float,
    })
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def fetch_chunk(symbol: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.debug(
                "[%s] request attempt=%d start=%s end=%s",
                symbol, attempt, fmt_dt(from_ms(start_ms)), fmt_dt(from_ms(end_ms))
            )
            resp = requests.get(BASE_URL + ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            rows = resp.json()
            log.debug("[%s] response rows=%d", symbol, len(rows))
            return rows
        except requests.RequestException as e:
            last_err = e
            wait_s = min(2 ** attempt, 8)
            log.warning("[%s] request failed attempt %d/%d: %s", symbol, attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                raise
            log.info("[%s] retrying in %ss", symbol, wait_s)
            time.sleep(wait_s)

    raise last_err


def fetch_range(symbol: str, start_dt: datetime, end_dt: datetime, max_bars: int | None = None) -> pd.DataFrame:
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)

    all_rows = []
    page = 0

    log.info("[%s] range start=%s end=%s max_bars=%s",
             symbol, fmt_dt(start_dt), fmt_dt(end_dt), max_bars if max_bars else "none")

    while start_ms < end_ms:
        remaining = None if max_bars is None else max_bars - len(all_rows)
        if remaining is not None and remaining <= 0:
            log.info("[%s] reached range max bars", symbol)
            break

        rows = fetch_chunk(symbol, start_ms, end_ms)
        if not rows:
            log.info("[%s] no rows returned, ending range", symbol)
            break

        if remaining is not None and len(rows) > remaining:
            rows = rows[:remaining]

        all_rows.extend(rows)
        page += 1

        first_open = from_ms(rows[0][0])
        last_open = from_ms(rows[-1][0])

        log.info(
            "[%s] page=%d rows=%d cumulative=%d span=%s -> %s",
            symbol, page, len(rows), len(all_rows), fmt_dt(first_open), fmt_dt(last_open)
        )

        next_start_ms = rows[-1][0] + INTERVAL_MS
        if next_start_ms <= start_ms:
            log.warning("[%s] pagination stalled", symbol)
            break

        start_ms = next_start_ms

        if len(rows) < LIMIT:
            log.info("[%s] final page for this range", symbol)
            break

        time.sleep(SLEEP_BETWEEN)

    df = normalize_df(all_rows)

    if df.empty:
        log.warning("[%s] range finished with no data", symbol)
    else:
        log.info("[%s] range complete rows=%d first=%s last=%s",
                 symbol, len(df), df.index.min(), df.index.max())

    return df


def parse_custom_regimes(items: list[str] | None) -> list[tuple[str, datetime, datetime]]:
    if not items:
        return []

    regimes = []
    for item in items:
        name, start_s, end_s = item.split(":")
        start_dt = parse_day(start_s)
        end_dt = parse_day(end_s) + timedelta(days=1) - timedelta(milliseconds=1)
        if end_dt <= start_dt:
            raise ValueError(f"Invalid regime: {item}")
        regimes.append((name, start_dt, end_dt))
    return regimes


def get_regimes(custom_regimes: list[tuple[str, datetime, datetime]] | None):
    if custom_regimes:
        return custom_regimes

    return [
        (name, parse_day(start_s), parse_day(end_s) + timedelta(days=1) - timedelta(milliseconds=1))
        for name, start_s, end_s in PRESET_REGIMES
    ]


def main(symbols: list[str] | None, regimes: list[tuple[str, datetime, datetime]], max_bars_per_regime: int | None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.warning("=" * 88)
    log.warning("Binance multi-regime 5m fetcher")
    log.warning("Output dir           : %s", OUTPUT_DIR)
    log.warning("Interval             : %s", INTERVAL)
    log.warning("Symbols              : %d", len(symbols))
    log.warning("Regimes              : %d", len(regimes))
    log.warning("Max bars per regime  : %s", max_bars_per_regime if max_bars_per_regime else "none")
    for i, (name, start_dt, end_dt) in enumerate(regimes, start=1):
        log.warning("Regime %d             : %-24s %s -> %s",
                    i, name, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    log.warning("=" * 88)

    summary = []

    for s_idx, symbol in enumerate(symbols, start=1):
        symbol_dir = os.path.join(OUTPUT_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        log.warning("")
        log.warning("[%d/%d] SYMBOL %s", s_idx, len(symbols), symbol)

        for r_idx, (regime_name, start_dt, end_dt) in enumerate(regimes, start=1):
            out_file = f"{symbol}_{regime_name}_{INTERVAL}.csv.gz"
            out_path = os.path.join(symbol_dir, out_file)

            log.warning("[%s] regime %d/%d -> %s", symbol, r_idx, len(regimes), regime_name)
            log.info("[%s] output path: %s", symbol, out_path)

            df = fetch_range(symbol, start_dt, end_dt, max_bars=max_bars_per_regime)

            if df.empty:
                log.warning("[%s] %s returned no data", symbol, regime_name)
                summary.append({
                    "symbol": symbol,
                    "regime": regime_name,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "status": "NO_DATA",
                    "file": out_file,
                })
                continue

            log.info("[%s] saving %d rows for regime %s", symbol, len(df), regime_name)
            df.to_csv(out_path, compression="gzip")

            start_s = df.index.min().strftime("%Y-%m-%d %H:%M")
            end_s = df.index.max().strftime("%Y-%m-%d %H:%M")
            size_kb = os.path.getsize(out_path) // 1024

            log.warning("[%s] saved regime=%s rows=%d span=%s -> %s size=%d KB",
                        symbol, regime_name, len(df), start_s, end_s, size_kb)

            summary.append({
                "symbol": symbol,
                "regime": regime_name,
                "rows": len(df),
                "start": start_s,
                "end": end_s,
                "status": "OK",
                "file": out_file,
            })

            time.sleep(0.25)

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    pd.DataFrame(summary).to_csv(manifest_path, index=False)

    log.warning("")
    log.warning("=" * 88)
    log.warning("DONE")
    log.warning("Manifest: %s", manifest_path)
    log.warning("Saved datasets: %d", sum(1 for x in summary if x["status"] == "OK"))
    log.warning("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance multi-regime OHLCV downloader")

    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to fetch, e.g. BTCUSDT ETHUSDT SOLUSDT")
    parser.add_argument("--custom-regimes", nargs="+", default=None,
                        help="Custom regimes in format NAME:YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--max-bars-per-regime", type=int, default=None,
                        help="Optional hard cap per regime")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v info, -vv debug)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only show errors")

    args = parser.parse_args()
    setup_logging(args.verbose, args.quiet)

    custom = parse_custom_regimes(args.custom_regimes) if args.custom_regimes else None
    regimes = get_regimes(custom)

    try:
        main(
            symbols=args.symbols,
            regimes=regimes,
            max_bars_per_regime=args.max_bars_per_regime,
        )
    except KeyboardInterrupt:
        log.error("Interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        log.exception("Fatal error: %s", e)
        raise SystemExit(1)
