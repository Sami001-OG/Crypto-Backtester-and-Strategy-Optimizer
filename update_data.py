"""
Incremental kline updater — tops up existing OHLCV CSVs to the latest CLOSED
bar using Binance's public REST API (stdlib only, no pip packages).

- Preserves each file's exact schema (12-col Binance Vision dump or 6-col
  open_time..volume) and its timestamp unit (ms or µs — Vision switched to µs
  on 2025-01-01, so newer 12-col files carry µs tails).
- Never writes the still-forming candle: a bar is appended only if its close
  time is in the past.
- Verifies continuity (each appended bar opens exactly one interval after the
  previous) and refuses to append across a gap.

Usage:
    python update_data.py                      # update the default file set
    python update_data.py BTC_1h.csv 1h        # update one file
"""

import csv
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Primary endpoint + the public data mirror (works where api.binance.com is
# geo-restricted). Same response format.
ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]

INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

# (path, binance symbol, interval) — one canonical copy in the crypto_data
# root; nothing is duplicated into the package folder.
DEFAULT_TARGETS = [
    (os.path.join(ROOT, "BTC_5m.csv"), "BTCUSDT", "5m"),
    (os.path.join(ROOT, "BTC_15m.csv"), "BTCUSDT", "15m"),
    (os.path.join(ROOT, "BTC_1h.csv"), "BTCUSDT", "1h"),
    (os.path.join(ROOT, "BTC_4h.csv"), "BTCUSDT", "4h"),
    (os.path.join(ROOT, "BTC_1d.csv"), "BTCUSDT", "1d"),
]


def fetch_klines(symbol, interval, start_ms, limit=1000):
    """One /api/v3/klines page starting at start_ms (inclusive)."""
    qs = f"?symbol={symbol}&interval={interval}&startTime={start_ms}&limit={limit}"
    last_err = None
    for base in ENDPOINTS:
        try:
            req = urllib.request.Request(base + qs, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:                       # try the mirror
            last_err = e
    raise RuntimeError(f"all endpoints failed for {symbol} {interval}: {last_err}")


def fetch_missing(symbol, interval, after_open_ms, now_ms):
    """All CLOSED klines with open_time > after_open_ms, in order."""
    step = INTERVAL_MS[interval]
    out = []
    cursor = after_open_ms + step
    while cursor + step <= now_ms + step:            # candidate bar exists
        batch = fetch_klines(symbol, interval, cursor)
        if not batch:
            break
        for k in batch:
            open_ms, close_ms = int(k[0]), int(k[6])
            if open_ms <= after_open_ms:
                continue
            if close_ms >= now_ms:                   # still-forming candle
                return out
            out.append(k)
        cursor = int(batch[-1][0]) + step
        if len(batch) < 1000:
            break
        time.sleep(0.15)                             # stay well under rate limits
    return out


def detect_unit(open_time_str):
    """'ms' for 13-digit stamps, 'us' for 16-digit."""
    v = float(open_time_str)
    return "us" if v > 1e14 else "ms"


def read_tail(path):
    """Header, column count, last row of a CSV (memory-light line scan)."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        last_line = None
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        raise ValueError(f"{path} has no data rows")
    return header, last_line.rstrip("\n").split(",")


def to_row(k, ncols, unit):
    """Convert one API kline (ms) to the file's schema and timestamp unit."""
    open_ms, close_ms = int(k[0]), int(k[6])
    if unit == "us":
        open_t = open_ms * 1000
        close_t = (close_ms + 1) * 1000 - 1          # Binance µs convention: ...999999
    else:
        open_t, close_t = open_ms, close_ms
    full = [str(open_t), k[1], k[2], k[3], k[4], k[5],
            str(close_t), k[7], str(k[8]), k[9], k[10], str(k[11])]
    return full[:ncols]


def update_file(path, symbol, interval, now_ms):
    name = os.path.relpath(path, ROOT)
    if not os.path.exists(path):
        print(f"  {name:<55} MISSING — skipped (run download_data.py first)")
        return 0
    header, last = read_tail(path)
    ncols = len(header)
    unit = detect_unit(last[0])
    last_open = float(last[0])
    while last_open > 1e14:                          # µs → ms
        last_open /= 1000.0
    last_open_ms = int(last_open)

    klines = fetch_missing(symbol, interval, last_open_ms, now_ms)
    if not klines:
        print(f"  {name:<55} up to date")
        return 0

    # Continuity guard: first new bar must open exactly one interval after
    # the file's last bar (crypto trades 24/7 — any hole is a data problem).
    step = INTERVAL_MS[interval]
    if int(klines[0][0]) != last_open_ms + step:
        raise RuntimeError(
            f"{name}: gap between file end ({last_open_ms}) and first API bar "
            f"({klines[0][0]}) — refusing to append across a hole")
    for a, b in zip(klines, klines[1:]):
        if int(b[0]) - int(a[0]) != step:
            raise RuntimeError(f"{name}: API data has an internal gap at {a[0]}→{b[0]}")

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k in klines:
            w.writerow(to_row(k, ncols, unit))
    first_new = int(klines[0][0])
    last_new = int(klines[-1][0])
    print(f"  {name:<55} +{len(klines)} bars "
          f"({time.strftime('%Y-%m-%d %H:%M', time.gmtime(first_new/1000))} → "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(last_new/1000))} UTC)")
    return len(klines)


def main():
    now_ms = int(time.time() * 1000)
    if len(sys.argv) == 3:
        targets = [(os.path.join(HERE, sys.argv[1]), "BTCUSDT", sys.argv[2])]
    else:
        targets = DEFAULT_TARGETS
    print(f"Updating {len(targets)} file(s) to the last closed bar "
          f"({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(now_ms/1000))})")
    total = 0
    for path, symbol, interval in targets:
        total += update_file(path, symbol, interval, now_ms)
    print(f"Done — {total} bars appended.")


if __name__ == "__main__":
    main()
