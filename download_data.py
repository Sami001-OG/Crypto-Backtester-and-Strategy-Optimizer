"""
Download BTC, SOL, ETH kline data from Binance Vision (data.binance.vision)
Timeframes: 5m, 15m, 1h, 4h
Period: Last 6 years (June 2020 - April 2026)
Uses only Python standard library - no pip packages needed.
"""

import urllib.request
import zipfile
import csv
import os
import io
import time

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SYMBOLS = ["BTCUSDT", "SOLUSDT", "ETHUSDT"]
INTERVALS = ["5m", "15m", "1h", "4h"]

# Build list of all months from June 2020 to April 2026
MONTHS = []
for y in range(2020, 2027):
    for m in range(1, 13):
        if (y == 2020 and m < 6):
            continue
        if (y == 2026 and m > 4):
            break
        MONTHS.append((str(y), f"{m:02d}"))

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base_vol",
    "taker_buy_quote_vol", "ignore"
]

total_months = len(MONTHS)
print(f"Will download {total_months} months per symbol/interval combo")
print(f"Total downloads: {len(SYMBOLS)} symbols x {len(INTERVALS)} intervals x {total_months} months = {len(SYMBOLS)*len(INTERVALS)*total_months}")


def download_and_extract(symbol, interval, year, month):
    """Download a single zip from Binance Vision and return CSV rows."""
    filename = f"{symbol}-{interval}-{year}-{month}"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}.zip"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=60)
        data = response.read()
    except Exception as e:
        print(f"    SKIP {year}-{month}: {e}")
        return []

    rows = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as csvfile:
                        reader = csv.reader(io.TextIOWrapper(csvfile, encoding="utf-8"))
                        for row in reader:
                            if row:
                                rows.append(row)
    except Exception as e:
        print(f"    SKIP {year}-{month} (extract error): {e}")
        return []

    return rows


def main():
    grand_total = 0
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            coin = symbol.replace("USDT", "")
            out_file = os.path.join(OUTPUT_DIR, f"{coin}_{interval}.csv")
            print(f"\n=== {coin} / {interval} ({total_months} months) ===")
            all_rows = []
            ok_count = 0

            for i, (year, month) in enumerate(MONTHS):
                rows = download_and_extract(symbol, interval, year, month)
                if rows:
                    all_rows.extend(rows)
                    ok_count += 1
                # Progress every 12 months
                if (i + 1) % 12 == 0:
                    print(f"    Progress: {i+1}/{total_months} months, {len(all_rows)} rows so far")

            if not all_rows:
                print(f"  No data for {coin} {interval}, skipping.")
                continue

            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                writer.writerows(all_rows)

            grand_total += len(all_rows)
            print(f"  => Saved {len(all_rows):,} rows ({ok_count}/{total_months} months) -> {coin}_{interval}.csv")

    print(f"\n=== ALL DONE ===")
    print(f"Grand total: {grand_total:,} rows across all files")
    print(f"Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
