"""
Preprocess all crypto CSV files:
- Convert open_time/close_time from Unix ms to readable datetime
- Add 'datetime' column (alias for readable open_time)
- Add 'returns' column (percent change of close)
Run once: python preprocess_data.py
"""

import csv
import os
import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def ms_to_datetime(ms_str):
    """Convert Unix milliseconds (or microseconds) string to 'YYYY-MM-DD HH:MM:SS'."""
    # Already converted
    if "-" in str(ms_str) and ":" in str(ms_str):
        return ms_str
    try:
        val = int(float(ms_str))
        # Binance uses 13-digit ms timestamps, but some data has 16-digit (microseconds)
        if val > 1e15:
            val = val // 1000  # microseconds -> milliseconds
        if val > 1e12:
            val = val // 1000  # milliseconds -> seconds
        return datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return ms_str


def preprocess_file(filepath):
    fname = os.path.basename(filepath)
    print(f"Processing {fname}...")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        print(f"  Skipping {fname} — empty.")
        return

    # Check if already preprocessed — but also check if timestamps need fixing
    has_datetime = "datetime" in header
    needs_fix = False
    if has_datetime and rows:
        # Check if any open_time values are still numeric (unconverted)
        sample = rows[-1][header.index("open_time")]
        if sample.isdigit() or (sample.replace(".", "").isdigit()):
            needs_fix = True

    if has_datetime and not needs_fix:
        print(f"  {fname} already preprocessed, skipping.")
        return

    # Find column indices
    ot_idx = header.index("open_time")
    ct_idx = header.index("close_time")
    close_idx = header.index("close")

    # Build new rows
    if has_datetime:
        dt_idx = header.index("datetime")
        ret_idx = header.index("returns")
        new_header = header
    else:
        new_header = header + ["datetime", "returns"]
        dt_idx = None
        ret_idx = None

    new_rows = []
    prev_close = None

    for row in rows:
        # Convert timestamps
        row[ot_idx] = ms_to_datetime(row[ot_idx])
        row[ct_idx] = ms_to_datetime(row[ct_idx])

        dt = row[ot_idx]
        close = float(row[close_idx])

        if prev_close is not None and prev_close != 0:
            ret = f"{((close - prev_close) / prev_close) * 100:.6f}"
        else:
            ret = "0.000000"

        if dt_idx is not None:
            row[dt_idx] = dt
            row[ret_idx] = ret
        else:
            row.append(dt)
            row.append(ret)
        new_rows.append(row)
        prev_close = close

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    print(f"  Done: {len(new_rows):,} rows, added datetime + returns columns.")


def main():
    import re
    pattern = re.compile(r"^(BTC|ETH|SOL)_((5m|15m|1h|4h)(_1y)?)\.csv$")
    csv_files = [f for f in os.listdir(DATA_DIR) if pattern.match(f)]
    csv_files.sort()
    print(f"Found {len(csv_files)} CSV files to preprocess.\n")

    for fname in csv_files:
        preprocess_file(os.path.join(DATA_DIR, fname))

    print("\nAll files preprocessed.")


if __name__ == "__main__":
    main()
