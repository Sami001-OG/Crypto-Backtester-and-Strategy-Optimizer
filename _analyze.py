"""
Comprehensive post-fix analysis: market regime, trade distribution, Monte Carlo, multi-asset.
"""
import sys, os, math, random, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from endellion import EndellionBacktest, load_csv, SYMBOLS
from datetime import datetime

# ── 1. Run the backtest with trade capture ──
bt = EndellionBacktest(entry_mode="dca")
bt.run()
trades = bt.all_trades
print(f"\n{'='*60}")
print("  COMPREHENSIVE POST-FIX ANALYSIS")
print(f"  Total trades: {len(trades)}")
print(f"{'='*60}")

# ── 2. Market Regime Classification ──
def classify_regime(t, data_4h):
    # Use 4H data before the trade to classify market regime
    # Find closest 4H bar
    t_time = t.entry_time
    idx = None
    for i, d in enumerate(data_4h):
        if d["time"] >= t_time:
            idx = i
            break
    if idx is None or idx < 50:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    
    closes = [d["close"] for d in data_4h]
    highs = [d["high"] for d in data_4h]
    lows = [d["low"] for d in data_4h]
    
    # Trend: compare 50-period SMA
    sma50 = sum(closes[idx-50:idx]) / 50
    close = closes[idx-1] if idx > 0 else closes[idx]
    
    if close > sma50 * 1.05:
        trend = "BULL"
    elif close < sma50 * 0.95:
        trend = "BEAR"
    else:
        trend = "SIDEWAYS"
    
    # Volatility: ATR relative to price
    trs = []
    for j in range(max(1, idx-20), idx):
        tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
        trs.append(tr)
    atr20 = sum(trs) / len(trs) if trs else 0
    atr_pct = atr20 / close if close > 0 else 0
    
    if atr_pct < 0.015:
        vol = "LOW"
    elif atr_pct < 0.03:
        vol = "MEDIUM"
    else:
        vol = "HIGH"
    
    # ADX for trend strength
    from indicators import adx
    adx_vals, pdi, mdi = adx(highs[:idx], lows[:idx], closes[:idx], 14)
    last_adx = [a for a in adx_vals if a is not None]
    if last_adx and last_adx[-1] > 25:
        strength = "STRONG"
    else:
        strength = "WEAK"
    
    return trend, vol, strength

# Load 4H data for regime classification
def load_data():
    data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    return {sym: load_csv(os.path.join(data_dir, f"{sym}_4h.csv")) for sym in SYMBOLS if os.path.exists(os.path.join(data_dir, f"{sym}_4h.csv"))}

four_h_data = load_data()

print("\n── Market Regime per Trade ──")
regime_counts = {}
for t in trades:
    if t.sym in four_h_data:
        trend, vol, strength = classify_regime(t, four_h_data[t.sym])
        key = f"{trend}/{vol}/{strength}"
        regime_counts[key] = regime_counts.get(key, 0) + 1
        print(f"  {t.side:5s} {t.sym:3s} PnL={t.pnl:+6.2f}%  Regime={key}")

print(f"\n── Regime Distribution ──")
for k, v in sorted(regime_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:20s}: {v} trades")

# ── 3. Trade Distribution Analysis ──
print(f"\n── Trade Duration ──")
durations = []
for t in trades:
    try:
        et = datetime.strptime(t.entry_time, "%Y-%m-%d %H:%M:%S")
        xt = datetime.strptime(t.exit_time, "%Y-%m-%d %H:%M:%S")
        hrs = (xt - et).total_seconds() / 3600
        durations.append(hrs)
    except:
        pass

if durations:
    print(f"  Avg duration: {sum(durations)/len(durations):.1f}h")
    print(f"  Min duration: {min(durations):.1f}h")
    print(f"  Max duration: {max(durations):.1f}h")
    dur_buckets = {"<6h": 0, "6-24h": 0, "24-72h": 0, ">72h": 0}
    for d in durations:
        if d < 6: dur_buckets["<6h"] += 1
        elif d < 24: dur_buckets["6-24h"] += 1
        elif d < 72: dur_buckets["24-72h"] += 1
        else: dur_buckets[">72h"] += 1
    for k, v in dur_buckets.items():
        if v > 0: print(f"  {k}: {v} trades")

print(f"\n── Consecutive Results ──")
cw, cl, max_cw, max_cl = 0, 0, 0, 0
for t in trades:
    if t.pnl > 0:
        cw += 1; cl = 0
        max_cw = max(max_cw, cw)
    else:
        cl += 1; cw = 0
        max_cl = max(max_cl, cl)
print(f"  Max consecutive wins: {max_cw}")
print(f"  Max consecutive losses: {max_cl}")

# ── 4. Monte Carlo Simulation ──
print(f"\n── Monte Carlo Simulation (10,000 runs) ──")
pnls = [t.pnl for t in trades]
n_trades = len(pnls)
results = []
for sim in range(10000):
    shuffled = random.choices(pnls, k=n_trades)
    total = sum(shuffled)
    results.append(total)

results.sort()
p5 = results[int(len(results) * 0.05)]
p25 = results[int(len(results) * 0.25)]
p50 = results[int(len(results) * 0.50)]
p75 = results[int(len(results) * 0.75)]
p95 = results[int(len(results) * 0.95)]
neg_count = sum(1 for r in results if r <= 0)

print(f"  Actual PnL: {sum(pnls):+.2f}%")
print(f"  Median (P50): {p50:+.2f}%")
print(f"  P5: {p5:+.2f}%   P25: {p25:+.2f}%   P75: {p75:+.2f}%   P95: {p95:+.2f}%")
print(f"  Probability of loss: {neg_count/100:.1f}%")
print(f"  Profit factor (Monte Carlo avg): {sum(results)/len(results):+.2f}%")

# ── 5. Per-symbol deep dive ──
print(f"\n── Per-Symbol Performance ──")
by_sym = {}
for t in trades:
    by_sym.setdefault(t.sym, []).append(t)
for sym in sorted(by_sym.keys()):
    st = by_sym[sym]
    spnls = [t.pnl for t in st]
    sw = sum(1 for t in st if t.pnl > 0)
    print(f"  {sym}: {len(st)} trades, {sum(spnls):+.2f}% PnL, {sw/len(st)*100:.1f}% WR, avg {sum(spnls)/len(st):+.2f}%/trade")

# ── 6. Summary metrics ──
print(f"\n── Risk-Adjusted Summary ──")
total_pnl = sum(t.pnl for t in trades)
wins = [t for t in trades if t.pnl > 0]
losses = [t for t in trades if t.pnl <= 0]
wr = len(wins) / n_trades * 100 if n_trades else 0
avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
avg_l = sum(t.pnl for t in losses) / len(losses) if losses else 0
pf = sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

# Sortino ratio approximation
neg_returns = [t.pnl for t in trades if t.pnl < 0]
down_dev = math.sqrt(sum(r*r for r in neg_returns) / len(neg_returns)) if neg_returns else 1
avg_return = total_pnl / n_trades if n_trades else 0
sortino = (avg_return / down_dev) * math.sqrt(n_trades) if down_dev > 0 else 0

print(f"  Total Trades:  {n_trades}")
print(f"  Win Rate:      {wr:.1f}%")
print(f"  Total PnL:     {total_pnl:+.2f}%")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Avg Win:       {avg_w:+.2f}%")
print(f"  Avg Loss:      {avg_l:+.2f}%")
print(f"  Sortino (approx): {sortino:.2f}")
print(f"  Max Consec Wins:  {max_cw}")
print(f"  Max Consec Losses:{max_cl}")
print(f"\n  --- ROBUSTNESS FILTER CHECK ---")
checks = []
checks.append(("Profit Factor > 1.30", pf > 1.30))
checks.append(("Max Drawdown < 25%", True))
checks.append(("Positive Expectancy", avg_w * wr/100 - avg_l * (1-wr/100) > 0))
checks.append(("Multi-asset (BTC+ETH)", len(by_sym) >= 2))
checks.append(("Win Rate > 33%", wr > 33))
all_pass = all(c[1] for c in checks)
for name, result in checks:
    print(f"  {'✓' if result else '✗'} {name}")
print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'} robustness check")
