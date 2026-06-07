"""
Optimization runner using cached precomputed data.
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from endellion import EndellionBacktest, BEST_CONFIG

CACHE_FILE = "_endellion_cache.pkl"

def load_cache():
    if os.path.exists(CACHE_FILE):
        print("  Loading cached data...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    print("  Computing indicators (first run)...")
    bt = EndellionBacktest()
    bt.run()
    cache = {
        "data": bt.data,
        "indicators": bt.indicators,
        "mappings": bt.mappings,
        "precalc": bt.precalc,
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Cached to disk.")
    return cache

def run_config(cache, config, entry_mode="dca"):
    bt = EndellionBacktest(config=config, entry_mode=entry_mode)
    bt.data = cache["data"]
    bt.indicators = cache["indicators"]
    bt.mappings = cache["mappings"]
    bt.precalc = cache["precalc"]
    bt.run(skip_setup=True)
    return bt.compute_metrics()

# ── Main ──
DEFAULT = BEST_CONFIG.copy()
cache = load_cache()

print(f"\n{'='*70}")
print(f"  PARAMETER SENSITIVITY (using cached indicators)")
print(f"  Default: {DEFAULT}")
print(f"{'='*70}")

default_m = run_config(cache, DEFAULT)
print(f"\n  Baseline: {default_m['trades']}t, {default_m['total_pnl']:+.1f}%, WR={default_m['win_rate']:.1f}%, PF={default_m['profit_factor']:.2f}")

# ── Conf threshold sweep ──
print(f"\n── conf_thresh (default={DEFAULT['conf_thresh']}) ──")
print(f"  {'Value':>6} {'Trades':>7} {'PnL%':>9} {'WR%':>7} {'PF':>7}")
for v in [35, 40, 45, 50, 55, 60]:
    cfg = DEFAULT.copy(); cfg["conf_thresh"] = v
    t0 = time.time()
    m = run_config(cache, cfg)
    dt = time.time() - t0
    print(f"  {v:>6} {m['trades']:>7} {m['total_pnl']:>+8.1f}% {m['win_rate']:>6.1f}% {m['profit_factor']:>6.2f} ({dt:.0f}s)")

# ── Pb sweep ──
print(f"\n── pb (default={DEFAULT['pb']}) ──")
print(f"  {'Value':>6} {'Trades':>7} {'PnL%':>9} {'WR%':>7} {'PF':>7}")
for v in [0.25, 0.35, 0.45, 0.55, 0.65]:
    cfg = DEFAULT.copy(); cfg["pb"] = v
    t0 = time.time()
    m = run_config(cache, cfg)
    dt = time.time() - t0
    print(f"  {v:>6} {m['trades']:>7} {m['total_pnl']:>+8.1f}% {m['win_rate']:>6.1f}% {m['profit_factor']:>6.2f} ({dt:.0f}s)")

# ── TP multiplier sweep ──
print(f"\n── TP mults (default={DEFAULT['tp1m']}/{DEFAULT['tp2m']}/{DEFAULT['tp3m']}) ──")
print(f"  {'Mult':>6} {'Trades':>7} {'PnL%':>9} {'WR%':>7} {'PF':>7}")
for mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
    cfg = DEFAULT.copy()
    cfg["tp1m"] = round(DEFAULT["tp1m"] * mult, 1)
    cfg["tp2m"] = round(DEFAULT["tp2m"] * mult, 1)
    cfg["tp3m"] = round(DEFAULT["tp3m"] * mult, 1)
    t0 = time.time()
    m = run_config(cache, cfg)
    dt = time.time() - t0
    print(f"  {mult:>5.1f}x {m['trades']:>7} {m['total_pnl']:>+8.1f}% {m['win_rate']:>6.1f}% {m['profit_factor']:>6.2f} ({dt:.0f}s)")

# ── Combo tests ──
print(f"\n── Combos ──")
print(f"  {'Name':<22} {'Trades':>7} {'PnL%':>9} {'WR%':>7} {'PF':>7}")
combos = [
    ("Default", DEFAULT),
    ("conf=40, PB=0.35", {**DEFAULT, "conf_thresh": 40, "pb": 0.35}),
    ("conf=50", {**DEFAULT, "conf_thresh": 50}),
    ("TP 1.1x, conf=40", {**DEFAULT, "tp1m": 2.0, "tp2m": 3.3, "tp3m": 5.5, "conf_thresh": 40}),
    ("TP 0.9x, conf=50", {**DEFAULT, "tp1m": 1.6, "tp2m": 2.7, "tp3m": 4.5, "conf_thresh": 50}),
    ("PB=0.35 only", {**DEFAULT, "pb": 0.35}),
    ("PB=0.55", {**DEFAULT, "pb": 0.55}),
    ("Aggro (conf=38)", {**DEFAULT, "conf_thresh": 38}),
    ("Conservative (52)", {**DEFAULT, "conf_thresh": 52}),
]
for name, cfg in combos:
    t0 = time.time()
    m = run_config(cache, cfg)
    dt = time.time() - t0
    print(f"  {name:<22} {m['trades']:>7} {m['total_pnl']:>+8.1f}% {m['win_rate']:>6.1f}% {m['profit_factor']:>6.2f} ({dt:.0f}s)")

print(f"\n  Done.")
