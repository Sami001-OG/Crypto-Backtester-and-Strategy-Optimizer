"""
TradingView-parity verification suite.

Every expectation below is hand-computed from TradingView's documented
formulas (Pine v5 reference implementations and the Strategy Tester help
pages) — not from this codebase. Run:  python test_tv_parity.py
"""

import math
import indicators as ind
from backtester import Backtester

PASS = 0
FAIL = 0
FAILURES = []


def check(name, got, want, tol=1e-9):
    global PASS, FAIL
    ok = False
    if got is None and want is None:
        ok = True
    elif got is None or want is None:
        ok = False
    elif isinstance(want, float) or isinstance(got, float):
        if want == float("inf") or want == float("-inf"):
            ok = got == want
        else:
            ok = abs(got - want) <= tol * max(1.0, abs(want))
    else:
        ok = got == want
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"  FAIL {name}: got {got!r}, want {want!r}")


def bars(rows):
    """rows: (o, h, l, c) or (o, h, l, c, datetime)."""
    out = []
    for i, r in enumerate(rows):
        o, h, l, c = r[:4]
        dt = r[4] if len(r) > 4 else f"2024-01-{i+1:02d} 00:00:00"
        out.append({"datetime": dt, "open_time": "", "open": float(o),
                    "high": float(h), "low": float(l), "close": float(c),
                    "volume": 1.0, "close_time": "", "quote_volume": 0.0,
                    "trades": 0})
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1. Indicator parity (hand-computed Pine expectations)
# ══════════════════════════════════════════════════════════════════════════

def test_sma_ema_rma():
    # ta.sma(x, 3) on [1..5]
    s = ind.sma([1, 2, 3, 4, 5], 3)
    check("sma[0]", s[0], None)
    check("sma[2]", s[2], 2.0)
    check("sma[4]", s[4], 4.0)

    # ta.ema(x, 3): na,na,2 (SMA seed), then 0.5*src + 0.5*prev
    e = ind.ema([1, 2, 3, 4, 5], 3)
    check("ema seed", e[2], 2.0)
    check("ema[3]", e[3], 3.0)          # 0.5*4 + 0.5*2
    check("ema[4]", e[4], 4.0)          # 0.5*5 + 0.5*3

    # ta.rma(x, 3): seed 2, then (src + 2*prev)/3
    r = ind._rma([1, 2, 3, 4, 5], 3)
    check("rma seed", r[2], 2.0)
    check("rma[3]", r[3], 8.0 / 3.0)
    check("rma[4]", r[4], (5 + 2 * 8.0 / 3.0) / 3.0)

    # na-aware seeding: leading Nones shift the seed (Pine: na until a full
    # window of non-na values exists)
    e2 = ind.ema([None, None, 1, 2, 3, 4], 3)
    check("ema na-led seed idx", e2[3], None)
    check("ema na-led seed", e2[4], 2.0)
    check("ema na-led next", e2[5], 3.0)


def test_rsi():
    # RSI(2) on [1,2,3,2,4]; changes: +1,+1,-1,+2
    # rma(up,2):  seed@2 = 1.0 ; then 0.5*0+0.5*1 = 0.5 ; 0.5*2+0.5*0.5 = 1.25
    # rma(down,2): seed@2 = 0.0 ; then 0.5*1+0 = 0.5 ; 0.5*0+0.25 = 0.25
    r = ind.rsi([1, 2, 3, 2, 4], 2)
    check("rsi[1]", r[1], None)
    check("rsi[2]", r[2], 100.0)                      # avg loss 0 → 100
    check("rsi[3]", r[3], 50.0)                       # rs = 1
    check("rsi[4]", r[4], 100.0 - 100.0 / 6.0)        # rs = 5


def test_macd():
    # Signal line must seed like Pine: EMA over the first 9 non-na MACD values.
    closes = [float(100 + math.sin(i / 3.0) * 5) for i in range(60)]
    macd_l, sig_l, hist = ind.macd(closes, 12, 26, 9)
    first_macd = next(i for i, v in enumerate(macd_l) if v is not None)
    check("macd first idx", first_macd, 25)
    first_sig = next(i for i, v in enumerate(sig_l) if v is not None)
    check("signal first idx", first_sig, 25 + 9 - 1)
    seed = sum(macd_l[25:34]) / 9.0
    check("signal seed = SMA of first 9 macd", sig_l[33], seed)
    k = 2.0 / 10.0
    check("signal recursion", sig_l[34], k * macd_l[34] + (1 - k) * sig_l[33])


def test_atr_tr():
    h = [10.0, 12.0, 11.0]
    l = [9.0, 10.5, 9.5]
    c = [9.5, 11.0, 10.0]
    tr = ind.true_range(h, l, c, handle_na=True)
    check("tr[0] = high-low", tr[0], 1.0)
    check("tr[1]", tr[1], max(1.5, abs(12 - 9.5), abs(10.5 - 9.5)))  # 2.5
    a = ind.atr(h, l, c, 2)
    check("atr seed = sma(tr,2)", a[1], (1.0 + 2.5) / 2.0)
    # DMI's TR variable is na on bar 0 (ta.tr without handle_na)
    tr2 = ind.true_range(h, l, c, handle_na=False)
    check("dmi tr[0] = na", tr2[0], None)


def test_adx_structure():
    import random
    rng = random.Random(7)
    n = 120
    h, l, c = [], [], []
    px = 100.0
    for _ in range(n):
        px *= 1 + rng.uniform(-0.02, 0.02)
        hi = px * (1 + rng.uniform(0, 0.01))
        lo = px * (1 - rng.uniform(0, 0.01))
        h.append(hi); l.append(lo); c.append(px)
    period = 14
    adx_v, pdi, mdi = ind.adx(h, l, c, period)
    # trur = rma(tr with na@0) seeds on bars 1..14 → first DI at index 14.
    first_di = next(i for i, v in enumerate(pdi) if v is not None)
    check("first +DI idx = period", first_di, period)
    # adx = 100 * rma(dx, 14); dx starts at 14 → adx at 14+14-1 = 27.
    first_adx = next(i for i, v in enumerate(adx_v) if v is not None)
    check("first ADX idx = 2*period-1", first_adx, 2 * period - 1)
    # Reference recomputation, straight from the Pine source, independent code:
    tr = ind.true_range(h, l, c, handle_na=False)
    pdm = [None] * n
    mdm = [None] * n
    for i in range(1, n):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0

    def rma_ref(src, length):
        out = [None] * len(src)
        prev = None
        for i, v in enumerate(src):
            if v is None:
                continue
            if prev is None:
                window = src[i:i + length]
                if all(x is not None for x in window) and len(window) == length:
                    prev = sum(window) / length
                    out[i + length - 1] = prev
                    for j, w in enumerate(src[i + length:], start=i + length):
                        prev = (w + (length - 1) * prev) / length
                        out[j] = prev
                break
        return out

    trr = rma_ref(tr, period)
    prr = rma_ref(pdm, period)
    mrr = rma_ref(mdm, period)
    dx = [None] * n
    for i in range(n):
        if trr[i] is None or trr[i] == 0:
            continue
        p = 100 * prr[i] / trr[i]
        m = 100 * mrr[i] / trr[i]
        s = p + m
        dx[i] = abs(p - m) / (1.0 if s == 0 else s)
    adx_ref = rma_ref(dx, period)
    for i in (27, 60, 119):
        check(f"ADX[{i}] vs independent ref", adx_v[i], 100 * adx_ref[i], tol=1e-12)
    check("pdi[50] vs ref", pdi[50], 100 * prr[50] / trr[50], tol=1e-12)


def test_supertrend():
    # Hand-built: uptrending closes then a crash; period=2, mult=1.
    h = [11.0, 12.0, 13.0, 14.0, 15.0, 10.0]
    l = [9.0, 10.0, 11.0, 12.0, 13.0, 6.0]
    c = [10.0, 11.5, 12.5, 13.5, 14.5, 7.0]
    st, d = ind.supertrend(h, l, c, 2, 1.0)
    # atr(2): tr = [2, 2, 2, 2, 2, 7.5]; seed@1 = 2; rma after.
    check("st[0] na", st[0], None)
    # Seed bar i=1: hl2=11, bands 9..13, direction seeds DOWNTREND → st = upper.
    check("seed dir = downtrend", d[1], -1)
    check("seed st = upper band", st[1], 13.0)
    # i=2: atr=2, hl2=12 → basic upper 14; ratchet: 14 < 13? no; close[1]=11.5 > 13? no
    # → upper stays 13. close 12.5 > 13? no → still downtrend, st = 13.
    check("st[2] ratcheted upper", st[2], 13.0)
    check("dir[2]", d[2], -1)
    # i=3: atr=2, hl2=13 → basic upper 15 (stays 13); close 13.5 > 13 → FLIP UP.
    check("dir[3] flips up", d[3], 1)
    # lower band: basic = 11; prev lower ratchet ... st = final lower = 11.
    check("st[3] = lower band", st[3], 11.0)
    # i=4: atr=2, hl2=14 → basic lower 12 > 11 → ratchet up to 12; close 14.5 ≥ 12 stays up.
    check("st[4] lower ratchets up", st[4], 12.0)
    check("dir[4]", d[4], 1)
    # i=5: tr = max(10-6, |10-14.5|, |6-14.5|) = 8.5 → atr = (8.5 + 2)/2 = 5.25;
    # hl2=8 → basic upper 13.25. Upper had re-ratcheted to 16 at i=4 (close[3]
    # 13.5 broke above 13), so 13.25 < 16 → upper = 13.25. close 7 < lower(12)
    # → FLIP DOWN; st = upper = 13.25.
    check("dir[5] flips down", d[5], -1)
    check("st[5]", st[5], 13.25)


def test_stochastic():
    h = [10, 11, 12, 13, 14.0]
    l = [8, 9, 10, 11, 12.0]
    c = [9, 10, 11, 12, 13.0]
    k, d = ind.stochastic(h, l, c, 3, 3, 1)
    # raw K @2: hh=12, ll=8 → (11-8)/4*100 = 75; @3: hh=13,ll=9 → 75; @4: 75
    check("stoch k[2]", k[2], 75.0)
    check("stoch d first", d[2], None)
    check("stoch d[4]", d[4], 75.0)
    # zero-range window → na (Pine division by zero), not 50
    k2, _ = ind.stochastic([5, 5, 5], [5, 5, 5], [5, 5, 5], 3, 3, 1)
    check("stoch zero-range = na", k2[2], None)


def test_vwap_sessions():
    rows = [
        (10, 10, 10, 10, "2024-01-01 22:00:00"),
        (20, 20, 20, 20, "2024-01-01 23:00:00"),
        (30, 30, 30, 30, "2024-01-02 00:00:00"),   # new UTC day → reset
        (40, 40, 40, 40, "2024-01-02 01:00:00"),
    ]
    data = bars(rows)
    h = [r["high"] for r in data]; l = [r["low"] for r in data]
    c = [r["close"] for r in data]; v = [1.0] * 4
    vw = ind.vwap(h, l, c, v, anchors=ind.day_keys(data))
    check("vwap day1 bar2", vw[1], 15.0)
    check("vwap resets on new day", vw[2], 30.0)
    check("vwap day2 bar2", vw[3], 35.0)
    vw_cum = ind.vwap(h, l, c, v)      # anchors=None → cumulative
    check("vwap cumulative", vw_cum[3], 25.0)


def test_bollinger_keltner_donchian():
    vals = [1.0, 2, 3, 4, 5]
    u, m, lo = ind.bollinger_bands(vals, 3, 2.0)
    # window [3,4,5]: mean 4, population var = 2/3
    std = math.sqrt(2.0 / 3.0)
    check("bb mid", m[4], 4.0)
    check("bb upper (population std)", u[4], 4.0 + 2 * std)

    h = [10, 11, 12, 13, 14.0]; l = [8, 9, 10, 11, 12.0]; c = [9, 10, 11, 12, 13.0]
    ku, kb, kl = ind.keltner_channels(h, l, c, 3, 2.0, 2)
    # basis = EMA(close,3) — close source (TV default), NOT typical price
    e = ind.ema(c, 3)
    check("keltner basis = ema(close)", kb[3], e[3])

    du, dl, db = ind.donchian_channels(h, l, 3)
    check("donchian upper", du[4], 14.0)
    check("donchian lower", dl[4], 10.0)
    check("donchian basis", db[4], 12.0)


def test_pivots():
    highs = [1, 2, 5, 2, 1, 1, 1.0]
    lows = [1, 1, 1, 1, 0.5, 1, 1.0]
    ph, pl = ind.pivot_highs_lows(highs, lows, 2, 2)
    check("pivot high at bar 2", ph[2], 5.0)
    check("pivot low at bar 4", pl[4], 0.5)
    conf = ind.pivots_confirmed(ph, 2)
    check("pivot confirmed 2 bars later", conf[4], 5.0)
    check("pivot not visible early", conf[2], None)
    # ties are NOT pivots (strict comparison, like Pine)
    ph2, _ = ind.pivot_highs_lows([1, 3, 3, 1, 1.0], [1, 1, 1, 1, 1.0], 1, 1)
    check("tie is not a pivot", ph2[1], None)
    check("tie is not a pivot(2)", ph2[2], None)


# ══════════════════════════════════════════════════════════════════════════
# 2. Broker-emulator parity (hand-computed TV expectations)
# ══════════════════════════════════════════════════════════════════════════

def test_market_order_timing_and_sizing():
    # Signal on bar0 close → fill bar1 open. qty = 100% * equity / CLOSE of
    # the signal bar (TV sizes at order creation), NOT the fill price.
    data = bars([(100, 101, 99, 100), (102, 103, 101, 102), (102, 104, 100, 103)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0, long_only=True)
    res = bt.run([1, 1, 1])
    check("one open trade (no force close)", len(res["trades"]), 0)
    ot = res["open_trade"]
    check("entry price = bar1 open", ot["entry_price"], 102.0)
    check("qty = equity/close0", ot["qty"], 100.0)          # 10000/100
    # equity at end = cash + qty*close2 = (10000-10200) + 100*103 = 10100
    check("final equity", res["final_equity"], 10100.0)
    check("open pnl", res["open_pnl"], 100.0)
    check("net profit excludes open trade", res["net_profit"], 0.0)
    # Buy & hold: first entry at 102, last close 103
    check("buy&hold from first entry", res["buy_and_hold_return_pct"],
          (103 - 102) / 102 * 100)


def test_exit_fills_next_open():
    data = bars([(100, 101, 99, 100), (102, 103, 101, 102),
                 (104, 105, 103, 104), (106, 107, 105, 106)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0)
    res = bt.run([1, 1, 0, 0])       # exit signal at bar2 close → fill bar3 open
    check("one closed trade", len(res["trades"]), 1)
    t = res["trades"][0]
    check("exit at bar3 open", t["exit_price"], 106.0)
    check("exit reason", t["exit_reason"], "signal")
    check("pnl", t["pnl"], 100.0 * (106 - 102))


def test_gap_through_stop_fills_at_open():
    # Long from 100; SL 5% → 95. Bar 2 OPENS at 90 → TV fills at the open.
    data = bars([(100, 100, 100, 100), (100, 101, 99, 100), (90, 92, 88, 91)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0, stop_loss_pct=5)
    res = bt.run([1, 1, 1])
    t = res["trades"][0]
    check("gap stop fills at open", t["exit_price"], 90.0)
    check("gap stop reason", t["exit_reason"], "stop_loss")


def test_same_bar_sl_tp_path_low_first():
    # Entry bar: o=100 h=111 l=94. open is closer to the LOW (11 > 6) →
    # path O→L→H→C → the 5% stop (95) fills before the 10% target (110).
    data = bars([(100, 100, 100, 100), (100, 111, 94, 100)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0,
                    stop_loss_pct=5, take_profit_pct=10)
    res = bt.run([1, 1])
    t = res["trades"][0]
    check("low-first path → SL", t["exit_reason"], "stop_loss")
    check("SL fill at stop price", t["exit_price"], 95.0)
    check("same-bar exit (entry bar)", t["bars_held"], 0)


def test_same_bar_sl_tp_path_high_first():
    # o=100 h=105 l=90 → open closer to HIGH (5 <= 10) → path O→H→L→C →
    # the 4% target (104) fills before the 5% stop (95).
    data = bars([(100, 100, 100, 100), (100, 105, 90, 100)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0,
                    stop_loss_pct=5, take_profit_pct=4)
    res = bt.run([1, 1])
    t = res["trades"][0]
    check("high-first path → TP", t["exit_reason"], "take_profit")
    check("TP fill at limit price", t["exit_price"], 104.0)


def test_short_sl_tp():
    data = bars([(100, 100, 100, 100), (100, 107, 99, 106)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0,
                    stop_loss_pct=5, long_only=False)
    res = bt.run([-1, -1])
    t = res["trades"][0]
    check("short side", t["side"], "SHORT")
    check("short stop above entry", t["exit_price"], 105.0)
    check("short stop reason", t["exit_reason"], "stop_loss")
    check("short pnl", t["pnl"], 100.0 * (100 - 105))


def test_slippage_rules():
    # 2 ticks × 0.5 tick_size = 1.0 adverse. Market entry slips UP;
    # limit TP must NOT slip; stop SL slips DOWN (sell).
    data = bars([(100, 100, 100, 100), (100, 100, 100, 100),
                 (100, 112, 99, 100)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0,
                    slippage_ticks=2, tick_size=0.5, take_profit_pct=10)
    res = bt.run([1, 1, 1])
    t = res["trades"][0]
    check("entry slipped up", t["entry_price"], 101.0)
    check("TP = limit, no slip", t["exit_price"], 101.0 * 1.10)

    data2 = bars([(100, 100, 100, 100), (100, 100, 100, 100),
                  (100, 101, 90, 95)])
    bt2 = Backtester(data2, initial_capital=10000, fee_pct=0.0,
                     slippage_ticks=2, tick_size=0.5, stop_loss_pct=5)
    res2 = bt2.run([1, 1, 1])
    t2 = res2["trades"][0]
    # entry 101 → stop 95.95, in range; stop is slipped like a market order.
    check("stop slipped down", t2["exit_price"], 101.0 * 0.95 - 1.0)


def test_commission():
    data = bars([(100, 100, 100, 100), (100, 100, 100, 100),
                 (100, 100, 100, 100), (110, 110, 110, 110)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.1)
    res = bt.run([1, 1, 0, 0])
    t = res["trades"][0]
    qty = t["qty"]
    check("qty sized on signal close", qty, 100.0)   # 100% * 10000 / 100
    entry_fee = qty * 100 * 0.001
    exit_fee = qty * 110 * 0.001
    check("trade pnl net of both fees", t["pnl"], qty * 10 - entry_fee - exit_fee)
    check("commission_paid", res["commission_paid"], entry_fee + exit_fee)
    # equity when flat == cash: initial + net
    check("final equity realises fees", res["final_equity"],
          10000 + t["pnl"])


def test_reversal_sizing_includes_open_pnl():
    # Long 100 @ bar1 open (100). Signal flips short at bar2 close (c=120):
    # equity = cash + 100*120 = 10000 + 2000 = 12000 → short qty = 12000/120 = 100.
    # Both legs fill at bar3 open (121).
    data = bars([(100, 101, 99, 100), (100, 121, 99, 120),
                 (121, 122, 118, 119), (119, 120, 117, 118)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0, long_only=False)
    res = bt.run([1, -1, -1, -1])
    check("closed long", len(res["trades"]), 1)
    t = res["trades"][0]
    check("long exit at bar2 open... (reversal next open)", t["exit_price"], 121.0)
    ot = res["open_trade"]
    check("short entry same open", ot["entry_price"], 121.0)
    check("short qty from equity at signal close", ot["qty"], 100.0)


def test_qty_step():
    data = bars([(100, 100, 100, 100), (103, 103, 103, 103), (103, 103, 103, 103)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0, qty_step=1.0)
    res = bt.run([1, 1, 1])
    check("qty floored to step", res["open_trade"]["qty"], 100.0)  # 10000/100 = 100 exact
    bt2 = Backtester(data, initial_capital=10050, fee_pct=0.0, qty_step=1.0)
    res2 = bt2.run([1, 1, 1])
    check("qty floored down", res2["open_trade"]["qty"], 100.0)    # 100.5 → 100
    bt3 = Backtester(data, initial_capital=50, fee_pct=0.0, qty_step=1.0)
    res3 = bt3.run([1, 1, 1])
    check("sub-step order skipped", res3["open_trade"], None)


def test_intrabar_drawdown_tv_example():
    # TV's documented formula: buy 44 contracts @34.08 (fixed qty, no fees).
    # Bar lows 33.55 then 30.67 → max DD = 44*(34.08-30.67) = 150.04.
    data = bars([
        (34.00, 34.05, 33.90, 34.00),
        (34.08, 34.08, 33.55, 34.00),      # entry bar: dd 44*0.53 = 23.32
        (34.00, 34.05, 30.67, 31.00),      # dd 44*3.41 = 150.04
        (31.00, 31.50, 30.90, 31.20),
    ])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0,
                    qty_type="fixed", qty_value=44)
    res = bt.run([1, 1, 1, 1])
    check("TV intrabar max drawdown $", res["max_drawdown_dollar"],
          44 * (34.08 - 30.67), tol=1e-9)
    check("TV intrabar max DD %", res["max_drawdown_pct"],
          44 * (34.08 - 34.08) + 44 * (34.08 - 30.67) / 10000 * 100, tol=1e-9)


def test_intrabar_runup():
    data = bars([
        (100, 100, 100, 100),
        (100, 120, 99, 101),               # in position: high 120 seen intrabar
        (101, 102, 100, 101),
    ])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0)
    res = bt.run([1, 1, 1])
    # Bar1: open (100) is closer to the low (99) → assumed path O→L→H→C, so the
    # equity trough 9900 happens BEFORE the 12000 peak: run-up = 12000 - 9900.
    check("run-up uses intrabar high", res["max_runup_dollar"], 2100.0)


def test_monthly_sharpe_sortino():
    # Daily bars across 4 calendar months; equity moves only via close-to-close
    # position P&L. We hand-compute monthly equity returns → TV Sharpe/Sortino.
    rows = []
    # Month 1 (Jan): flat at 10000 (no trade yet; signal fires at month end).
    for d in range(1, 32):
        rows.append((100, 100, 100, 100, f"2024-01-{d:02d} 00:00:00"))
    # Month 2 (Feb): in position, price 100 → 110. Feb 1 opens at exactly 100
    # so the fill price equals the sizing price and equity gains exactly +10%.
    for i, d in enumerate(range(1, 30)):
        px = 100 + i * 10.0 / 28.0
        rows.append((px, px, px, px, f"2024-02-{d:02d} 00:00:00"))
    # Month 3 (Mar): price 110 → 99 (equity 11000 → 9900 = exactly -10%).
    for i, d in enumerate(range(1, 32)):
        px = 110 - i * 11.0 / 30.0
        rows.append((px, px, px, px, f"2024-03-{d:02d} 00:00:00"))
    # Month 4 (Apr): one bar so March is a COMPLETED month.
    rows.append((99, 99, 99, 99, "2024-04-01 00:00:00"))
    data = bars(rows)
    sigs = [0] * 30 + [1] * (len(rows) - 30)   # signal on Jan 31 → fill Feb 1
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0)
    res = bt.run(sigs)
    # Monthly returns: Jan 0%, Feb +10%, Mar -10%. April incomplete → excluded.
    m = res["monthly_returns"]
    check("3 completed months", len(m), 3)
    check("jan", m[0], 0.0)
    check("feb", m[1], 0.10, tol=1e-12)
    check("mar", m[2], -0.10, tol=1e-12)
    mean = (0.0 + 0.10 - 0.10) / 3.0
    var = (0.0 - mean) ** 2 + (0.10 - mean) ** 2 + (-0.10 - mean) ** 2
    std = math.sqrt(var / 3.0)                      # population stdev (TV)
    rfr = 0.02 / 12.0
    check("TV sharpe", res["sharpe_ratio"], (mean - rfr) / std, tol=1e-9)
    down_dev = math.sqrt((-0.10) ** 2 / 3.0)        # negatives only, / N
    check("TV sortino", res["sortino_ratio"], (mean - rfr) / down_dev, tol=1e-9)


def test_breakeven_and_percent_profitable():
    data = bars([(100, 100, 100, 100), (100, 100, 100, 100),
                 (100, 100, 100, 100),                       # trade 1: breakeven
                 (100, 100, 100, 100), (110, 110, 110, 110),  # trade 2 entry @110
                 (115, 115, 115, 115)])                       # trade 2 exit @115: win
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0)
    res = bt.run([1, 1, 0, 1, 0, 0])
    check("two closed trades", res["num_trades"], 2)
    check("one win", res["num_winning"], 1)
    check("breakeven is NOT a loss", res["num_losing"], 0)
    check("breakeven counted", res["num_even"], 1)
    check("percent profitable = wins/total", res["win_rate_pct"], 50.0)


def test_process_orders_on_close():
    data = bars([(100, 101, 99, 100), (102, 103, 101, 102)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0,
                    process_orders_on_close=True)
    res = bt.run([1, 1])
    check("fills on the signal close", res["open_trade"]["entry_price"], 100.0)


def test_force_close_end():
    data = bars([(100, 100, 100, 100), (100, 100, 100, 100), (110, 110, 110, 110)])
    bt = Backtester(data, initial_capital=10000, fee_pct=0.0, force_close_end=True)
    res = bt.run([1, 1, 1])
    check("force-closed at last close", len(res["trades"]), 1)
    check("end_of_data reason", res["trades"][0]["exit_reason"], "end_of_data")
    check("no open trade", res["open_trade"], None)


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print(f)
    return FAIL == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
