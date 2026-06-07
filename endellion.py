"""
Endellion Multi-Timeframe Trading Strategy — Full Implementation

Entry logic (modified per user request):
  - Signal triggers -> entry executes immediately at close price (100% of signals taken)
  - If price later reaches the PB (pullback) level, a DCA add of equal size is triggered
  - Average entry = (close_entry + pb_entry) / 2
  - TPs/SL recalculated from the averaged entry
"""

import os
import csv
import math
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indicators as ind

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["4h", "1h", "15m", "5m"]
TF_MIN_BARS = {"4h": 200, "1h": 100, "15m": 200, "5m": 50}

BEST_CONFIG = {
    "conf_thresh": 45,
    "use_btc_filter": True,
    "strict_htf": True,
    "htf_neutral_skip": True,
    "use_1h_control": True,
    "pb": 0.45,
    "tp1m": 1.8,
    "tp2m": 3.0,
    "tp3m": 5.0,
    "a1": 0.5,
    "a2": 0.3,
}


def compute_15m_heavy_arrays(I):
    """Precompute expensive per-bar arrays for 15m analysis only."""
    n = len(I["close"])
    closes = I["close"]; highs = I["high"]; lows = I["low"]
    opens = I["open"]; volumes = I["volume"]
    bb_w = I["bb_width"]

    # BB 20th percentile (50-bar window) and squeeze
    bb_20p = [None] * n
    if n >= 50:
        from bisect import insort
        valid_bw = [x for x in bb_w[:50] if x is not None]
        window_sorted = sorted(valid_bw)
        for j in range(49, n):
            if window_sorted:
                p20_idx = max(0, int(len(window_sorted) * 0.2) - 1)
                bb_20p[j] = window_sorted[p20_idx]
            if j + 1 < n:
                old_v = bb_w[j - 49]; new_v = bb_w[j + 1]
                if old_v is not None:
                    try:
                        idx = window_sorted.index(old_v)
                        window_sorted.pop(idx)
                    except ValueError:
                        pass
                if new_v is not None:
                    insort(window_sorted, new_v)
    I["bb_20p_width"] = bb_20p
    squeeze = [False] * n
    for j in range(49, n):
        if bb_w[j] is not None and bb_20p[j] is not None:
            squeeze[j] = bb_w[j] < bb_20p[j]
    I["squeeze"] = squeeze

    # Precompute tr_vals once for reuse
    tr_all = [0.0] * n
    for j in range(1, n):
        tr_all[j] = max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))

    # RSI divergence & MACD divergence combined (0=neutral, 1=bullish, -1=bearish)
    rsi_div = [0] * n
    macd_div = [0] * n
    for j in range(50, n):
        ps = closes[j - 50:j + 1]
        vs = I["rsi14"][j - 50:j + 1]
        hv = I["macd_5_34_5_hist"][j - 50:j + 1]
        r_pivots = []
        for k in range(5, 45):
            if k >= 46 - 5:
                break
            pk = ps[k]
            if all(pk >= ps[k - kk] and pk >= ps[k + kk] for kk in range(1, 6)):
                r_pivots.append((pk, vs[k], "high"))
            if all(pk <= ps[k - kk] and pk <= ps[k + kk] for kk in range(1, 6)):
                r_pivots.append((pk, vs[k], "low"))
        if len(r_pivots) >= 2:
            lp, lv, lt = r_pivots[-1]; pp, pv, pt = r_pivots[-2]
            if lt == "low" and pt == "low" and lp < pp and lv > pv:
                rsi_div[j] = 1
            elif lt == "high" and pt == "high" and lp > pp and lv < pv:
                rsi_div[j] = -1

        m_pivots = []
        for k in range(1, len(hv) - 2):
            vk = hv[k]; vk_1 = hv[k - 1]; vk1 = hv[k + 1]
            if vk is not None and vk_1 is not None and vk1 is not None:
                if (vk > vk_1 and vk > vk1) or (vk < vk_1 and vk < vk1):
                    m_pivots.append((vk, k))
        last_h = hv[-1]; prev_h = hv[-2]
        if len(m_pivots) >= 2:
            lv, li = m_pivots[-1]; pv, pi = m_pivots[-2]
            if lv < 0 and lv < pv:
                macd_div[j] = 1 if ps[li] < ps[pi] else -1
            elif lv > 0 and lv > pv:
                macd_div[j] = -1 if ps[li] < ps[pi] else 1
        if macd_div[j] == 0:
            if last_h is not None and prev_h is not None:
                if last_h > 0 and prev_h < 0:
                    macd_div[j] = 1
                elif last_h < 0 and prev_h > 0:
                    macd_div[j] = -1
    I["rsi_divergence"] = rsi_div
    I["macd_divergence"] = macd_div

    # LG / SFP / FG with direct indexing (no double slices)
    lg_arr = [0] * n; sfp_arr = [0] * n; fg_arr = [0] * n
    for j in range(25, n):
        w0 = j - 25  # window start
        # avg_atr and avg_vol for this window
        if j >= 39:
            atr_sum = 0
            for t in range(j - 13, j + 1):
                atr_sum += tr_all[t]
            avg_atr = atr_sum / 14.0
        else:
            atr_vals = [tr_all[t] for t in range(w0 + 1, j + 1)]
            avg_atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0
        avg_vol = sum(volumes[w0:j + 1]) / 26.0

        # LG
        for s in range(1, min(5, j + 1)):
            idx = j - s
            if idx < 16:
                break
            lb_start = max(w0, idx - 15)
            lb_lo = min(lows[lb_start:idx])
            lb_hi = max(highs[lb_start:idx])
            bl = lows[idx]; bh = highs[idx]
            bc = closes[idx]; bo = opens[idx]
            if bl < lb_lo and bc > lb_lo:
                has_disp = (bc > bo and (bc - bo) > avg_atr * 0.6) or (j > idx and closes[j] > bh and volumes[j] > avg_vol * 0.8)
                if has_disp:
                    lg_arr[j] = 1; break
            if bh > lb_hi and bc < lb_hi:
                has_disp = (bo > bc and (bo - bc) > avg_atr * 0.6) or (j > idx and closes[j] < bl and volumes[j] > avg_vol * 0.8)
                if has_disp:
                    lg_arr[j] = -1; break

        # SFP
        if j >= 17:
            s_start = max(w0, j - 17)
            s_end = max(s_start + 1, j - 2)
            if s_start < s_end:
                sfp_lo = min(lows[s_start:s_end])
                sfp_hi = max(highs[s_start:s_end])
                lc = closes[j]
                if lows[j] < sfp_lo and lc > sfp_lo:
                    sfp_arr[j] = 1
                elif highs[j] > sfp_hi and lc < sfp_hi:
                    sfp_arr[j] = -1

        # FG
        for s in range(2, min(6, j + 1)):
            idx = j - s
            if idx < 16:
                break
            lb_start = max(w0, idx - 15)
            lb_lo = min(lows[lb_start:idx])
            lb_hi = max(highs[lb_start:idx])
            bl = lows[idx]; bh = highs[idx]; bc = closes[idx]
            if bl < lb_lo and bc > lb_lo:
                for k in range(idx + 1, j):
                    if closes[k] < bl:
                        fg_arr[j] = -1; break
                if fg_arr[j] != 0: break
            if bh > lb_hi and bc < lb_hi:
                for k in range(idx + 1, j):
                    if closes[k] > bh:
                        fg_arr[j] = 1; break
                if fg_arr[j] != 0: break
    I["lg"] = lg_arr; I["sfp"] = sfp_arr; I["fg"] = fg_arr

    # Sweep cooldown
    cooldown_arr = [False] * n
    for j in range(16, n):
        for s in range(2, 5):
            sc_idx = j - s
            if sc_idx < 16:
                continue
            lb_lo = min(lows[sc_idx - 15:sc_idx])
            lb_hi = max(highs[sc_idx - 15:sc_idx])
            sc_l = lows[sc_idx]; sc_h = highs[sc_idx]; sc_c = closes[sc_idx]
            if sc_l < lb_lo and sc_c > lb_lo:
                if not (closes[j] >= sc_c and lows[j - 2] >= sc_c and closes[j] > sc_c):
                    cooldown_arr[j] = True; break
            if sc_h > lb_hi and sc_c < lb_hi:
                if not (closes[j] <= sc_c and highs[j - 2] <= sc_c and closes[j] < sc_c):
                    cooldown_arr[j] = True; break
    I["sweep_cooldown"] = cooldown_arr

    return I


def load_csv(filepath, max_bars=None):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        ci = {name: idx for idx, name in enumerate(header)}
        t_col = ci.get("datetime", ci.get("open_time", 0))
        o_col = ci["open"]
        h_col = ci["high"]
        l_col = ci["low"]
        c_col = ci["close"]
        v_col = ci["volume"]
        for line in f:
            parts = line.strip().split(",")
            rows.append({
                "time": parts[t_col],
                "open": float(parts[o_col]),
                "high": float(parts[h_col]),
                "low": float(parts[l_col]),
                "close": float(parts[c_col]),
                "volume": float(parts[v_col]),
            })
    if max_bars and len(rows) > max_bars:
        rows = rows[-max_bars:]
    return rows


def time_to_epoch(t_str):
    return int(datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").timestamp())


# ─── Extended indicator computations ───

def compute_all_indicators(data):
    n = len(data)
    closes = [d["close"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    opens = [d["open"] for d in data]
    volumes = [d["volume"] for d in data]

    I = {}

    I["ema9"] = ind.ema(closes, 9)
    I["ema10"] = ind.ema(closes, 10)
    I["ema20"] = ind.ema(closes, 20)
    I["ema21"] = ind.ema(closes, 21)
    I["ema30"] = ind.ema(closes, 30)
    I["ema50"] = ind.ema(closes, 50)
    I["ema100"] = ind.ema(closes, 100)
    I["ema200"] = ind.ema(closes, 200)
    I["rsi14"] = ind.rsi(closes, 14)
    I["rsi21"] = ind.rsi(closes, 21)
    I["atr14"] = ind.atr(highs, lows, closes, 14)
    I["atr7"] = ind.atr(highs, lows, closes, 7)

    _, _, hist_12_26_9 = ind.macd(closes, 12, 26, 9)
    I["macd_hist"] = hist_12_26_9

    _, _, hist_5_34_5 = ind.macd(closes, 5, 34, 5)
    I["macd_5_34_5_hist"] = hist_5_34_5

    bb_u, bb_m, bb_l = ind.bollinger_bands(closes, 30, 2.0)
    I["bb_30_2_upper"] = bb_u
    I["bb_30_2_middle"] = bb_m
    I["bb_30_2_lower"] = bb_l

    adx14, pdi14, mdi14 = ind.adx(highs, lows, closes, 14)
    I["adx_14_adx"] = adx14
    I["adx_14_pdi"] = pdi14
    I["adx_14_mdi"] = mdi14

    adx7, pdi7, mdi7 = ind.adx(highs, lows, closes, 7)
    I["adx_7_adx"] = adx7
    I["adx_7_pdi"] = pdi7
    I["adx_7_mdi"] = mdi7

    adx20, pdi20, mdi20 = ind.adx(highs, lows, closes, 20)
    I["adx_20_adx"] = adx20
    I["adx_20_pdi"] = pdi20
    I["adx_20_mdi"] = mdi20

    I["obv"] = ind.obv(closes, volumes)
    I["vol_sma_20"] = ind.volume_sma(volumes, 20)

    # Swing highs/lows: 4-left, 2-right
    swing_high = [None] * n
    swing_low = [None] * n
    for i in range(4, n - 2):
        if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and
            highs[i] >= highs[i-3] and highs[i] >= highs[i-4] and
            highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
            swing_high[i] = highs[i]
        if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and
            lows[i] <= lows[i-3] and lows[i] <= lows[i-4] and
            lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
            swing_low[i] = lows[i]
    I["swing_high"] = swing_high
    I["swing_low"] = swing_low

    # BOS
    bos = [0] * n
    last_sh = None
    last_sl = None
    for i in range(n):
        if swing_high[i] is not None:
            last_sh = swing_high[i]
        if swing_low[i] is not None:
            last_sl = swing_low[i]
        if i >= 8:
            if last_sh is not None and closes[i] > last_sh:
                bos[i] = 1
                last_sh = highs[i]
            elif last_sl is not None and closes[i] < last_sl:
                bos[i] = -1
                last_sl = lows[i]
    I["bos"] = bos

    # MSS swings: 4-left, 3-right
    mss_high = [None] * n
    mss_low = [None] * n
    for i in range(4, n - 3):
        if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and
            highs[i] >= highs[i-3] and highs[i] >= highs[i-4] and
            highs[i] >= highs[i+1] and highs[i] >= highs[i+2] and highs[i] >= highs[i+3]):
            mss_high[i] = highs[i]
        if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and
            lows[i] <= lows[i-3] and lows[i] <= lows[i-4] and
            lows[i] <= lows[i+1] and lows[i] <= lows[i+2] and lows[i] <= lows[i+3]):
            mss_low[i] = lows[i]
    I["mss_high"] = mss_high
    I["mss_low"] = mss_low

    mss = [0] * n
    last_mss_sh = None
    last_mss_sl = None
    for i in range(n):
        if mss_high[i] is not None:
            last_mss_sh = mss_high[i]
        if mss_low[i] is not None:
            last_mss_sl = mss_low[i]
        if i >= 10:
            if last_mss_sh is not None and closes[i] > last_mss_sh:
                mss[i] = 1
                last_mss_sh = highs[i]
            elif last_mss_sl is not None and closes[i] < last_mss_sl:
                mss[i] = -1
                last_mss_sl = lows[i]
    I["mss"] = mss

    # SuperTrend(7, 3)
    st_val, st_dir = ind.supertrend(highs, lows, closes, 7, 3.0)
    I["supertrend_value"] = st_val
    I["supertrend_trend"] = st_dir

    # Volume Profile (50-bar rolling) — fixed grid for O(1) sliding window
    vp_poc = [None] * n
    vp_vah = [None] * n
    vp_val = [None] * n
    if n > 50:
        vp_bins_n = 30
        min_vp = min(lows)
        max_vp = max(highs)
        vp_range = max_vp - min_vp
        if vp_range == 0:
            vp_range = 1
        vp_bin_sz = vp_range / vp_bins_n
        bin_idx_lookup = [0] * n
        for j in range(n):
            mid = (highs[j] + lows[j]) / 2.0
            bidx = int((mid - min_vp) / vp_bin_sz)
            bin_idx_lookup[j] = max(0, min(vp_bins_n - 1, bidx))
        bins = [0.0] * vp_bins_n
        for j in range(50):
            bins[bin_idx_lookup[j]] += volumes[j]
        def _vp_value_area(b, bi, bsz, mn):
            tv = sum(b)
            if tv <= 0:
                return None, None, None
            target = tv * 0.7
            acc = b[bi]
            lo = hi = bi
            while acc < target and (lo > 0 or hi < vp_bins_n - 1):
                if lo > 0:
                    lo -= 1; acc += b[lo]
                if acc >= target: break
                if hi < vp_bins_n - 1:
                    hi += 1; acc += b[hi]
            return (mn + (bi * bsz) + bsz / 2.0,
                    mn + (hi * bsz) + bsz,
                    mn + (lo * bsz))
        poc = max(range(vp_bins_n), key=lambda b: bins[b])
        pc, va_h, va_l = _vp_value_area(bins, poc, vp_bin_sz, min_vp)
        vp_poc[49] = pc; vp_vah[49] = va_h; vp_val[49] = va_l
        for i in range(50, n):
            bins[bin_idx_lookup[i - 50]] -= volumes[i - 50]
            bins[bin_idx_lookup[i]] += volumes[i]
            if i % 50 == 0:
                bins = [0.0] * vp_bins_n
                for j in range(i - 49, i + 1):
                    bins[bin_idx_lookup[j]] += volumes[j]
            poc = max(range(vp_bins_n), key=lambda b: bins[b])
            pc, va_h, va_l = _vp_value_area(bins, poc, vp_bin_sz, min_vp)
            vp_poc[i] = pc; vp_vah[i] = va_h; vp_val[i] = va_l
    I["vp_poc"] = vp_poc
    I["vp_vah"] = vp_vah
    I["vp_val"] = vp_val

    # Order Flow (14-period) — precomputed cumulative sums for O(1) per bar
    order_flow = ["neutral"] * n
    if n > 14:
        buy_vol = [0.0] * n
        sell_vol = [0.0] * n
        for j in range(n):
            r = highs[j] - lows[j]
            if r == 0:
                buy_vol[j] = volumes[j] / 2.0
                sell_vol[j] = volumes[j] / 2.0
            else:
                buy_vol[j] = volumes[j] * (closes[j] - lows[j]) / r
                sell_vol[j] = volumes[j] * (highs[j] - closes[j]) / r
        cum_buy = [0.0] * (n + 1)
        cum_sell = [0.0] * (n + 1)
        for j in range(n):
            cum_buy[j + 1] = cum_buy[j] + buy_vol[j]
            cum_sell[j + 1] = cum_sell[j] + sell_vol[j]
        for i in range(14, n):
            buying = cum_buy[i + 1] - cum_buy[i - 14]
            selling = cum_sell[i + 1] - cum_sell[i - 14]
            ratio = buying / (buying + selling) if (buying + selling) > 0 else 0.5
            if ratio > 0.55:
                order_flow[i] = "bullish"
            elif ratio < 0.45:
                order_flow[i] = "bearish"
            else:
                order_flow[i] = "neutral"
    I["order_flow"] = order_flow

    # ── Precomputed arrays for fast per-bar analysis ──
    # BB width array
    bb_w = [None] * n
    for j in range(n):
        bu = I["bb_30_2_upper"][j]; bm = I["bb_30_2_middle"][j]; bl = I["bb_30_2_lower"][j]
        if bu is not None and bm is not None and bl is not None and bm != 0:
            bb_w[j] = (bu - bl) / bm
    I["bb_width"] = bb_w

    # Rolling recent_high_20 / recent_low_20 (monotonic deques for O(1))
    rh20 = [None] * n; rl20 = [None] * n
    from collections import deque
    max_q, min_q = deque(), deque()
    for j in range(n):
        c = closes[j]
        while max_q and max_q[-1][1] <= c:
            max_q.pop()
        max_q.append((j, c))
        while max_q and max_q[0][0] <= j - 20:
            max_q.popleft()
        while min_q and min_q[-1][1] >= c:
            min_q.pop()
        min_q.append((j, c))
        while min_q and min_q[0][0] <= j - 20:
            min_q.popleft()
        if j >= 19:
            rh20[j] = max_q[0][1]
            rl20[j] = min_q[0][1]
    I["recent_high_20"] = rh20
    I["recent_low_20"] = rl20

    # ATR20 average (running sum)
    avg_a20 = [None] * n
    if n > 20:
        atr_v = I["atr14"]
        run_sum = 0.0; run_cnt = 0
        for j in range(n):
            v = atr_v[j]
            if v is not None:
                run_sum += v; run_cnt += 1
            if j >= 20:
                old = atr_v[j - 20]
                if old is not None:
                    run_sum -= old; run_cnt -= 1
            if run_cnt == 20:
                avg_a20[j] = run_sum / 20.0
    I["avg_atr20"] = avg_a20

    # Store raw data for convenience
    I["close"] = closes
    I["high"] = highs
    I["low"] = lows
    I["open"] = opens
    I["volume"] = volumes
    I["time"] = [d["time"] for d in data]

    return I


def map_timeframe(ltf_data, htf_data, ltf_times=None, htf_times=None):
    n_ltf = len(ltf_data)
    mapping = [-1] * n_ltf
    hi = 0
    if ltf_times is not None and htf_times is not None:
        for i in range(n_ltf):
            ltf_t = ltf_times[i]
            while hi < len(htf_data) - 1 and htf_times[hi + 1] <= ltf_t:
                hi += 1
            if htf_times[hi] <= ltf_t:
                mapping[i] = hi
    else:
        for i in range(n_ltf):
            ltf_t = ltf_data[i]["time"]
            while hi < len(htf_data) - 1 and htf_data[hi + 1]["time"] <= ltf_t:
                hi += 1
            if htf_data[hi]["time"] <= ltf_t:
                mapping[i] = hi
    return mapping


def detect_divergence(prices, values):
    n = len(prices)
    if n < 50:
        return "none"
    pivots = []
    for i in range(5, n - 5):
        if all(prices[i] >= prices[i-j] and prices[i] >= prices[i+j] for j in range(1, 6)):
            pivots.append({"price": prices[i], "value": values[i], "type": "high"})
        if all(prices[i] <= prices[i-j] and prices[i] <= prices[i+j] for j in range(1, 6)):
            pivots.append({"price": prices[i], "value": values[i], "type": "low"})
    if len(pivots) < 2:
        return "none"
    last = pivots[-1]
    prev = pivots[-2]
    if last["type"] == "low" and prev["type"] == "low":
        if last["price"] < prev["price"] and last["value"] > prev["value"]:
            return "regular_bullish"
    if last["type"] == "high" and prev["type"] == "high":
        if last["price"] > prev["price"] and last["value"] < prev["value"]:
            return "regular_bearish"
    return "none"


def detect_macd_divergence(prices, hist_values):
    n = len(prices)
    if n < 50:
        return "none"
    last_hist = hist_values[-1]
    prev_hist = hist_values[-2]
    cross_bullish = last_hist > 0 and prev_hist < 0
    cross_bearish = last_hist < 0 and prev_hist > 0
    pivots = []
    for i in range(1, n - 2):
        if (hist_values[i] > hist_values[i-1] and hist_values[i] > hist_values[i+1]) or \
           (hist_values[i] < hist_values[i-1] and hist_values[i] < hist_values[i+1]):
            pivots.append({"value": hist_values[i], "index": i})
    if len(pivots) < 2:
        if cross_bullish:
            return "regular_bullish"
        if cross_bearish:
            return "regular_bearish"
        return "none"
    last_p = pivots[-1]
    prev_p = pivots[-2]
    if last_p["value"] < 0 and last_p["value"] < prev_p["value"]:
        if prices[last_p["index"]] > prices[prev_p["index"]]:
            return "hidden_bullish"
        if prices[last_p["index"]] < prices[prev_p["index"]]:
            return "regular_bullish"
    if last_p["value"] > 0 and last_p["value"] > prev_p["value"]:
        if prices[last_p["index"]] < prices[prev_p["index"]]:
            return "hidden_bearish"
        if prices[last_p["index"]] > prices[prev_p["index"]]:
            return "regular_bearish"
    if cross_bullish:
        return "regular_bullish"
    if cross_bearish:
        return "regular_bearish"
    return "none"


def detect_lg_sfp_fg(data_slice):
    n = len(data_slice)
    lg = "neutral"
    sfp = "neutral"
    fg = "neutral"
    if n < 20:
        return lg, sfp, fg
    closes = [d["close"] for d in data_slice]
    highs = [d["high"] for d in data_slice]
    lows = [d["low"] for d in data_slice]
    opens = [d["open"] for d in data_slice]
    volumes = [d["volume"] for d in data_slice]
    # Fast ATR approximation over small window
    tr_vals = []
    for j in range(1, n):
        tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
        tr_vals.append(tr)
    if n >= 15:
        avg_atr = sum(tr_vals[-14:]) / 14.0
    else:
        avg_atr = sum(tr_vals) / len(tr_vals) if tr_vals else 0
    avg_vol = sum(volumes[-25:]) / min(25, n)

    # Liquidity Grab
    for s in range(1, min(5, n)):
        idx = n - s
        if idx < 16:
            break
        lookback_lows = lows[idx-15:idx]
        lookback_highs = highs[idx-15:idx]
        if not lookback_lows:
            continue
        min_low = min(lookback_lows)
        max_high = max(lookback_highs)
        bar_low = lows[idx]
        bar_high = highs[idx]
        bar_close = closes[idx]
        bar_open = opens[idx]

        if bar_low < min_low and bar_close > min_low:
            has_displacement = False
            has_high_sweep = False
            for j in range(idx + 1, n):
                if abs(closes[j] - opens[j]) > avg_atr * 0.5 and volumes[j] > avg_vol * 0.8:
                    has_displacement = True
                if closes[j] > bar_high:
                    has_high_sweep = True
            if bar_close > bar_open and (bar_close - bar_open) > avg_atr * 0.6:
                has_displacement = True
            if idx > 0 and bar_close > highs[idx - 1]:
                has_high_sweep = True
            if has_displacement and has_high_sweep:
                lg = "bullish"
                break

        if bar_high > max_high and bar_close < max_high:
            has_displacement = False
            has_low_sweep = False
            for j in range(idx + 1, n):
                if abs(closes[j] - opens[j]) > avg_atr * 0.5 and volumes[j] > avg_vol * 0.8:
                    has_displacement = True
                if closes[j] < bar_low:
                    has_low_sweep = True
            if bar_open > bar_close and (bar_open - bar_close) > avg_atr * 0.6:
                has_displacement = True
            if idx > 0 and bar_close < lows[idx - 1]:
                has_low_sweep = True
            if has_displacement and has_low_sweep:
                lg = "bearish"
                break

    # SFP
    if n >= 17:
        max_high_sfp = max(highs[-17:-2]) if len(highs[-17:-2]) > 0 else 0
        min_low_sfp = min(lows[-17:-2]) if len(lows[-17:-2]) > 0 else 0
        last = data_slice[-1]
        prev = data_slice[-2]
        if (last["low"] < min_low_sfp and last["close"] > min_low_sfp) or \
           (prev["low"] < min_low_sfp and prev["close"] > min_low_sfp and last["close"] > min_low_sfp):
            sfp = "bullish"
        if (last["high"] > max_high_sfp and last["close"] < max_high_sfp) or \
           (prev["high"] > max_high_sfp and prev["close"] < max_high_sfp and last["close"] < max_high_sfp):
            sfp = "bearish"

    # Failed Grab
    for s in range(2, min(6, n)):
        idx = n - s
        if idx < 16:
            break
        lookback_lows = lows[idx-15:idx]
        lookback_highs = highs[idx-15:idx]
        if not lookback_lows:
            continue
        min_low = min(lookback_lows)
        max_high = max(lookback_highs)
        bar_low = lows[idx]
        bar_high = highs[idx]
        bar_close = closes[idx]
        if bar_low < min_low and bar_close > min_low:
            for j in range(idx + 1, n):
                if closes[j] < bar_low:
                    fg = "bearish"
                    break
        if bar_high > max_high and bar_close < max_high:
            for j in range(idx + 1, n):
                if closes[j] > bar_high:
                    fg = "bullish"
                    break

    return lg, sfp, fg


# ─── Analysis Functions ───

def compute_htf_direction(htf_idx, I):
    ls = 0.0
    ss = 0.0
    close = I["close"][htf_idx]
    ema200 = I["ema200"][htf_idx]
    ema9 = I["ema9"][htf_idx]
    ema50 = I["ema50"][htf_idx]
    hist = I["macd_hist"][htf_idx]
    prev_hist = I["macd_hist"][htf_idx - 1] if htf_idx > 0 else 0
    adx = I["adx_14_adx"][htf_idx]
    pdi = I["adx_14_pdi"][htf_idx]
    mdi = I["adx_14_mdi"][htf_idx]

    if close is None or ema200 is None:
        return "NEUTRAL"

    if close > ema200:
        ls += 30
    elif close < ema200:
        ss += 30

    if ema9 is not None and ema50 is not None:
        if ema9 > ema50:
            ls += 20
        elif ema9 < ema50:
            ss += 20

    if hist is not None and prev_hist is not None:
        if hist > 0 and hist > prev_hist:
            ls += 25
        elif hist > 0:
            ls += 10
        elif hist < 0 and hist < prev_hist:
            ss += 25
        elif hist < 0:
            ss += 10

    if adx is not None and pdi is not None and mdi is not None:
        if adx > 25 and pdi > mdi:
            ls += 15
        elif adx > 25 and mdi > pdi:
            ss += 15
        elif adx > 20 and adx <= 25 and pdi > mdi:
            ls += 5
        elif adx > 20 and adx <= 25 and mdi > pdi:
            ss += 5

    if ls >= 50:
        return "LONG"
    if ss >= 50:
        return "SHORT"
    return "NEUTRAL"


def compute_1h_control(htf_bias, h1_idx, I):
    bias = htf_bias if htf_bias != "NEUTRAL" else "LONG"
    hist = I["macd_hist"][h1_idx]
    prev_hist = I["macd_hist"][h1_idx - 1] if h1_idx > 0 else 0
    rsi = I["rsi14"][h1_idx]
    close = I["close"][h1_idx]
    ema20 = I["ema20"][h1_idx]

    if any(x is None for x in [hist, prev_hist]):
        return "WAIT"

    if bias == "LONG":
        if hist < 0 and hist < prev_hist and (close is not None and hist < close * -0.0001):
            return "VETO"
        if hist > 0 and hist > prev_hist and rsi is not None and rsi > 50 and close is not None and ema20 is not None and close > ema20:
            return "CONTINUATION"
        if hist <= 0 or hist <= prev_hist:
            return "WAIT"
        return "WAIT"
    else:
        if hist > 0 and hist > prev_hist and (close is not None and hist > close * 0.0001):
            return "VETO"
        if hist < 0 and hist < prev_hist and rsi is not None and rsi < 50 and close is not None and ema20 is not None and close < ema20:
            return "CONTINUATION"
        if hist >= 0 or hist >= prev_hist:
            return "WAIT"
        return "WAIT"


def analyze_15m(i, I, data):
    close = I["close"][i]
    high = I["high"][i]
    low = I["low"][i]
    open_p = I["open"][i]
    volume = I["volume"][i]
    prev_close = I["close"][i - 1] if i > 0 else close

    ema20 = I["ema20"][i]; ema50 = I["ema50"][i]; ema200 = I["ema200"][i]
    atr14 = I["atr14"][i]; rsi21 = I["rsi21"][i]
    adx = I["adx_14_adx"][i]; pdi = I["adx_14_pdi"][i]; mdi = I["adx_14_mdi"][i]
    hist = I["macd_hist"][i]
    prev_hist = I["macd_hist"][i - 1] if i > 0 else None
    prev_prev_hist = I["macd_hist"][i - 2] if i > 1 else None
    vol_sma_20 = I["vol_sma_20"][i]; obv = I["obv"][i]
    obv_prev = I["obv"][i - 1] if i > 0 else None
    bos = I["bos"][i]; mss = I["mss"][i]; st_trend = I["supertrend_trend"][i]
    vp_vah = I["vp_vah"][i]; vp_val = I["vp_val"][i]
    order_flow = I["order_flow"][i]

    if any(x is None for x in (close, ema20, ema50, ema200, atr14, rsi21)):
        return ("NO TRADE", 0, 0)

    recent_low_20 = I["recent_low_20"][i] if I["recent_low_20"][i] is not None else low
    recent_high_20 = I["recent_high_20"][i] if I["recent_high_20"][i] is not None else high

    bb_width = I["bb_width"][i]
    bb_width_prev = I["bb_width"][i - 1] if i > 0 and I["bb_width"][i - 1] is not None else 0
    is_high_volatility = (bb_width is not None and bb_width_prev is not None and bb_width_prev > 0
                          and bb_width > bb_width_prev * 1.5) if bb_width is not None else False

    is_trending = adx is not None and adx > 25
    is_trending_up = is_trending and pdi is not None and mdi is not None and pdi > mdi
    is_trending_down = is_trending and pdi is not None and mdi is not None and mdi > pdi

    is_squeeze = I["squeeze"][i] if i >= 50 else False

    volume_spike = volume / vol_sma_20 if vol_sma_20 and vol_sma_20 > 0 else 0
    avg_atr20 = I["avg_atr20"][i] if I["avg_atr20"][i] is not None else atr14
    atr_expansion = atr14 / avg_atr20 if avg_atr20 > 0 else 1.0

    # ── Layer scores ──
    adx_score = 0.0
    if is_trending_up:
        adx_score = min((adx - 15) / 25.0, 1.0)
    elif is_trending_down:
        adx_score = max(-(adx - 15) / 25.0, -1.0)
    bos_score = float(bos) if bos in (1, -1) else 0.0

    ema_score = 0.0
    if close > ema20 and ema20 > ema50 and ema50 > ema200:
        ema_score = 1.0
    elif close < ema20 and ema20 < ema50 and ema50 < ema200:
        ema_score = -1.0
    elif close > ema50:
        ema_score = 0.5
    elif close < ema50:
        ema_score = -0.5

    macd_score = 0.0
    if prev_hist is not None and hist is not None:
        if prev_hist < 0 and hist > 0:
            macd_score = 1.5
        elif prev_hist > 0 and hist < 0:
            macd_score = -1.5
        elif hist > 0:
            if prev_prev_hist is not None and hist > prev_hist and prev_hist > prev_prev_hist:
                macd_score = 1.0
            elif prev_hist is not None and hist < prev_hist:
                macd_score = -0.5
            else:
                macd_score = 0.5
        elif hist < 0:
            if prev_prev_hist is not None and hist < prev_hist and prev_hist < prev_prev_hist:
                macd_score = -1.0
            elif prev_hist is not None and hist > prev_hist:
                macd_score = 0.5
            else:
                macd_score = -0.5

    rsi_score = 0.0
    if is_trending_up:
        if rsi21 < 45:
            rsi_score = 1.0
        elif 55 <= rsi21 <= 70:
            rsi_score = 1.0
        elif rsi21 > 70:
            rsi_score = -0.5
        else:
            rsi_score = 0.0
    elif is_trending_down:
        if rsi21 > 55:
            rsi_score = -1.0
        elif 30 <= rsi21 <= 45:
            rsi_score = -1.0
        elif rsi21 < 30:
            rsi_score = 0.5
        else:
            rsi_score = 0.0
    else:
        if rsi21 < 35:
            rsi_score = 1.0
        elif rsi21 > 65:
            rsi_score = -1.0
        else:
            rsi_score = 0.0

    displacement_score = 0.0
    if close > open_p and abs(close - open_p) > atr14 * 0.8 and vol_sma_20 and volume > vol_sma_20:
        displacement_score = 1.0
    elif close < open_p and abs(close - open_p) > atr14 * 0.8 and vol_sma_20 and volume > vol_sma_20:
        displacement_score = -1.0

    vol_score = 0.0
    if vol_sma_20 and volume > vol_sma_20 * 1.2 and close > prev_close:
        vol_score = 1.0
    elif vol_sma_20 and volume > vol_sma_20 * 1.2 and close < prev_close:
        vol_score = -1.0

    obv_score = 0.0
    if obv is not None and obv_prev is not None:
        if obv > obv_prev:
            obv_score = 1.0
        elif obv < obv_prev:
            obv_score = -1.0

    # ── Evidence Buckets ──
    tb = ema_score * 12.5
    if st_trend is not None:
        if st_trend == 1:
            tb += 7.5
        elif st_trend == -1:
            tb -= 7.5
    if is_trending_up:
        tb += 5
    if is_trending_down:
        tb -= 5
    tb = max(-25, min(25, tb))

    sb = 15 if bos == 1 else -15 if bos == -1 else 0
    if mss is not None:
        if mss == 1:
            sb += 10
        elif mss == -1:
            sb -= 10
    sb = max(-25, min(25, sb))

    mb = macd_score * 6 + rsi_score * 5

    r_div = I["rsi_divergence"][i]
    m_div = I["macd_divergence"][i]
    has_bullish_div = r_div == 1 or m_div == 1
    has_bearish_div = r_div == -1 or m_div == -1
    mb = max(-20, min(20, mb))

    lg_s = I["lg"][i]; sfp_s = I["sfp"][i]; fg_s = I["fg"][i]
    lib = 0
    if lg_s == 1:
        lib += 10
    elif lg_s == -1:
        lib -= 10
    if sfp_s == 1:
        lib += 5
    elif sfp_s == -1:
        lib -= 5
    lib = max(-15, min(15, lib))

    vb = 0
    if order_flow == "bullish":
        vb += 4
    elif order_flow == "bearish":
        vb -= 4
    if volume_spike > 1.5 and close > prev_close:
        vb += 4
    if volume_spike > 1.5 and close < prev_close:
        vb -= 4
    if vp_vah is not None and close > vp_vah:
        vb += 3
    if vp_val is not None and close < vp_val:
        vb -= 3
    if volume_spike >= 1.3 and close > prev_close:
        vb += 4
    elif volume_spike >= 1.1 and close > prev_close:
        vb += 2
    elif volume_spike >= 1.3 and close < prev_close:
        vb -= 4
    elif volume_spike >= 1.1 and close < prev_close:
        vb -= 2
    elif volume_spike < 0.8:
        vb += -2 if close > prev_close else 2
    vb = max(-15, min(15, vb))

    combined = tb + sb + mb + lib + vb
    raw_confidence = abs(combined)

    if atr_expansion < 0.6:
        vm = 0.90
    elif atr_expansion < 0.8:
        vm = 0.95
    elif atr_expansion <= 1.8:
        vm = 1.00
    elif atr_expansion <= 2.5:
        vm = 0.95
    else:
        vm = 0.90
    vol_adjusted_confidence = raw_confidence * vm

    penalty = 0.0
    if combined > 0:
        if has_bearish_div: penalty += 15
        if sfp_s == -1: penalty += 15
        if mss is not None and mss == -1: penalty += 20
        if fg_s == -1: penalty += 15
    elif combined < 0:
        if has_bullish_div: penalty += 15
        if sfp_s == 1: penalty += 15
        if mss is not None and mss == 1: penalty += 20
        if fg_s == 1: penalty += 15

    if I["sweep_cooldown"][i]:
        penalty += 20

    confidence = max(0.0, vol_adjusted_confidence - penalty)
    confidence = min(100.0, confidence)

    fs = combined / 100.0
    is_market_regime_violation = (fs > 0 and mss is not None and mss == -1 and bos != 1) or \
                                 (fs < 0 and mss is not None and mss == 1 and bos != -1)

    signal = "NO TRADE"
    if confidence >= 55 and not is_market_regime_violation:
        signal = "LONG" if fs > 0 else "SHORT"

    if signal == "LONG" and abs(close - recent_high_20) < atr14 * 0.2:
        signal = "NO TRADE"
    if signal == "SHORT" and abs(close - recent_low_20) < atr14 * 0.2:
        signal = "NO TRADE"

    entry_price = close
    if signal != "NO TRADE":
        sl_price = close * 0.985 if signal == "LONG" else close * 1.015
        if is_high_volatility:
            multiplier = 2.2
            sl_price = entry_price - multiplier * atr14 if signal == "LONG" else entry_price + multiplier * atr14
        elif is_squeeze:
            multiplier = 1.25
            sl_price = entry_price - multiplier * atr14 if signal == "LONG" else entry_price + multiplier * atr14
        elif is_trending:
            if signal == "LONG":
                structural_sl = recent_low_20 - 0.25 * atr14
                sl_price = max(structural_sl, entry_price - 3.0 * atr14)
                sl_price = min(sl_price, entry_price - 1.5 * atr14)
            else:
                structural_sl = recent_high_20 + 0.25 * atr14
                sl_price = min(structural_sl, entry_price + 3.0 * atr14)
                sl_price = max(sl_price, entry_price + 1.5 * atr14)
        else:
            if signal == "LONG":
                structural_sl = recent_low_20 - 0.2 * atr14
                sl_price = max(structural_sl, entry_price - 2.0 * atr14)
                sl_price = min(sl_price, entry_price - 1.5 * atr14)
            else:
                structural_sl = recent_high_20 + 0.2 * atr14
                sl_price = min(structural_sl, entry_price + 2.0 * atr14)
                sl_price = max(sl_price, entry_price + 1.5 * atr14)

        risk = abs(entry_price - sl_price)
        if risk / entry_price > 0.12:
            signal = "NO TRADE"
    else:
        sl_price = 0

    sl_price = sl_price if signal != "NO TRADE" else 0
    return (signal, confidence, sl_price)


def validate_5m(signal, idx, I):
    close = I["close"][idx]
    ema10 = I["ema10"][idx]
    ema30 = I["ema30"][idx]
    bos = I["bos"][idx]
    open_p = I["open"][idx]
    atr14 = I["atr14"][idx]
    vol_sma_20 = I["vol_sma_20"][idx]
    volume = I["volume"][idx]
    adx7_adx = I["adx_7_adx"][idx]
    adx7_pdi = I["adx_7_pdi"][idx]
    adx7_mdi = I["adx_7_mdi"][idx]
    rsi14 = I["rsi14"][idx]

    score = 0
    if signal == "LONG":
        if close is not None and ema10 is not None and ema30 is not None:
            if close > ema10 and ema10 > ema30:
                score += 25
        if bos == 1:
            score += 25
        if (adx7_adx is not None and adx7_pdi is not None and adx7_mdi is not None and
            adx7_adx > 15 and adx7_pdi > adx7_mdi) or \
           (rsi14 is not None and 50 < rsi14 < 70):
            score += 20
        is_displacement_up = (close > open_p and abs(close - open_p) > atr14 * 0.8 and
                              vol_sma_20 and volume > vol_sma_20)
        if is_displacement_up or (vol_sma_20 and volume > vol_sma_20):
            score += 15
        return score >= 40
    else:
        if close is not None and ema10 is not None and ema30 is not None:
            if close < ema10 and ema10 < ema30:
                score += 25
        if bos == -1:
            score += 25
        if (adx7_adx is not None and adx7_mdi is not None and adx7_pdi is not None and
            adx7_adx > 15 and adx7_mdi > adx7_pdi) or \
           (rsi14 is not None and 30 < rsi14 < 50):
            score += 20
        is_displacement_down = (close < open_p and abs(close - open_p) > atr14 * 0.8 and
                                vol_sma_20 and volume > vol_sma_20)
        if is_displacement_down or (vol_sma_20 and volume > vol_sma_20):
            score += 15
        return score >= 40


def compute_btc_trend(idx, I):
    close = I["close"][idx]
    ema20 = I["ema20"][idx]
    ema50 = I["ema50"][idx]
    if close is not None and ema20 is not None and ema50 is not None:
        if close > ema20 and ema20 > ema50:
            return "LONG"
        if close < ema20 and ema20 < ema50:
            return "SHORT"
    return "NEUTRAL"


# ─── Position ───

class Position:
    def __init__(self, side, sym, entry_time, entry_price, sl, tp1, tp2, tp3, a1, a2, conf,
                 dca_price=None, initial_sl=None):
        self.side = side
        self.sym = sym
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.a1 = a1
        self.a2 = a2
        self.conf = conf
        self.csl = sl
        self.ht1 = False
        self.ht2 = False
        self.pnl = 0.0
        self.closed = False
        self.exit_reason = ""
        self.exit_time = ""
        self.exit_price = 0
        self.dca_price = dca_price
        self.dca_added = False
        self.initial_sl = initial_sl or sl
        self.initial_entry = entry_price


# ─── Main Backtest ───

class EndellionBacktest:
    def __init__(self, config=None, entry_mode="dca"):
        # entry_mode: "dca" = enter at close + DCA at PB
        #            "pullback" = enter only at PB level (original spec)
        self.config = config or BEST_CONFIG.copy()
        self.entry_mode = entry_mode
        self.data = {}
        self.indicators = {}
        self.mappings = {}
        self.precalc = {}
        self.all_trades = []

    def load_data(self):
        data_dir = DATA_DIR
        from concurrent.futures import ThreadPoolExecutor, as_completed

        SAFE_MAX = {"4h": 2400, "1h": 9000, "15m": 36000, "5m": 110000}

        def _load_one(sym, tf):
            filename = f"{sym}_{tf}.csv"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                rows = load_csv(filepath, max_bars=SAFE_MAX.get(tf))
                return (sym, tf, rows)
            return (sym, tf, [])

        futures = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for sym in SYMBOLS:
                for tf in TIMEFRAMES:
                    futures[ex.submit(_load_one, sym, tf)] = (sym, tf)
            for fut in as_completed(futures):
                sym, tf, rows = fut.result()
                if sym not in self.data:
                    self.data[sym] = {}
                self.data[sym][tf] = rows
                print(f"  Loaded {sym}_{tf}.csv: {len(rows)} bars")

        if not self.data.get("BTC", {}).get("15m"):
            raise FileNotFoundError("BTC 15m data required")

        # Trim to last 1 year using string comparison (ISO timestamps sort chronologically)
        self._cached_times = {}
        for sym in SYMBOLS:
            self._cached_times[sym] = {}
            d15 = self.data[sym].get("15m", [])
            if len(d15) < 35000:
                continue
            # Build cutoff string once
            last_time_str = d15[-1]["time"]
            from datetime import datetime, timedelta
            last_dt = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
            cutoff_dt = last_dt - timedelta(days=366)
            cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
            for tf in TIMEFRAMES:
                d = self.data[sym].get(tf, [])
                if not d:
                    continue
                trimmed = [r for r in d if r["time"] >= cutoff_str]
                self.data[sym][tf] = trimmed
                self._cached_times[sym][tf] = [r["time"] for r in trimmed]
                saved = len(d) - len(trimmed)
                if saved:
                    print(f"  Trimmed {sym} {tf}: {len(d)} -> {len(trimmed)} bars")

    def precompute(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = []
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                d = self.data[sym].get(tf, [])
                if len(d) < TF_MIN_BARS[tf]:
                    print(f"  Skipping {sym} {tf}: {len(d)} bars, min {TF_MIN_BARS[tf]}")
                    if sym not in self.indicators:
                        self.indicators[sym] = {}
                    self.indicators[sym][tf] = None
                else:
                    tasks.append((sym, tf, d))

        def _compute(sym, tf, d):
            print(f"  Computing indicators for {sym} {tf} ({len(d)} bars)...")
            I = compute_all_indicators(d)
            if tf == "15m" and len(d) >= 200:
                I = compute_15m_heavy_arrays(I)
            return (sym, tf, I)

        results = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_compute, s, t, d): (s, t) for s, t, d in tasks}
            for fut in as_completed(futs):
                sym, tf, I = fut.result()
                results[(sym, tf)] = I

        for sym in SYMBOLS:
            self.indicators[sym] = {}
            for tf in TIMEFRAMES:
                self.indicators[sym][tf] = results.get((sym, tf))

    def build_mappings(self):
        for sym in SYMBOLS:
            d15m = self.data[sym].get("15m", [])
            if not d15m:
                continue
            t15m = self._cached_times.get(sym, {}).get("15m", None)
            self.mappings[sym] = {}
            for tf in ["4h", "1h", "5m"]:
                htf_data = self.data[sym].get(tf, [])
                if htf_data:
                    t_htf = self._cached_times.get(sym, {}).get(tf, None)
                    self.mappings[sym][f"{tf}_to_15m"] = map_timeframe(d15m, htf_data, t15m, t_htf)
                    print(f"  Mapped {sym} {tf} -> 15m")
            btc_data = self.data.get("BTC", {}).get("15m", [])
            if btc_data:
                t_btc = self._cached_times.get("BTC", {}).get("15m", None)
                self.mappings[sym]["btc_15m_to_15m"] = map_timeframe(d15m, btc_data, t15m, t_btc)
                print(f"  Mapped BTC 15m -> {sym} 15m")

    def precalc_signals(self):
        dbg = {"total": 0, "raw_signal": 0, "conf_ge_55": 0}
        for sym in SYMBOLS:
            d15m = self.data[sym].get("15m", [])
            I15m = self.indicators[sym].get("15m")
            if not d15m or I15m is None:
                continue
            n = len(d15m)
            pre = {
                "htf": [None] * n, "ctrl": [None] * n,
                "signal": [None] * n, "conf": [0.0] * n, "sl": [0.0] * n,
                "btc": [None] * n, "ltf_l": [False] * n, "ltf_s": [False] * n,
            }
            self.precalc[sym] = pre
            print(f"  Pre-calc signals for {sym} ({n} bars)...")
            m4 = self.mappings[sym]["4h_to_15m"]
            m1 = self.mappings[sym]["1h_to_15m"]
            m5 = self.mappings[sym]["5m_to_15m"]
            mb = self.mappings[sym]["btc_15m_to_15m"]
            I4h = self.indicators[sym].get("4h")
            I1h = self.indicators[sym].get("1h")
            I5m = self.indicators[sym].get("5m")
            I_btc = self.indicators.get("BTC", {}).get("15m")
            if not all([I4h, I1h, I5m, I_btc]):
                continue
            for i in range(200, n - 1):
                htf_idx = m4[i]; h1_idx = m1[i]; m5_idx = m5[i]; btc_idx = mb[i]
                if m5_idx < 0 or htf_idx < 0 or h1_idx < 0 or btc_idx < 0:
                    continue
                if htf_idx < 200 or h1_idx < 100 or m5_idx < 50 or btc_idx < 50:
                    continue
                dbg["total"] += 1
                htf_dir = compute_htf_direction(htf_idx, I4h)
                ctrl = compute_1h_control(htf_dir, h1_idx, I1h)
                sig, conf, sl_price = analyze_15m(i, I15m, d15m)
                if sig != "NO TRADE":
                    dbg["raw_signal"] += 1
                if conf >= 55:
                    dbg["conf_ge_55"] += 1
                btc_trend = compute_btc_trend(btc_idx, I_btc)
                ltf_l = validate_5m("LONG", m5_idx, I5m)
                ltf_s = validate_5m("SHORT", m5_idx, I5m)
                pre["htf"][i] = htf_dir
                pre["ctrl"][i] = ctrl
                pre["signal"][i] = sig
                pre["conf"][i] = conf
                pre["sl"][i] = sl_price
                pre["btc"][i] = btc_trend
                pre["ltf_l"][i] = ltf_l
                pre["ltf_s"][i] = ltf_s
        print(f"  DEBUG: {dbg['total']} bars analyzed, {dbg['raw_signal']} raw signals (conf>=55 + no regime violation), {dbg['conf_ge_55']} bars with conf>=55")

    def run(self, skip_setup=False):
        mode_name = {"dca": "Entry at close + DCA at PB",
                     "pullback": "Entry ONLY at PB level (limit order)"}
        print("\n========================================")
        print("  ENDELLION STRATEGY BACKTEST")
        print(f"  Mode: {mode_name.get(self.entry_mode, self.entry_mode)}")
        print(f"  Config: {self.config}")
        print("========================================\n")

        if not skip_setup:
            self.load_data()
            self.precompute()
            self.build_mappings()
            self.precalc_signals()

        total_trades = {sym: [] for sym in SYMBOLS}
        dcounters = {sym: {"passed_filters": 0, "no_trade": 0, "htf_neutral": 0, "wait": 0, "veto": 0,
                           "strict_htf": 0, "btc_filter": 0, "ltf_fail": 0, "conf_fail": 0,
                           "risk_fail": 0, "entered": 0, "already_in_pos": 0} for sym in SYMBOLS}

        for sym in SYMBOLS:
            d15m = self.data[sym].get("15m", [])
            I15m = self.indicators[sym].get("15m")
            if not d15m or I15m is None or sym not in self.precalc:
                continue

            n = len(d15m)
            pos = None
            pending_order = None  # (side, pb_price, sl, tp1, tp2, tp3, a1, a2, conf, entry_time, entry_bar)

            for i in range(200, n):
                # ── Manage open position ──
                if pos is not None and not pos.closed:
                    self._manage_position(pos, d15m, i, I15m)
                    if pos.closed:
                        total_trades[sym].append(pos)
                        pos = None
                    continue

                # ── Check pending order (pullback mode) ──
                if pending_order is not None and self.entry_mode == "pullback":
                    side, pb_price, sl, tp1, tp2, tp3, a1, a2, conf, e_time, e_bar = pending_order
                    bar = d15m[i]
                    hit = False
                    if side == "LONG" and bar["low"] <= pb_price:
                        hit = True
                        entry_px = min(pb_price, bar["open"])
                    elif side == "SHORT" and bar["high"] >= pb_price:
                        hit = True
                        entry_px = max(pb_price, bar["open"])

                    if hit:
                        risk = abs(entry_px - sl)
                        if risk / entry_px <= 0.05:
                            pos = Position(side, sym, bar["time"], entry_px, sl,
                                           tp1, tp2, tp3, a1, a2, conf,
                                           initial_sl=sl)
                            pos.entry_price = entry_px
                        pending_order = None
                    elif i - e_bar > 4:
                        pending_order = None  # cancel after 5 bars (~75min)
                    continue

                # ── Check for new signal ──
                sig = self.precalc[sym]
                if sig["signal"][i] is None or sig["signal"][i] == "NO TRADE":
                    dcounters[sym]["no_trade"] += 1
                    continue

                dcounters[sym]["passed_filters"] += 1

                # Apply filters
                if self.config["htf_neutral_skip"] and sig["htf"][i] == "NEUTRAL":
                    dcounters[sym]["htf_neutral"] += 1
                    continue
                if self.config["use_1h_control"] and sig["ctrl"][i] == "WAIT":
                    dcounters[sym]["wait"] += 1
                    continue
                if sig["ctrl"][i] == "VETO":
                    dcounters[sym]["veto"] += 1
                    continue
                if self.config["strict_htf"] and sig["htf"][i] != sig["signal"][i]:
                    dcounters[sym]["strict_htf"] += 1
                    continue
                if self.config["use_btc_filter"] and sym != "BTC" and sig["btc"][i] != "NEUTRAL":
                    if sig["signal"][i] == "LONG" and sig["btc"][i] == "SHORT":
                        dcounters[sym]["btc_filter"] += 1
                        continue
                    if sig["signal"][i] == "SHORT" and sig["btc"][i] == "LONG":
                        dcounters[sym]["btc_filter"] += 1
                        continue
                if sig["signal"][i] == "LONG" and not sig["ltf_l"][i]:
                    dcounters[sym]["ltf_fail"] += 1
                    continue
                if sig["signal"][i] == "SHORT" and not sig["ltf_s"][i]:
                    dcounters[sym]["ltf_fail"] += 1
                    continue
                if sig["conf"][i] < self.config["conf_thresh"]:
                    dcounters[sym]["conf_fail"] += 1
                    continue

                # ── ENTRY LOGIC ──
                close = I15m["close"][i]
                sl = sig["sl"][i]
                if sl == 0:
                    sl = close * 0.985 if sig["signal"][i] == "LONG" else close * 1.015

                pb_factor = self.config["pb"]
                distance = abs(close - sl)
                pb_price = close - distance * pb_factor if sig["signal"][i] == "LONG" else close + distance * pb_factor

                risk = abs(close - sl)
                if risk / close > 0.05:
                    continue

                tp1m = self.config["tp1m"]
                tp2m = self.config["tp2m"]
                tp3m = self.config["tp3m"]
                a1 = self.config["a1"]
                a2 = self.config["a2"]

                if sig["signal"][i] == "LONG":
                    tp1 = close + risk * tp1m
                    tp2 = close + risk * tp2m
                    tp3 = close + risk * tp3m
                else:
                    tp1 = close - risk * tp1m
                    tp2 = close - risk * tp2m
                    tp3 = close - risk * tp3m

                entry_time = I15m["time"][i]

                dcounters[sym]["entered"] += 1

                if self.entry_mode == "dca":
                    pos = Position(sig["signal"][i], sym, entry_time, close, sl,
                                   tp1, tp2, tp3, a1, a2, sig["conf"][i],
                                   dca_price=pb_price, initial_sl=sl)
                else:
                    # Original pullback: entry at PB-adjusted price immediately
                    entry_px = pb_price
                    risk_pb = abs(entry_px - sl)
                    if risk_pb / entry_px > 0.05:
                        dcounters[sym]["risk_fail"] += 1
                        continue
                    dcounters[sym]["entered"] = dcounters[sym].get("entered", 0)
                    if sig["signal"][i] == "LONG":
                        tp1 = entry_px + risk_pb * tp1m
                        tp2 = entry_px + risk_pb * tp2m
                        tp3 = entry_px + risk_pb * tp3m
                    else:
                        tp1 = entry_px - risk_pb * tp1m
                        tp2 = entry_px - risk_pb * tp2m
                        tp3 = entry_px - risk_pb * tp3m
                    pos = Position(sig["signal"][i], sym, entry_time, entry_px, sl,
                                   tp1, tp2, tp3, a1, a2, sig["conf"][i],
                                   initial_sl=sl)

        # Collect all trades
        for sym in SYMBOLS:
            for t in total_trades[sym]:
                self.all_trades.append(t)

        self._last_pipeline_counts = dcounters

        print(f"\n  DEBUG pipeline per symbol:")
        for sym in SYMBOLS:
            c = dcounters[sym]
            print(f"    {sym}: signals={c['passed_filters']} "
                  f"htf_n={c['htf_neutral']} wait={c['wait']} veto={c['veto']} "
                  f"strict={c['strict_htf']} btc={c['btc_filter']} "
                  f"ltf={c['ltf_fail']} conf={c['conf_fail']} "
                  f"risk={c['risk_fail']} → ENTERED={c['entered']}")

        return self.all_trades

    def _manage_position(self, pos, d15m, i, I15m):
        bar = d15m[i]
        high = bar["high"]
        low = bar["low"]
        time_str = bar["time"]

        # Check DCA trigger
        if pos.dca_price is not None and not pos.dca_added:
            pb_hit = False
            if pos.side == "LONG" and low <= pos.dca_price:
                pb_hit = True
            elif pos.side == "SHORT" and high >= pos.dca_price:
                pb_hit = True
            if pb_hit:
                old_entry = pos.entry_price
                new_avg = (old_entry + pos.dca_price) / 2.0
                old_risk = abs(old_entry - pos.initial_sl)
                new_risk_from_avg = abs(new_avg - pos.initial_sl)

                if pos.side == "LONG":
                    risk_mult = abs(pos.tp1 - old_entry) / old_risk if old_risk > 0 else 1.8
                    risk_mult2 = abs(pos.tp2 - old_entry) / old_risk if old_risk > 0 else 3.0
                    risk_mult3 = abs(pos.tp3 - old_entry) / old_risk if old_risk > 0 else 5.0
                else:
                    risk_mult = abs(old_entry - pos.tp1) / old_risk if old_risk > 0 else 1.8
                    risk_mult2 = abs(old_entry - pos.tp2) / old_risk if old_risk > 0 else 3.0
                    risk_mult3 = abs(old_entry - pos.tp3) / old_risk if old_risk > 0 else 5.0

                if pos.side == "LONG":
                    pos.tp1 = new_avg + new_risk_from_avg * risk_mult
                    pos.tp2 = new_avg + new_risk_from_avg * risk_mult2
                    pos.tp3 = new_avg + new_risk_from_avg * risk_mult3
                else:
                    pos.tp1 = new_avg - new_risk_from_avg * risk_mult
                    pos.tp2 = new_avg - new_risk_from_avg * risk_mult2
                    pos.tp3 = new_avg - new_risk_from_avg * risk_mult3

                pos.entry_price = new_avg
                pos.sl = pos.initial_sl
                pos.dca_added = True

        # LONG position management
        if pos.side == "LONG":
            if low <= pos.csl:
                if not pos.ht1:
                    pos.pnl = (pos.csl - pos.entry_price) / pos.entry_price * 100.0
                    pos.exit_reason = "stop_loss"
                else:
                    pos.pnl = 0.0
                    pos.exit_reason = "breakeven"
                pos.exit_price = pos.csl
                pos.exit_time = time_str
                pos.closed = True
                return

            if not pos.ht1 and high >= pos.tp1:
                pos.ht1 = True
                pos.pnl += (pos.tp1 - pos.entry_price) / pos.entry_price * 100.0 * pos.a1
                pos.csl = pos.entry_price
                return

            if pos.ht1 and not pos.ht2 and high >= pos.tp2:
                pos.ht2 = True
                pos.pnl += (pos.tp2 - pos.entry_price) / pos.entry_price * 100.0 * pos.a2
                return

            if high >= pos.tp3:
                remaining = 1.0 - (pos.a1 if pos.ht1 else 0) - (pos.a2 if pos.ht2 else 0)
                if remaining > 0:
                    pos.pnl += (pos.tp3 - pos.entry_price) / pos.entry_price * 100.0 * remaining
                pos.exit_reason = "take_profit"
                pos.exit_price = pos.tp3
                pos.exit_time = time_str
                pos.closed = True
                return

            entry_epoch = time_to_epoch(pos.entry_time)
            bar_epoch = time_to_epoch(time_str)
            if bar_epoch - entry_epoch > 86400:
                remaining = 1.0 - (pos.a1 if pos.ht1 else 0) - (pos.a2 if pos.ht2 else 0)
                close_price = I15m["close"][i]
                pos.pnl += (close_price - pos.entry_price) / pos.entry_price * 100.0 * remaining
                pos.exit_reason = "timed_out"
                pos.exit_price = close_price
                pos.exit_time = time_str
                pos.closed = True
                return

        # SHORT position management
        else:
            if high >= pos.csl:
                if not pos.ht1:
                    pos.pnl = (pos.entry_price - pos.csl) / pos.entry_price * 100.0
                    pos.exit_reason = "stop_loss"
                else:
                    pos.pnl = 0.0
                    pos.exit_reason = "breakeven"
                pos.exit_price = pos.csl
                pos.exit_time = time_str
                pos.closed = True
                return

            if not pos.ht1 and low <= pos.tp1:
                pos.ht1 = True
                pos.pnl += (pos.entry_price - pos.tp1) / pos.entry_price * 100.0 * pos.a1
                pos.csl = pos.entry_price
                return

            if pos.ht1 and not pos.ht2 and low <= pos.tp2:
                pos.ht2 = True
                pos.pnl += (pos.entry_price - pos.tp2) / pos.entry_price * 100.0 * pos.a2
                return

            if low <= pos.tp3:
                remaining = 1.0 - (pos.a1 if pos.ht1 else 0) - (pos.a2 if pos.ht2 else 0)
                if remaining > 0:
                    pos.pnl += (pos.entry_price - pos.tp3) / pos.entry_price * 100.0 * remaining
                pos.exit_reason = "take_profit"
                pos.exit_price = pos.tp3
                pos.exit_time = time_str
                pos.closed = True
                return

            entry_epoch = time_to_epoch(pos.entry_time)
            bar_epoch = time_to_epoch(time_str)
            if bar_epoch - entry_epoch > 86400:
                remaining = 1.0 - (pos.a1 if pos.ht1 else 0) - (pos.a2 if pos.ht2 else 0)
                close_price = I15m["close"][i]
                pos.pnl += (pos.entry_price - close_price) / pos.entry_price * 100.0 * remaining
                pos.exit_reason = "timed_out"
                pos.exit_price = close_price
                pos.exit_time = time_str
                pos.closed = True
                return

    def compute_metrics(self):
        all_t = self.all_trades
        total = len(all_t)
        if total == 0:
            return {"trades": 0, "win_rate": 0, "total_pnl": 0.0, "profit_factor": 0,
                    "avg_pnl": 0, "dca_pct": 0}

        wins = [t for t in all_t if t.pnl > 0]
        losses = [t for t in all_t if t.pnl <= 0]
        win_rate = len(wins) / total * 100.0
        total_pnl = sum(t.pnl for t in all_t)
        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else (float("inf") if total_profit > 0 else 0)
        avg_pnl = total_pnl / total if total > 0 else 0
        dca_count = sum(1 for t in all_t if t.dca_added)
        dca_pct = dca_count / total * 100.0 if total > 0 else 0
        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        max_win = max((t.pnl for t in all_t), default=0)
        max_loss = min((t.pnl for t in all_t), default=0)

        sym_data = {}
        for sym in SYMBOLS:
            st = [t for t in all_t if t.sym == sym]
            if st:
                sw = [t for t in st if t.pnl > 0]
                sym_data[sym] = {
                    "trades": len(st),
                    "pnl": sum(t.pnl for t in st),
                    "win_rate": len(sw) / len(st) * 100.0 if st else 0,
                }

        return {
            "trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "dca_count": dca_count,
            "dca_pct": dca_pct,
            "by_symbol": sym_data,
        }


if __name__ == "__main__":
    bt = EndellionBacktest()
    bt.run()
