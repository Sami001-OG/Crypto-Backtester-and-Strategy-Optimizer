"""
Technical indicators — exact ports of TradingView's Pine Script `ta.*` built-ins.

Pine parity notes (each verified against the Pine v5/v6 reference implementations):
  - EMA / RMA replicate Pine's na-handling: the value is na until a full window
    of non-na source values exists, then seeds with the SMA of that window
    (`sum := na(sum[1]) ? ta.sma(src, length) : alpha*src + (1-alpha)*sum[1]`).
  - RSI = 100 - 100/(1 + rma(gains)/rma(losses)), Wilder's smoothing.
  - MACD signal line is an EMA of the MACD line (na-led, SMA-seeded like Pine).
  - Bollinger Bands use population stdev (divide by n) like ta.stdev.
  - ATR uses ta.tr(handle_na=true): the first bar's TR is high-low.
  - ADX ports the built-in DMI exactly: ta.tr (na on bar 0, NOT high-low),
    RMA on +DM/-DM/TR, fixnan() on the DIs, and DX guarded with
    `sum == 0 ? 1 : sum` before the final RMA.
  - SuperTrend is a line-by-line port of Pine v5 `ta.supertrend` (nz() band
    seeding, close[1] ratchet rule, first value on the first ATR bar).
    NOTE: direction convention here is 1 = uptrend / -1 = downtrend, which is
    the NEGATION of TV's return value (TV uses -1 for uptrend). Values match.
  - Stochastic matches the built-in indicator: raw %K = ta.stoch (na when the
    window range is zero), %K = SMA(raw, smooth_k), %D = SMA(%K, d_period).
  - VWAP is session-anchored like TV (resets each UTC day by default via
    `anchors`); hlc3 source.
  - Keltner Channels use the TV defaults: EMA(close, 20) basis, ATR(10) bands.
  - pivot_highs_lows marks the pivot AT the pivot bar (like plotting
    `ta.pivothigh` offset back); a pivot is only KNOWN right_bars later — use
    pivots_confirmed() in strategies to avoid lookahead.

All functions accept/return plain lists; None represents Pine's na.
"""

import math
import datetime

_UTC = datetime.timezone.utc


# ── Core smoothing primitives (Pine-exact, na-aware) ────────────────────────

def _sliding_sma(arr, length):
    """ta.sma: na until the window holds `length` consecutive non-na values."""
    n = len(arr)
    out = [None] * n
    if length <= 0 or n < length:
        return out
    total = 0.0
    nones = 0
    for i in range(n):
        v = arr[i]
        if v is None:
            nones += 1
        else:
            total += v
        if i >= length:
            old = arr[i - length]
            if old is None:
                nones -= 1
            else:
                total -= old
        if i >= length - 1 and nones == 0:
            out[i] = total / length
    return out


def _recursive_ma(src, length, alpha):
    """
    Shared Pine pattern for ta.ema / ta.rma:
        sum := na(sum[1]) ? ta.sma(src, length) : alpha*src + (1-alpha)*sum[1]
    na source values poison the running value, after which it re-seeds from
    the next full SMA window — exactly like Pine.
    """
    n = len(src)
    out = [None] * n
    if length <= 0 or n == 0:
        return out
    total = 0.0
    nones = 0
    prev = None
    for i in range(n):
        v = src[i]
        if v is None:
            nones += 1
        else:
            total += v
        if i >= length:
            old = src[i - length]
            if old is None:
                nones -= 1
            else:
                total -= old
        if prev is None:
            if i >= length - 1 and nones == 0:
                prev = total / length
                out[i] = prev
        else:
            if v is None:
                prev = None            # poisoned; will re-seed via SMA
            else:
                prev = alpha * v + (1.0 - alpha) * prev
                out[i] = prev
    return out


def _rma(src, length):
    """Wilder's RMA — identical to Pine ta.rma (alpha = 1/length)."""
    return _recursive_ma(src, length, 1.0 / length)


def _ema_vec(src, length):
    """Pine ta.ema (alpha = 2/(length+1), SMA-seeded, na-aware)."""
    return _recursive_ma(src, length, 2.0 / (length + 1))


def _fixnan(arr):
    """Pine fixnan(): replace na with the last non-na value."""
    out = list(arr)
    last = None
    for i, v in enumerate(out):
        if v is None:
            out[i] = last
        else:
            last = v
    return out


# ── Moving averages ──────────────────────────────────────────────────────────

def sma(values, period):
    return _sliding_sma(values, period)


def ema(values, period):
    return _ema_vec(values, period)


# ── Oscillators ──────────────────────────────────────────────────────────────

def rsi(values, period=14):
    """ta.rsi: 100 - 100/(1 + rma(up, n)/rma(down, n)). 100 when avg loss is 0."""
    n = len(values)
    up = [None] * n
    down = [None] * n
    for i in range(1, n):
        d = values[i] - values[i - 1]
        up[i] = d if d > 0.0 else 0.0
        down[i] = -d if d < 0.0 else 0.0
    avg_up = _rma(up, period)
    avg_down = _rma(down, period)
    out = [None] * n
    for i in range(n):
        u, d = avg_up[i], avg_down[i]
        if u is None or d is None:
            continue
        if d == 0.0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + u / d)
    return out


def macd(values, fast=12, slow=26, signal_period=9):
    """ta.macd: EMA(fast) - EMA(slow); signal = EMA of the MACD line."""
    ema_fast = _ema_vec(values, fast)
    ema_slow = _ema_vec(values, slow)
    n = len(values)
    macd_line = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    signal_line = _ema_vec(macd_line, signal_period)
    histogram = [None] * n
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, histogram


def stochastic(highs, lows, closes, k_period=14, d_period=3, smooth_k=1):
    """
    TV built-in Stochastic: raw = ta.stoch(close, high, low, k_period)
    (na when the window's high == low, matching Pine's division-by-zero na),
    %K = SMA(raw, smooth_k), %D = SMA(%K, d_period). TV defaults: 14, 3, 1.
    """
    n = len(closes)
    raw = [None] * n
    for i in range(k_period - 1, n):
        h_max = max(highs[i - k_period + 1:i + 1])
        l_min = min(lows[i - k_period + 1:i + 1])
        if h_max != l_min:
            raw[i] = (closes[i] - l_min) / (h_max - l_min) * 100.0
    k_values = _sliding_sma(raw, smooth_k) if smooth_k > 1 else raw
    d_values = _sliding_sma(k_values, d_period)
    return k_values, d_values


# ── Bands / channels ─────────────────────────────────────────────────────────

def bollinger_bands(values, period=20, num_std=2.0):
    """ta.bb: SMA basis ± num_std * population stdev (ta.stdev divides by n)."""
    n = len(values)
    middle = _sliding_sma(values, period)
    upper = [None] * n
    lower = [None] * n
    if n >= period:
        sum_x = sum(values[:period])
        sum_x2 = sum(x * x for x in values[:period])
        for i in range(period - 1, n):
            mean = middle[i]
            variance = sum_x2 / period - mean * mean
            if variance < 1e-15:
                variance = 0.0
            std = math.sqrt(variance)
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std
            if i + 1 < n:
                old_val = values[i - period + 1]
                new_val = values[i + 1]
                sum_x += new_val - old_val
                sum_x2 += new_val * new_val - old_val * old_val
    return upper, middle, lower


def bollinger_band_width(upper, lower, middle):
    n = len(upper)
    result = [None] * n
    for i in range(n):
        if upper[i] is not None and lower[i] is not None and middle[i] and middle[i] != 0:
            result[i] = (upper[i] - lower[i]) / middle[i]
    return result


def donchian_channels(highs, lows, period=20):
    """TV Donchian Channels: highest/lowest of the last `period` bars (inclusive)."""
    n = len(highs)
    upper = [None] * n
    lower = [None] * n
    basis = [None] * n
    for i in range(period - 1, n):
        upper[i] = max(highs[i - period + 1:i + 1])
        lower[i] = min(lows[i - period + 1:i + 1])
        basis[i] = (upper[i] + lower[i]) / 2.0
    return upper, lower, basis


def keltner_channels(highs, lows, closes, ema_period=20, atr_mult=2.0, atr_period=10):
    """
    TV built-in Keltner Channels defaults: basis = EMA(CLOSE, 20) (close
    source, exponential MA), bands = basis ± 2 * ATR(10).
    """
    basis = _ema_vec(closes, ema_period)
    atr_vals = atr(highs, lows, closes, atr_period)
    n = len(closes)
    upper = [None] * n
    lower = [None] * n
    for i in range(n):
        if basis[i] is not None and atr_vals[i] is not None:
            upper[i] = basis[i] + atr_mult * atr_vals[i]
            lower[i] = basis[i] - atr_mult * atr_vals[i]
    return upper, basis, lower


# ── Volatility / trend strength ──────────────────────────────────────────────

def true_range(highs, lows, closes, handle_na=True):
    """ta.tr: bar 0 is high-low when handle_na else na (the `ta.tr` variable)."""
    n = len(highs)
    tr = [None] * n
    if n == 0:
        return tr
    if handle_na:
        tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
    return tr


def atr(highs, lows, closes, period=14):
    """ta.atr = rma(ta.tr(handle_na=true), period)."""
    return _rma(true_range(highs, lows, closes, handle_na=True), period)


def adx(highs, lows, closes, period=14, adx_smoothing=None):
    """
    Exact port of TV's built-in DMI / ta.adx:

        up = ta.change(high);  down = -ta.change(low)          // na on bar 0
        plusDM  = na(up)   ? na : (up > down and up > 0 ? up : 0)
        minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
        trur  = ta.rma(ta.tr, len)                             // ta.tr: na bar 0
        plus  = fixnan(100 * ta.rma(plusDM, len) / trur)
        minus = fixnan(100 * ta.rma(minusDM, len) / trur)
        sum   = plus + minus
        adx   = 100 * ta.rma(abs(plus - minus) / (sum == 0 ? 1 : sum), lensig)

    Returns (adx, plus_di, minus_di).
    """
    lensig = adx_smoothing if adx_smoothing else period
    n = len(closes)
    tr = true_range(highs, lows, closes, handle_na=False)
    plus_dm = [None] * n
    minus_dm = [None] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    trur = _rma(tr, period)
    pdm_rma = _rma(plus_dm, period)
    mdm_rma = _rma(minus_dm, period)

    plus_raw = [None] * n
    minus_raw = [None] * n
    for i in range(n):
        if trur[i] is None or trur[i] == 0 or pdm_rma[i] is None:
            continue
        plus_raw[i] = 100.0 * pdm_rma[i] / trur[i]
        minus_raw[i] = 100.0 * mdm_rma[i] / trur[i]
    plus_di = _fixnan(plus_raw)
    minus_di = _fixnan(minus_raw)

    dx = [None] * n
    for i in range(n):
        p, m = plus_di[i], minus_di[i]
        if p is None or m is None:
            continue
        s = p + m
        dx[i] = abs(p - m) / (1.0 if s == 0 else s)
    adx_out = [None] * n
    for i, v in enumerate(_rma(dx, lensig)):
        if v is not None:
            adx_out[i] = 100.0 * v
    return adx_out, plus_di, minus_di


def supertrend(highs, lows, closes, period=10, multiplier=3.0):
    """
    Line-by-line port of Pine v5 ta.supertrend(factor, atrPeriod), hl2 source:

        upperBand = hl2 + factor * atr;  lowerBand = hl2 - factor * atr
        lowerBand := lowerBand > nz(lowerBand[1]) or close[1] < nz(lowerBand[1])
                     ? lowerBand : lowerBand[1]
        upperBand := upperBand < nz(upperBand[1]) or close[1] > nz(upperBand[1])
                     ? upperBand : upperBand[1]
        if na(atr[1])                 → seed as DOWNTREND (ST = upper band)
        else if superTrend[1] == upperBand[1]
                                      → uptrend only when close > upperBand
        else                          → downtrend only when close < lowerBand

    Values match TV exactly. Direction convention: 1 = uptrend (ST is the
    lower band / support), -1 = downtrend — this is the NEGATION of TV's
    ta.supertrend direction output (TV returns -1 for uptrend).
    """
    n = len(closes)
    atr_vals = atr(highs, lows, closes, period)
    st = [None] * n
    direction = [None] * n
    final_upper = [None] * n
    final_lower = [None] * n

    for i in range(n):
        if atr_vals[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2.0
        basic_upper = hl2 + multiplier * atr_vals[i]
        basic_lower = hl2 - multiplier * atr_vals[i]

        prev_lower = final_lower[i - 1] if i > 0 and final_lower[i - 1] is not None else 0.0
        prev_upper = final_upper[i - 1] if i > 0 and final_upper[i - 1] is not None else 0.0
        prev_close = closes[i - 1] if i > 0 else None

        if basic_lower > prev_lower or (prev_close is not None and prev_close < prev_lower):
            final_lower[i] = basic_lower
        else:
            final_lower[i] = prev_lower
        if basic_upper < prev_upper or (prev_close is not None and prev_close > prev_upper):
            final_upper[i] = basic_upper
        else:
            final_upper[i] = prev_upper

        if i == 0 or atr_vals[i - 1] is None:
            direction[i] = -1                      # TV seeds as downtrend
        elif st[i - 1] == final_upper[i - 1]:      # was downtrend (ST on upper)
            direction[i] = 1 if closes[i] > final_upper[i] else -1
        else:                                       # was uptrend (ST on lower)
            direction[i] = -1 if closes[i] < final_lower[i] else 1
        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return st, direction


# ── Volume ───────────────────────────────────────────────────────────────────

def vwap(highs, lows, closes, volumes, anchors=None):
    """
    ta.vwap with session anchoring: hlc3 source, cumulative price*volume /
    volume, RESET whenever the `anchors` key changes (TV's default anchor is
    the session — one UTC day for 24/7 crypto). anchors=None gives a single
    cumulative VWAP over the whole dataset (previous behavior).
    Use day_keys(data) to build daily anchors.
    """
    n = len(closes)
    result = [None] * n
    cum_vol = 0.0
    cum_pv = 0.0
    prev_key = object()
    for i in range(n):
        if anchors is not None:
            key = anchors[i]
            if key is not None and key != prev_key:
                cum_vol = 0.0
                cum_pv = 0.0
                prev_key = key
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_vol += volumes[i]
        cum_pv += tp * volumes[i]
        if cum_vol > 0:
            result[i] = cum_pv / cum_vol
    return result


def obv(closes, volumes):
    """On-Balance Volume (ta.obv)."""
    n = len(closes)
    result = [0.0] * n
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            result[i] = result[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            result[i] = result[i - 1] - volumes[i]
        else:
            result[i] = result[i - 1]
    return result


def volume_sma(volumes, period=20):
    return _sliding_sma(volumes, period)


# ── Time helpers ─────────────────────────────────────────────────────────────

_DT_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


def _row_epoch(row):
    try:
        v = float(row.get("open_time", ""))
        while v > 1e11:        # normalize ms (~1e12) / µs (~1e15) to seconds
            v /= 1000.0
        if v > 1e8:
            return v
    except (TypeError, ValueError):
        pass
    s = row.get("datetime", "")
    for fmt in _DT_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt).replace(tzinfo=_UTC).timestamp()
        except (ValueError, TypeError):
            continue
    return None


def day_keys(data):
    """
    One key per bar identifying its UTC day — the session anchor TV uses for
    VWAP on 24/7 crypto symbols. Bars with unparseable timestamps inherit the
    previous bar's key (no spurious resets).
    """
    keys = []
    last = None
    for row in data:
        ep = _row_epoch(row)
        if ep is not None:
            last = int(ep // 86400.0)
        keys.append(last)
    return keys


# ── Structure (pivots / swings) ──────────────────────────────────────────────

def pivot_highs_lows(highs, lows, left_bars=5, right_bars=5):
    """
    TradingView ta.pivothigh / ta.pivotlow (strict > / < against both sides).
    The value is stored AT the pivot bar, but in Pine it only becomes known
    `right_bars` later — feed strategies through pivots_confirmed() to avoid
    lookahead bias.
    """
    n = len(highs)
    ph = [None] * n
    pl = [None] * n
    for i in range(left_bars, n - right_bars):
        h_val = highs[i]
        if all(highs[i - j] < h_val for j in range(1, left_bars + 1)) and \
           all(highs[i + j] < h_val for j in range(1, right_bars + 1)):
            ph[i] = h_val
        l_val = lows[i]
        if all(lows[i - j] > l_val for j in range(1, left_bars + 1)) and \
           all(lows[i + j] > l_val for j in range(1, right_bars + 1)):
            pl[i] = l_val
    return ph, pl


def pivots_confirmed(pivots, right_bars):
    """
    Shift pivot values to the bar where TV would CONFIRM them (pivot bar +
    right_bars) — the earliest bar a strategy may act on them without
    lookahead. Mirrors reading ta.pivothigh() (non-na `right_bars` after).
    """
    n = len(pivots)
    out = [None] * n
    for i, v in enumerate(pivots):
        if v is not None and i + right_bars < n:
            out[i + right_bars] = v
    return out


def swing_highs_lows(highs, lows, lookback=5):
    """
    Centered swing marks. WARNING: uses future bars (lookahead) — the mark at
    bar i needs bars i+1..i+lookback. For live-tradeable logic, shift with
    pivots_confirmed(sh, lookback).
    """
    n = len(highs)
    sh = [None] * n
    sl = [None] * n
    for i in range(lookback, n - lookback):
        if highs[i] > max(highs[i - lookback:i]) and highs[i] > max(highs[i + 1:i + lookback + 1]):
            sh[i] = highs[i]
        if lows[i] < min(lows[i - lookback:i]) and lows[i] < min(lows[i + 1:i + lookback + 1]):
            sl[i] = lows[i]
    return sh, sl


def detect_bos(closes, highs, lows, lookback=5):
    """
    Break-of-structure vs the last CONFIRMED swing (swings are only acted on
    once their right-hand window has fully printed — no lookahead in the
    break test itself).
    """
    n = len(closes)
    bos = [0] * n
    sh_raw, sl_raw = swing_highs_lows(highs, lows, lookback)
    sh = pivots_confirmed(sh_raw, lookback)
    sl = pivots_confirmed(sl_raw, lookback)
    last_sh = None
    last_sl = None
    for i in range(n):
        if sh[i] is not None:
            last_sh = sh[i]
        if sl[i] is not None:
            last_sl = sl[i]
        if last_sh is not None and closes[i] > last_sh:
            bos[i] = 1
            last_sh = None
        elif last_sl is not None and closes[i] < last_sl:
            bos[i] = -1
            last_sl = None
    return bos


# ── Batch compute ────────────────────────────────────────────────────────────

def compute_all(data):
    """
    Compute all standard indicators for a dataset in one pass.
    Returns a dict of indicator arrays keyed by name.
    """
    n = len(data)
    closes = [data[i]["close"] for i in range(n)]
    highs = [data[i]["high"] for i in range(n)]
    lows = [data[i]["low"] for i in range(n)]
    volumes = [data[i]["volume"] for i in range(n)]

    out = {}

    # Moving averages
    out["sma_20"] = _sliding_sma(closes, 20)
    out["sma_50"] = _sliding_sma(closes, 50)
    out["sma_200"] = _sliding_sma(closes, 200)
    out["ema_12"] = _ema_vec(closes, 12)
    out["ema_26"] = _ema_vec(closes, 26)
    out["ema_50"] = _ema_vec(closes, 50)

    # RSI
    out["rsi_14"] = rsi(closes, 14)

    # MACD
    macd_l, sig_l, hist_l = macd(closes, 12, 26, 9)
    out["macd_line"] = macd_l
    out["macd_signal"] = sig_l
    out["macd_hist"] = hist_l

    # Bollinger
    bb_u, bb_m, bb_l = bollinger_bands(closes, 20, 2.0)
    out["bb_upper"] = bb_u
    out["bb_middle"] = bb_m
    out["bb_lower"] = bb_l
    out["bb_width"] = bollinger_band_width(bb_u, bb_l, bb_m)

    # ATR
    out["atr_14"] = atr(highs, lows, closes, 14)

    # Supertrend
    st_val, st_dir = supertrend(highs, lows, closes, 10, 3.0)
    out["supertrend_val"] = st_val
    out["supertrend_dir"] = st_dir

    # ADX
    adx_v, pdi, mdi = adx(highs, lows, closes, 14)
    out["adx"] = adx_v
    out["plus_di"] = pdi
    out["minus_di"] = mdi

    # Stochastic
    stok_k, stok_d = stochastic(highs, lows, closes, 14, 3)
    out["stoch_k"] = stok_k
    out["stoch_d"] = stok_d

    # VWAP (session-anchored per UTC day, like TV)
    out["vwap"] = vwap(highs, lows, closes, volumes, anchors=day_keys(data))

    # Volume
    out["obv"] = obv(closes, volumes)
    out["vol_sma_20"] = _sliding_sma(volumes, 20)

    # Donchian
    dc_u, dc_l, dc_b = donchian_channels(highs, lows, 20)
    out["donchian_upper"] = dc_u
    out["donchian_lower"] = dc_l
    out["donchian_basis"] = dc_b

    # Keltner
    kc_u, kc_b, kc_l = keltner_channels(highs, lows, closes, 20, 2.0, 10)
    out["keltner_upper"] = kc_u
    out["keltner_basis"] = kc_b
    out["keltner_lower"] = kc_l

    return out
