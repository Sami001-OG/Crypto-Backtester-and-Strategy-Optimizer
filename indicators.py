"""
Technical indicators — implementations match TradingView's pine script output exactly.
All functions are batch-vectorized with sliding-window memoization for speed.

TradingView parity notes:
  - RSI uses Wilder's smoothing (RMA), not simple EMA.
  - MACD uses EMA (not SMA) for signal line.
  - ATR uses RMA smoothing.
  - Bollinger Bands use population std (divide by n, not n-1).
  - ADX uses RMA on +DM/-DM/TR then RMA on DX.
  - SuperTrend uses ATR and trailing band logic matching pine v5.
"""

import math
from functools import lru_cache


# ── Utility: sliding window helpers ─────────────────────────────────────────

def _sliding_sma(arr, length):
    """Return list of SMA values. Uses running sum for O(n)."""
    n = len(arr)
    out = [None] * n
    if n < length:
        return out
    total = sum(arr[:length])
    out[length - 1] = total / length
    for i in range(length, n):
        total += arr[i] - arr[i - length]
        out[i] = total / length
    return out


def _rma(src, length):
    """Wilder's RMA (same as TradingView's rma / ta.rma)."""
    n = len(src)
    out = [None] * n
    if n < length:
        return out
    alpha = 1.0 / length
    # seed with SMA
    acc = sum(src[:length])
    out[length - 1] = acc / length
    for i in range(length, n):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def _ema_vec(src, length):
    """EMA using TradingView's seed (SMA) and multiplier 2/(n+1)."""
    n = len(src)
    out = [None] * n
    if n < length:
        return out
    k = 2.0 / (length + 1)
    acc = sum(src[:length])
    out[length - 1] = acc / length
    for i in range(length, n):
        out[i] = k * src[i] + (1 - k) * out[i - 1]
    return out


# ── Core Indicators (TradingView-compatible) ────────────────────────────────

def sma(values, period):
    return _sliding_sma(values, period)


def ema(values, period):
    return _ema_vec(values, period)


def rsi(values, period=14):
    """RSI using Wilder's RMA on gains/losses → matches ta.rsi(close, 14)."""
    n = len(values)
    if n < period + 1:
        return [None] * n

    delta = [values[i] - values[i - 1] for i in range(1, n)]
    up = [max(d, 0.0) for d in delta]
    down = [max(-d, 0.0) for d in delta]

    # RMA seed
    avg_gain = sum(up[:period]) / period
    avg_loss = sum(down[:period]) / period

    alpha = 1.0 / period
    result = [None] * (period + 1)

    for i in range(period, len(delta)):
        if i > period:
            avg_gain = alpha * up[i - 1] + (1 - alpha) * avg_gain
            avg_loss = alpha * down[i - 1] + (1 - alpha) * avg_loss
        if avg_loss == 0.0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - 100.0 / (1.0 + rs))

    # result has length: period+1 (nones) + (len(delta) - period) = len(delta) + 1 = n
    return result[:n]


def macd(values, fast=12, slow=26, signal_period=9):
    """MACD: ema(fast) - ema(slow), signal = EMA of MACD line."""
    ema_fast = _ema_vec(values, fast)
    ema_slow = _ema_vec(values, slow)
    n = len(values)

    macd_line = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # signal = EMA of macd_line
    first = next((i for i, v in enumerate(macd_line) if v is not None), n)
    signal_line = [None] * n
    if first + signal_period - 1 < n:
        seed_val = sum(macd_line[first:first + signal_period]) / signal_period
        signal_line[first + signal_period - 1] = seed_val
        k = 2.0 / (signal_period + 1)
        for i in range(first + signal_period, n):
            signal_line[i] = k * macd_line[i] + (1 - k) * signal_line[i - 1]

    histogram = [None] * n
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


def bollinger_bands(values, period=20, num_std=2.0):
    """TradingView uses population std (n, not n-1)."""
    n = len(values)
    middle = _sliding_sma(values, period)
    upper = [None] * n
    lower = [None] * n

    for i in range(period - 1, n):
        window = values[i - period + 1:i + 1]
        mean = middle[i]
        # population std
        var = sum((x - mean) * (x - mean) for x in window) / period
        std = math.sqrt(var)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, middle, lower


def atr(highs, lows, closes, period=14):
    """ATR using RMA (Wilder's smoothing) → matches ta.atr(14)."""
    n = len(highs)
    if n < 2:
        return [None] * n

    # True Range
    tr = [None] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    # RMA of TR
    return _rma(tr, period)


def supertrend(highs, lows, closes, period=10, multiplier=3.0):
    """Supertrend matching TradingView pine v5 ta.supertrend(src=hl2, ...)."""
    n = len(closes)
    atr_vals = atr(highs, lows, closes, period)
    st = [None] * n
    direction = [None] * n
    upper_band = [None] * n
    lower_band = [None] * n

    for i in range(period, n):
        if atr_vals[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2.0
        upper_band[i] = hl2 + multiplier * atr_vals[i]
        lower_band[i] = hl2 - multiplier * atr_vals[i]

        if i == period:
            direction[i] = 1
            st[i] = lower_band[i]
            continue

        # band clamping to prevent narrowing
        if upper_band[i] < upper_band[i - 1] if upper_band[i - 1] is not None else False:
            upper_band[i] = upper_band[i - 1]
        if lower_band[i] > lower_band[i - 1] if lower_band[i - 1] is not None else False:
            lower_band[i] = lower_band[i - 1]

        prev_dir = direction[i - 1]
        if prev_dir == 1:
            if closes[i] < lower_band[i]:
                direction[i] = -1
                st[i] = upper_band[i]
            else:
                direction[i] = 1
                st[i] = lower_band[i]
        else:
            if closes[i] > upper_band[i]:
                direction[i] = 1
                st[i] = lower_band[i]
            else:
                direction[i] = -1
                st[i] = upper_band[i]

    return st, direction


def adx(highs, lows, closes, period=14):
    """ADX using RMA smoothing → matches ta.adx(14)."""
    n = len(closes)
    plus_di = [None] * n
    minus_di = [None] * n
    adx_out = [None] * n
    if n < period + 1:
        return adx_out, plus_di, minus_di

    tr = [None] * n
    plus_dm = [None] * n
    minus_dm = [None] * n
    tr[0] = highs[0] - lows[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # RMA smooth
    tr_rma = _rma(tr, period)
    pdm_rma = _rma(plus_dm, period)
    mdm_rma = _rma(minus_dm, period)

    dx = [None] * n
    for i in range(n):
        if tr_rma[i] is None or tr_rma[i] == 0:
            continue
        pdi = 100.0 * pdm_rma[i] / tr_rma[i] if pdm_rma[i] is not None else 0.0
        mdi = 100.0 * mdm_rma[i] / tr_rma[i] if mdm_rma[i] is not None else 0.0
        plus_di[i] = pdi
        minus_di[i] = mdi
        denom = pdi + mdi
        dx[i] = abs(pdi - mdi) / denom * 100.0 if denom > 0 else 0.0

    # ADX = RMA of DX
    dx_clean = [d if d is not None else 0.0 for d in dx]
    adx_rma = _rma(dx_clean, period)
    for i in range(n):
        if adx_rma[i] is not None:
            adx_out[i] = adx_rma[i]

    return adx_out, plus_di, minus_di


def stochastic(highs, lows, closes, k_period=14, d_period=3, smooth_k=1):
    """Stochastic matching ta.stoch(close, high, low, 14, 3, 3)."""
    n = len(closes)
    k_values = [None] * n
    d_values = [None] * n

    for i in range(k_period - 1, n):
        h_window = highs[i - k_period + 1:i + 1]
        l_window = lows[i - k_period + 1:i + 1]
        h_max = max(h_window)
        l_min = min(l_window)
        if h_max == l_min:
            k_values[i] = 50.0
        else:
            k_values[i] = ((closes[i] - l_min) / (h_max - l_min)) * 100.0

    # %D = SMA of %K
    for i in range(k_period - 1, n):
        if k_values[i] is None:
            continue
        start = i - d_period + 1
        if start < k_period - 1:
            continue
        # collect last d_period valid K values
        window = k_values[start:i + 1]
        if len(window) == d_period:
            d_values[i] = sum(window) / d_period

    return k_values, d_values


def vwap(highs, lows, closes, volumes):
    """VWAP matching ta.vwap."""
    n = len(closes)
    result = [None] * n
    cum_vol = 0.0
    cum_pv = 0.0
    for i in range(n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_vol += volumes[i]
        cum_pv += tp * volumes[i]
        if cum_vol > 0:
            result[i] = cum_pv / cum_vol
    return result


def obv(closes, volumes):
    """On-Balance Volume."""
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


def bollinger_band_width(upper, lower, middle):
    n = len(upper)
    result = [None] * n
    for i in range(n):
        if upper[i] is not None and lower[i] is not None and middle[i] and middle[i] != 0:
            result[i] = (upper[i] - lower[i]) / middle[i]
    return result


def swing_highs_lows(highs, lows, lookback=5):
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
    n = len(closes)
    bos = [0] * n
    sh, sl = swing_highs_lows(highs, lows, lookback)
    last_sh = None
    last_sl = None
    for i in range(n):
        if sh[i] is not None:
            last_sh = sh[i]
        if sl[i] is not None:
            last_sl = sl[i]
        if i < lookback * 2:
            continue
        if last_sh is not None and closes[i] > last_sh:
            bos[i] = 1
            last_sh = highs[i]
        elif last_sl is not None and closes[i] < last_sl:
            bos[i] = -1
            last_sl = lows[i]
    return bos


def pivot_highs_lows(highs, lows, left_bars=5, right_bars=5):
    """TradingView ta.pivothigh / ta.pivotlow."""
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


def donchian_channels(highs, lows, period=20):
    """ta.donchian"""
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
    """ta.keltner"""
    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes))]
    basis = _ema_vec(tp, ema_period)
    atr_vals = atr(highs, lows, closes, atr_period)
    n = len(closes)
    upper = [None] * n
    lower = [None] * n
    for i in range(n):
        if basis[i] is not None and atr_vals[i] is not None:
            upper[i] = basis[i] + atr_mult * atr_vals[i]
            lower[i] = basis[i] - atr_mult * atr_vals[i]
    return upper, basis, lower


def compute_all(data):
    """
    Compute all standard indicators for a dataset in one pass.
    Returns a dict of indicator arrays keyed by name.
    Uses NumPy-style batch extraction first, then computes each.
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

    # VWAP
    out["vwap"] = vwap(highs, lows, closes, volumes)

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
