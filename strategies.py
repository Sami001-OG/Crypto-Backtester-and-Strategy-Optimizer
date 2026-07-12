"""
Trading strategies — each follows the uniform interface:

    def generate_signals(data, **params) -> list[int]

    data: list of dicts with keys: open, high, low, close, volume, datetime
    returns: list[int] of signals (1=long, -1=short/exit, 0=flat)

Includes both classic strategies and multi-timeframe composite strategies.
"""

import indicators as ind

# ═════════════════════════════════════════════════════════════════════════
# Classic Single-Timeframe Strategies
# ═════════════════════════════════════════════════════════════════════════

def sma_crossover(data, fast_period=10, slow_period=50):
    closes = [d["close"] for d in data]
    fast = ind.sma(closes, fast_period)
    slow = ind.sma(closes, slow_period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (fast[i], slow[i], fast[i - 1], slow[i - 1]):
            continue
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            signals[i] = 1
        elif fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def ema_crossover(data, fast_period=12, slow_period=26):
    closes = [d["close"] for d in data]
    fast = ind.ema(closes, fast_period)
    slow = ind.ema(closes, slow_period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (fast[i], slow[i], fast[i - 1], slow[i - 1]):
            continue
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            signals[i] = 1
        elif fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def golden_cross(data, fast_period=50, slow_period=200):
    """Classic Golden Cross / Death Cross: 50/200 SMA crossover."""
    closes = [d["close"] for d in data]
    fast = ind.sma(closes, fast_period)
    slow = ind.sma(closes, slow_period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (fast[i], slow[i], fast[i - 1], slow[i - 1]):
            continue
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            signals[i] = 1
        elif fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def rsi_strategy(data, period=14, oversold=30, overbought=70):
    closes = [d["close"] for d in data]
    rsi_vals = ind.rsi(closes, period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if rsi_vals[i] is None or rsi_vals[i - 1] is None:
            continue
        if rsi_vals[i - 1] < oversold and rsi_vals[i] >= oversold:
            signals[i] = 1
        elif rsi_vals[i - 1] < overbought and rsi_vals[i] >= overbought:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def macd_strategy(data, fast=12, slow=26, signal_period=9):
    closes = [d["close"] for d in data]
    macd_line, signal_line, histogram = ind.macd(closes, fast, slow, signal_period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if histogram[i] is None or histogram[i - 1] is None:
            continue
        if histogram[i] > 0 and histogram[i - 1] <= 0:
            signals[i] = 1
        elif histogram[i] < 0 and histogram[i - 1] >= 0:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def bollinger_band_strategy(data, period=20, num_std=2.0):
    closes = [d["close"] for d in data]
    upper, middle, lower = ind.bollinger_bands(closes, period, num_std)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if upper[i] is None or lower[i] is None:
            continue
        if closes[i] <= lower[i]:
            signals[i] = 1
        elif closes[i] >= upper[i]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def supertrend_strategy(data, period=10, multiplier=3.0):
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    closes = [d["close"] for d in data]
    st_vals, direction = ind.supertrend(highs, lows, closes, period, multiplier)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if direction[i] is None or direction[i - 1] is None:
            continue
        if direction[i] == 1 and direction[i - 1] == -1:
            signals[i] = 1
        elif direction[i] == -1 and direction[i - 1] == 1:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def stochastic_strategy(data, k_period=14, d_period=3, oversold=20, overbought=80):
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    closes = [d["close"] for d in data]
    k_vals, d_vals = ind.stochastic(highs, lows, closes, k_period, d_period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (k_vals[i], d_vals[i], k_vals[i - 1], d_vals[i - 1]):
            continue
        if k_vals[i] > d_vals[i] and k_vals[i - 1] <= d_vals[i - 1] and k_vals[i] < oversold:
            signals[i] = 1
        elif k_vals[i] < d_vals[i] and k_vals[i - 1] >= d_vals[i - 1] and k_vals[i] > overbought:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def adx_strategy(data, period=14, threshold=25):
    """Buy when +DI crosses above -DI while ADX > threshold (trend is strong)."""
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    closes = [d["close"] for d in data]
    adx_vals, plus_di, minus_di = ind.adx(highs, lows, closes, period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (adx_vals[i], plus_di[i], minus_di[i], plus_di[i - 1], minus_di[i - 1]):
            continue
        if plus_di[i] > minus_di[i] and plus_di[i - 1] <= minus_di[i - 1] and adx_vals[i] > threshold:
            signals[i] = 1
        elif plus_di[i] < minus_di[i] and plus_di[i - 1] >= minus_di[i - 1] and adx_vals[i] > threshold:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def donchian_breakout(data, period=20):
    """
    Donchian breakout: go long when price closes above the PRIOR bar's Donchian
    high, short when it closes below the prior bar's Donchian low. Comparing to
    the previous bar's channel (which excludes the current bar) is what makes a
    breakout detectable — the current bar's own high always defines its channel
    top, so ``close > upper[i]`` can never be true.
    """
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    closes = [d["close"] for d in data]
    dc_upper, dc_lower, dc_basis = ind.donchian_channels(highs, lows, period)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if dc_upper[i - 1] is None or dc_lower[i - 1] is None:
            continue
        if closes[i] > dc_upper[i - 1]:
            signals[i] = 1
        elif closes[i] < dc_lower[i - 1]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


def vwap_reversion(data, lookback=20):
    """Mean reversion around VWAP with Bollinger confirmation."""
    closes = [d["close"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    volumes = [d["volume"] for d in data]
    vwap_vals = ind.vwap(highs, lows, closes, volumes)
    bb_upper, bb_mid, bb_lower = ind.bollinger_bands(closes, lookback, 2.0)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (vwap_vals[i], bb_upper[i], bb_lower[i]):
            continue
        if closes[i] < vwap_vals[i] and closes[i] <= bb_lower[i]:
            signals[i] = 1
        elif closes[i] > vwap_vals[i] and closes[i] >= bb_upper[i]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


# ═════════════════════════════════════════════════════════════════════════
# Multi-Timeframe Composite Strategies
# ═════════════════════════════════════════════════════════════════════════

def mtf_alignment(data, trend_tf_data=None):
    """
    Multi-timeframe alignment: use higher-TF trend (separate data)
    as filter, and execute on lower-TF signals.

    If trend_tf_data is None, compute SMA 50/200 on the same data as the trend filter.
    """
    closes = [d["close"] for d in data]
    sma50 = ind.sma(closes, 50)
    sma200 = ind.sma(closes, 200)

    if trend_tf_data is not None:
        tf_closes = [d["close"] for d in trend_tf_data]
        tf_sma50 = ind.sma(tf_closes, 50)
        tf_sma200 = ind.sma(tf_closes, 200)
        # Map TF trend to current timeframe by index ratio
        ratio = len(data) // len(trend_tf_data) if trend_tf_data else 1
        trend_bullish = [False] * len(data)
        for i in range(len(data)):
            tf_idx = min(i // ratio, len(tf_closes) - 1)
            if tf_sma50[tf_idx] is not None and tf_sma200[tf_idx] is not None:
                trend_bullish[i] = tf_sma50[tf_idx] > tf_sma200[tf_idx]
    else:
        trend_bullish = [False] * len(data)
        for i in range(len(data)):
            if sma50[i] is not None and sma200[i] is not None:
                trend_bullish[i] = sma50[i] > sma200[i]

    # RSI entry on lower TF
    rsi_vals = ind.rsi(closes, 14)
    signals = [0] * len(data)
    for i in range(1, len(data)):
        if rsi_vals[i] is None or rsi_vals[i - 1] is None:
            continue
        if trend_bullish[i]:
            if rsi_vals[i - 1] < 30 and rsi_vals[i] >= 30:
                signals[i] = 1
            elif rsi_vals[i - 1] > 70 and rsi_vals[i] <= 70:
                signals[i] = 0
            else:
                signals[i] = signals[i - 1]
        else:
            signals[i] = 0
    return signals


def bband_squeeze_breakout(data, period=20, squeeze_threshold=0.03):
    """
    Bollinger Band squeeze breakout: enter when BB width expands after compression.
    """
    closes = [d["close"] for d in data]
    bb_upper, bb_mid, bb_lower = ind.bollinger_bands(closes, period, 2.0)
    bb_width = ind.bollinger_band_width(bb_upper, bb_lower, bb_mid)

    signals = [0] * len(data)
    squeeze = False
    trigger_bar = -10

    for i in range(1, len(data)):
        if bb_width[i] is None:
            continue

        if bb_width[i] < squeeze_threshold:
            squeeze = True
            trigger_bar = i

        if squeeze and i > trigger_bar:
            if bb_width[i] > bb_width[i - 1] and closes[i] > bb_upper[i]:
                signals[i] = 1
                squeeze = False
            elif bb_width[i] > bb_width[i - 1] and closes[i] < bb_lower[i]:
                signals[i] = -1
                squeeze = False
            else:
                signals[i] = signals[i - 1]
        else:
            signals[i] = signals[i - 1]

    return signals


def triple_ema_cross(data, period1=5, period2=20, period3=50):
    """
    Triple EMA: fast crosses above mid AND mid crosses above slow → long.
    Reverse → short.
    """
    closes = [d["close"] for d in data]
    e1 = ind.ema(closes, period1)
    e2 = ind.ema(closes, period2)
    e3 = ind.ema(closes, period3)

    signals = [0] * len(data)
    for i in range(1, len(data)):
        if None in (e1[i], e2[i], e3[i]):
            continue
        # Bullish alignment: price > ema1 > ema2 > ema3
        if closes[i] > e1[i] > e2[i] > e3[i]:
            signals[i] = 1
        elif closes[i] < e1[i] < e2[i] < e3[i]:
            signals[i] = -1
        else:
            signals[i] = signals[i - 1]
    return signals


# ═════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "sma_crossover": sma_crossover,
    "ema_crossover": ema_crossover,
    "golden_cross": golden_cross,
    "rsi": rsi_strategy,
    "macd": macd_strategy,
    "bollinger": bollinger_band_strategy,
    "supertrend": supertrend_strategy,
    "stochastic": stochastic_strategy,
    "adx": adx_strategy,
    "donchian": donchian_breakout,
    "vwap_reversion": vwap_reversion,
    "mtf_alignment": mtf_alignment,
    "bband_squeeze": bband_squeeze_breakout,
    "triple_ema": triple_ema_cross,
}
