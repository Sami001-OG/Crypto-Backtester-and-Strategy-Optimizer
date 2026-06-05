# Backtester and Optimizer

A TradingView-compatible backtesting framework for crypto strategies. Pure Python, zero external dependencies.

---

## Files

| File | Purpose |
|------|---------|
| `backtester.py` | Core engine — bar-magnitude execution, 30+ metrics, stop/take-profit |
| `indicators.py` | 31 technical indicators matching TradingView's pine script output |
| `strategies.py` | 14 built-in strategies plus multi-timeframe composites |
| `optimizer.py` | Parallel grid search, walk-forward, and adaptive random search |
| `download_data.py` | Downloads monthly kline CSVs from Binance Vision |
| `preprocess_data.py` | Converts raw timestamps, adds datetime/returns columns |

---

## Quick Start

```python
from backtester import Backtester, load_data
from strategies import ema_crossover

data = load_data("BTC_1h.csv")
signals = ema_crossover(data, 12, 26)

bt = Backtester(data, initial_capital=10000, fee_pct=0.04)
results = bt.run(signals)
Backtester.print_report(results)
```

Output:

```
            STRATEGY TESTER REPORT
═══════════════════════════════════════════════
  ── PERFORMANCE ──
  Net Profit                       $13,336.94
  Net Profit %                        133.37%
  CAGR                                 15.41%
  ── RISK ──
  Max Drawdown %                       63.52%
  ── RATIOS ──
  Sharpe Ratio                          0.557
  Sortino Ratio                         0.433
  Profit Factor                          1.10
  ── TRADE STATISTICS ──
  Total Trades                            914
  Win Rate                             27.68%
  Avg Trade Duration                    28.8h
```

---

## Execution Model

Matches **TradingView's Strategy Tester** exactly:

1. **Bar-magnitude**: signal generated at bar `i` → executes at bar `i+1` open (no lookahead bias)
2. **Stop/Take-profit**: checked intra-bar using high/low — stop uses low for longs, take-profit uses high
3. **Commission**: applied on every fill as % of order value (TV default: 0.04% for crypto)
4. **Slippage**: configurable as % of price or tick count

---

## Indicators (31 total)

All match TradingView's pine script `ta.*` output:

| Indicator | TV Equivalent | Notes |
|-----------|---------------|-------|
| SMA | `ta.sma` | Running sum, O(n) |
| EMA | `ta.ema` | SMA seed, `2/(n+1)` multiplier |
| RSI | `ta.rsi` | Wilder's RMA smoothing, not EMA |
| MACD | `ta.macd` | EMA of MACD for signal line |
| Bollinger Bands | `ta.bb` | Population std (n), not sample std |
| ATR | `ta.atr` | RMA (Wilder's) smoothing |
| SuperTrend | `ta.supertrend` | Band clamping, hl2 source |
| ADX | `ta.adx` | RMA on +DM/-DM, RMA on DX |
| Stochastic | `ta.stoch` | SMA of %K for %D |
| VWAP | `ta.vwap` | Cumulative typical price × volume |
| Donchian | `ta.donchian` | 20-period by default |
| Keltner | `ta.keltner` | EMA basis, ATR multiplier |

Batch compute all at once:

```python
from indicators import compute_all
all_indicators = compute_all(data)
# all_indicators["rsi_14"], all_indicators["macd_line"], etc.
```

---

## Strategies (14 built-in)

| Strategy | Parameters |
|----------|------------|
| `sma_crossover` | fast_period, slow_period |
| `ema_crossover` | fast_period, slow_period |
| `golden_cross` | fast_period=50, slow_period=200 |
| `rsi` | period, oversold=30, overbought=70 |
| `macd` | fast=12, slow=26, signal_period=9 |
| `bollinger` | period=20, num_std=2 |
| `supertrend` | period=10, multiplier=3 |
| `stochastic` | k_period=14, d_period=3, oversold=20, overbought=80 |
| `adx` | period=14, threshold=25 |
| `donchian` | period=20 |
| `vwap_reversion` | lookback=20 |
| `mtf_alignment` | Higher-TF trend filter on lower-TF RSI entries |
| `bband_squeeze` | Bollinger squeeze breakout |
| `triple_ema` | Triple EMA alignment |

Add your own — just write a function that returns `list[int]` (1=long, -1=short, 0=flat) and register it in the `STRATEGIES` dict.

---

## Optimizer

### Grid Search (parallel)

```python
from optimizer import grid_search

best = grid_search(
    "BTC_1h.csv", "sma_crossover",
    {"fast_period": [10, 20, 50], "slow_period": [100, 200]},
    rank_by="sharpe_ratio",  # or "total_return_pct", "profit_factor"
    parallel=True,
)
```

### Walk-Forward

```python
from optimizer import walk_forward

params, train_metrics, test_metrics = walk_forward(
    "BTC_1h.csv", "rsi",
    {"period": [7, 14, 21], "oversold": [25, 30], "overbought": [70, 75]},
    train_ratio=0.7,
)
```

### Rolling Walk-Forward (N folds)

```python
from optimizer import rolling_walk_forward

fold_results = rolling_walk_forward(
    "BTC_1h.csv", "ema_crossover",
    {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    windows=4,
)
```

### Adaptive Random Search

For large parameter spaces (>10⁶ combos):

```python
from optimizer import adaptive_search

best = adaptive_search(
    "BTC_1h.csv", "bollinger",
    {"period": (5, 100, "int"), "num_std": (1.0, 4.0, "float")},
    iterations=500,
)
```

---

## Metrics (30+)

All returned in the results dict:

| Metric | Key | Description |
|--------|-----|-------------|
| Net Profit | `net_profit` | Final equity minus initial |
| Return % | `total_return_pct` | Total return as percentage |
| Buy & Hold % | `buy_and_hold_return_pct` | Benchmark return |
| CAGR % | `cagr_pct` | Compound annual growth rate |
| Max Drawdown % | `max_drawdown_pct` | Largest peak-to-trough decline |
| Max Drawdown $ | `max_drawdown_dollar` | Dollar value of max DD |
| Sharpe Ratio | `sharpe_ratio` | Risk-adjusted return (annualized) |
| Sortino Ratio | `sortino_ratio` | Downside risk-adjusted return |
| Calmar Ratio | `calmar_ratio` | Return / max drawdown |
| Profit Factor | `profit_factor` | Gross profit / gross loss |
| Win Rate | `win_rate_pct` | Percentage of winning trades |
| Expectancy | `expectancy` | Average expected P&L per trade |
| Exposure | `exposure_pct` | Percentage of time in market |
| Total Fees | `total_fees` | Sum of all commissions paid |
| Long P&L | `long_pnl` | Net P&L from long trades |
| Short P&L | `short_pnl` | Net P&L from short trades |

---

## Data Pipeline

```
download_data.py ──► raw CSVs ──► preprocess_data.py ──► clean CSVs
  (Binance Vision)                                          (for backtest)
```

- **Download**: `python download_data.py` — fetches monthly zips from Binance Vision for BTC, ETH, SOL at 5m/15m/1h/4h
- **Preprocess**: `python preprocess_data.py` — converts Unix ms timestamps to `datetime`, adds `returns` column
- Expected CSV columns: `open_time, open, high, low, close, volume, close_time, quote_volume, trades, datetime, returns`

---

## Backtester Parameters

```python
Backtester(
    data,                          # list of dicts from load_data()
    initial_capital=10000,         # starting capital in USD
    fee_pct=0.04,                  # commission per trade (%)
    slippage_ticks=0,              # slippage in ticks
    slippage_pct=0.0,              # slippage as % of price
    stop_loss_pct=None,            # stop loss % (e.g. 2 = 2%)
    take_profit_pct=None,          # take profit % (e.g. 5 = 5%)
    long_only=True,                # restrict to long trades only
    pyramiding=1,                  # max concurrent positions
    qty_type="equity_pct",         # "equity_pct" or "fixed"
    qty_value=100,                 # % of equity or fixed units
)
```

---

## Requirements

Python 3.7+ — **no pip packages required** (standard library only: `csv`, `os`, `math`, `json`, `datetime`, `urllib`, `zipfile`, `io`, `itertools`, `concurrent.futures`)
