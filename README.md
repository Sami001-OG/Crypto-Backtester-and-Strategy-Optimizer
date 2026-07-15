# Backtester and Optimizer

A TradingView-compatible backtesting framework for crypto strategies. Pure Python, zero external dependencies.

The engine replicates **TradingView's Strategy Tester** — the broker emulator's fill rules, percent-of-equity sizing, intrabar drawdown, and the Performance Summary formulas — so a strategy ported from Pine Script produces the same trades and the same headline numbers here.

---

## Files

| File | Purpose |
|------|---------|
| `backtester.py` | Core engine — TV broker-emulator execution, Performance Summary metrics |
| `indicators.py` | Technical indicators, exact ports of Pine `ta.*` built-ins |
| `strategies.py` | 16 built-in strategies plus multi-timeframe composites |
| `optimizer.py` | Parallel grid search, walk-forward, and adaptive random search |
| `test_tv_parity.py` | 115 hand-computed TV-parity checks — run `python test_tv_parity.py` |
| `download_data.py` | Downloads full-history monthly kline CSVs from Binance Vision |
| `update_data.py` | Incrementally tops up the CSVs to the latest closed bar (Binance REST) |
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

---

## Execution model (TradingView broker emulator)

Every rule below matches TV's documented behavior:

1. **Order timing** — market orders are created on the signal bar's close and fill at the **next bar's open** (entries and signal exits alike). Set `process_orders_on_close=True` for TV's same-close fill option.
2. **Position sizing** (`qty_type="equity_pct"`) — quantity is fixed when the order is *created*: `qty = pct% × strategy.equity / signal-bar close`, where equity = initial + closed net profit + open position P&L. The order then fills at the next open with that quantity — TV's documented "sizing drift", reproduced deliberately. `qty_step` rounds down to the symbol's minimum order size (TV: 0.000001 for BTCUSD) and skips sub-minimum orders.
3. **Stop-loss / take-profit** — resting stop/limit orders, active from the moment the entry fills (they can close the trade on the **entry bar itself**, like `strategy.exit` placed with the entry).
4. **Gap rule** — if a bar *opens* beyond the stop/limit price, the fill happens at the open, not at the order price.
5. **Intrabar path assumption** — when both SL and TP sit inside one bar, the emulator assumes price moved **open → high → low → close** if the open is closer to the high, else **open → low → high → close**, and fills whichever order that path touches first. (TV Premium's Bar Magnifier is not emulated — that needs lower-timeframe data.)
6. **Slippage** — `slippage_ticks` (TV-style) applies adversely to market and stop fills; **limit fills are never slipped**. `slippage_pct` is a non-TV convenience with the same rules.
7. **Commission** — `fee_pct` % of order value on every fill, deducted from equity (TV "commission percent").
8. **Margin** — none (TV Pine v5 default `margin_long = margin_short = 0`): orders always fill, fees can push cash slightly negative, and no margin-call liquidation is simulated. A 1x short that gets run over can take equity below zero — same as TV with margin 0.
9. **End of data** — an open position is *left open* (TV behavior): `net_profit` counts closed trades only, the open trade is reported in `open_trade` / `open_pnl`, and `final_equity` includes it. Set `force_close_end=True` for realized-only accounting.

### Signals

A strategy returns a *state* list, one int per bar: `1` = want long, `-1` = want short (or exit-to-flat when `long_only=True`), `0` = flat. The engine diffs that state against the current position each close and issues the orders — equivalent to calling `strategy.entry()` while the condition holds.

---

## Metrics (TV Performance Summary)

| TV row | Key | Notes |
|--------|-----|-------|
| Net Profit | `net_profit`, `net_profit_pct` | Closed trades only, commissions included |
| Open P&L | `open_pnl`, `open_trade` | Unrealized, reported separately like TV |
| Gross Profit / Loss | `gross_profit`, `gross_loss` | |
| Buy & Hold Return | `buy_and_hold_return_pct` | All funds in at the **first trade's entry**, held to the last close |
| Max Equity Drawdown | `max_drawdown_dollar`, `max_drawdown_pct` | **Intrabar**: uses bar low/high equity while in a position, path-ordered on exit bars (TV's documented formula) |
| Max Equity Run-up | `max_runup_dollar`, `max_runup_pct` | Intrabar, mirror of drawdown |
| Sharpe Ratio | `sharpe_ratio` | TV formula: **monthly** equity returns, `(MR − RFR/12) / population stdev`, default `risk_free_rate=2` %/yr, **not annualized**. Needs ≥ 2 completed calendar months (`months_used`) |
| Sortino Ratio | `sortino_ratio` | Same, denominator = downside deviation `sqrt(Σ r<0 r² / N)` |
| Profit Factor | `profit_factor` | gross profit / gross loss |
| Commission Paid | `commission_paid` (= `total_fees`) | |
| Total / Winning / Losing Trades | `num_trades`, `num_winning`, `num_losing`, `num_even` | Breakeven trades count in the total, neither win nor loss |
| Percent Profitable | `win_rate_pct` | wins / total closed |
| Avg P&L, Avg Win, Avg Loss, Ratio | `avg_pnl`, `avg_win`, `avg_loss`, `ratio_avg_win_loss` | |
| Largest Win / Loss | `max_win_dollar`, `max_loss_dollar`, `largest_win_pct`, `largest_loss_pct` | % is trade P&L over entry value |
| Avg # Bars in Trades / Wins / Losses | `avg_bars_held`, `avg_bars_win`, `avg_bars_loss` | |

Extras beyond TV (kept for the optimizer): `cagr_pct`, `calmar_ratio`, `expectancy`, `exposure_pct`, `avg_trade_duration_hours`, `monthly_returns`, `long_pnl` / `short_pnl`, `total_return_pct` (includes open P&L).

---

## Indicators — exact Pine `ta.*` ports

| Indicator | TV equivalent | Parity details |
|-----------|---------------|----------------|
| SMA | `ta.sma` | na-aware window (na until a full clean window) |
| EMA | `ta.ema` | Pine recursion: na until seeded with the SMA of the first full non-na window, `alpha = 2/(n+1)` |
| RMA | `ta.rma` | Same, `alpha = 1/n` (Wilder) |
| RSI | `ta.rsi` | `100 − 100/(1 + rma(up)/rma(down))`; 100 when avg loss is 0 |
| MACD | `ta.macd` | Signal = EMA of the (na-led) MACD line, SMA-seeded like Pine |
| Bollinger | `ta.bb` | Population stdev (÷ n) |
| ATR | `ta.atr` | `rma(ta.tr(handle_na=true))` — first bar TR = high − low |
| ADX / DMI | built-in DMI | Line-by-line port: `ta.tr` **na on bar 0**, RMA chains, `fixnan()` on the DIs, DX guarded with `sum == 0 ? 1 : sum`. First ADX value lands at bar `2·len − 1`, like TV |
| SuperTrend | `ta.supertrend` | Line-by-line port of the Pine v5 source (nz() band seeding, close[1] ratchets, first value on the first ATR bar, seeded as downtrend). **Convention:** here `direction = 1` means uptrend — the *negation* of TV's return (TV uses −1 for up). Values are identical |
| Stochastic | built-in Stochastic | raw `ta.stoch` (na when window range is 0), `%K = sma(raw, smooth_k)`, `%D = sma(%K, d)`; TV defaults 14/3/1 |
| VWAP | `ta.vwap` | **Session-anchored** — resets each UTC day via `anchors=day_keys(data)`; hlc3 source. `anchors=None` = cumulative |
| Donchian | Donchian Channels | highest/lowest incl. current bar |
| Keltner | built-in KC | Basis = **EMA(close, 20)** (close source, TV default), bands ± 2 × ATR(10) |
| Pivots | `ta.pivothigh/low` | Strict comparisons; value stored at the pivot bar. Use `pivots_confirmed(ph, right_bars)` to shift values to the bar TV would confirm them — **required to avoid lookahead** |

`swing_highs_lows()` is centered (uses future bars) — research only; `detect_bos()` now acts only on confirmed swings.

**Timestamps:** `load_data` / `day_keys` normalize Unix timestamps in seconds, **milliseconds or microseconds** (Binance switched klines to µs on 2025-01-01, so one CSV can mix units — handled).

---

## Strategies (16 built-in)

`sma_crossover`, `ema_crossover`, `golden_cross`, `rsi`, `macd`, `bollinger`, `supertrend`, `stochastic`, `adx`, `donchian`, `vwap_reversion` (session VWAP), `mtf_alignment`, `bband_squeeze`, `triple_ema`, `calm_trend`, `pulse_futures`.

Add your own — write a function returning `list[int]` (1/-1/0 state per bar) and register it in `STRATEGIES`.

---

## Optimizer

Brute-force grid search, walk-forward (train/test, no leak), rolling walk-forward, and adaptive random search. Parallelism uses **processes**, so pure-Python backtests use every core; data loads once per worker.

```python
from optimizer import grid_search, walk_forward, rolling_walk_forward, adaptive_search

best = grid_search(
    "BTC_1h.csv", "sma_crossover",
    {"fast_period": [10, 20, 50], "slow_period": [100, 200]},
    rank_by="sharpe_ratio",   # TV monthly Sharpe; or total_return_pct, calmar_ratio…
    min_trades=10,
    parallel=True,
)
best_params = best[0]["params"]
```

`walk_forward(...)` grid-searches the train slice only, then evaluates out-of-sample. `adaptive_search(...)` handles huge spaces with random exploration + refinement.

---

## Backtester parameters

```python
Backtester(
    data,                          # list of dicts from load_data()
    initial_capital=10000,
    fee_pct=0.04,                  # commission % per fill
    slippage_ticks=0,              # TV slippage (market & stop fills only)
    slippage_pct=0.0,              # extra % slippage, same rules (non-TV)
    tick_size=0.01,
    stop_loss_pct=None,            # % from entry fill (stop order)
    take_profit_pct=None,          # % from entry fill (limit order)
    long_only=True,
    qty_type="equity_pct",         # or "fixed" (contracts)
    qty_value=100,
    qty_step=None,                 # min order size (BTCUSD on TV: 1e-6)
    process_orders_on_close=False, # TV option: fill on the signal close
    force_close_end=False,         # True = realized-only (non-TV)
    risk_free_rate=2.0,            # annual %, for Sharpe/Sortino (TV: 2)
)
```

---

## Verification

`python test_tv_parity.py` — 115 checks, every expectation hand-computed from TV's documented formulas (Pine v5 reference sources and Strategy Tester help pages): EMA/RMA/RSI/MACD/ADX/SuperTrend/Stochastic/VWAP values, gap fills, same-bar SL/TP path ordering, sizing-at-creation, the TV intrabar-drawdown worked example, monthly Sharpe/Sortino, breakeven-trade accounting, and more.

Known gaps vs TV (documented, deliberate): no Bar Magnifier (needs LTF data), no margin-call liquidation (TV v5 default margin = 0 has none either), Pine v6 default margin (100%) not modeled, `calc_on_every_tick` not modeled.

---

## Layout

This folder contains **only the framework** (engine, indicators, strategies, optimizer, tests, data pipeline). Everything built *with* it — `endellion.py`, `run_regime_futures.py`, the `research/` experiments — lives in the `crypto_data` root alongside the market data, and imports the framework from here.

## Data pipeline

All market data lives in **one canonical location: the `crypto_data` root folder** (the parent of this package). `load_data("BTC_1h.csv")` resolves there automatically; nothing is duplicated into this folder.

```
download_data.py ──► full-history CSVs ──► update_data.py (incremental top-up)
  (Binance Vision monthly zips)              (Binance REST, last CLOSED bar)
```

- `python update_data.py` — appends only the missing bars to every `BTC_*.csv` (5m/15m/1h/4h/1d), never writes the still-forming candle, refuses to append across gaps, and preserves each file's schema and timestamp unit (ms or µs).
- Required CSV columns: `open, high, low, close, volume`; optional `datetime` (labels) and `open_time` (Unix s/ms/µs — used for sessions and monthly Sharpe buckets).
- Known data hole: Binance halted spot trading 2023-03-24 ≈12:35–14:00 UTC, so no 5m/15m/1h candles exist for that window — TradingView's Binance charts have the identical gap.

## Requirements

Python 3.7+ — no pip packages (standard library only).
