"""
Core backtesting engine — replicates TradingView's Strategy Tester (broker emulator).

TradingView parity (each rule below matches TV's documented behavior):

  Order execution (broker emulator):
    - Market orders are created at bar i close and fill at bar i+1 open
      (process_orders_on_close=False, TV default). Entries AND signal exits.
    - Stop-loss / take-profit are resting stop/limit orders, active from the
      moment the entry fills — they CAN fill on the entry bar itself.
    - Gap rule: if the bar OPENS beyond a stop/limit price, the order fills at
      the open, not at the order price.
    - Intra-bar path assumption: if the open is closer to the high, price is
      assumed to move open → high → low → close; otherwise open → low → high →
      close. This decides which of SL/TP fills first when both are inside one
      bar, exactly like TV.
    - Slippage (ticks and/or %) is applied adversely to market and stop fills.
      Limit fills are NOT slipped (they fill at the limit price or better).

  Position sizing (strategy.percent_of_equity):
    - Quantity is computed when the order is CREATED (signal bar close), using
      strategy.equity = initial + closed net profit + open position profit,
      divided by the signal bar's CLOSE — not the fill price. The order then
      fills at the next open with that fixed quantity (TV's documented
      "sizing drift").
    - Quantity is rounded DOWN to `qty_step` (the symbol's minimum quantity,
      e.g. 0.000001 for BTCUSD on TV); orders smaller than one step are skipped.
    - No margin check (TV default margin_long/short = 0): fees may push cash
      slightly negative, as in TV.

  Metrics (Strategy Tester "Performance Summary"):
    - Net profit counts CLOSED trades only; an open position at the end of data
      is reported separately as open P&L (TV leaves it open — no force close).
    - Max drawdown / max run-up use INTRA-BAR equity extremes (bar low/high
      while in a position, path-ordered on exit bars), per TV's
      "Max equity drawdown (intrabar)" formula.
    - Sharpe / Sortino use MONTHLY equity returns and a 2%-annual risk-free
      rate divided by 12, exactly like TV: SR = (MR - RFR) / SD with population
      standard deviation; Sortino divides by the downside deviation
      sqrt(sum(r<0: r^2) / N). Not annualized (TV reports the monthly ratio).
    - Buy & hold return: all funds buy at the FIRST TRADE's entry and hold to
      the last close.
    - Percent profitable = winning / total closed; breakeven trades count in
      the total but are neither wins nor losses.
    - Commission is charged on every fill as % of order value.

  Extras beyond TV (kept for the optimizer): CAGR, Calmar, exposure %,
  timeframe-aware annualization. Bar Magnifier is not emulated (needs
  lower-timeframe data).
"""

import csv
import os
import math
import datetime

# All market data lives in the crypto_data root (one canonical copy).
# Relative filenames resolve there first, then fall back to this folder.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_PKG_DIR, ".."))

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
_DT_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")
_UTC = datetime.timezone.utc


def _parse_dt(s):
    for fmt in _DT_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _epoch_of(datetime_str, open_time):
    """
    Best-effort bar epoch (seconds, UTC). Handles seconds, milliseconds AND
    microseconds (Binance switched kline timestamps to µs on 2025-01-01, so a
    single CSV can mix ms and µs rows). Returns None if unparseable.
    """
    try:
        v = float(open_time)
        while v > 1e11:       # ms (~1e12) or µs (~1e15) → normalize to seconds
            v /= 1000.0
        if v > 1e8:           # sane epoch-seconds range (>1973)
            return v
    except (TypeError, ValueError):
        pass
    dt = _parse_dt(datetime_str)
    if dt is not None:
        return dt.replace(tzinfo=_UTC).timestamp()
    return None


def load_data(filename):
    """
    Load an OHLCV CSV. Required columns: open, high, low, close, volume.
    A ``datetime`` column is used for labels when present; otherwise it is
    synthesized (UTC) from ``open_time`` (Unix ms). ``close_time``,
    ``quote_volume`` and ``trades`` are optional and default sensibly.
    """
    if os.path.isabs(filename):
        filepath = filename
    else:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            local = os.path.join(_PKG_DIR, filename)
            if os.path.exists(local):
                filepath = local
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        ci = {name.strip(): idx for idx, name in enumerate(header)}

        for req in ("open", "high", "low", "close", "volume"):
            if req not in ci:
                raise ValueError(f"CSV missing required column '{req}'. Found: {list(ci)}")

        def col(name, default=None):
            return ci[name] if name in ci else default

        dt_col = col("datetime")
        ot_col = col("open_time")
        o_col, h_col, l_col, c_col, v_col = ci["open"], ci["high"], ci["low"], ci["close"], ci["volume"]
        ct_col = col("close_time")
        qv_col = col("quote_volume")
        tr_col = col("trades")

        for row in reader:
            if not row:
                continue
            open_time = row[ot_col] if ot_col is not None else (row[dt_col] if dt_col is not None else "")
            if dt_col is not None:
                dt_str = row[dt_col]
            else:
                # Synthesize a readable UTC datetime from open_time (s/ms/µs).
                dt_str = open_time
                ep = _epoch_of(None, open_time)
                if ep is not None:
                    dt_str = datetime.datetime.fromtimestamp(ep, _UTC).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({
                "datetime": dt_str,
                "open_time": open_time,
                "open": float(row[o_col]),
                "high": float(row[h_col]),
                "low": float(row[l_col]),
                "close": float(row[c_col]),
                "volume": float(row[v_col]),
                "close_time": row[ct_col] if ct_col is not None else "",
                "quote_volume": float(row[qv_col]) if qv_col is not None and row[qv_col] != "" else 0.0,
                "trades": int(float(row[tr_col])) if tr_col is not None and row[tr_col] != "" else 0,
            })
    return rows


def _infer_interval_seconds(epochs):
    """Median gap between consecutive bar epochs (seconds). Unit-safe because
    the epochs are already normalized (handles mixed ms/µs source files)."""
    diffs = []
    prev = None
    for ep in epochs:
        if ep is not None:
            if prev is not None:
                d = ep - prev
                if d > 0:
                    diffs.append(d)
            prev = ep
        if len(diffs) >= 500:
            break
    if diffs:
        return sorted(diffs)[len(diffs) // 2]
    return 3600.0


class Backtester:
    """
    TradingView-compatible backtester (see module docstring for the exact
    broker-emulator rules replicated).

    Parameters:
        initial_capital   - starting capital
        fee_pct           - commission per fill, % of order value (TV
                            "Commission" percent type)
        slippage_ticks    - slippage in ticks, applied to market & stop fills
                            (TV `slippage`); limit fills are never slipped
        slippage_pct      - extra slippage as % of price (non-TV convenience)
        tick_size         - price value of one tick (default 0.01)
        stop_loss_pct     - stop loss % from entry fill (e.g. 2 = 2%)
        take_profit_pct   - take profit % from entry fill (e.g. 5 = 5%)
        long_only         - if True, short signals flatten instead of shorting
        qty_type          - "equity_pct" (TV percent-of-equity) or "fixed"
        qty_value         - % of equity (0-100] or fixed units (contracts)
        qty_step          - round order size DOWN to this step and skip orders
                            below it (TV min quantity; BTCUSD = 1e-6). None =
                            no rounding.
        process_orders_on_close - fill market orders on the signal bar's close
                            instead of the next open (TV option)
        force_close_end   - close any open position at the last close (NOT
                            TV behavior; off by default)
        risk_free_rate    - annual risk-free %, for Sharpe/Sortino (TV: 2)
    """

    def __init__(self, data, initial_capital=10000, fee_pct=0.04,
                 slippage_ticks=0, slippage_pct=0.0, tick_size=0.01,
                 stop_loss_pct=None, take_profit_pct=None,
                 long_only=True, pyramiding=1,
                 qty_type="equity_pct", qty_value=100,
                 qty_step=None, process_orders_on_close=False,
                 force_close_end=False, risk_free_rate=2.0):
        if not data:
            raise ValueError("data is empty")
        self.data = data
        self.initial_capital = float(initial_capital)
        self.fee_rate = fee_pct / 100.0
        self.slippage_rate = slippage_pct / 100.0
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.stop_loss_pct = stop_loss_pct / 100.0 if stop_loss_pct else None
        self.take_profit_pct = take_profit_pct / 100.0 if take_profit_pct else None
        self.long_only = long_only
        self.pyramiding = pyramiding  # reserved (single-entry model)
        self.qty_type = qty_type
        self.qty_value = qty_value
        self.qty_step = qty_step
        self.process_orders_on_close = process_orders_on_close
        self.force_close_end = force_close_end
        self.risk_free_rate = risk_free_rate

        n = len(data)
        self._open = [data[i]["open"] for i in range(n)]
        self._high = [data[i]["high"] for i in range(n)]
        self._low = [data[i]["low"] for i in range(n)]
        self._close = [data[i]["close"] for i in range(n)]
        self._time = [data[i]["datetime"] for i in range(n)]
        self._n = n
        self._epoch = [_epoch_of(data[i].get("datetime"), data[i].get("open_time"))
                       for i in range(n)]
        self._interval = _infer_interval_seconds(self._epoch)

    # ── Fill-price helpers ───────────────────────────────────────────────────

    def _slip(self, price, is_buy):
        """Adverse slippage for market/stop fills: buys up, sells down."""
        if self.slippage_ticks > 0:
            adj = self.slippage_ticks * self.tick_size
            price = price + adj if is_buy else price - adj
        if self.slippage_rate > 0:
            price = price * (1.0 + self.slippage_rate) if is_buy else price * (1.0 - self.slippage_rate)
        return price

    def _plan_qty(self, equity, price):
        """
        Contracts for a new entry, sized like TV at ORDER CREATION time:
        (qty_value% of strategy.equity) / signal-bar close. Rounded down to
        qty_step; <= 0 (or < one step) means the order is skipped.
        """
        if self.qty_type == "fixed":
            q = float(self.qty_value)
        else:
            if price <= 0 or equity <= 0:
                return 0.0
            q = equity * (self.qty_value / 100.0) / price
        if self.qty_step:
            q = math.floor(q / self.qty_step) * self.qty_step
        return max(0.0, q)

    # ── Intra-bar path simulation (TV broker emulator) ──────────────────────

    @staticmethod
    def _bar_path(o, h, l, c):
        """TV assumption: open closer to high → O,H,L,C; else O,L,H,C."""
        if (h - o) <= (o - l):
            return (h, l)
        return (l, h)

    def _simulate_exit(self, side, o, h, l, c, stop_px, limit_px):
        """
        Walk the assumed intra-bar path with resting stop/limit exit orders.

        Returns (reason, raw_fill, checkpoints) where checkpoints is the
        chronological list of prices the position actually saw this bar
        (used for intra-bar equity drawdown/run-up). reason is None if the
        position survives the bar.
        """
        # Gap rule: order price crossed at the open → fill at the open.
        if side == 1:
            if stop_px is not None and o <= stop_px:
                return "stop_loss", o, [o]
            if limit_px is not None and o >= limit_px:
                return "take_profit", o, [o]
        else:
            if stop_px is not None and o >= stop_px:
                return "stop_loss", o, [o]
            if limit_px is not None and o <= limit_px:
                return "take_profit", o, [o]

        checkpoints = [o]
        cur = o
        for leg_end in self._bar_path(o, h, l, c):
            if leg_end >= cur:      # ascending leg
                if side == 1 and limit_px is not None and cur < limit_px <= leg_end:
                    checkpoints.append(limit_px)
                    return "take_profit", limit_px, checkpoints
                if side == -1 and stop_px is not None and cur < stop_px <= leg_end:
                    checkpoints.append(stop_px)
                    return "stop_loss", stop_px, checkpoints
            else:                   # descending leg
                if side == 1 and stop_px is not None and leg_end <= stop_px < cur:
                    checkpoints.append(stop_px)
                    return "stop_loss", stop_px, checkpoints
                if side == -1 and limit_px is not None and leg_end <= limit_px < cur:
                    checkpoints.append(limit_px)
                    return "take_profit", limit_px, checkpoints
            cur = leg_end
            checkpoints.append(cur)
        return None, None, checkpoints

    # ── Main run loop ────────────────────────────────────────────────────────

    def run(self, signals):
        n = self._n
        if len(signals) != n:
            raise ValueError(f"Signal length ({len(signals)}) != data length ({n})")

        cash = self.initial_capital
        qty = 0.0
        entry_price = 0.0
        entry_fee = 0.0
        entry_time = ""
        entry_bar = 0
        pos_side = 0                 # 1=long, -1=short, 0=flat
        total_fees = 0.0
        pending = None               # (target_side, planned_qty or None)
        bars_in_market = 0
        first_entry_price = None

        equity_curve = [0.0] * n
        trades = []

        # Intra-bar equity extremes → TV max drawdown / max run-up.
        peak = self.initial_capital
        trough = self.initial_capital
        max_dd = 0.0
        max_dd_pct = 0.0
        max_ru = 0.0
        max_ru_pct = 0.0

        def touch(eq):
            """Feed one chronological equity checkpoint into DD/run-up."""
            nonlocal peak, trough, max_dd, max_dd_pct, max_ru, max_ru_pct
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
            if eq < trough:
                trough = eq
            ru = eq - trough
            if ru > max_ru:
                max_ru = ru
                max_ru_pct = (ru / trough * 100.0) if trough > 0 else 0.0

        def pos_equity(price):
            if pos_side == 1:
                return cash + qty * price
            if pos_side == -1:
                return cash + qty * (entry_price - price)
            return cash

        def open_position(side, raw_price, planned_qty, bar):
            nonlocal cash, qty, entry_price, entry_fee, entry_time, entry_bar
            nonlocal pos_side, total_fees, first_entry_price
            fill = self._slip(raw_price, side == 1)
            if fill <= 0 or planned_qty is None or planned_qty <= 0:
                return
            if self.qty_step and planned_qty < self.qty_step:
                return                      # TV skips sub-minimum orders
            fee = planned_qty * fill * self.fee_rate
            qty = planned_qty
            entry_fee = fee
            entry_price = fill
            entry_time = self._time[bar]
            entry_bar = bar
            pos_side = side
            total_fees += fee
            if first_entry_price is None:
                first_entry_price = raw_price   # buy & hold benchmark basis
            if side == 1:
                cash -= qty * fill + fee        # may dip negative by the fee (TV margin=0)
            else:
                cash -= fee                     # short at 1x: P&L realised on cover
            touch(pos_equity(raw_price))

        def close_position(fill, bar, reason):
            nonlocal cash, qty, entry_price, entry_fee, pos_side, total_fees
            exit_fee = fill * qty * self.fee_rate
            total_fees += exit_fee
            if pos_side == 1:
                gross = (fill - entry_price) * qty
                cash += qty * fill - exit_fee
            else:
                gross = (entry_price - fill) * qty
                cash += gross - exit_fee
            net = gross - entry_fee - exit_fee
            side_str = "LONG" if pos_side == 1 else "SHORT"
            trades.append(self._trade(entry_time, self._time[bar], side_str,
                                      entry_price, fill, qty, net, reason,
                                      entry_bar, bar))
            pos_side = 0
            qty = 0.0
            entry_price = 0.0
            entry_fee = 0.0
            touch(cash)

        def execute_market(target, planned_qty, raw_price, bar):
            """Fill pending market order(s) at raw_price (an open, or a close
            when process_orders_on_close=True)."""
            if pos_side != 0 and target != pos_side:
                touch(pos_equity(raw_price))    # equity seen at the fill price pre-exit
                fill = self._slip(raw_price, pos_side == -1)
                close_position(fill, bar, "signal")
            if target != 0 and pos_side == 0:
                open_position(target, raw_price, planned_qty, bar)

        for i in range(n):
            o = self._open[i]
            h = self._high[i]
            l = self._low[i]
            c = self._close[i]
            in_market_this_bar = pos_side != 0

            # ── 1. Fill pending market orders at this bar's open ──
            if pending is not None:
                target, planned_qty = pending
                pending = None
                execute_market(target, planned_qty, o, i)
                in_market_this_bar = in_market_this_bar or pos_side != 0

            # ── 2. Resting stop/limit exits (active from the entry bar) ──
            if pos_side != 0 and (self.stop_loss_pct or self.take_profit_pct):
                if pos_side == 1:
                    stop_px = entry_price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct else None
                    limit_px = entry_price * (1.0 + self.take_profit_pct) if self.take_profit_pct else None
                else:
                    stop_px = entry_price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct else None
                    limit_px = entry_price * (1.0 - self.take_profit_pct) if self.take_profit_pct else None
                reason, raw_fill, checkpoints = self._simulate_exit(
                    pos_side, o, h, l, c, stop_px, limit_px)
                for px in checkpoints:
                    touch(pos_equity(px))
                if reason is not None:
                    if reason == "stop_loss":
                        # Stops are slipped like market orders (TV).
                        fill = self._slip(raw_fill, pos_side == -1)
                    else:
                        fill = raw_fill          # limit orders never slip
                    close_position(fill, i, reason)
            elif pos_side != 0:
                # No exit orders: position sees the whole bar, path-ordered.
                for px in self._bar_path(o, h, l, c):
                    touch(pos_equity(px))

            # ── 3. Mark equity at the close ──
            equity_curve[i] = pos_equity(c)
            touch(equity_curve[i])
            if in_market_this_bar or pos_side != 0:
                bars_in_market += 1

            # ── 4. Evaluate signal → create orders (sized on THIS close) ──
            sig = signals[i]
            if sig == 1:
                desired = 1
            elif sig == -1:
                desired = 0 if self.long_only else -1
            else:
                desired = 0
            if desired != pos_side:
                planned_qty = None
                if desired != 0:
                    planned_qty = self._plan_qty(equity_curve[i], c)
                pending = (desired, planned_qty)
            else:
                pending = None

            if self.process_orders_on_close and pending is not None:
                target, planned_qty = pending
                pending = None
                execute_market(target, planned_qty, c, i)
                equity_curve[i] = pos_equity(c)

        # ── End of data: TV leaves the position open (open P&L reported) ──
        open_trade = None
        open_pnl = 0.0
        if pos_side != 0:
            if self.force_close_end:
                touch(pos_equity(self._close[-1]))
                fill = self._slip(self._close[-1], pos_side == -1)
                close_position(fill, n - 1, "end_of_data")
                equity_curve[-1] = cash
            else:
                last_c = self._close[-1]
                open_pnl = (qty * (last_c - entry_price) if pos_side == 1
                            else qty * (entry_price - last_c))
                open_trade = {
                    "entry_time": entry_time,
                    "side": "LONG" if pos_side == 1 else "SHORT",
                    "entry_price": entry_price,
                    "qty": qty,
                    "open_pnl": open_pnl - entry_fee,
                    "entry_bar": entry_bar,
                    "bars_held": (n - 1) - entry_bar,
                }

        dd_state = (max_dd, max_dd_pct, max_ru, max_ru_pct)
        return self._compute_metrics(equity_curve, trades, total_fees,
                                     bars_in_market, dd_state,
                                     first_entry_price, open_trade, open_pnl)

    def _trade(self, entry_t, exit_t, side, entry_p, exit_p, qty, pnl, reason,
               entry_bar, exit_bar):
        inv_val = entry_p * qty
        return {
            "entry_time": entry_t,
            "exit_time": exit_t,
            "side": side,
            "entry_price": entry_p,
            "exit_price": exit_p,
            "qty": qty,
            "pnl": pnl,
            "pnl_pct": pnl / inv_val * 100.0 if inv_val > 0 else 0.0,
            "exit_reason": reason,
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "bars_held": exit_bar - entry_bar,
        }

    # ── Monthly returns (TV Sharpe/Sortino basis) ────────────────────────────

    def _monthly_returns(self, equity_curve):
        """
        Equity returns per completed calendar month (UTC), TV-style: the month
        return is month-end equity over the previous month-end equity (first
        partial month is measured from initial capital). The trailing,
        incomplete month is excluded — TV only uses completed periods.
        """
        n = len(equity_curve)
        returns = []
        prev_key = None
        base = self.initial_capital
        for i in range(n):
            ep = self._epoch[i]
            if ep is None:
                return []
            d = datetime.datetime.fromtimestamp(ep, _UTC)
            key = (d.year, d.month)
            if prev_key is not None and key != prev_key:
                end_eq = equity_curve[i - 1]
                if base <= 0:
                    # % returns are meaningless from non-positive equity
                    # (blown account) — stop collecting rather than skip
                    # months and produce a distorted series.
                    break
                returns.append(end_eq / base - 1.0)
                base = end_eq
            prev_key = key
        return returns

    # ── Metrics (TV Performance Summary + extras) ────────────────────────────

    def _compute_metrics(self, equity_curve, trades, total_fees, bars_in_market,
                         dd_state, first_entry_price, open_trade, open_pnl):
        n = len(equity_curve)
        initial = self.initial_capital
        final = equity_curve[-1] if n else initial
        max_dd_dollar, max_dd_pct, max_ru_dollar, max_ru_pct = dd_state

        # Net profit counts CLOSED trades only (TV); open P&L is separate.
        net_profit = sum(t["pnl"] for t in trades)
        net_profit_pct = (net_profit / initial) * 100.0 if initial > 0 else 0.0
        total_return_pct = ((final - initial) / initial) * 100.0 if initial > 0 else 0.0

        # Buy & hold: all funds in at the first trade's entry, held to the end.
        if first_entry_price and first_entry_price > 0:
            bnh_return = (self._close[-1] - first_entry_price) / first_entry_price * 100.0
        elif len(self._close) > 1 and self._close[0] > 0:
            bnh_return = (self._close[-1] - self._close[0]) / self._close[0] * 100.0
        else:
            bnh_return = 0.0
        bnh_profit = initial * bnh_return / 100.0

        # ── Trade statistics (TV: breakeven is neither win nor loss) ──
        num_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        evens = num_trades - len(wins) - len(losses)
        win_count = len(wins)
        lose_count = len(losses)
        win_rate_pct = (win_count / num_trades * 100.0) if num_trades > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        avg_win = gross_profit / win_count if win_count > 0 else 0.0
        avg_loss = gross_loss / lose_count if lose_count > 0 else 0.0
        avg_pnl = net_profit / num_trades if num_trades > 0 else 0.0
        ratio_avg_win_loss = avg_win / avg_loss if avg_loss > 0 else (
            float("inf") if avg_win > 0 else 0.0)

        max_win = max((t["pnl"] for t in trades), default=0.0)
        max_loss = min((t["pnl"] for t in trades), default=0.0)
        largest_win_pct = max((t["pnl_pct"] for t in trades), default=0.0)
        largest_loss_pct = min((t["pnl_pct"] for t in trades), default=0.0)

        max_consec_w = max_consec_l = cw = cl = 0
        for t in trades:
            if t["pnl"] > 0:
                cw += 1; cl = 0
                max_consec_w = max(max_consec_w, cw)
            elif t["pnl"] < 0:
                cl += 1; cw = 0
                max_consec_l = max(max_consec_l, cl)
            else:
                cw = 0; cl = 0

        # ── Durations (bar-based, timeframe-aware) ──
        hours_per_bar = self._interval / 3600.0
        avg_bars_held = (sum(t["bars_held"] for t in trades) / num_trades
                         if num_trades else 0.0)
        avg_bars_win = (sum(t["bars_held"] for t in wins) / win_count
                        if win_count else 0.0)
        avg_bars_loss = (sum(t["bars_held"] for t in losses) / lose_count
                         if lose_count else 0.0)
        avg_duration_hrs = avg_bars_held * hours_per_bar
        exposure_pct = (bars_in_market / n * 100.0) if n > 0 else 0.0

        # ── Sharpe / Sortino: monthly returns, RFR/12, population stdev (TV) ──
        monthly = self._monthly_returns(equity_curve)
        rfr_month = (self.risk_free_rate / 100.0) / 12.0
        sharpe = 0.0
        sortino = 0.0
        if len(monthly) >= 2:
            mean_r = sum(monthly) / len(monthly)
            var_r = sum((r - mean_r) ** 2 for r in monthly) / len(monthly)
            std_r = math.sqrt(var_r)
            if std_r > 1e-12:
                sharpe = (mean_r - rfr_month) / std_r
            downside = [r for r in monthly if r < 0]
            if downside:
                down_dev = math.sqrt(sum(r * r for r in downside) / len(monthly))
                if down_dev > 1e-12:
                    sortino = (mean_r - rfr_month) / down_dev
            elif mean_r > rfr_month:
                sortino = float("inf")

        # ── Extras (not part of TV's summary) ──
        calmar = total_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0
        years = (n * self._interval) / SECONDS_PER_YEAR
        if years > 1e-9 and initial > 0 and final > 0:
            cagr = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr = 0.0
        if num_trades > 0:
            expectancy = (win_count / num_trades) * avg_win - (lose_count / num_trades) * avg_loss
        else:
            expectancy = 0.0

        long_pnl = sum(t["pnl"] for t in trades if t["side"] == "LONG")
        short_pnl = sum(t["pnl"] for t in trades if t["side"] == "SHORT")

        eq_dicts = [{"datetime": self._time[i], "equity": equity_curve[i]} for i in range(n)]

        return {
            "initial_capital": initial,
            "final_equity": final,
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "total_return_pct": total_return_pct,
            "open_pnl": open_pnl,
            "open_trade": open_trade,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "buy_and_hold_return_pct": bnh_return,
            "buy_and_hold_profit": bnh_profit,
            "total_fees": total_fees,
            "commission_paid": total_fees,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_dollar": max_dd_dollar,
            "max_runup_pct": max_ru_pct,
            "max_runup_dollar": max_ru_dollar,
            "num_trades": num_trades,
            "num_winning": win_count,
            "num_losing": lose_count,
            "num_even": evens,
            "win_rate_pct": win_rate_pct,
            "percent_profitable": win_rate_pct,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "ratio_avg_win_loss": ratio_avg_win_loss,
            "max_win_dollar": max_win,
            "max_loss_dollar": max_loss,
            "largest_win_pct": largest_win_pct,
            "largest_loss_pct": largest_loss_pct,
            "max_consecutive_wins": max_consec_w,
            "max_consecutive_losses": max_consec_l,
            "expectancy": expectancy,
            "avg_trade_duration_hours": avg_duration_hrs,
            "avg_bars_held": avg_bars_held,
            "avg_bars_win": avg_bars_win,
            "avg_bars_loss": avg_bars_loss,
            "exposure_pct": exposure_pct,
            "bar_interval_seconds": self._interval,
            "monthly_returns": monthly,
            "months_used": len(monthly),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "cagr_pct": cagr,
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "equity_curve": eq_dicts,
            "trades": trades,
        }

    @staticmethod
    def print_report(results):
        label_w = 28
        val_w = 22

        def num(x, fmt="{:,.2f}"):
            if x == float("inf"):
                return "inf"
            if x == float("-inf"):
                return "-inf"
            return fmt.format(x)

        def row(label, value):
            print(f"  {label:<{label_w}} {value:>{val_w}}")

        print()
        print("=" * 55)
        print("       STRATEGY TESTER — PERFORMANCE SUMMARY")
        print("=" * 55)
        print()
        print("  ── OVERVIEW ──")
        row("Net Profit", f"${num(results['net_profit'])} ({num(results['net_profit_pct'])}%)")
        if results.get("open_trade"):
            row("Open P&L", f"${num(results['open_pnl'])}")
        row("Gross Profit", f"${num(results.get('gross_profit', 0.0))}")
        row("Gross Loss", f"${num(results.get('gross_loss', 0.0))}")
        row("Buy & Hold Return", f"{num(results['buy_and_hold_return_pct'])}%")
        row("Max Equity Drawdown", f"${num(results['max_drawdown_dollar'])} ({num(results['max_drawdown_pct'])}%)")
        row("Max Equity Run-up", f"${num(results.get('max_runup_dollar', 0.0))} ({num(results.get('max_runup_pct', 0.0))}%)")
        row("Initial Capital", f"${num(results['initial_capital'])}")
        row("Final Equity", f"${num(results['final_equity'])}")
        row("Commission Paid", f"${num(results['total_fees'])}")
        print()
        print("  ── RATIOS (TV: monthly returns, 2% RFR) ──")
        months = results.get("months_used", 0)
        if months >= 2:
            row("Sharpe Ratio", num(results['sharpe_ratio'], "{:.3f}"))
            row("Sortino Ratio", num(results['sortino_ratio'], "{:.3f}"))
        else:
            row("Sharpe Ratio", "n/a (<2 months)")
            row("Sortino Ratio", "n/a (<2 months)")
        pf = results['profit_factor']
        row("Profit Factor", "inf" if pf == float("inf") else f"{pf:.3f}")
        print()
        print("  ── TRADES ANALYSIS ──")
        total_incl_open = results['num_trades'] + (1 if results.get("open_trade") else 0)
        row("Total Trades", f"{total_incl_open}")
        row("Total Closed Trades", f"{results['num_trades']}")
        row("Winning Trades", f"{results['num_winning']}")
        row("Losing Trades", f"{results['num_losing']}")
        if results.get("num_even"):
            row("Breakeven Trades", f"{results['num_even']}")
        row("Percent Profitable", f"{num(results['win_rate_pct'])}%")
        row("Avg P&L per Trade", f"${num(results['avg_pnl'])}")
        row("Avg Winning Trade", f"${num(results['avg_win'])}")
        row("Avg Losing Trade", f"${num(results['avg_loss'])}")
        rw = results.get("ratio_avg_win_loss", 0.0)
        row("Ratio Avg Win / Loss", "inf" if rw == float("inf") else f"{rw:.3f}")
        row("Largest Winning Trade", f"${num(results['max_win_dollar'])} ({num(results['largest_win_pct'])}%)")
        row("Largest Losing Trade", f"${num(results['max_loss_dollar'])} ({num(results['largest_loss_pct'])}%)")
        row("Max Consec Wins", f"{results['max_consecutive_wins']}")
        row("Max Consec Losses", f"{results['max_consecutive_losses']}")
        row("Avg # Bars in Trades", f"{num(results['avg_bars_held'], '{:.1f}')}")
        row("Avg # Bars in Wins", f"{num(results.get('avg_bars_win', 0.0), '{:.1f}')}")
        row("Avg # Bars in Losses", f"{num(results.get('avg_bars_loss', 0.0), '{:.1f}')}")
        print()
        print("  ── EXTRAS (beyond TV) ──")
        row("CAGR", f"{num(results['cagr_pct'])}%")
        row("Calmar Ratio", f"{num(results['calmar_ratio'], '{:.3f}')}")
        row("Expectancy", f"${num(results['expectancy'])}")
        row("Avg Trade Duration", f"{num(results['avg_trade_duration_hours'], '{:.1f}')}h")
        row("Time in Market", f"{num(results['exposure_pct'], '{:.1f}')}%")

        if results.get("long_pnl") or results.get("short_pnl"):
            print()
            print("  ── BY SIDE ──")
            row("Long P&L", f"${num(results['long_pnl'])}")
            row("Short P&L", f"${num(results['short_pnl'])}")

        print()
        print("=" * 55)

    @staticmethod
    def save_results(results, prefix="backtest"):
        eq_file = os.path.join(DATA_DIR, f"{prefix}_equity.csv")
        with open(eq_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["datetime", "equity"])
            writer.writeheader()
            writer.writerows(results["equity_curve"])
        print(f"Equity curve saved: {eq_file}")

        trades_file = os.path.join(DATA_DIR, f"{prefix}_trades.csv")
        if results["trades"]:
            fields = ["entry_time", "exit_time", "side", "entry_price", "exit_price",
                      "qty", "pnl", "pnl_pct", "exit_reason", "entry_bar", "exit_bar",
                      "bars_held"]
            with open(trades_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results["trades"])
            print(f"Trade log saved:   {trades_file}")


if __name__ == "__main__":
    import strategies
    data = load_data("BTC_1h.csv")
    signals = strategies.sma_crossover(data, 20, 50)
    bt = Backtester(data, initial_capital=10000, fee_pct=0.04)
    results = bt.run(signals)
    Backtester.print_report(results)
