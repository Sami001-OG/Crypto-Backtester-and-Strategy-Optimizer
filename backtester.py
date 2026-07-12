"""
Core backtesting engine — matches TradingView's bar-magnitude execution model.

TradingView parity:
  - Signals generated at bar i (close) execute at bar i+1 open (no lookahead bias).
    Entries AND exits both fill at the next bar's open — symmetric, no bias.
  - Stop-loss / take-profit checked intra-bar using high/low (SL before TP).
  - Commission charged on every fill (% of order value).
  - Cash-based accounting: equity when flat == cash. Works for any position size
    (qty_value < 100 keeps the remainder in cash) and for shorts (1x, margin-locked).
  - Timeframe-aware ratios: bar interval is inferred from timestamps, so Sharpe,
    Sortino, CAGR and exposure are correct for 5m / 15m / 1h / 4h / 1d data.
  - O(n) running computation, pre-extracted arrays for speed.
"""

import csv
import os
import math
import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
_DT_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


def load_data(filename):
    """
    Load an OHLCV CSV. Required columns: open, high, low, close, volume.
    A ``datetime`` column is used for labels/interval detection when present;
    otherwise ``open_time`` (Unix ms) is used. ``close_time``, ``quote_volume``
    and ``trades`` are optional and default sensibly when missing.
    """
    filepath = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
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

        t_col = col("datetime", col("open_time", 0))
        ot_col = col("open_time")
        o_col, h_col, l_col, c_col, v_col = ci["open"], ci["high"], ci["low"], ci["close"], ci["volume"]
        ct_col = col("close_time")
        qv_col = col("quote_volume")
        tr_col = col("trades")

        for row in reader:
            if not row:
                continue
            rows.append({
                "datetime": row[t_col],
                "open_time": row[ot_col] if ot_col is not None else row[t_col],
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


def _parse_dt(s):
    for fmt in _DT_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _infer_interval_seconds(times, open_times):
    """
    Infer the bar interval (seconds) from the median gap between bars.
    Tries the ``datetime`` strings first, then numeric ``open_time`` (ms or s).
    Falls back to 3600s (1h) if nothing parses.
    """
    diffs = []
    # Try parsed datetimes
    prev = None
    for s in times[:200]:
        dt = _parse_dt(s)
        if dt is not None:
            if prev is not None:
                d = (dt - prev).total_seconds()
                if d > 0:
                    diffs.append(d)
            prev = dt
    if not diffs and open_times:
        prev_v = None
        for s in open_times[:200]:
            try:
                v = float(s)
            except (ValueError, TypeError):
                prev_v = None
                continue
            if prev_v is not None:
                d = v - prev_v
                if d > 0:
                    diffs.append(d)
            prev_v = v
        if diffs:
            med = sorted(diffs)[len(diffs) // 2]
            # open_time is usually milliseconds; normalise to seconds
            return med / 1000.0 if med > 1e5 else med
    if diffs:
        return sorted(diffs)[len(diffs) // 2]
    return 3600.0


class Backtester:
    """
    TradingView-compatible backtester with bar-magnitude execution.

    Execution model (matches TV):
      1. Signal at bar i (close) schedules a transition executed at bar i+1 open.
      2. Stop-loss / take-profit checked intra-bar: SL uses bar low (long) /
         high (short); TP the opposite. SL is checked before TP.
      3. Commission applied on both entry and exit (% of order value).
      4. Cash accounting: equity == cash whenever flat, for any qty_value / side.

    Parameters:
        initial_capital   - starting capital
        fee_pct           - commission per fill (%) — TV default 0.04% for crypto
        slippage_ticks    - slippage in ticks (tick_size configurable)
        slippage_pct      - slippage as % of price
        tick_size         - price value of one tick (default 0.01)
        stop_loss_pct     - stop loss % from entry (e.g. 2 = 2%)
        take_profit_pct   - take profit % from entry (e.g. 5 = 5%)
        long_only         - if True, short signals just flatten instead of shorting
        qty_type          - "equity_pct" (default) or "fixed" (fixed units)
        qty_value         - % of equity (0-100) or fixed units
    """

    def __init__(self, data, initial_capital=10000, fee_pct=0.04,
                 slippage_ticks=0, slippage_pct=0.0, tick_size=0.01,
                 stop_loss_pct=None, take_profit_pct=None,
                 long_only=True, pyramiding=1,
                 qty_type="equity_pct", qty_value=100):
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

        n = len(data)
        self._open = [data[i]["open"] for i in range(n)]
        self._high = [data[i]["high"] for i in range(n)]
        self._low = [data[i]["low"] for i in range(n)]
        self._close = [data[i]["close"] for i in range(n)]
        self._time = [data[i]["datetime"] for i in range(n)]
        self._n = n
        self._interval = _infer_interval_seconds(
            self._time, [data[i].get("open_time") for i in range(n)]
        )

    def _slip(self, price, is_buy):
        """Adverse slippage: buys fill higher, sells fill lower."""
        if self.slippage_ticks > 0:
            adj = self.slippage_ticks * self.tick_size
            return price + adj if is_buy else price - adj
        if self.slippage_rate > 0:
            return price * (1.0 + self.slippage_rate) if is_buy else price * (1.0 - self.slippage_rate)
        return price

    def _size(self, cash, price):
        """Notional (in quote currency) to allocate to a new position."""
        if self.qty_type == "fixed":
            notional = self.qty_value * price          # qty_value = fixed units
        else:
            notional = cash * (self.qty_value / 100.0)  # qty_value = % of equity
        return max(0.0, min(notional, cash))

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
        pos_side = 0            # 1=long, -1=short, 0=flat
        total_fees = 0.0
        pending = None          # desired side to establish at the next bar's open
        bars_in_market = 0

        equity_curve = [0.0] * n
        trades = []

        def open_position(side, raw_price, bar):
            nonlocal cash, qty, entry_price, entry_fee, entry_time, entry_bar, pos_side, total_fees
            price = self._slip(raw_price, side == 1)
            if price <= 0:
                return
            notional = self._size(cash, price)
            if notional <= 0:
                return
            fee = notional * self.fee_rate
            qty = notional / price
            entry_fee = fee
            entry_price = price
            entry_time = self._time[bar]
            entry_bar = bar
            pos_side = side
            total_fees += fee
            if side == 1:
                cash -= notional + fee      # long ties up notional + pays fee
            else:
                cash -= fee                 # short: 1x margin held against notional, pay fee

        def close_position(raw_price, bar, reason):
            nonlocal cash, qty, entry_price, entry_fee, pos_side, total_fees
            # Long exit is a SELL (slips down); short cover is a BUY (slips up).
            price = self._slip(raw_price, pos_side == -1)
            exit_fee = price * qty * self.fee_rate
            total_fees += exit_fee
            if pos_side == 1:
                gross = (price - entry_price) * qty
                cash += qty * price - exit_fee   # return notional + P&L, minus exit fee
            else:
                gross = (entry_price - price) * qty
                cash += gross - exit_fee          # realise short P&L, minus exit fee
            net = gross - entry_fee - exit_fee
            side_str = "LONG" if pos_side == 1 else "SHORT"
            trades.append(self._trade(entry_time, self._time[bar], side_str,
                                      entry_price, price, qty, net, reason,
                                      entry_bar, bar))
            pos_side = 0
            qty = 0.0
            entry_price = 0.0
            entry_fee = 0.0

        for i in range(n):
            o = self._open[i]
            h = self._high[i]
            l = self._low[i]
            c = self._close[i]

            # ── 1. Execute pending transition at this bar's open ──
            if pending is not None:
                target = pending
                pending = None
                if pos_side != 0 and pos_side != target:
                    close_position(o, i, "signal")
                if target != 0 and pos_side == 0:
                    open_position(target, o, i)

            # ── 2. Intra-bar stop-loss / take-profit ──
            if pos_side == 1 and (self.stop_loss_pct or self.take_profit_pct):
                sl = entry_price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct else None
                tp = entry_price * (1.0 + self.take_profit_pct) if self.take_profit_pct else None
                if sl is not None and l <= sl:
                    close_position(sl, i, "stop_loss")
                elif tp is not None and h >= tp:
                    close_position(tp, i, "take_profit")
            elif pos_side == -1 and (self.stop_loss_pct or self.take_profit_pct):
                sl = entry_price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct else None
                tp = entry_price * (1.0 - self.take_profit_pct) if self.take_profit_pct else None
                if sl is not None and h >= sl:
                    close_position(sl, i, "stop_loss")
                elif tp is not None and l <= tp:
                    close_position(tp, i, "take_profit")

            # ── 3. Mark-to-market equity at close ──
            if pos_side == 1:
                equity_curve[i] = cash + qty * c
                bars_in_market += 1
            elif pos_side == -1:
                equity_curve[i] = cash + (entry_price - c) * qty
                bars_in_market += 1
            else:
                equity_curve[i] = cash

            # ── 4. Evaluate signal → schedule transition for next bar open ──
            if i + 1 < n:
                sig = signals[i]
                if sig == 1:
                    desired = 1
                elif sig == -1:
                    desired = 0 if self.long_only else -1
                else:
                    desired = 0
                pending = desired if desired != pos_side else None

        # ── Force close any open position at the last close ──
        if pos_side != 0:
            close_position(self._close[-1], n - 1, "end_of_data")
            equity_curve[-1] = cash

        return self._compute_metrics(equity_curve, trades, total_fees, bars_in_market)

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

    def _compute_metrics(self, equity_curve, trades, total_fees, bars_in_market):
        n = len(equity_curve)
        initial = self.initial_capital
        final = equity_curve[-1] if n else initial

        net_profit = final - initial
        total_return_pct = (net_profit / initial) * 100.0 if initial > 0 else 0.0

        bnh_return = ((self._close[-1] - self._close[0]) / self._close[0] * 100.0
                      if len(self._close) > 1 and self._close[0] > 0 else 0.0)

        # ── Max drawdown (track peak value at the trough) ──
        peak = initial
        peak_at_trough = initial
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                peak_at_trough = peak
        max_dd_pct = max_dd * 100.0
        max_dd_dollar = peak_at_trough * max_dd

        # ── Trade statistics ──
        num_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_count = len(wins)
        lose_count = len(losses)
        win_rate_pct = (win_count / num_trades * 100.0) if num_trades > 0 else 0.0

        total_profit = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = float("inf") if total_profit > 0 else 0.0

        avg_win = total_profit / win_count if win_count > 0 else 0.0
        avg_loss = total_loss / lose_count if lose_count > 0 else 0.0
        avg_pnl = sum(t["pnl"] for t in trades) / num_trades if num_trades > 0 else 0.0

        max_win = max((t["pnl"] for t in trades), default=0.0)
        max_loss = min((t["pnl"] for t in trades), default=0.0)
        largest_win_pct = max((t["pnl_pct"] for t in trades), default=0.0)
        largest_loss_pct = min((t["pnl_pct"] for t in trades), default=0.0)

        max_consec_w = max_consec_l = cw = cl = 0
        for t in trades:
            if t["pnl"] > 0:
                cw += 1; cl = 0
                max_consec_w = max(max_consec_w, cw)
            else:
                cl += 1; cw = 0
                max_consec_l = max(max_consec_l, cl)

        # ── Durations (bar-based, timeframe-aware) ──
        hours_per_bar = self._interval / 3600.0
        if trades:
            avg_bars_held = sum(t["bars_held"] for t in trades) / num_trades
        else:
            avg_bars_held = 0.0
        avg_duration_hrs = avg_bars_held * hours_per_bar
        exposure_pct = (bars_in_market / n * 100.0) if n > 0 else 0.0

        # ── Per-bar returns ──
        bar_returns = []
        for i in range(1, n):
            prev = equity_curve[i - 1]
            if prev > 0:
                bar_returns.append((equity_curve[i] - prev) / prev)

        periods_per_year = SECONDS_PER_YEAR / self._interval if self._interval > 0 else 8760.0

        # ── Sharpe / Sortino (annualised) ──
        sharpe = 0.0
        sortino = 0.0
        if len(bar_returns) > 5:
            mean_r = sum(bar_returns) / len(bar_returns)
            var_r = sum((r - mean_r) ** 2 for r in bar_returns) / len(bar_returns)
            std_r = math.sqrt(var_r)
            ann = math.sqrt(periods_per_year)
            if std_r > 1e-12:
                sharpe = (mean_r / std_r) * ann
            downside = [r for r in bar_returns if r < 0]
            if downside:
                down_std = math.sqrt(sum(r * r for r in downside) / len(downside))
                if down_std > 1e-12:
                    sortino = (mean_r / down_std) * ann
            elif mean_r > 0:
                sortino = float("inf")

        # ── Calmar ──
        calmar = total_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

        # ── CAGR (timeframe-aware) ──
        years = (n * self._interval) / SECONDS_PER_YEAR
        if years > 1e-9 and initial > 0 and final > 0:
            cagr = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr = 0.0

        # ── Expectancy ──
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
            "net_profit_pct": total_return_pct,
            "total_return_pct": total_return_pct,
            "buy_and_hold_return_pct": bnh_return,
            "total_fees": total_fees,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_dollar": max_dd_dollar,
            "num_trades": num_trades,
            "num_winning": win_count,
            "num_losing": lose_count,
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win_dollar": max_win,
            "max_loss_dollar": max_loss,
            "largest_win_pct": largest_win_pct,
            "largest_loss_pct": largest_loss_pct,
            "max_consecutive_wins": max_consec_w,
            "max_consecutive_losses": max_consec_l,
            "expectancy": expectancy,
            "avg_trade_duration_hours": avg_duration_hrs,
            "avg_bars_held": avg_bars_held,
            "exposure_pct": exposure_pct,
            "bar_interval_seconds": self._interval,
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

        def fmt(x):
            if x == float("inf"):
                return "inf"
            if x == float("-inf"):
                return "-inf"
            return x

        def row(label, value):
            print(f"  {label:<{label_w}} {value:>{val_w}}")

        print()
        print("=" * 55)
        print("            STRATEGY TESTER REPORT")
        print("=" * 55)
        print()
        print("  ── PERFORMANCE ──")
        row("Net Profit", f"${results['net_profit']:,.2f}")
        row("Net Profit %", f"{results['net_profit_pct']:.2f}%")
        row("Initial Capital", f"${results['initial_capital']:,.2f}")
        row("Final Equity", f"${results['final_equity']:,.2f}")
        row("Buy & Hold Return", f"{results['buy_and_hold_return_pct']:.2f}%")
        row("Total Fees", f"${results['total_fees']:,.2f}")
        row("CAGR", f"{results['cagr_pct']:.2f}%")
        print()
        print("  ── RISK ──")
        row("Max Drawdown %", f"{results['max_drawdown_pct']:.2f}%")
        row("Max Drawdown $", f"${results['max_drawdown_dollar']:,.2f}")
        print()
        print("  ── RATIOS ──")
        row("Sharpe Ratio", f"{fmt(results['sharpe_ratio']):.3f}" if results['sharpe_ratio'] not in (float('inf'), float('-inf')) else "inf")
        row("Sortino Ratio", f"{fmt(results['sortino_ratio']):.3f}" if results['sortino_ratio'] not in (float('inf'), float('-inf')) else "inf")
        row("Calmar Ratio", f"{results['calmar_ratio']:.3f}")
        pf = results['profit_factor']
        row("Profit Factor", "inf" if pf == float("inf") else f"{pf:.2f}")
        print()
        print("  ── TRADE STATISTICS ──")
        row("Total Trades", f"{results['num_trades']}")
        row("Winning Trades", f"{results['num_winning']}")
        row("Losing Trades", f"{results['num_losing']}")
        row("Win Rate", f"{results['win_rate_pct']:.2f}%")
        row("Avg P/L per Trade", f"${results['avg_pnl']:,.2f}")
        row("Avg Win", f"${results['avg_win']:,.2f}")
        row("Avg Loss", f"${results['avg_loss']:,.2f}")
        row("Largest Win", f"${results['max_win_dollar']:,.2f}")
        row("Largest Loss", f"${results['max_loss_dollar']:,.2f}")
        row("Largest Win %", f"{results['largest_win_pct']:.2f}%")
        row("Largest Loss %", f"{results['largest_loss_pct']:.2f}%")
        row("Max Consec Wins", f"{results['max_consecutive_wins']}")
        row("Max Consec Losses", f"{results['max_consecutive_losses']}")
        row("Expectancy", f"${results['expectancy']:,.2f}")
        row("Avg Trade Duration", f"{results['avg_trade_duration_hours']:.1f}h")
        row("Time in Market", f"{results['exposure_pct']:.1f}%")

        if results.get("long_pnl") or results.get("short_pnl"):
            print()
            print("  ── BY SIDE ──")
            row("Long P/L", f"${results['long_pnl']:,.2f}")
            row("Short P/L", f"${results['short_pnl']:,.2f}")

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
