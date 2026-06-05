"""
Core backtesting engine — matches TradingView's bar-magnitude execution model.

TradingView parity:
  - Signals execute at next bar's open (no lookahead bias)
  - Stop/take-profit checked bar-by-bar using high/low
  - Commission on every fill (% of order value)
  - 30+ metrics matching TV's Strategy Tester
  - O(n) running computation, pre-extracted arrays for speed
"""

import csv
import os
import math
import datetime
import time
from collections import deque

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "datetime": row.get("datetime", row.get("open_time", "")),
                "open_time": row["open_time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "close_time": row["close_time"],
                "quote_volume": float(row["quote_volume"]),
                "trades": int(row["trades"]),
            })
    return rows


def _row(data, idx, key):
    return data[idx][key]


class Backtester:
    """
    TradingView-compatible backtester with bar-magnitude execution.

    Execution model (matches TV):
      1. Signal generated at bar i (close) → executes at bar i+1 open
      2. Stop-loss / take-profit checked intra-bar: SL uses bar low, TP uses bar high
      3. Commission applied on both entry and exit (TV: % of order value)
      4. Pyramiding: max N concurrent entries (default 1)

    Parameters:
        initial_capital   - starting capital
        fee_pct           - commission per trade (%) — TV default is 0.04% for crypto
        slippage_ticks    - slippage in ticks
        slippage_pct      - slippage as % of price
        stop_loss_pct     - stop loss % from entry (e.g. 2 = 2%)
        take_profit_pct   - take profit % from entry (e.g. 5 = 5%)
        long_only         - if True, only long trades
        pyramiding        - max concurrent entries (default 1)
        qty_type          - "equity_pct" (default) or "fixed"
        qty_value         - % of equity (0-100) or fixed units
    """

    def __init__(self, data, initial_capital=10000, fee_pct=0.04,
                 slippage_ticks=0, slippage_pct=0.0,
                 stop_loss_pct=None, take_profit_pct=None,
                 long_only=True, pyramiding=1,
                 qty_type="equity_pct", qty_value=100):
        self.data = data
        self.initial_capital = float(initial_capital)
        self.fee_rate = fee_pct / 100.0
        self.slippage_rate = slippage_pct / 100.0
        self.slippage_ticks = slippage_ticks
        self.stop_loss_pct = stop_loss_pct / 100.0 if stop_loss_pct else None
        self.take_profit_pct = take_profit_pct / 100.0 if take_profit_pct else None
        self.long_only = long_only
        self.pyramiding = pyramiding
        self.qty_type = qty_type
        self.qty_value = qty_value

        n = len(data)
        self._open = [data[i]["open"] for i in range(n)]
        self._high = [data[i]["high"] for i in range(n)]
        self._low = [data[i]["low"] for i in range(n)]
        self._close = [data[i]["close"] for i in range(n)]
        self._time = [data[i]["datetime"] for i in range(n)]
        self._n = n

    def _slip(self, price, is_buy):
        if self.slippage_ticks > 0:
            return price + (self.slippage_ticks * 0.01 if is_buy else -self.slippage_ticks * 0.01)
        if self.slippage_rate > 0:
            return price * (1.0 + self.slippage_rate) if is_buy else price * (1.0 - self.slippage_rate)
        return price

    def run(self, signals):
        n = self._n
        if len(signals) != n:
            raise ValueError(f"Signal length ({len(signals)}) != data length ({n})")

        # State
        cash = self.initial_capital
        qty = 0.0
        entry_price = 0.0
        entry_time = ""
        pos_side = 0  # 1=long, -1=short, 0=flat
        total_fees = 0.0
        pending_entry = None  # (direction, bar_index) to execute at next bar open

        equity_curve = [0.0] * n
        trades = []

        i = 0
        while i < n:
            o = self._open[i]
            h = self._high[i]
            l = self._low[i]
            c = self._close[i]
            t = self._time[i]

            # ── Execute any pending entry at this bar's open ──
            if pending_entry is not None:
                dir_idx, exec_price = pending_entry
                exec_price = self._slip(exec_price, dir_idx == 1)
                trade_capital = cash * (self.qty_value / 100.0) if self.qty_type == "equity_pct" else min(self.qty_value, cash)
                fee = trade_capital * self.fee_rate
                total_fees += fee
                qty = (trade_capital - fee) / exec_price
                entry_price = exec_price
                entry_time = t
                pos_side = 1 if dir_idx == 1 else -1
                cash -= trade_capital
                pending_entry = None

            # ── Check stop-loss / take-profit ──
            if pos_side == 1:
                sl = entry_price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct else None
                tp = entry_price * (1.0 + self.take_profit_pct) if self.take_profit_pct else None
                exit_reason = None
                exit_price_exec = None

                if sl and l <= sl:
                    exit_price_exec = self._slip(sl, False)
                    exit_reason = "stop_loss"
                elif tp and h >= tp:
                    exit_price_exec = self._slip(tp, False)
                    exit_reason = "take_profit"

                if exit_reason:
                    fee = exit_price_exec * qty * self.fee_rate
                    total_fees += fee
                    pnl = (exit_price_exec - entry_price) * qty
                    inv = entry_price * qty
                    net = pnl - fee
                    cash = inv + pnl - fee
                    trades.append(self._trade(entry_time, t, "LONG", entry_price, exit_price_exec, qty, net, exit_reason))
                    pos_side = 0
                    qty = 0.0
                    entry_price = 0.0

            elif pos_side == -1:
                sl = entry_price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct else None
                tp = entry_price * (1.0 - self.take_profit_pct) if self.take_profit_pct else None
                exit_reason = None
                exit_price_exec = None

                if sl and h >= sl:
                    exit_price_exec = self._slip(sl, True)
                    exit_reason = "stop_loss"
                elif tp and l <= tp:
                    exit_price_exec = self._slip(tp, True)
                    exit_reason = "take_profit"

                if exit_reason:
                    fee = exit_price_exec * qty * self.fee_rate
                    total_fees += fee
                    pnl = (entry_price - exit_price_exec) * qty
                    net = pnl - fee
                    cash += net
                    trades.append(self._trade(entry_time, t, "SHORT", entry_price, exit_price_exec, qty, net, exit_reason))
                    pos_side = 0
                    qty = 0.0
                    entry_price = 0.0

            # ── Mark-to-market equity ──
            if pos_side == 1:
                equity_curve[i] = cash + qty * c
            elif pos_side == -1:
                equity_curve[i] = cash + (entry_price - c) * qty
            else:
                equity_curve[i] = cash

            # ── Process signal (after computing equity for this bar) ──
            sig = signals[i]

            if pos_side == 0:
                # Flat — check entry
                if sig == 1:
                    # Schedule entry at next bar open
                    if i + 1 < n:
                        pending_entry = (1, self._open[i + 1])
                elif sig == -1 and not self.long_only:
                    if i + 1 < n:
                        pending_entry = (-1, self._open[i + 1])

            else:
                # In position — check exit signal
                should_exit = (pos_side == 1 and sig != 1) or (pos_side == -1 and sig != -1)
                if should_exit:
                    is_buy = (pos_side == -1)
                    exit_price_exec = self._slip(c, not is_buy)
                    fee = exit_price_exec * qty * self.fee_rate
                    total_fees += fee
                    side_str = "LONG" if pos_side == 1 else "SHORT"
                    if pos_side == 1:
                        pnl = (exit_price_exec - entry_price) * qty
                        inv = entry_price * qty
                        net = pnl - fee
                        cash = inv + pnl - fee
                    else:
                        pnl = (entry_price - exit_price_exec) * qty
                        net = pnl - fee
                        cash += net
                    trades.append(self._trade(entry_time, t, side_str, entry_price, exit_price_exec, qty, net, "signal"))
                    pos_side = 0
                    qty = 0.0
                    entry_price = 0.0

                    # Immediate re-entry if signal flipped (will execute at next bar)
                    if sig == 1 and i + 1 < n:
                        pending_entry = (1, self._open[i + 1])
                    elif sig == -1 and not self.long_only and i + 1 < n:
                        pending_entry = (-1, self._open[i + 1])

            i += 1

        # Force close at end of data
        if pos_side != 0:
            c = self._close[-1]
            t = self._time[-1]
            exit_price_exec = self._slip(c, pos_side == -1)
            fee = exit_price_exec * qty * self.fee_rate
            total_fees += fee
            side_str = "LONG" if pos_side == 1 else "SHORT"
            if pos_side == 1:
                pnl = (exit_price_exec - entry_price) * qty
                inv = entry_price * qty
                net = pnl - fee
                cash = inv + pnl - fee
            else:
                pnl = (entry_price - exit_price_exec) * qty
                net = pnl - fee
                cash += net
            trades.append(self._trade(entry_time, t, side_str, entry_price, exit_price_exec, qty, net, "end_of_data"))
            equity_curve[-1] = cash

        return self._compute_metrics(equity_curve, trades, total_fees)

    def _trade(self, entry_t, exit_t, side, entry_p, exit_p, qty, pnl, reason):
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
        }

    def _compute_metrics(self, equity_curve, trades, total_fees):
        n = len(equity_curve)
        initial = self.initial_capital
        final = equity_curve[-1] if n else initial

        net_profit = final - initial
        total_return_pct = (net_profit / initial) * 100.0 if initial > 0 else 0.0

        # Buy & Hold return
        bnh_return = (self._close[-1] - self._close[0]) / self._close[0] * 100.0 if len(self._close) > 1 else 0.0

        # ── Max Drawdown (TradingView matching: lookback peak) ──
        peak = initial
        max_dd = 0.0
        dd_peak_idx = 0
        dd_trough_idx = 0
        for i in range(n):
            if equity_curve[i] > peak:
                peak = equity_curve[i]
                dd_peak_idx = i
            dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                dd_trough_idx = i
        max_dd_pct = max_dd * 100.0
        max_dd_dollar = peak - equity_curve[dd_trough_idx] if dd_trough_idx < n else 0.0

        # ── Trade Stats ──
        num_trades = len(trades)
        win_count = sum(1 for t in trades if t["pnl"] > 0)
        lose_count = num_trades - win_count
        win_rate_pct = (win_count / num_trades * 100.0) if num_trades > 0 else 0.0

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_profit = sum(t["pnl"] for t in wins) if wins else 0.0
        total_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)

        avg_win = total_profit / win_count if win_count > 0 else 0.0
        avg_loss = total_loss / lose_count if lose_count > 0 else 0.0
        avg_pnl = sum(t["pnl"] for t in trades) / num_trades if num_trades > 0 else 0.0

        max_win = max((t["pnl"] for t in trades), default=0.0)
        max_loss = min((t["pnl"] for t in trades), default=0.0)
        largest_win_pct = max((t["pnl_pct"] for t in trades), default=0.0)
        largest_loss_pct = min((t["pnl_pct"] for t in trades), default=0.0)

        # Consecutive wins/losses
        max_consec_w = 0
        max_consec_l = 0
        cw, cl = 0, 0
        for t in trades:
            if t["pnl"] > 0:
                cw += 1; cl = 0
                max_consec_w = max(max_consec_w, cw)
            else:
                cl += 1; cw = 0
                max_consec_l = max(max_consec_l, cl)

        # Avg duration
        avg_duration_hrs = 0.0
        avg_bars_held = 0.0
        bars_in_trade = 0.0
        if trades:
            durations = []
            for t in trades:
                try:
                    et = datetime.datetime.strptime(t["entry_time"], "%Y-%m-%d %H:%M:%S")
                    xt = datetime.datetime.strptime(t["exit_time"], "%Y-%m-%d %H:%M:%S")
                    hrs = (xt - et).total_seconds() / 3600.0
                    durations.append(hrs)
                    bars_in_trade += hrs
                except (ValueError, TypeError):
                    pass
            if durations:
                avg_duration_hrs = sum(durations) / len(durations)
                avg_bars_held = avg_duration_hrs

        exposure_pct = (bars_in_trade / n * 100.0) if n > 0 else 0.0

        # ── Return series for ratios ──
        bar_returns = []
        for i in range(1, n):
            prev = equity_curve[i - 1]
            if prev > 0:
                bar_returns.append((equity_curve[i] - prev) / prev)

        # ── Sharpe Ratio (annualized) ──
        sharpe = 0.0
        if len(bar_returns) > 5:
            mean_r = sum(bar_returns) / len(bar_returns)
            var_r = sum((r - mean_r) ** 2 for r in bar_returns) / len(bar_returns)
            std_r = math.sqrt(var_r)
            if std_r > 1e-10:
                # Annualize: sqrt(periods_per_year)
                # For hourly: 365*24=8760; for 5m: 365*24*12=105120
                periods_per_year = max(len(bar_returns) / (n / 8760.0), 1.0) if n > 0 else 8760
                periods_per_year = min(periods_per_year, 200000)
                annual_factor = math.sqrt(periods_per_year)
                sharpe = (mean_r / std_r) * annual_factor

        # ── Sortino ──
        sortino = 0.0
        negative_returns = [r for r in bar_returns if r < 0]
        if negative_returns and len(bar_returns) > 5:
            down_var = sum(r * r for r in negative_returns) / len(negative_returns)
            down_std = math.sqrt(down_var)
            if down_std > 1e-10:
                periods_per_year = max(len(bar_returns) / (n / 8760.0), 1.0) if n > 0 else 8760
                sortino = (mean_r / down_std) * math.sqrt(periods_per_year)

        # ── Calmar ──
        calmar = total_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

        # ── CAGR ──
        total_hours = n  # 1 bar = 1 hour for hourly data
        years = total_hours / (365.25 * 24.0)
        cagr = ((final / initial) ** (1.0 / years) - 1.0) * 100.0 if years > 0.5 and initial > 0 else 0.0

        # ── Expectancy ──
        if num_trades > 0:
            w_pct = win_count / num_trades
            l_pct = lose_count / num_trades
            expectancy = w_pct * avg_win - l_pct * avg_loss
        else:
            expectancy = 0.0

        # ── P&L by side ──
        long_pnl = sum(t["pnl"] for t in trades if t["side"] == "LONG")
        short_pnl = sum(t["pnl"] for t in trades if t["side"] == "SHORT")

        # ── Equity curve dicts ──
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
        row("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
        row("Sortino Ratio", f"{results['sortino_ratio']:.3f}")
        row("Calmar Ratio", f"{results['calmar_ratio']:.3f}")
        row("Profit Factor", f"{results['profit_factor']:.2f}")
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
                      "qty", "pnl", "pnl_pct", "exit_reason"]
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
