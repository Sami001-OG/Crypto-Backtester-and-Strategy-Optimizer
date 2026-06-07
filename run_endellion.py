"""
Run Endellion backtest in both entry modes for comparison.

Mode 1 — "dca":       Entry at close (100% signals taken) + DCA at PB
Mode 2 — "pullback":  Entry only at PB level (original spec limit order)
"""

from endellion import EndellionBacktest, BEST_CONFIG, SYMBOLS


def print_results(m, label):
    print("\n" + "=" * 62)
    print(f"  {label}")
    print("=" * 62)
    print(f"  Total Trades:       {m['trades']}")
    print(f"  Winning Trades:     {m['wins']}")
    print(f"  Losing Trades:      {m['losses']}")
    print(f"  Win Rate:           {m['win_rate']:.1f}%")
    print(f"  Total PnL:          {m['total_pnl']:+.1f}%")
    print(f"  Profit Factor:      {m['profit_factor']:.2f}")
    print(f"  Avg PnL/Trade:      {m['avg_pnl']:+.2f}%")
    print(f"  Avg Win:            {m['avg_win']:+.2f}%")
    print(f"  Avg Loss:           {m['avg_loss']:+.2f}%")
    print(f"  Max Win:            {m['max_win']:+.2f}%")
    print(f"  Max Loss:           {m['max_loss']:+.2f}%")
    print(f"  DCA Added:          {m['dca_count']} ({m['dca_pct']:.1f}%)")
    print()
    print("  Per-Symbol:")
    for sym, sd in m.get("by_symbol", {}).items():
        print(f"    {sym}: {sd['trades']} trades, {sd['pnl']:+.1f}% PnL, {sd['win_rate']:.1f}% WR")
    print("=" * 62)


if __name__ == "__main__":
    # Mode 1: DCA
    bt1 = EndellionBacktest(entry_mode="dca")
    bt1.run()
    m1 = bt1.compute_metrics()
    print_results(m1, "MODE: Entry at Close + DCA at PB")

    # Mode 2: Original pullback
    bt2 = EndellionBacktest(entry_mode="pullback")
    bt2.run()
    m2 = bt2.compute_metrics()
    print_results(m2, "MODE: Entry ONLY at PB Level (original)")

    # Summary comparison
    print("\n" + "=" * 62)
    print("  COMPARISON SUMMARY")
    print("=" * 62)
    print(f"  {'Metric':<30} {'DCA Mode':>14} {'Pullback':>14}")
    print(f"  {'Trades':<30} {m1['trades']:>14} {m2['trades']:>14}")
    print(f"  {'Win Rate':<30} {m1['win_rate']:>13.1f}% {m2['win_rate']:>13.1f}%")
    print(f"  {'Total PnL':<30} {m1['total_pnl']:>+13.1f}% {m2['total_pnl']:>+13.1f}%")
    print(f"  {'Profit Factor':<30} {m1['profit_factor']:>14.2f} {m2['profit_factor']:>14.2f}")
    print(f"  {'DCA Hit %':<30} {m1['dca_pct']:>13.1f}% {'N/A':>14}")
    print("=" * 62)
