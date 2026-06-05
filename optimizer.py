"""
Parameter optimizer — parallel grid search, walk-forward with rolling windows,
and adaptive search.

Usage:
    from optimizer import grid_search, walk_forward, adaptive_search
    best = grid_search("BTC_1h.csv", "sma_crossover",
                        {"fast_period": [5,10,20], "slow_period": [30,50,100]})
"""

import sys
import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import Backtester, load_data
import strategies as strat


def _run_single(data, strategy_fn, params, initial_capital, fee_pct,
                stop_loss_pct, take_profit_pct, long_only):
    """Worker: run one parameter combination."""
    try:
        signals = strategy_fn(data, **params)
        bt = Backtester(
            data,
            initial_capital=initial_capital,
            fee_pct=fee_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            long_only=long_only,
        )
        metrics = bt.run(signals)
        summary = {k: v for k, v in metrics.items()
                   if k not in ("equity_curve", "trades", "trade_pnls_pct")}
        summary["params"] = params
        return summary, None
    except Exception as e:
        return None, str(e)


def grid_search(filename, strategy_name, param_grid, initial_capital=10000,
                fee_pct=0.04, stop_loss_pct=None, take_profit_pct=None,
                long_only=True, rank_by="sharpe_ratio", top_n=10,
                parallel=True, max_workers=None):
    """
    Exhaustive grid search over parameter combinations with parallel execution.

    Returns sorted list of (params_dict, metrics_dict).
    """
    strategy_fn = strat.STRATEGIES.get(strategy_name)
    if not strategy_fn:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strat.STRATEGIES.keys())}")

    data = load_data(filename)
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"Grid search: {strategy_name} on {filename} ({len(data):,} bars)")
    print(f"Testing {len(combos)} parameter combinations...")

    results = []
    errors = []
    workers = max_workers or min(cpu_count(), 8)

    if parallel and len(combos) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for combo in combos:
                params = dict(zip(keys, combo))
                fut = executor.submit(
                    _run_single, data, strategy_fn, params,
                    initial_capital, fee_pct, stop_loss_pct, take_profit_pct, long_only
                )
                futures[fut] = params

            done = 0
            for fut in as_completed(futures):
                summary, err = fut.result()
                done += 1
                if summary:
                    results.append(summary)
                if err:
                    errors.append((futures[fut], err))
                if done % 50 == 0 or done == len(combos):
                    print(f"  {done}/{len(combos)} tested...")
    else:
        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            summary, err = _run_single(data, strategy_fn, params,
                                       initial_capital, fee_pct, stop_loss_pct,
                                       take_profit_pct, long_only)
            if summary:
                results.append(summary)
            if err:
                errors.append((params, err))
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(combos)} tested...")

    # Sort
    reverse = rank_by not in ("max_drawdown_pct",)
    results.sort(key=lambda r: r.get(rank_by, 0), reverse=reverse)

    # Print
    print(f"\n{'='*80}")
    print(f"  TOP {min(top_n, len(results))} RESULTS (ranked by {rank_by})")
    print(f"{'='*80}")
    header = f"  {'#':<4} {'Return%':>9} {'MaxDD%':>8} {'Win%':>7} {'Sharpe':>8} {'PF':>6} {'Trades':>7}  Params"
    print(header)
    print(f"  {'-'*4} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*7}  {'-'*30}")
    for i, r in enumerate(results[:top_n]):
        print(f"  {i+1:<4} {r['total_return_pct']:>8.2f}% {r['max_drawdown_pct']:>7.2f}% "
              f"{r['win_rate_pct']:>6.2f}% {r['sharpe_ratio']:>8.2f} "
              f"{r.get('profit_factor', 0):>6.2f} {r['num_trades']:>7}  {r['params']}")
    print(f"{'='*80}\n")

    if errors:
        print(f"  {len(errors)} errors skipped.")

    return results


def walk_forward(filename, strategy_name, param_grid, initial_capital=10000,
                 fee_pct=0.04, train_ratio=0.7, rank_by="sharpe_ratio",
                 stop_loss_pct=None, take_profit_pct=None, long_only=True,
                 parallel=True):
    """
    Walk-forward optimization: train on first train_ratio, test on rest.
    """
    strategy_fn = strat.STRATEGIES.get(strategy_name)
    if not strategy_fn:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    full_data = load_data(filename)
    split_idx = int(len(full_data) * train_ratio)
    train_data = full_data[:split_idx]
    test_data = full_data[split_idx:]

    print(f"Walk-forward: {strategy_name} on {filename}")
    print(f"  Training: {len(train_data):,} bars | Testing: {len(test_data):,} bars")

    # Grid search on training
    best = grid_search(
        filename, strategy_name, param_grid,
        initial_capital=initial_capital, fee_pct=fee_pct,
        rank_by=rank_by, top_n=1, parallel=parallel
    )

    if not best:
        print("No valid params found on training set.")
        return None, None, None

    best_params = best[0]["params"]
    print(f"\n  Best params (training): {best_params}")

    # Evaluate on test set
    signals = strategy_fn(test_data, **best_params)
    bt = Backtester(test_data, initial_capital=initial_capital, fee_pct=fee_pct,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                    long_only=long_only)
    test_metrics = bt.run(signals)

    print("\n  ── TRAINING SET ──")
    Backtester.print_report(best[0])
    print("\n  ── TEST SET (out-of-sample) ──")
    Backtester.print_report(test_metrics)

    return best_params, best[0], test_metrics


def rolling_walk_forward(filename, strategy_name, param_grid, initial_capital=10000,
                         fee_pct=0.04, windows=4, rank_by="sharpe_ratio",
                         stop_loss_pct=None, take_profit_pct=None, long_only=True,
                         parallel=True):
    """
    Rolling walk-forward: slide a window across the data.
    Train on window N, test on window N+1. Average results across all folds.

    Returns aggregate out-of-sample metrics and per-fold breakdown.
    """
    strategy_fn = strat.STRATEGIES.get(strategy_name)
    if not strategy_fn:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    full_data = load_data(filename)
    fold_size = len(full_data) // windows
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"Rolling walk-forward: {strategy_name} on {filename}")
    print(f"  {windows} folds, {fold_size:,} bars each, {len(combos)} param combos")

    all_test_trades = []
    all_test_equity = []
    fold_results = []

    for fold in range(windows - 1):
        train_start = fold * fold_size
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min((fold + 2) * fold_size, len(full_data))

        train_data = full_data[train_start:train_end]
        test_data = full_data[test_start:test_end]

        # Find best params on this fold's training
        best_score = float("-inf")
        best_params = None

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                signals = strategy_fn(train_data, **params)
                bt = Backtester(train_data, initial_capital=initial_capital, fee_pct=fee_pct,
                                stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                                long_only=long_only)
                metrics = bt.run(signals)
                score = metrics.get(rank_by, 0)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            continue

        # Test
        signals = strategy_fn(test_data, **best_params)
        bt = Backtester(test_data, initial_capital=initial_capital, fee_pct=fee_pct,
                        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                        long_only=long_only)
        test_res = bt.run(signals)
        fold_results.append({"fold": fold, "params": best_params,
                             "test_metrics": test_res})
        all_test_trades.extend(test_res["trades"])
        print(f"  Fold {fold+1}/{windows-1}: return={test_res['total_return_pct']:.2f}% "
              f"sharpe={test_res['sharpe_ratio']:.2f} params={best_params}")

    # Aggregate
    if all_test_trades:
        wins = [t for t in all_test_trades if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in all_test_trades)
        print(f"\n  ── AGGREGATE OOS ──")
        print(f"  Total Trades: {len(all_test_trades)}")
        print(f"  Win Rate: {len(wins)/len(all_test_trades)*100:.1f}%")
        print(f"  Total PnL: ${total_pnl:,.2f}")

    return fold_results


def adaptive_search(filename, strategy_name, param_ranges, initial_capital=10000,
                    fee_pct=0.04, iterations=200, rank_by="sharpe_ratio",
                    stop_loss_pct=None, take_profit_pct=None, long_only=True):
    """
    Adaptive random search: sample random params from ranges, ideal for large spaces.

    param_ranges: dict of param_name -> (low, high, step OR "int"/"float"/"bool")
        e.g. {"fast_period": (5, 100, "int"), "slow_period": (50, 300, "int")}
    """
    import random

    strategy_fn = strat.STRATEGIES.get(strategy_name)
    if not strategy_fn:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    data = load_data(filename)
    print(f"Adaptive search: {strategy_name} on {filename} | {iterations} iterations")

    results = []
    rng = random.Random(42)

    for it in range(iterations):
        params = {}
        for name, rng_spec in param_ranges.items():
            low, high, dtype = rng_spec
            if dtype == "int":
                params[name] = rng.randint(int(low), int(high))
            elif dtype == "float":
                params[name] = rng.uniform(low, high)
            elif dtype == "bool":
                params[name] = rng.choice([True, False])
            elif dtype == "choice":
                params[name] = rng.choice(rng_spec[3])  # extra arg: list of choices

        summary, err = _run_single(data, strategy_fn, params,
                                   initial_capital, fee_pct, stop_loss_pct,
                                   take_profit_pct, long_only)
        if summary:
            results.append(summary)

        if (it + 1) % 50 == 0:
            print(f"  {it + 1}/{iterations}...")

    reverse = rank_by not in ("max_drawdown_pct",)
    results.sort(key=lambda r: r.get(rank_by, 0), reverse=reverse)

    print(f"\n{'='*80}")
    print(f"  TOP 10 RESULTS (ranked by {rank_by})")
    print(f"{'='*80}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>2}. Return={r['total_return_pct']:>8.2f}%  "
              f"DD={r['max_drawdown_pct']:>6.2f}%  "
              f"Sharpe={r['sharpe_ratio']:>7.2f}  "
              f"PF={r.get('profit_factor',0):>6.2f}  "
              f"Trades={r['num_trades']:>5}  {r['params']}")
    print(f"{'='*80}\n")
    return results


if __name__ == "__main__":
    grid_search(
        "BTC_1h.csv", "sma_crossover",
        {"fast_period": [5, 10, 15, 20], "slow_period": [30, 50, 100, 200]},
        rank_by="total_return_pct",
        parallel=True,
    )
