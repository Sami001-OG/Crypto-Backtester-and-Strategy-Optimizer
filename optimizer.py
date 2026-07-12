"""
Brute-force parameter optimizer for the backtester.

Given a strategy (which uses one or more indicators) and a grid of parameter
values, the optimizer backtests *every* combination and ranks the results, so
you can find the best-performing parameter set for that strategy/indicator.

Features:
  - grid_search          : exhaustive parallel grid over all combinations
  - walk_forward         : train on the first slice, test out-of-sample (no leak)
  - rolling_walk_forward : N anchored folds, aggregate out-of-sample metrics
  - adaptive_search      : random sampling + refinement for huge spaces

Parallelism uses processes (not threads) so pure-Python backtests actually run
on multiple cores. Data is loaded once per worker via an initializer, so the
big OHLCV list is never re-pickled per task.

Usage:
    from optimizer import grid_search
    best = grid_search("BTC_1h.csv", "sma_crossover",
                        {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
                        rank_by="sharpe_ratio")
    print(best[0]["params"], best[0]["sharpe_ratio"])
"""

import sys
import os
import itertools
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import Backtester, load_data
import strategies as strat

# Metrics where a SMALLER value is better (everything else: larger is better).
_LOWER_IS_BETTER = {"max_drawdown_pct", "max_drawdown_dollar", "total_fees"}

_DROP_KEYS = ("equity_curve", "trades", "trade_pnls_pct")

# ── Per-worker state (populated by the process initializer) ──────────────────
_WORKER = {}


def _init_worker(filename):
    """Runs once per worker process: load the dataset a single time."""
    _WORKER["data"] = load_data(filename)


def _score_key(rank_by):
    """Return a sort key that pushes NaN/None to the worst position."""
    lower = rank_by in _LOWER_IS_BETTER

    def key(r):
        v = r.get(rank_by, None)
        if v is None:
            v = float("inf") if lower else float("-inf")
        # inf profit_factor etc. sort naturally; guard against NaN
        if v != v:  # NaN
            v = float("inf") if lower else float("-inf")
        return v
    return key, (not lower)  # (key, reverse)


def _run_single(data, strategy_fn, params, bt_kwargs):
    """Backtest one parameter combination. Returns (summary|None, error|None)."""
    try:
        signals = strategy_fn(data, **params)
        bt = Backtester(data, **bt_kwargs)
        metrics = bt.run(signals)
        summary = {k: v for k, v in metrics.items() if k not in _DROP_KEYS}
        summary["params"] = params
        return summary, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _eval_task(task):
    """Process-pool entry point. Reads shared data from the worker cache."""
    strategy_name, params, bt_kwargs, sl_bounds = task
    data = _WORKER["data"]
    if sl_bounds is not None:
        data = data[sl_bounds[0]:sl_bounds[1]]
    strategy_fn = strat.STRATEGIES[strategy_name]
    return _run_single(data, strategy_fn, params, bt_kwargs)


def _bt_kwargs(initial_capital, fee_pct, stop_loss_pct, take_profit_pct,
               long_only, extra=None):
    kw = {
        "initial_capital": initial_capital,
        "fee_pct": fee_pct,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "long_only": long_only,
    }
    if extra:
        kw.update(extra)
    return kw


def _expand_grid(param_grid):
    keys = list(param_grid.keys())
    values = [v if isinstance(v, (list, tuple)) else [v] for v in param_grid.values()]
    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]
    return combos


def _rank(results, rank_by, min_trades):
    filtered = [r for r in results if r.get("num_trades", 0) >= min_trades]
    if not filtered and results:
        # Nothing met min_trades — fall back so the caller still gets output.
        filtered = results
    key, reverse = _score_key(rank_by)
    filtered.sort(key=key, reverse=reverse)
    return filtered


def _print_table(results, rank_by, top_n):
    print(f"\n{'='*88}")
    print(f"  TOP {min(top_n, len(results))} RESULTS (ranked by {rank_by})")
    print(f"{'='*88}")
    print(f"  {'#':<4} {'Return%':>9} {'MaxDD%':>8} {'Win%':>7} {'Sharpe':>8} "
          f"{'PF':>6} {'Trades':>7}  Params")
    print(f"  {'-'*4} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*7}  {'-'*34}")
    for i, r in enumerate(results[:top_n]):
        pf = r.get("profit_factor", 0)
        pf_s = " inf" if pf == float("inf") else f"{pf:6.2f}"
        print(f"  {i+1:<4} {r['total_return_pct']:>8.2f}% {r['max_drawdown_pct']:>7.2f}% "
              f"{r['win_rate_pct']:>6.2f}% {r['sharpe_ratio']:>8.2f} "
              f"{pf_s} {r['num_trades']:>7}  {r['params']}")
    print(f"{'='*88}\n")


def grid_search(filename, strategy_name, param_grid, initial_capital=10000,
                fee_pct=0.04, stop_loss_pct=None, take_profit_pct=None,
                long_only=True, rank_by="sharpe_ratio", top_n=10,
                min_trades=1, parallel=True, max_workers=None, bt_extra=None,
                _data_slice=None, _verbose=True):
    """
    Exhaustive brute-force grid search over every parameter combination.

    Backtests each combo, then ranks by ``rank_by`` (higher is better except for
    drawdown/fees, which rank ascending). ``min_trades`` filters out degenerate
    combos that only fired a handful of trades — a basic overfitting guard.

    Returns the ranked list of per-combo summary dicts, each with a ``params``
    key. ``best = grid_search(...)[0]`` is the winner.
    """
    if strategy_name not in strat.STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Available: {list(strat.STRATEGIES.keys())}")
    strategy_fn = strat.STRATEGIES[strategy_name]

    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("param_grid produced no combinations")

    bt_kwargs = _bt_kwargs(initial_capital, fee_pct, stop_loss_pct,
                           take_profit_pct, long_only, bt_extra)

    if _verbose:
        # Only load here for the bar count / message; workers load their own copy.
        n_bars = len(load_data(filename)) if _data_slice is None else (_data_slice[1] - _data_slice[0])
        print(f"Grid search: {strategy_name} on {filename} ({n_bars:,} bars)")
        print(f"Testing {len(combos)} parameter combinations "
              f"({'parallel' if parallel and len(combos) > 1 else 'serial'})...")

    results, errors = [], []
    workers = max_workers or max(1, min(cpu_count(), 8))

    if parallel and len(combos) > 1:
        tasks = [(strategy_name, p, bt_kwargs, _data_slice) for p in combos]
        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker,
                                     initargs=(filename,)) as ex:
                futs = {ex.submit(_eval_task, t): t[1] for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    summary, err = fut.result()
                    done += 1
                    if summary:
                        results.append(summary)
                    if err:
                        errors.append((futs[fut], err))
                    if _verbose and (done % 50 == 0 or done == len(combos)):
                        print(f"  {done}/{len(combos)} tested...")
        except Exception as e:
            # Fall back to serial if the process pool can't start (restricted env).
            if _verbose:
                print(f"  Process pool unavailable ({e}); running serial.")
            parallel = False

    if not parallel or len(combos) == 1:
        data = load_data(filename)
        if _data_slice is not None:
            data = data[_data_slice[0]:_data_slice[1]]
        for idx, params in enumerate(combos):
            summary, err = _run_single(data, strategy_fn, params, bt_kwargs)
            if summary:
                results.append(summary)
            if err:
                errors.append((params, err))
            if _verbose and (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(combos)} tested...")

    ranked = _rank(results, rank_by, min_trades)

    if _verbose:
        _print_table(ranked, rank_by, top_n)
        if errors:
            print(f"  {len(errors)} combos raised errors (e.g. {errors[0][1]}).")

    return ranked


def walk_forward(filename, strategy_name, param_grid, initial_capital=10000,
                 fee_pct=0.04, train_ratio=0.7, rank_by="sharpe_ratio",
                 stop_loss_pct=None, take_profit_pct=None, long_only=True,
                 min_trades=1, parallel=True):
    """
    Walk-forward optimization with NO lookahead: grid-search on the first
    ``train_ratio`` of the data only, then evaluate the winning params on the
    held-out remainder.

    Returns (best_params, train_summary, test_metrics).
    """
    if strategy_name not in strat.STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy_fn = strat.STRATEGIES[strategy_name]

    full_data = load_data(filename)
    n = len(full_data)
    split_idx = int(n * train_ratio)
    if split_idx < 2 or split_idx >= n:
        raise ValueError(f"train_ratio={train_ratio} leaves an empty train/test split")

    print(f"Walk-forward: {strategy_name} on {filename}")
    print(f"  Train: bars 0..{split_idx:,} | Test: bars {split_idx:,}..{n:,}")

    # Grid-search on the TRAIN slice only (this is the leak fix).
    train_ranked = grid_search(
        filename, strategy_name, param_grid,
        initial_capital=initial_capital, fee_pct=fee_pct,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        long_only=long_only, rank_by=rank_by, top_n=1, min_trades=min_trades,
        parallel=parallel, _data_slice=(0, split_idx), _verbose=False,
    )
    if not train_ranked:
        print("No valid params found on training set.")
        return None, None, None

    best = train_ranked[0]
    best_params = best["params"]
    print(f"  Best params (in-sample): {best_params}")

    # Evaluate on the out-of-sample test slice.
    test_data = full_data[split_idx:]
    signals = strategy_fn(test_data, **best_params)
    bt = Backtester(test_data, initial_capital=initial_capital, fee_pct=fee_pct,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                    long_only=long_only)
    test_metrics = bt.run(signals)

    print("\n  ── IN-SAMPLE (train) ──")
    Backtester.print_report(best)
    print("\n  ── OUT-OF-SAMPLE (test) ──")
    Backtester.print_report(test_metrics)

    return best_params, best, test_metrics


def rolling_walk_forward(filename, strategy_name, param_grid, initial_capital=10000,
                         fee_pct=0.04, windows=4, rank_by="sharpe_ratio",
                         stop_loss_pct=None, take_profit_pct=None, long_only=True,
                         min_trades=1, parallel=True):
    """
    Anchored rolling walk-forward. Split the data into ``windows`` equal folds;
    for each fold N, grid-search on fold N (train) and evaluate on fold N+1
    (out-of-sample). Aggregate the out-of-sample results across all folds.

    Returns a list of per-fold dicts: {fold, params, test_metrics}.
    """
    if strategy_name not in strat.STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy_fn = strat.STRATEGIES[strategy_name]

    full_data = load_data(filename)
    n = len(full_data)
    fold_size = n // windows
    if fold_size < 2:
        raise ValueError(f"{windows} folds is too many for {n} bars")
    combos = _expand_grid(param_grid)

    print(f"Rolling walk-forward: {strategy_name} on {filename}")
    print(f"  {windows} folds, {fold_size:,} bars each, {len(combos)} param combos")

    fold_results = []
    all_test_trades = []
    agg_return = []

    for fold in range(windows - 1):
        train_start, train_end = fold * fold_size, (fold + 1) * fold_size
        test_start, test_end = train_end, min((fold + 2) * fold_size, n)

        ranked = grid_search(
            filename, strategy_name, param_grid,
            initial_capital=initial_capital, fee_pct=fee_pct,
            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
            long_only=long_only, rank_by=rank_by, top_n=1, min_trades=min_trades,
            parallel=parallel, _data_slice=(train_start, train_end), _verbose=False,
        )
        if not ranked:
            print(f"  Fold {fold+1}: no valid params on training slice, skipping.")
            continue
        best_params = ranked[0]["params"]

        test_data = full_data[test_start:test_end]
        signals = strategy_fn(test_data, **best_params)
        bt = Backtester(test_data, initial_capital=initial_capital, fee_pct=fee_pct,
                        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                        long_only=long_only)
        test_res = bt.run(signals)
        fold_results.append({"fold": fold, "params": best_params, "test_metrics": test_res})
        all_test_trades.extend(test_res["trades"])
        agg_return.append(test_res["total_return_pct"])
        print(f"  Fold {fold+1}/{windows-1}: OOS return={test_res['total_return_pct']:>7.2f}% "
              f"sharpe={test_res['sharpe_ratio']:>6.2f}  params={best_params}")

    if all_test_trades:
        wins = sum(1 for t in all_test_trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in all_test_trades)
        avg_ret = sum(agg_return) / len(agg_return) if agg_return else 0.0
        print(f"\n  ── AGGREGATE OUT-OF-SAMPLE ──")
        print(f"  Folds evaluated : {len(fold_results)}")
        print(f"  Total OOS trades: {len(all_test_trades)}")
        print(f"  OOS win rate    : {wins/len(all_test_trades)*100:.1f}%")
        print(f"  Avg fold return : {avg_ret:+.2f}%")
        print(f"  Total OOS PnL   : ${total_pnl:,.2f}")

    return fold_results


def adaptive_search(filename, strategy_name, param_ranges, initial_capital=10000,
                    fee_pct=0.04, iterations=200, rank_by="sharpe_ratio",
                    stop_loss_pct=None, take_profit_pct=None, long_only=True,
                    min_trades=1, refine_frac=0.4, seed=42):
    """
    Random search with adaptive refinement — for spaces too large to grid fully.

    param_ranges: dict of name -> spec, where spec is one of
        (low, high, "int")
        (low, high, "float")
        (low, high, "bool")            # low/high ignored
        (low, high, "choice", [a, b])  # sample from the explicit list

    Phase 1 (exploration) samples ``iterations*(1-refine_frac)`` random points.
    Phase 2 (refinement) samples the remainder from a shrunken window centred on
    the best point so far, honing in on the optimum.

    Returns the ranked list of summary dicts.
    """
    if strategy_name not in strat.STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy_fn = strat.STRATEGIES[strategy_name]

    data = load_data(filename)
    bt_kwargs = _bt_kwargs(initial_capital, fee_pct, stop_loss_pct,
                           take_profit_pct, long_only)
    rng = random.Random(seed)
    key, reverse = _score_key(rank_by)

    print(f"Adaptive search: {strategy_name} on {filename} | {iterations} iterations")

    def sample(center=None, shrink=1.0):
        params = {}
        for name, spec in param_ranges.items():
            low, high, dtype = spec[0], spec[1], spec[2]
            if dtype == "bool":
                params[name] = rng.choice([True, False])
            elif dtype == "choice":
                params[name] = rng.choice(spec[3])
            else:
                if center is not None and name in center and dtype in ("int", "float"):
                    span = (high - low) * shrink / 2.0
                    lo = max(low, center[name] - span)
                    hi = min(high, center[name] + span)
                else:
                    lo, hi = low, high
                if dtype == "int":
                    params[name] = rng.randint(int(round(lo)), int(round(hi)))
                else:
                    params[name] = round(rng.uniform(lo, hi), 6)
        return params

    results = []
    seen = set()
    explore_n = max(1, int(iterations * (1.0 - refine_frac)))
    best_params = None

    for it in range(iterations):
        if it < explore_n or best_params is None:
            params = sample()
        else:
            # Shrink the window as refinement progresses.
            progress = (it - explore_n) / max(1, iterations - explore_n)
            shrink = max(0.05, 0.5 * (1.0 - progress))
            params = sample(center=best_params, shrink=shrink)

        sig = tuple(sorted(params.items()))
        if sig in seen:
            continue
        seen.add(sig)

        summary, _ = _run_single(data, strategy_fn, params, bt_kwargs)
        if summary and summary.get("num_trades", 0) >= min_trades:
            results.append(summary)
            # Track the current best so refinement samples around it.
            cur_best = max(results, key=key) if reverse else min(results, key=key)
            best_params = cur_best["params"]

        if (it + 1) % 50 == 0:
            print(f"  {it + 1}/{iterations}...")

    ranked = _rank(results, rank_by, min_trades)
    _print_table(ranked, rank_by, 10)
    return ranked


if __name__ == "__main__":
    grid_search(
        "BTC_1h.csv", "sma_crossover",
        {"fast_period": [5, 10, 15, 20], "slow_period": [30, 50, 100, 200]},
        rank_by="total_return_pct",
        parallel=True,
    )
