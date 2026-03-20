#!/usr/bin/env python3
"""Benchmark data generation speed at various depths.

Runs a small number of games at each depth and reports:
- Games/sec and positions/sec
- Estimated wall time for 1M games
- Per-game timing with verbose progress

Usage:
    python scripts/benchmark_depths.py [--games 50] [--workers 14] [--depths 8 9 10 11 12 13 14 15]

Designed for GCP VM benchmarking — logs are verbose so you can poll with `tail -f`.
"""

import argparse
import time
import sys
import os
import json
from datetime import datetime, timedelta

def run_benchmark(depths, games_per_depth, workers, random_moves):
    from tonnesjakk.nnue import DataGenerator

    results = {}
    trainer = DataGenerator()

    print(f"={'=' * 70}")
    print(f"NNUE Data Generation Benchmark")
    print(f"  Games per depth: {games_per_depth}")
    print(f"  Workers: {workers}")
    print(f"  Random moves: {random_moves}")
    print(f"  Depths to test: {depths}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"={'=' * 70}")
    print(flush=True)

    for depth in depths:
        print(f"\n--- Depth {depth} ({games_per_depth} games, {workers} workers) ---")
        print(flush=True)

        save_path = f"/tmp/bench_d{depth}.bin"
        # Clean up any previous benchmark files
        for f in [save_path, save_path.replace('.bin', '_meta.json')]:
            if os.path.exists(f):
                os.remove(f)

        t0 = time.time()
        try:
            _, _, stats = trainer.generate_dataset(
                num_games=games_per_depth,
                depth=depth,
                random_opening_moves=random_moves,
                use_search_scores=True,
                augment=True,
                verbose=True,
                save_every=max(1, games_per_depth // 10),
                save_path=save_path,
                workers=workers,
            )
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  ERROR at depth {depth}: {e}")
            print(flush=True)
            results[depth] = {"error": str(e)}
            continue

        total_games = stats.white_wins + stats.black_wins + stats.draws
        games_per_sec = total_games / elapsed if elapsed > 0 else 0
        pos_per_sec = stats.total_positions / elapsed if elapsed > 0 else 0
        pos_per_game = stats.total_positions / total_games if total_games > 0 else 0

        # Estimate time for various dataset sizes
        est_100k = timedelta(seconds=100_000 / games_per_sec) if games_per_sec > 0 else None
        est_500k = timedelta(seconds=500_000 / games_per_sec) if games_per_sec > 0 else None
        est_1m = timedelta(seconds=1_000_000 / games_per_sec) if games_per_sec > 0 else None
        est_2m = timedelta(seconds=2_000_000 / games_per_sec) if games_per_sec > 0 else None

        results[depth] = {
            "elapsed_sec": round(elapsed, 1),
            "total_games": total_games,
            "total_positions": stats.total_positions,
            "games_per_sec": round(games_per_sec, 3),
            "positions_per_sec": round(pos_per_sec, 1),
            "positions_per_game": round(pos_per_game, 1),
            "balance": f"W{stats.white_wins}/B{stats.black_wins}/D{stats.draws}",
            "est_100k_games": str(est_100k).split('.')[0] if est_100k else "N/A",
            "est_500k_games": str(est_500k).split('.')[0] if est_500k else "N/A",
            "est_1M_games": str(est_1m).split('.')[0] if est_1m else "N/A",
            "est_2M_games": str(est_2m).split('.')[0] if est_2m else "N/A",
        }

        print(f"\n  RESULTS depth {depth}:")
        print(f"    Time: {elapsed:.1f}s for {total_games} games")
        print(f"    Speed: {games_per_sec:.3f} games/sec, {pos_per_sec:.0f} positions/sec")
        print(f"    Avg positions/game: {pos_per_game:.1f} (with augment)")
        print(f"    Balance: {results[depth]['balance']}")
        print(f"    --- Time estimates ---")
        print(f"    100K games: {results[depth]['est_100k_games']}")
        print(f"    500K games: {results[depth]['est_500k_games']}")
        print(f"    1M games:   {results[depth]['est_1M_games']}")
        print(f"    2M games:   {results[depth]['est_2M_games']}")
        print(flush=True)

        # Clean up benchmark files
        for f in [save_path, save_path.replace('.bin', '_meta.json')]:
            if os.path.exists(f):
                os.remove(f)

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Depth':>6} {'Games/s':>10} {'Pos/s':>10} {'Pos/game':>10} {'Est 1M':>14} {'Est 2M':>14}")
    print(f"{'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 14} {'-' * 14}")
    for depth in depths:
        r = results.get(depth, {})
        if "error" in r:
            print(f"{depth:>6} {'ERROR':>10}")
            continue
        print(f"{depth:>6} {r['games_per_sec']:>10.3f} {r['positions_per_sec']:>10.0f} "
              f"{r['positions_per_game']:>10.1f} {r['est_1M_games']:>14} {r['est_2M_games']:>14}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(flush=True)

    # Save results to JSON for later reference
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"meta": {"games_per_depth": games_per_depth, "workers": workers,
                            "random_moves": random_moves, "timestamp": datetime.now().isoformat()},
                   "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NNUE data generation at various depths")
    parser.add_argument("--games", type=int, default=50, help="Games per depth (default: 50)")
    parser.add_argument("--workers", type=int, default=14, help="Parallel workers (default: 14)")
    parser.add_argument("--random-moves", type=int, default=6, help="Random opening moves (default: 6)")
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 9, 10, 11, 12, 13, 14, 15],
                        help="Depths to benchmark (default: 8-15)")
    args = parser.parse_args()

    run_benchmark(args.depths, args.games, args.workers, args.random_moves)
