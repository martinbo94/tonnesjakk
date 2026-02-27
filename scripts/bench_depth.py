#!/usr/bin/env python3
"""
Depth scaling benchmark for Tonnesjakk.

Measures two things:
1. ELO gain per depth level (round-robin tournament, same weights, different depths)
2. Speed: games/second at each depth (using play_alphabeta_games from Rust)

Usage:
    python scripts/bench_depth.py                    # depths 4-9, 50 games/pair
    python scripts/bench_depth.py --depths 5 6 7 8 9 10 --games 100
    python scripts/bench_depth.py --speed-only       # just benchmark speed
"""

import argparse
import multiprocessing
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tonnesjakk import Board, Engine
from tonnesjakk._core import MCTSEngine as _RustMCTSEngine


def is_white_turn(board: Board) -> bool:
    return "White" in repr(board.current_player)


def play_match(depth_a: int, depth_b: int, num_games: int, max_moves: int = 100) -> tuple:
    """Play depth_a vs depth_b. Returns (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0

    for game_idx in range(num_games):
        a_is_white = (game_idx % 2 == 0)
        board = Board()
        engine_a = Engine()
        engine_b = Engine()

        # Random opening (2 or 4 moves)
        for _ in range(random.choice([2, 4])):
            moves = board.generate_moves()
            if not moves or board.check_winner() is not None:
                break
            board.make_move(random.choice(moves))

        move_count = 0
        while board.check_winner() is None and move_count < max_moves:
            white_turn = is_white_turn(board)
            is_a_turn = (white_turn == a_is_white)
            depth = depth_a if is_a_turn else depth_b
            engine = engine_a if is_a_turn else engine_b

            sr = engine.search(board, depth)
            if sr.best_move is None:
                break
            board.make_move(sr.best_move)
            move_count += 1

        winner = board.check_winner()
        if winner is None:
            draws += 1
        elif ("White" in repr(winner)) == a_is_white:
            wins_a += 1
        else:
            wins_b += 1

    return wins_a, wins_b, draws


def elo_diff(wins, losses, draws):
    """Compute ELO difference from score percentage."""
    total = wins + losses + draws
    if total == 0:
        return 0.0
    score = (wins + 0.5 * draws) / total
    if score >= 1.0:
        return 400.0
    if score <= 0.0:
        return -400.0
    import math
    return 400.0 * math.log10(score / (1.0 - score))


def _match_worker(args):
    """Worker for parallel matchups."""
    d_a, d_b, num_games, seed = args
    random.seed(seed)
    t0 = time.time()
    w_a, w_b, draws = play_match(d_a, d_b, num_games)
    elapsed = time.time() - t0
    return d_a, d_b, w_a, w_b, draws, elapsed


def bench_speed(depths: list, num_games: int = 20):
    """Benchmark alphabeta self-play game speed at each depth."""
    print()
    print("=" * 70)
    print("  SPEED BENCHMARK (alphabeta self-play games)")
    print("=" * 70)
    print(f"  {'Depth':>5s}  {'Games':>5s}  {'Time(s)':>8s}  {'Games/s':>8s}  {'Examples':>8s}  {'Ex/game':>8s}  {'Slowdown':>8s}")
    print("  " + "-" * 64)

    base_time = None
    for depth in depths:
        engine = _RustMCTSEngine(200, 1.0)
        t0 = time.time()
        results = engine.play_alphabeta_games(num_games, depth=depth, random_opening=4, max_moves=80)
        elapsed = time.time() - t0

        total_examples = sum(len(r.examples) for r in results)
        games_per_sec = num_games / elapsed
        ex_per_game = total_examples / max(num_games, 1)

        if base_time is None:
            base_time = elapsed
        slowdown = elapsed / base_time

        print(f"  {depth:>5d}  {num_games:>5d}  {elapsed:>8.2f}  {games_per_sec:>8.1f}  "
              f"{total_examples:>8d}  {ex_per_game:>8.1f}  {slowdown:>7.1f}x")

    print()


def main():
    parser = argparse.ArgumentParser(description="Depth scaling benchmark")
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9],
                        help="Depths to test (default: 4 5 6 7 8 9)")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per matchup for ELO (default: 50)")
    parser.add_argument("--speed-games", type=int, default=20,
                        help="Games for speed benchmark (default: 20)")
    parser.add_argument("--speed-only", action="store_true",
                        help="Only run speed benchmark, skip ELO tournament")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for tournament (default: CPU count)")
    args = parser.parse_args()

    random.seed(args.seed)
    depths = sorted(args.depths)
    num_workers = args.workers or multiprocessing.cpu_count()

    # Speed benchmark (single-threaded — measures per-core performance)
    bench_speed(depths, num_games=args.speed_games)

    if args.speed_only:
        return

    # ELO tournament: all depth pairs, parallelized
    matchups = []
    for i in range(len(depths)):
        for j in range(i + 1, len(depths)):
            seed = args.seed + i * 100 + j
            matchups.append((depths[i], depths[j], args.games, seed))

    print("=" * 70)
    print("  DEPTH ELO TOURNAMENT (round-robin)")
    print("=" * 70)
    print(f"  Depths: {depths}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Matchups: {len(matchups)}, workers: {min(num_workers, len(matchups))}")
    print()

    t_start = time.time()
    pool_size = min(num_workers, len(matchups))
    if pool_size > 1:
        with multiprocessing.Pool(processes=pool_size) as pool:
            match_results = pool.map(_match_worker, matchups)
    else:
        match_results = [_match_worker(m) for m in matchups]

    results = {}
    for d_a, d_b, w_a, w_b, draws, elapsed in match_results:
        results[(d_a, d_b)] = (w_a, w_b, draws)
        total = w_a + w_b + draws
        score_a = (w_a + 0.5 * draws) / total * 100
        elo = elo_diff(w_a, w_b, draws)
        print(f"  Depth {d_a} vs {d_b}: D{d_a} {w_a}W-{draws}D-{w_b}L ({score_a:.0f}%) "
              f"ELO diff: {elo:+.0f}  [{elapsed:.0f}s]")

    print(f"\n  Tournament completed in {time.time() - t_start:.0f}s")

    # Summary table
    print()
    print("  SUMMARY")
    print(f"  {'Depth':>5s}  {'vs D-1':>12s}  {'ELO vs D-1':>10s}  {'vs D+1':>12s}  {'ELO vs D+1':>10s}")
    print("  " + "-" * 55)
    for d in depths:
        # vs depth-1
        lower = ""
        lower_elo = ""
        if d - 1 in depths:
            key = (d - 1, d)
            if key in results:
                w_low, w_high, dr = results[key]
                lower = f"{w_high}W-{dr}D-{w_low}L"
                lower_elo = f"{elo_diff(w_high, w_low, dr):+.0f}"

        # vs depth+1
        upper = ""
        upper_elo = ""
        if d + 1 in depths:
            key = (d, d + 1)
            if key in results:
                w_low, w_high, dr = results[key]
                upper = f"{w_low}W-{dr}D-{w_high}L"
                upper_elo = f"{elo_diff(w_low, w_high, dr):+.0f}"

        print(f"  {d:>5d}  {lower:>12s}  {lower_elo:>10s}  {upper:>12s}  {upper_elo:>10s}")

    # Compute cumulative ELO via chain
    print()
    print(f"  CUMULATIVE ELO (anchored at depth {depths[0]} = 0)")
    cum_elo = {depths[0]: 0.0}
    for i in range(1, len(depths)):
        key = (depths[i - 1], depths[i])
        if key in results:
            w_low, w_high, dr = results[key]
            gain = elo_diff(w_high, w_low, dr)
            cum_elo[depths[i]] = cum_elo[depths[i - 1]] + gain

    print(f"  {'Depth':>5s}  {'Cumulative ELO':>14s}  {'Marginal ELO':>12s}")
    print("  " + "-" * 35)
    prev = 0.0
    for d in depths:
        if d in cum_elo:
            marginal = cum_elo[d] - prev
            prev = cum_elo[d]
            marginal_str = f"{marginal:+.0f}" if d != depths[0] else "-"
            print(f"  {d:>5d}  {cum_elo[d]:>+14.0f}  {marginal_str:>12s}")

    print()


if __name__ == "__main__":
    main()
