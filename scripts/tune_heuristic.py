#!/usr/bin/env python3
"""
Heuristic tuning tournament for Tonnesjakk.

Runs a round-robin tournament between different heuristic weight configurations
to find the strongest evaluation parameters. Each configuration defines 5 tunable
weights: progress, center_pail, blocking, scored, and threat.

Usage:
    python scripts/tune_heuristic.py                        # full tournament, 100 games/pair, depth 7
    python scripts/tune_heuristic.py --games 50 --depth 5   # faster, shallower
    python scripts/tune_heuristic.py --time-ms 100          # time-based instead of depth
    python scripts/tune_heuristic.py --configs custom.json   # load configs from file
"""

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tonnesjakk import Board, Engine


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

CONFIGS: Dict[str, Dict[str, int]] = {
    "baseline":     {"progress": 100, "center_pail": 10, "blocking": 15, "scored": 500, "threat": 200},
    "aggressive":   {"progress": 170, "center_pail":  5, "blocking":  5, "scored": 400, "threat": 300},
    "defensive":    {"progress":  80, "center_pail": 25, "blocking": 35, "scored": 500, "threat": 150},
    "rusher":       {"progress": 200, "center_pail":  0, "blocking":  0, "scored": 350, "threat": 200},
    "consolidator": {"progress":  80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150},
    "tactical":     {"progress": 100, "center_pail":  5, "blocking": 10, "scored": 500, "threat": 350},
    "positional":   {"progress": 120, "center_pail": 20, "blocking": 20, "scored": 500, "threat": 150},
    "greedy":       {"progress": 100, "center_pail": 10, "blocking": 15, "scored": 800, "threat": 350},
    "minimalist":   {"progress": 150, "center_pail":  0, "blocking":  0, "scored": 400, "threat": 100},
    "allrounder":   {"progress": 130, "center_pail": 15, "blocking": 20, "scored": 550, "threat": 250},
    # New eval feature variants (based on consolidator weights)
    "cons_thr2_lo":  {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "threat2": 30},
    "cons_thr2_mid": {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "threat2": 60},
    "cons_thr2_hi":  {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "threat2": 100},
    "cons_ablk_lo":  {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "adj_blocking": 5},
    "cons_ablk_mid": {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "adj_blocking": 10},
    "cons_mob_lo":   {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "mobility": 5},
    "cons_mob_mid":  {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "mobility": 15},
    "cons_combo":    {"progress": 80, "center_pail": 15, "blocking": 20, "scored": 700, "threat": 150, "threat2": 50, "adj_blocking": 8, "mobility": 10},
}


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

def make_engine_with_weights(weights: Dict[str, int]) -> Engine:
    """Create an engine with custom heuristic weights."""
    engine = Engine()
    engine.weight_progress = weights["progress"]
    engine.weight_center_pail = weights["center_pail"]
    engine.weight_blocking = weights["blocking"]
    engine.weight_scored = weights["scored"]
    engine.weight_threat = weights["threat"]
    engine.weight_threat2 = weights.get("threat2", 0)
    engine.weight_adj_blocking = weights.get("adj_blocking", 0)
    engine.weight_mobility = weights.get("mobility", 0)
    engine.weight_passed = weights.get("passed", 0)
    engine.weight_trapped = weights.get("trapped", 0)
    engine.weight_score_accel = weights.get("score_accel", 0)
    engine.weight_eg_threat = weights.get("eg_threat", 0)
    engine.weight_jump = weights.get("jump", 0)
    return engine


def is_white_turn(board: Board) -> bool:
    return "White" in repr(board.current_player)


# ---------------------------------------------------------------------------
# Single matchup runner
# ---------------------------------------------------------------------------

def run_matchup(
    name_a: str,
    weights_a: Dict[str, int],
    name_b: str,
    weights_b: Dict[str, int],
    num_games: int,
    depth: Optional[int],
    time_ms: Optional[int],
    max_moves: int = 80,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Play a match between two heuristic configurations.

    Returns (wins_a, wins_b, draws).
    Colors alternate each game. Random openings (2-4 moves) for variety.
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    if seed is not None:
        random.seed(seed)

    for game_idx in range(num_games):
        a_is_white = (game_idx % 2 == 0)

        board = Board()
        engine_a = make_engine_with_weights(weights_a)
        engine_b = make_engine_with_weights(weights_b)

        # Random opening (2 or 4 random moves — must be even for fairness)
        opening_moves = random.choice([2, 4])
        for _ in range(opening_moves):
            moves = board.generate_moves()
            if not moves or board.check_winner() is not None:
                break
            board.make_move(random.choice(moves))

        move_count = 0
        while board.check_winner() is None and move_count < max_moves:
            white_turn = is_white_turn(board)
            is_a_turn = (white_turn == a_is_white)
            current_engine = engine_a if is_a_turn else engine_b

            if depth is not None:
                sr = current_engine.search(board, depth)
            else:
                sr = current_engine.search_timed(board, time_ms)

            if sr.best_move is None:
                break
            board.make_move(sr.best_move)
            move_count += 1

        winner = board.check_winner()
        if winner is None:
            draws += 1
            tag = "D"
        else:
            white_won = "White" in repr(winner)
            if white_won == a_is_white:
                wins_a += 1
                tag = "A"
            else:
                wins_b += 1
                tag = "B"

        if verbose:
            total = game_idx + 1
            print(
                f"    Game {total:>3}/{num_games}: {tag}  "
                f"({move_count:>2} moves)  "
                f"| {name_a}={wins_a} {name_b}={wins_b} D={draws}",
                flush=True,
            )

    return wins_a, wins_b, draws


# ---------------------------------------------------------------------------
# ELO computation — iterative maximum likelihood estimation
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    names: List[str],
    results: Dict[Tuple[str, str], Tuple[int, int, int]],
    anchor: str = "baseline",
    anchor_elo: float = 1500.0,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Compute ELO ratings from round-robin results using iterative MLE.

    The algorithm repeatedly adjusts each player's rating to make the observed
    results most likely given the current ratings of all opponents.

    Args:
        names: List of config names.
        results: Dict mapping (name_a, name_b) -> (wins_a, wins_b, draws).
        anchor: Config to anchor at anchor_elo.
        anchor_elo: ELO value for the anchor config.
        iterations: Number of optimization passes.

    Returns:
        Dict mapping config name -> ELO rating.
    """
    ratings = {name: anchor_elo for name in names}

    for _ in range(iterations):
        new_ratings = {}
        for player in names:
            # Collect total actual score and expected score vs all opponents
            actual_score = 0.0
            expected_score = 0.0
            total_games = 0

            for opponent in names:
                if player == opponent:
                    continue

                # Look up result from either direction
                if (player, opponent) in results:
                    w, l, d = results[(player, opponent)]
                elif (opponent, player) in results:
                    l, w, d = results[(opponent, player)]
                else:
                    continue

                n = w + l + d
                if n == 0:
                    continue

                actual_score += w + 0.5 * d
                total_games += n

                # Expected score based on current ratings
                diff = ratings[player] - ratings[opponent]
                exp_win_rate = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
                expected_score += exp_win_rate * n

            if total_games == 0:
                new_ratings[player] = ratings[player]
                continue

            # Adjust rating: if actual > expected, rating should increase
            # Use a damped update for stability
            if expected_score > 0:
                adjustment = 400.0 * math.log10(actual_score / expected_score)
                new_ratings[player] = ratings[player] + adjustment * 0.5
            else:
                new_ratings[player] = ratings[player]

        # Re-anchor
        if anchor in new_ratings:
            offset = anchor_elo - new_ratings[anchor]
            for name in new_ratings:
                new_ratings[name] += offset

        ratings = new_ratings

    return ratings


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def print_tournament_results(
    names: List[str],
    configs: Dict[str, Dict[str, int]],
    results: Dict[Tuple[str, str], Tuple[int, int, int]],
    ratings: Dict[str, float],
    wall_time: float,
) -> None:
    """Print a formatted tournament results table."""
    # Sort by ELO descending
    ranked = sorted(names, key=lambda n: ratings[n], reverse=True)

    print()
    print("=" * 100)
    print("  TOURNAMENT RESULTS")
    print("=" * 100)
    print()

    # ELO ranking table
    header = (
        f"  {'Rank':>4s}  {'Config':14s}  {'ELO':>6s}  {'W':>4s}  {'L':>4s}  {'D':>4s}  "
        f"{'Score%':>6s}  {'Prog':>4s}  {'Cntr':>4s}  {'Blck':>4s}  {'Scrd':>4s}  {'Thrt':>4s}  "
        f"{'Tht2':>4s}  {'ABlk':>4s}  {'Mob':>4s}  "
        f"{'Pass':>4s}  {'Trap':>4s}  {'SAcl':>4s}  {'EgTh':>4s}  {'Jump':>4s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rank, name in enumerate(ranked, 1):
        cfg = configs[name]
        total_w = 0
        total_l = 0
        total_d = 0
        for opp in names:
            if name == opp:
                continue
            if (name, opp) in results:
                w, l, d = results[(name, opp)]
                total_w += w
                total_l += l
                total_d += d
            elif (opp, name) in results:
                l2, w2, d2 = results[(opp, name)]
                total_w += w2
                total_l += l2
                total_d += d2

        total = total_w + total_l + total_d
        score_pct = (total_w + 0.5 * total_d) / max(total, 1) * 100

        print(
            f"  {rank:>4d}  {name:14s}  {ratings[name]:6.0f}  {total_w:4d}  {total_l:4d}  {total_d:4d}  "
            f"{score_pct:5.1f}%  {cfg['progress']:4d}  {cfg['center_pail']:4d}  {cfg['blocking']:4d}  "
            f"{cfg['scored']:4d}  {cfg['threat']:4d}  "
            f"{cfg.get('threat2', 0):4d}  {cfg.get('adj_blocking', 0):4d}  {cfg.get('mobility', 0):4d}  "
            f"{cfg.get('passed', 0):4d}  {cfg.get('trapped', 0):4d}  {cfg.get('score_accel', 0):4d}  {cfg.get('eg_threat', 0):4d}  {cfg.get('jump', 0):4d}"
        )

    print()

    # Head-to-head matrix
    print("  HEAD-TO-HEAD (rows = config A, columns = config B, cell = A's wins-draws-losses)")
    print()
    # Column headers
    col_width = max(len(n) for n in names) + 2
    header_row = " " * (col_width + 2)
    for name in ranked:
        header_row += f"{name:>{col_width}s}"
    print(header_row)

    for row_name in ranked:
        row = f"  {row_name:<{col_width}s}"
        for col_name in ranked:
            if row_name == col_name:
                cell = "---"
            elif (row_name, col_name) in results:
                w, l, d = results[(row_name, col_name)]
                cell = f"{w}-{d}-{l}"
            elif (col_name, row_name) in results:
                l, w, d = results[(col_name, row_name)]
                cell = f"{w}-{d}-{l}"
            else:
                cell = "n/a"
            row += f"{cell:>{col_width}s}"
        print(row)

    print()
    print(f"  Total wall time: {wall_time:.1f}s ({wall_time / 60:.1f} min)")
    print("=" * 100)
    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    names: List[str],
    configs: Dict[str, Dict[str, int]],
    results: Dict[Tuple[str, str], Tuple[int, int, int]],
    ratings: Dict[str, float],
    settings: Dict,
    output_path: Path,
) -> None:
    """Save tournament results to JSON."""
    ranked = sorted(names, key=lambda n: ratings[n], reverse=True)

    # Convert results dict (tuple keys) to serializable format
    matchups = []
    for (a, b), (w_a, w_b, d) in results.items():
        matchups.append({
            "config_a": a,
            "config_b": b,
            "wins_a": w_a,
            "wins_b": w_b,
            "draws": d,
        })

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings,
        "configs": configs,
        "rankings": [
            {"rank": i + 1, "name": name, "elo": round(ratings[name], 1), "params": configs[name]}
            for i, name in enumerate(ranked)
        ],
        "matchups": matchups,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Heuristic tuning tournament for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/tune_heuristic.py                        # 100 games/pair, depth 7
  python scripts/tune_heuristic.py --games 50 --depth 5   # faster, shallower
  python scripts/tune_heuristic.py --time-ms 100          # time-based
  python scripts/tune_heuristic.py --configs custom.json   # custom configs
""",
    )
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games per matchup (default: 100)")
    parser.add_argument("--depth", type=int, default=7,
                        help="Fixed search depth (default: 7). Ignored if --time-ms is set.")
    parser.add_argument("--time-ms", type=int, default=None,
                        help="Time per move in ms (overrides --depth)")
    parser.add_argument("--max-moves", type=int, default=100,
                        help="Max moves per game (default: 100)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Path to JSON file with custom configurations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON (default: scripts/tuning_results.json)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Only test each config against this baseline (skip round-robin)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel matchup workers (default: 1)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-game output")

    args = parser.parse_args()

    # Load configs
    if args.configs:
        config_path = Path(args.configs)
        if not config_path.exists():
            print(f"Error: config file not found: {args.configs}", file=sys.stderr)
            sys.exit(1)
        with open(config_path, "r") as f:
            configs = json.load(f)
        # Validate (threat2, adj_blocking, mobility are optional, default 0)
        for name, params in configs.items():
            required = {"progress", "center_pail", "blocking", "scored", "threat"}
            missing = required - set(params.keys())
            if missing:
                print(f"Error: config '{name}' missing keys: {missing}", file=sys.stderr)
                sys.exit(1)
    else:
        configs = CONFIGS

    names = list(configs.keys())
    n = len(names)

    # Build matchup pairs
    if args.baseline:
        if args.baseline not in configs:
            print(f"Error: baseline '{args.baseline}' not found in configs", file=sys.stderr)
            sys.exit(1)
        matchup_pairs = [
            (args.baseline, name)
            for name in names if name != args.baseline
        ]
    else:
        matchup_pairs = [
            (names[i], names[j])
            for i in range(n) for j in range(i + 1, n)
        ]

    total_matchups = len(matchup_pairs)

    # Determine search mode
    use_time = args.time_ms is not None
    search_desc = f"{args.time_ms}ms/move" if use_time else f"depth {args.depth}"

    print()
    print("=" * 70)
    print("  TONNESJAKK HEURISTIC TUNING TOURNAMENT")
    print("=" * 70)
    print(f"  Configs:  {n}")
    if args.baseline:
        print(f"  Mode:     baseline ({args.baseline}) vs each challenger")
    print(f"  Matchups: {total_matchups}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Search: {search_desc}")
    print(f"  Total games: {total_matchups * args.games}")
    print("=" * 70)
    print()

    if args.seed is not None:
        random.seed(args.seed)

    # Run matchups
    results: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    tournament_start = time.time()

    if args.workers > 1:
        # Parallel execution
        print(f"  Running {total_matchups} matchups with {args.workers} workers...\n")
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for name_a, name_b in matchup_pairs:
                future = executor.submit(
                    run_matchup,
                    name_a=name_a,
                    weights_a=configs[name_a],
                    name_b=name_b,
                    weights_b=configs[name_b],
                    num_games=args.games,
                    depth=args.depth if not use_time else None,
                    time_ms=args.time_ms if use_time else None,
                    max_moves=args.max_moves,
                    verbose=False,
                )
                futures[future] = (name_a, name_b)

            done_count = 0
            for future in as_completed(futures):
                name_a, name_b = futures[future]
                wins_a, wins_b, draws = future.result()
                results[(name_a, name_b)] = (wins_a, wins_b, draws)
                done_count += 1
                total = wins_a + wins_b + draws
                print(
                    f"  [{done_count}/{total_matchups}] {name_a} vs {name_b}: "
                    f"+{wins_a} ={draws} -{wins_b}  "
                    f"({(wins_a + 0.5 * draws) / max(total, 1):.1%})"
                )
    else:
        # Sequential execution
        matchup_num = 0
        for name_a, name_b in matchup_pairs:
            matchup_num += 1
            print(f"  Matchup {matchup_num}/{total_matchups}: {name_a} vs {name_b}")

            wins_a, wins_b, draws = run_matchup(
                name_a=name_a,
                weights_a=configs[name_a],
                name_b=name_b,
                weights_b=configs[name_b],
                num_games=args.games,
                depth=args.depth if not use_time else None,
                time_ms=args.time_ms if use_time else None,
                max_moves=args.max_moves,
                verbose=not args.quiet,
            )

            results[(name_a, name_b)] = (wins_a, wins_b, draws)

            total = wins_a + wins_b + draws
            print(
                f"  => {name_a} +{wins_a} ={draws} -{wins_b}  "
                f"({(wins_a + 0.5 * draws) / max(total, 1):.1%})"
            )
            print()

    wall_time = time.time() - tournament_start

    # Compute ELO ratings
    anchor = "baseline" if "baseline" in names else names[0]
    ratings = compute_elo_ratings(names, results, anchor=anchor)

    # Print results
    print_tournament_results(names, configs, results, ratings, wall_time)

    # Save results
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "tuning_results.json"
    )
    settings = {
        "games_per_matchup": args.games,
        "search": search_desc,
        "depth": args.depth if not use_time else None,
        "time_ms": args.time_ms,
        "max_moves": args.max_moves,
        "seed": args.seed,
    }
    save_results(names, configs, results, ratings, settings, output_path)

    # Print best config for easy copy-paste
    best = max(names, key=lambda n: ratings[n])
    print(f"\n  Best config: {best} (ELO {ratings[best]:.0f})")
    print(f"  Parameters: {json.dumps(configs[best])}")
    print()


if __name__ == "__main__":
    main()
