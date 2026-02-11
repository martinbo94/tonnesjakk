"""
Engine benchmark script for testing search improvements.

Generates deterministic benchmark positions (fixed seed), then searches each at
a fixed depth. Reports total nodes searched and time taken.
Fewer nodes at same depth = better pruning.

Usage:
    python scripts/bench_engine.py                    # Run benchmark (depth 8)
    python scripts/bench_engine.py --depth 7          # Custom depth
    python scripts/bench_engine.py --compare-depths   # Depth 6 vs 8 match (50 games)
    python scripts/bench_engine.py --games 100        # More games for depth comparison
"""

import sys
import os
import time
import json
import random
import argparse

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from tonnesjakk._core import Board, Engine


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_SEED = 42  # Fixed seed for reproducible positions


def generate_positions(num_positions=30, random_opening_moves=6, engine_moves=10):
    """Generate benchmark positions deterministically (fixed seed).

    Uses random openings followed by engine play to reach interesting mid-game
    positions. Same seed = same positions every time, even across engine versions
    (opening moves are random, not engine-dependent).
    """
    rng = random.Random(BENCH_SEED)
    positions = []
    engine = Engine()

    for i in range(num_positions + 10):  # Generate extras in case some end early
        board = Board()

        # Random opening moves (deterministic via fixed seed)
        for _ in range(random_opening_moves):
            moves = board.generate_moves()
            if not moves or board.check_winner() is not None:
                break
            mv = moves[rng.randint(0, len(moves) - 1)]
            board.make_move(mv)

        # Play a few engine moves to reach interesting positions
        for _ in range(engine_moves):
            if board.check_winner() is not None:
                break
            result = engine.search(board, 4)
            if result.best_move is None:
                break
            board.make_move(result.best_move)

        if board.check_winner() is None:
            positions.append(board)
            if len(positions) >= num_positions:
                break

    return positions


def run_bench(depth=8, num_positions=30):
    """Generate positions and search each at given depth.
    Returns (total_nodes, total_time, nps, positions_searched)."""
    positions = generate_positions(num_positions)
    engine = Engine()
    total_nodes = 0
    total_time = 0.0

    print(f"Searching {len(positions)} positions @ depth {depth}...")

    for i, board in enumerate(positions):
        start = time.perf_counter()
        result = engine.search_iterative(board, depth)
        elapsed = time.perf_counter() - start

        total_nodes += result.nodes_searched
        total_time += elapsed

        if (i + 1) % 10 == 0:
            print(f"  Position {i+1}/{len(positions)}: {result.nodes_searched:,} nodes, {elapsed:.2f}s")

    nps = int(total_nodes / total_time) if total_time > 0 else 0
    return total_nodes, total_time, nps, len(positions)


def compare_depths(depth_a=6, depth_b=8, num_games=50, random_moves=4):
    """Play depth_a vs depth_b to measure search quality.
    Uses a fixed seed for reproducibility."""
    rng = random.Random(BENCH_SEED + 1000)  # Different seed from bench positions
    engine = Engine()
    wins_a, wins_b, draws = 0, 0, 0

    for game in range(num_games):
        board = Board()

        # Random opening (deterministic)
        for _ in range(random_moves):
            moves = board.generate_moves()
            if not moves:
                break
            mv = moves[rng.randint(0, len(moves) - 1)]
            board.make_move(mv)

        # Alternate which depth plays white
        a_is_white = (game % 2 == 0)
        white_depth = depth_a if a_is_white else depth_b
        black_depth = depth_b if a_is_white else depth_a

        move_count = 0
        while board.check_winner() is None and move_count < 100:
            depth = white_depth if board.current_player == 1 else black_depth
            result = engine.search(board, depth)
            if result.best_move is None:
                break
            board.make_move(result.best_move)
            move_count += 1

        winner = board.check_winner()
        if winner is None or winner == 0:
            draws += 1
        elif (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
            wins_a += 1
        else:
            wins_b += 1

        if (game + 1) % 10 == 0:
            print(f"  Game {game+1}/{num_games}: D{depth_a} {wins_a}-{wins_b}-{draws} D{depth_b}")

    return wins_a, wins_b, draws


def main():
    parser = argparse.ArgumentParser(description='Engine benchmark')
    parser.add_argument('--depth', type=int, default=8, help='Search depth for bench')
    parser.add_argument('--positions', type=int, default=30, help='Number of positions')
    parser.add_argument('--compare-depths', action='store_true', help='Run depth 6 vs 8 match')
    parser.add_argument('--games', type=int, default=50, help='Games for depth comparison')
    args = parser.parse_args()

    if args.compare_depths:
        print(f"=== Depth comparison: D6 vs D8, {args.games} games ===")
        wa, wb, d = compare_depths(6, 8, args.games)
        total = wa + wb + d
        print(f"\nResult: D6 {wa}/{total} ({100*wa/total:.0f}%) - D8 {wb}/{total} ({100*wb/total:.0f}%) - Draws {d}")
        return

    print(f"=== Engine Benchmark @ depth {args.depth} ===")
    total_nodes, total_time, nps, num_pos = run_bench(args.depth, args.positions)

    print(f"\n{'='*50}")
    print(f"Total nodes:  {total_nodes:>15,}")
    print(f"Total time:   {total_time:>15.2f}s")
    print(f"Avg nodes/pos:{total_nodes // num_pos:>15,}")
    print(f"NPS:          {nps:>15,}")
    print(f"{'='*50}")

    # Save result for comparison
    branch = os.popen('git branch --show-current').read().strip()
    result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'branch': branch,
        'depth': args.depth,
        'positions': num_pos,
        'total_nodes': total_nodes,
        'total_time_s': round(total_time, 3),
        'nps': nps,
        'avg_nodes_per_pos': total_nodes // num_pos,
    }

    # Append to history
    history_file = os.path.join(SCRIPTS_DIR, 'bench_history.json')
    history = []
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
    history.append(result)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nResult saved ({branch})")
    if len(history) > 1:
        prev = history[-2]
        if prev.get('total_nodes'):
            node_diff = total_nodes - prev['total_nodes']
            pct = 100 * node_diff / prev['total_nodes']
            print(f"vs previous ({prev['branch']}): {node_diff:+,} nodes ({pct:+.1f}%)")


if __name__ == '__main__':
    main()
