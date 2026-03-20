#!/usr/bin/env python3
"""Round-robin tournament between heuristic engine at different depths.

Tests whether deeper search actually produces stronger play.
Each matchup plays N games with alternating colors and random openings.

Usage:
    python scripts/depth_tournament.py [--games 50] [--depths 4 6 8 10 12]
"""

import argparse
import random
import time
from itertools import combinations


def play_match(depth_a, depth_b, num_games, verbose=True):
    from tonnesjakk import Board, Engine

    engine_a = Engine()
    engine_b = Engine()

    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(num_games):
        white_is_a = (game_idx % 2 == 0)

        board = Board()
        engine_a.full_reset()
        engine_b.full_reset()

        # Random opening (6 random moves for diversity)
        for _ in range(random.randint(4, 8)):
            moves = board.generate_moves()
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        # Play game
        game_start = time.time()
        move_count = 0
        while board.check_winner() is None and move_count < 80:
            if time.time() - game_start > 120.0:
                break

            is_white_turn = "White" in repr(board.current_player)
            if is_white_turn == white_is_a:
                result = engine_a.search(board, depth_a)
            else:
                result = engine_b.search(board, depth_b)

            if result.best_move is None:
                break
            board.make_move(result.best_move)
            move_count += 1

        winner = board.check_winner()
        if winner is None:
            draws += 1
        elif ("White" in str(winner)) == white_is_a:
            wins_a += 1
        else:
            wins_b += 1

        if verbose and (game_idx + 1) % 10 == 0:
            total = wins_a + wins_b + draws
            print(f"    Game {game_idx+1}/{num_games}: d{depth_a} {wins_a}-{wins_b}-{draws} d{depth_b} "
                  f"({time.time() - game_start:.1f}s/game)", flush=True)

    return wins_a, wins_b, draws


def main():
    parser = argparse.ArgumentParser(description="Depth strength tournament")
    parser.add_argument("--games", type=int, default=50, help="Games per matchup (default: 50)")
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 10, 12],
                        help="Depths to test (default: 4 6 8 10 12)")
    args = parser.parse_args()

    depths = sorted(args.depths)
    matchups = list(combinations(depths, 2))

    print(f"Depth Strength Tournament")
    print(f"  Depths: {depths}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Matchups: {len(matchups)}")
    print(f"  Total games: {len(matchups) * args.games}")
    print(flush=True)

    results = {}
    # Track points: win=1, draw=0.5, loss=0
    points = {d: 0.0 for d in depths}
    total_games = {d: 0 for d in depths}

    for d_low, d_high in matchups:
        print(f"\n--- Depth {d_low} vs Depth {d_high} ({args.games} games) ---", flush=True)
        t0 = time.time()
        wins_low, wins_high, draws = play_match(d_low, d_high, args.games)
        elapsed = time.time() - t0

        wr_high = (wins_high + 0.5 * draws) / (wins_low + wins_high + draws) * 100
        results[(d_low, d_high)] = (wins_low, wins_high, draws)

        points[d_low] += wins_low + 0.5 * draws
        points[d_high] += wins_high + 0.5 * draws
        total_games[d_low] += args.games
        total_games[d_high] += args.games

        print(f"  Result: d{d_high} wins {wins_high}, d{d_low} wins {wins_low}, draws {draws} "
              f"(d{d_high} score: {wr_high:.0f}%) [{elapsed:.0f}s]", flush=True)

    # Summary table
    print(f"\n{'=' * 60}")
    print("RESULTS MATRIX (row vs column, row's score)")
    print(f"{'=' * 60}")

    header = f"{'':>8}" + "".join(f"{'d'+str(d):>10}" for d in depths)
    print(header)
    for d_row in depths:
        row = f"{'d'+str(d_row):>8}"
        for d_col in depths:
            if d_row == d_col:
                row += f"{'---':>10}"
            elif (d_row, d_col) in results:
                w, l, d = results[(d_row, d_col)]
                row += f"{f'{w}-{l}-{d}':>10}"
            elif (d_col, d_row) in results:
                l, w, d = results[(d_col, d_row)]
                row += f"{f'{w}-{l}-{d}':>10}"
        print(row)

    # Standings
    print(f"\n{'=' * 60}")
    print("STANDINGS")
    print(f"{'=' * 60}")
    print(f"{'Depth':>8} {'Points':>10} {'Games':>8} {'Score%':>10}")
    for d in sorted(depths, key=lambda x: -points[x]):
        pct = points[d] / total_games[d] * 100 if total_games[d] > 0 else 0
        print(f"{'d'+str(d):>8} {points[d]:>10.1f} {total_games[d]:>8} {pct:>9.1f}%")

    print(flush=True)


if __name__ == "__main__":
    main()
