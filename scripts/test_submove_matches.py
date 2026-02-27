"""
Test the sub-move implementation by playing self-play matches at various depths.
Verifies the engine still works correctly end-to-end after the pail/barrel split.

Usage:
    python scripts/test_submove_matches.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from tonnesjakk._core import Board, Engine, Player

DRAW_LIMIT = 100


def play_match(engine, depth, rng, game_num):
    board = Board()
    move_count = 0
    random_moves = 4

    # Random opening
    for _ in range(random_moves):
        moves = board.generate_moves()
        if not moves or board.check_winner() is not None:
            break
        mv = moves[rng.randint(0, len(moves) - 1)]
        board.make_move(mv)
        move_count += 1

    # Engine play
    while board.check_winner() is None and move_count < DRAW_LIMIT:
        moves = board.generate_moves()
        if not moves:
            break
        result = engine.search(board, depth)
        if result.best_move is None:
            break
        board.make_move(result.best_move)
        move_count += 1

    winner = board.check_winner()
    return board, move_count, winner


def winner_str(winner):
    if winner is None:
        return None
    return str(winner)


def main():
    engine = Engine()
    white_str = str(Player.White)
    black_str = str(Player.Black)

    for depth in [7, 8, 9]:
        rng = random.Random(42)
        w, d, b_ = 0, 0, 0
        total_time = 0.0
        total_moves = 0

        print(f"=== Depth {depth}: 100 matches (draw at {DRAW_LIMIT} moves) ===")

        for i in range(100):
            t0 = time.perf_counter()
            board, mc, winner = play_match(engine, depth, rng, i)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            total_moves += mc

            ws = winner_str(winner)
            if ws == white_str:
                w += 1
                result_char = "W"
            elif ws == black_str:
                b_ += 1
                result_char = "B"
            else:
                d += 1
                result_char = "D"

            print(f"  Game {i+1:3d}/100: {result_char}  {mc:3d} moves  {elapsed:.2f}s  "
                  f"(W:{board.white_scored} B:{board.black_scored})  "
                  f"Running: W:{w} D:{d} B:{b_}")

        avg_match = total_time / 100
        avg_move = total_time / total_moves * 1000 if total_moves else 0
        print(f"\nDepth {depth} summary:")
        print(f"  Score:      W:{w}  D:{d}  B:{b_}")
        print(f"  Time/match: {avg_match:.3f}s")
        print(f"  Time/move:  {avg_move:.1f}ms")
        print(f"  Total time: {total_time:.1f}s")
        print()


if __name__ == "__main__":
    main()
