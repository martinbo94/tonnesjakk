#!/usr/bin/env python3
"""
Diagnostic tool: play detailed games between WDL NNUE and heuristic,
capturing per-move evaluations, board states, and score trajectories
to identify where/why the NNUE model fails.
"""

import sys
import os
import random
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tonnesjakk import Board, Engine


def board_display(board):
    """Get a compact board display string."""
    arr = board.to_array()
    symbols = {0: '.', 1: 'W', -1: 'B', 2: 'w', -2: 'b'}
    lines = []
    for r, row in enumerate(arr):
        line = ' '.join(symbols.get(v, '?') for v in row)
        lines.append(f"  {5-r} | {line}")
    lines.append("    +-----------")
    lines.append("      0 1 2 3 4 5")
    return '\n'.join(lines)


def score_display(score):
    """Format a score for display."""
    if abs(score) > 90000:
        return f"{'+'if score>0 else ''}M{abs(score)//1000}"
    return f"{score:+d}"


def play_diagnostic_game(nnue_path, time_ms=200, game_num=1, seed=None):
    """
    Play one game with full per-move diagnostics.
    Both engines evaluate every position so we can compare.
    """
    if seed is not None:
        random.seed(seed + game_num)

    engine_nnue = Engine()
    engine_nnue.load_nnue(nnue_path)

    engine_heur = Engine()  # heuristic only

    # Alternate colors
    nnue_is_white = (game_num % 2 == 0)
    color_str = "White" if nnue_is_white else "Black"

    board = Board()
    engine_nnue.full_reset()
    engine_heur.full_reset()

    # Random opening (2-4 moves)
    opening_moves = random.randint(2, 4)
    opening_desc = []
    for _ in range(opening_moves):
        moves = board.generate_moves()
        if not moves or board.check_winner() is not None:
            break
        mv = random.choice(moves)
        opening_desc.append(repr(mv))
        board.make_move(mv)

    print(f"\n{'='*72}")
    print(f"DIAGNOSTIC GAME {game_num}  |  NNUE plays {color_str}")
    print(f"Opening ({len(opening_desc)} random moves): {', '.join(opening_desc)}")
    print(f"{'='*72}")
    print()
    print("Starting position:")
    print(board_display(board))
    print(f"  Score: W={board.white_scored} B={board.black_scored}")
    print()

    # Column headers
    print(f"{'Move':>4} {'Turn':>5} {'Who':>5} | {'Move':>12} | "
          f"{'NNUE':>7} {'Heur':>7} {'Diff':>7} | "
          f"{'Depth':>5} {'Nodes':>8} | {'Score':>5}")
    print('-' * 85)

    move_log = []
    move_count = 0
    max_moves = 80
    game_start = time.time()

    while board.check_winner() is None and move_count < max_moves:
        if time.time() - game_start > 120.0:
            print("  [TIMEOUT]")
            break

        white_turn = "White" in repr(board.current_player)
        is_nnue_turn = (white_turn == nnue_is_white)
        who = "NNUE" if is_nnue_turn else "HEUR"
        turn_str = "W" if white_turn else "B"

        # Both engines search to get their evaluations
        if is_nnue_turn:
            sr = engine_nnue.search_timed(board, time_ms)
            # Also get heuristic eval of same position for comparison
            sr_compare = engine_heur.search_timed(board, min(time_ms, 100))
        else:
            sr = engine_heur.search_timed(board, time_ms)
            # Also get NNUE eval of same position for comparison
            sr_compare = engine_nnue.search_timed(board, min(time_ms, 100))

        if sr.best_move is None:
            print("  [NO MOVES]")
            break

        # Get both evals (from the perspective of the side to move)
        if is_nnue_turn:
            nnue_score = sr.score
            heur_score = sr_compare.score
            played_depth = sr.depth
            played_nodes = sr.nodes_searched
        else:
            heur_score = sr.score
            nnue_score = sr_compare.score
            played_depth = sr.depth
            played_nodes = sr.nodes_searched

        diff = nnue_score - heur_score
        move_repr = repr(sr.best_move)

        # Flag large disagreements
        flag = ""
        if abs(diff) > 200:
            flag = " ***"
        elif abs(diff) > 100:
            flag = " **"
        elif abs(diff) > 50:
            flag = " *"

        move_entry = {
            'move_num': move_count + 1,
            'turn': turn_str,
            'who': who,
            'move': move_repr,
            'nnue_score': nnue_score,
            'heur_score': heur_score,
            'diff': diff,
            'depth': played_depth,
            'nodes': played_nodes,
            'white_scored': board.white_scored,
            'black_scored': board.black_scored,
        }
        move_log.append(move_entry)

        print(f"{move_count+1:>4} {turn_str:>5} {who:>5} | {move_repr:>12} | "
              f"{score_display(nnue_score):>7} {score_display(heur_score):>7} {diff:>+7d} | "
              f"{played_depth:>5} {played_nodes:>8,d} | "
              f"W{board.white_scored}-B{board.black_scored}{flag}")

        board.make_move(sr.best_move)
        move_count += 1

    # Game result
    winner = board.check_winner()
    game_time = time.time() - game_start
    print()
    print("Final position:")
    print(board_display(board))
    print(f"  Score: W={board.white_scored} B={board.black_scored}")

    if winner is None:
        result = "DRAW"
    elif "White" in repr(winner):
        result = "WHITE WINS"
    else:
        result = "BLACK WINS"

    nnue_won = False
    if winner is not None:
        white_won = "White" in repr(winner)
        nnue_won = (white_won == nnue_is_white)

    nnue_result = "WIN" if nnue_won else ("LOSS" if winner else "DRAW")

    print(f"\n  Result: {result} (NNUE {nnue_result}) in {move_count} moves ({game_time:.1f}s)")

    # Analysis
    print(f"\n--- Analysis ---")

    # Score trajectory summary
    nnue_scores = [m['nnue_score'] for m in move_log]
    heur_scores = [m['heur_score'] for m in move_log]
    diffs = [m['diff'] for m in move_log]

    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs, key=abs)
        big_disagreements = [(i+1, d) for i, d in enumerate(diffs) if abs(d) > 100]

        print(f"  Avg eval difference (NNUE - Heur): {avg_diff:+.0f} cp")
        print(f"  Max absolute difference: {max_diff:+d} cp")
        print(f"  Large disagreements (>100cp): {len(big_disagreements)}/{len(diffs)} moves")

        if big_disagreements:
            print(f"  Biggest disagreements at moves: {', '.join(f'#{m}({d:+d})' for m,d in sorted(big_disagreements, key=lambda x: -abs(x[1]))[:5])}")

    # Check for score collapse patterns
    nnue_moves = [m for m in move_log if m['who'] == 'NNUE']
    heur_moves = [m for m in move_log if m['who'] == 'HEUR']

    if nnue_moves:
        first_half = nnue_moves[:len(nnue_moves)//2]
        second_half = nnue_moves[len(nnue_moves)//2:]

        if first_half and second_half:
            avg_early = sum(m['nnue_score'] for m in first_half) / len(first_half)
            avg_late = sum(m['nnue_score'] for m in second_half) / len(second_half)
            print(f"  NNUE eval trend: early avg={avg_early:+.0f}, late avg={avg_late:+.0f}")

            avg_early_diff = sum(m['diff'] for m in first_half) / len(first_half)
            avg_late_diff = sum(m['diff'] for m in second_half) / len(second_half)
            print(f"  Eval disagreement trend: early={avg_early_diff:+.0f}, late={avg_late_diff:+.0f}")

    # When did scoring events happen?
    scoring_events = []
    for i, m in enumerate(move_log):
        if i > 0:
            prev = move_log[i-1]
            if m['white_scored'] > prev['white_scored']:
                scoring_events.append(f"  Move {m['move_num']}: White scores (now W{m['white_scored']}-B{m['black_scored']})")
            if m['black_scored'] > prev['black_scored']:
                scoring_events.append(f"  Move {m['move_num']}: Black scores (now W{m['white_scored']}-B{m['black_scored']})")

    if scoring_events:
        print(f"\n  Scoring timeline:")
        for ev in scoring_events:
            print(ev)

    return move_log, nnue_result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose WDL NNUE model weaknesses")
    parser.add_argument("model", help="Path to NNUE weights JSON")
    parser.add_argument("--games", type=int, default=3, help="Number of games (default: 3)")
    parser.add_argument("--time-ms", type=int, default=200, help="Time per move in ms (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"NNUE Diagnostic Tool")
    print(f"Model: {args.model}")
    print(f"Time control: {args.time_ms}ms/move")
    print(f"Games: {args.games}")
    print(f"Seed: {args.seed}")

    results = []
    all_logs = []

    for g in range(args.games):
        log, result = play_diagnostic_game(
            args.model,
            time_ms=args.time_ms,
            game_num=g,
            seed=args.seed
        )
        results.append(result)
        all_logs.append(log)

    # Summary
    print(f"\n{'='*72}")
    print(f"OVERALL SUMMARY ({args.games} games)")
    print(f"{'='*72}")
    wins = results.count("WIN")
    losses = results.count("LOSS")
    draws = results.count("DRAW")
    print(f"  NNUE: {wins}W {losses}L {draws}D")

    # Aggregate analysis
    all_diffs = []
    all_early_diffs = []
    all_late_diffs = []

    for log in all_logs:
        for m in log:
            all_diffs.append(m['diff'])
        half = len(log) // 2
        for m in log[:half]:
            all_early_diffs.append(m['diff'])
        for m in log[half:]:
            all_late_diffs.append(m['diff'])

    if all_diffs:
        print(f"\n  Overall eval disagreement (NNUE - Heur):")
        print(f"    Mean: {sum(all_diffs)/len(all_diffs):+.1f} cp")
        print(f"    Early game mean: {sum(all_early_diffs)/max(len(all_early_diffs),1):+.1f} cp")
        print(f"    Late game mean: {sum(all_late_diffs)/max(len(all_late_diffs),1):+.1f} cp")

        # Distribution of disagreements
        small = sum(1 for d in all_diffs if abs(d) < 50)
        medium = sum(1 for d in all_diffs if 50 <= abs(d) < 200)
        large = sum(1 for d in all_diffs if abs(d) >= 200)
        total = len(all_diffs)
        print(f"\n  Disagreement distribution:")
        print(f"    Small (<50cp):  {small:>4} ({100*small/total:.0f}%)")
        print(f"    Medium (50-200): {medium:>4} ({100*medium/total:.0f}%)")
        print(f"    Large (>200cp): {large:>4} ({100*large/total:.0f}%)")

        # Bias: does NNUE systematically over or under-evaluate?
        positive = sum(1 for d in all_diffs if d > 20)
        negative = sum(1 for d in all_diffs if d < -20)
        print(f"\n  Bias direction:")
        print(f"    NNUE higher than heur: {positive} times")
        print(f"    NNUE lower than heur:  {negative} times")


if __name__ == "__main__":
    main()
