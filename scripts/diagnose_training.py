"""
Diagnostic: Why can't AlphaZero beat depth-4 heuristic?

Compares two models (e.g. v16 vs v9) side-by-side:
  - Module 1: Eval games with per-move annotations (barrel advancement, repetitions, scoring)
  - Module 2: Replay buffer quality (value distribution, policy accuracy, calibration)
  - Module 3: Position deep dive on critical turning points from losses

Usage:
  # Compare v16 vs v9
  python scripts/diagnose_training.py \\
    alphazero_v16/latest_model.pt alphazero_v9/latest_model.pt \\
    --labels v16 v9 --hidden 128 64 --simulations 100 200

  # Single model
  python scripts/diagnose_training.py alphazero_v16/latest_model.pt \\
    --hidden 128 --simulations 100
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from tonnesjakk import Board
from tonnesjakk._core import MCTSEngine as _RustMCTSEngine
from tonnesjakk.alphazero import (
    BOARD_PLANES,
    BOARD_SIZE,
    POLICY_SIZE,
    make_network,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_move(m):
    if m.is_barrel_placement:
        to = m.barrel_to
        return f"place->({to.row},{to.col})"
    frm = m.barrel_from
    to = m.barrel_to
    if frm is None:
        # Pail-only move
        return f"pail->({to.row},{to.col})"
    return f"({frm.row},{frm.col})->({to.row},{to.col})"


def load_model(checkpoint_path, hidden=64, num_blocks=5):
    net = make_network("resnet", hidden=hidden, num_blocks=num_blocks)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


def make_batch_eval_fn(net, batch_size=8):
    plane_size = BOARD_PLANES * BOARD_SIZE * BOARD_SIZE
    input_buffer = torch.zeros(batch_size, plane_size, dtype=torch.float32)

    def batch_eval_fn(batch_planes):
        n = len(batch_planes)
        np_batch = np.array(batch_planes, dtype=np.float32)
        cpu_tensor = torch.as_tensor(np_batch)
        buf = input_buffer[:n]
        buf.copy_(cpu_tensor)
        with torch.no_grad():
            policy_logits, values = net(buf)
        return policy_logits.numpy().tolist(), values.numpy().tolist()

    return batch_eval_fn


def net_eval(net, board, engine):
    """Raw network policy + value (no MCTS)."""
    planes = engine.board_planes(board)
    planes_t = torch.tensor(planes, dtype=torch.float32).reshape(
        1, BOARD_PLANES, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        logits, value = net(planes_t)
    probs = logits.softmax(1)[0].cpu()
    return probs, value.item()


def barrel_advancement(move, is_white):
    """Rows advanced toward goal. Positive = forward."""
    if move.is_barrel_placement or move.barrel_from is None:
        return 0
    frm = move.barrel_from
    to = move.barrel_to
    if is_white:
        return frm.row - to.row  # White goes toward row 0
    else:
        return to.row - frm.row  # Black goes toward row 5


def entropy(p):
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Module 1: Eval Games
# ---------------------------------------------------------------------------

def _handle_pail_submove(board):
    """If the board is in pail-placement phase, pick a random center-biased pail move.

    This matches the Rust game loop (play_eval_match_impl) which handles pail
    sub-moves separately from barrel moves.  Returns the pail move made, or None
    if the current legal moves are not pail-only.
    """
    moves = board.generate_moves()
    if not moves or not moves[0].is_pail_only:
        return None
    # Center-biased weighting (matching Rust random_center_pail)
    weights = []
    for m in moves:
        to = m.barrel_to  # pail target stored here for pail-only moves
        dist = abs(to.row - 2.5) + abs(to.col - 2.5)
        w = max(6.0 - dist, 0.5)
        weights.append(w * w)
    weights = np.array(weights)
    weights /= weights.sum()
    idx = np.random.choice(len(moves), p=weights)
    board.make_move(moves[idx])
    return moves[idx]


def play_eval_games(net, simulations, depth, num_games, use_gumbel, label,
                    batch_size=16):
    """Play games vs heuristic, recording per-move data.

    Pail sub-moves are handled with center-biased random (matching Rust
    play_eval_match_impl).  Only barrel moves go through MCTS / heuristic
    search.  A fresh MCTS engine is created per search to avoid stale tree
    reuse across sub-moves.
    """
    eval_fn = make_batch_eval_fn(net, batch_size=batch_size)

    games = []
    for game_idx in range(num_games):
        net_is_white = (game_idx % 2 == 0)
        board = Board()
        move_log = []
        hash_history = []

        for move_num in range(80):
            winner = board.check_winner()
            if winner is not None:
                break

            # Handle pail sub-move first (if applicable)
            _handle_pail_submove(board)

            moves = board.generate_moves()
            if not moves:
                break

            is_white = str(board.current_player) == "Player.White"
            is_net_turn = (is_white == net_is_white)
            pos_hash = board.get_hash()
            hash_history.append(pos_hash)

            # Check repetition count
            rep_count = hash_history.count(pos_hash)

            entry = {
                "move_num": move_num,
                "is_white": is_white,
                "is_net_turn": is_net_turn,
                "white_scored": board.white_scored,
                "black_scored": board.black_scored,
                "pos_hash": pos_hash,
                "repetition": rep_count,
            }

            if is_net_turn:
                # Network + MCTS — fresh engine to avoid stale tree reuse
                engine = _RustMCTSEngine(simulations, 1.0, use_gumbel=use_gumbel,
                                         forward_only=False)
                net_probs, raw_value = net_eval(net, board, engine)
                result = engine.search_network_batched(
                    board, eval_fn, batch_size=batch_size)
                mcts_policy = np.array(result.policy_target, dtype=np.float32)
                best_move = result.best_move

                # Top-5 moves
                top5 = []
                for m in moves:
                    idx = m.policy_index()
                    mp = mcts_policy[idx] if idx < len(mcts_policy) else 0
                    np_ = net_probs[idx].item() if idx < len(net_probs) else 0
                    top5.append((m, mp, np_))
                top5.sort(key=lambda x: -x[1])
                top5 = top5[:5]

                entry["net_value"] = raw_value
                entry["mcts_value"] = result.root_value
                entry["top5"] = [(
                    _fmt_move(m), round(mp, 4), round(np_, 4)
                ) for m, mp, np_ in top5]
            else:
                # Heuristic — fresh engine to avoid stale tree reuse
                heuristic_engine = _RustMCTSEngine(200, 1.0)
                result = heuristic_engine.search_heuristic(board)
                best_move = result.best_move
                entry["heuristic_value"] = result.root_value

            adv = barrel_advancement(best_move, is_white)
            entry["move_str"] = _fmt_move(best_move)
            entry["advancement"] = adv

            move_log.append(entry)
            board.make_move(best_move)

        winner = board.check_winner()
        # Determine result from network's perspective
        if winner is None:
            # Check for draw by repetition or max moves
            net_result = "draw"
        else:
            winner_str = str(winner)
            if "White" in winner_str:
                net_result = "win" if net_is_white else "loss"
            else:
                net_result = "loss" if net_is_white else "win"

        # Check how game ended
        last_hash_counts = defaultdict(int)
        for h in hash_history:
            last_hash_counts[h] += 1
        has_3fold = any(c >= 3 for c in last_hash_counts.values())

        game_record = {
            "game_idx": game_idx,
            "net_is_white": net_is_white,
            "result": net_result,
            "winner": str(winner) if winner else None,
            "move_count": len(move_log),
            "moves": move_log,
            "three_fold": has_3fold,
            "final_white_scored": board.white_scored,
            "final_black_scored": board.black_scored,
        }
        games.append(game_record)

        # Progress
        w = sum(1 for g in games if g["result"] == "win")
        d = sum(1 for g in games if g["result"] == "draw")
        l = sum(1 for g in games if g["result"] == "loss")
        print(f"\r  [{label}] Game {game_idx+1}/{num_games}: "
              f"{w}W-{d}D-{l}L", end="", flush=True)

    print()
    return games


def print_eval_summary(games, label):
    """Print aggregate stats from eval games."""
    wins = [g for g in games if g["result"] == "win"]
    draws = [g for g in games if g["result"] == "draw"]
    losses = [g for g in games if g["result"] == "loss"]

    print(f"\n{'='*70}")
    print(f"  EVAL GAMES: {label}")
    print(f"{'='*70}")
    print(f"  W-D-L: {len(wins)}-{len(draws)}-{len(losses)}  "
          f"({len(games)} games vs depth-4 heuristic)")

    # Mean game length by outcome
    for name, group in [("Wins", wins), ("Draws", draws), ("Losses", losses)]:
        if group:
            lengths = [g["move_count"] for g in group]
            print(f"  {name}: avg {np.mean(lengths):.1f} moves "
                  f"(range {min(lengths)}-{max(lengths)})")

    # How games end
    three_fold = sum(1 for g in games if g["three_fold"])
    max_moves = sum(1 for g in games
                    if g["result"] == "draw" and not g["three_fold"])
    decisive = sum(1 for g in games if g["result"] != "draw")
    print(f"\n  Game endings:")
    print(f"    Decisive (winner):   {decisive}")
    print(f"    3-fold repetition:   {three_fold}")
    print(f"    Max moves (80):      {max_moves}")

    # Barrel advancement rate
    net_adv = []
    heur_adv = []
    for g in games:
        for m in g["moves"]:
            if m["is_net_turn"]:
                net_adv.append(m["advancement"])
            else:
                heur_adv.append(m["advancement"])

    print(f"\n  Barrel advancement (avg rows/move):")
    if net_adv:
        print(f"    Network:    {np.mean(net_adv):+.3f}  "
              f"(fwd: {sum(1 for a in net_adv if a > 0)}/{len(net_adv)}, "
              f"back: {sum(1 for a in net_adv if a < 0)}/{len(net_adv)}, "
              f"stay: {sum(1 for a in net_adv if a == 0)}/{len(net_adv)})")
    if heur_adv:
        print(f"    Heuristic:  {np.mean(heur_adv):+.3f}  "
              f"(fwd: {sum(1 for a in heur_adv if a > 0)}/{len(heur_adv)}, "
              f"back: {sum(1 for a in heur_adv if a < 0)}/{len(heur_adv)}, "
              f"stay: {sum(1 for a in heur_adv if a == 0)}/{len(heur_adv)})")

    # Scoring timeline
    first_score_moves = []
    for g in games:
        for m in g["moves"]:
            ws = m["white_scored"]
            bs = m["black_scored"]
            if ws > 0 or bs > 0:
                first_score_moves.append(m["move_num"])
                break
    if first_score_moves:
        print(f"\n  First barrel scored at move: "
              f"avg {np.mean(first_score_moves):.1f} "
              f"(range {min(first_score_moves)}-{max(first_score_moves)})")

    # Repetition stats
    rep_moves = []
    for g in games:
        for m in g["moves"]:
            if m["repetition"] >= 2:
                rep_moves.append(m["move_num"])
    if rep_moves:
        games_with_reps = sum(1 for g in games
                              if any(m["repetition"] >= 2 for m in g["moves"]))
        print(f"  Games with position repetitions: {games_with_reps}/{len(games)}")
        print(f"  First repetition at move: avg {np.mean(rep_moves):.1f}")


def print_game_replay(game, title):
    """Print detailed move-by-move replay of a game."""
    print(f"\n  {'─'*66}")
    print(f"  {title}")
    print(f"  Result: {game['result']}  |  "
          f"Net={'White' if game['net_is_white'] else 'Black'}  |  "
          f"Moves: {game['move_count']}  |  "
          f"Score: W{game['final_white_scored']}-B{game['final_black_scored']}")
    if game["three_fold"]:
        print(f"  ** 3-fold repetition detected **")
    print(f"  {'─'*66}")

    for m in game["moves"]:
        player = "W" if m["is_white"] else "B"
        who = "NET" if m["is_net_turn"] else "HEU"
        rep_mark = f" REP={m['repetition']}" if m["repetition"] >= 2 else ""
        adv_mark = f" adv={m['advancement']:+d}" if m["advancement"] != 0 else ""
        score = f" [W{m['white_scored']}-B{m['black_scored']}]"

        line = (f"  {m['move_num']:3d}. {player}/{who} {m['move_str']:20s}"
                f"{adv_mark}{rep_mark}{score}")

        if m["is_net_turn"]:
            line += (f"  net={m['net_value']:+.3f} "
                     f"mcts={m['mcts_value']:+.3f}")
            # Show top move
            if m["top5"]:
                top = m["top5"][0]
                if top[0] != m["move_str"]:
                    line += f"  (top: {top[0]} mcts={top[1]:.2f})"
        else:
            if "heuristic_value" in m:
                line += f"  heur={m['heuristic_value']:+.3f}"

        print(line)


def run_module1(models, args):
    """Module 1: Eval games with annotations."""
    print("\n" + "=" * 70)
    print("  MODULE 1: EVAL GAMES vs DEPTH-4 HEURISTIC")
    print("=" * 70)

    all_games = []
    for i, (net, label, sims, gumbel) in enumerate(models):
        games = play_eval_games(
            net, sims, args.eval_depth, args.eval_games,
            gumbel, label)
        all_games.append((label, games))
        print_eval_summary(games, label)

        # Print replays: 2 worst losses + best win
        losses = sorted(
            [g for g in games if g["result"] == "loss"],
            key=lambda g: g["move_count"])
        wins_sorted = sorted(
            [g for g in games if g["result"] == "win"],
            key=lambda g: -g["move_count"])

        print(f"\n  --- Detailed Replays for {label} ---")
        for j, g in enumerate(losses[:2]):
            print_game_replay(g, f"LOSS #{j+1} (game {g['game_idx']})")
        if wins_sorted:
            print_game_replay(wins_sorted[0],
                              f"BEST WIN (game {wins_sorted[0]['game_idx']})")
        elif losses:
            # No wins — show a draw instead
            draws = [g for g in games if g["result"] == "draw"]
            if draws:
                print_game_replay(draws[0],
                                  f"DRAW (game {draws[0]['game_idx']})")

    # Side-by-side comparison
    if len(all_games) == 2:
        print(f"\n{'='*70}")
        print(f"  COMPARISON: {all_games[0][0]} vs {all_games[1][0]}")
        print(f"{'='*70}")

        for label, games in all_games:
            w = sum(1 for g in games if g["result"] == "win")
            d = sum(1 for g in games if g["result"] == "draw")
            l = sum(1 for g in games if g["result"] == "loss")
            net_adv = [m["advancement"] for g in games for m in g["moves"]
                       if m["is_net_turn"]]
            three_fold = sum(1 for g in games if g["three_fold"])
            avg_len = np.mean([g["move_count"] for g in games])
            print(f"  {label:20s}: {w:2d}W-{d:2d}D-{l:2d}L  "
                  f"avg_len={avg_len:.0f}  "
                  f"adv={np.mean(net_adv):+.3f}  "
                  f"3fold={three_fold}")

    return all_games


# ---------------------------------------------------------------------------
# Module 2: Network Quality (replay buffer)
# ---------------------------------------------------------------------------

def run_module2(models, args):
    """Module 2: Replay buffer analysis."""
    print("\n" + "=" * 70)
    print("  MODULE 2: NETWORK QUALITY (REPLAY BUFFER)")
    print("=" * 70)

    for net, label, sims, gumbel in models:
        # Find buffer file alongside checkpoint
        # Try to get checkpoint path from label -> find matching dir
        buf_path = None
        for cp_path in args.checkpoints:
            p = Path(cp_path)
            candidate = p.parent / f"{p.name}.buffer.npz"
            if candidate.exists():
                buf_path = candidate
                break

        # Try each checkpoint
        for idx, cp_path in enumerate(args.checkpoints):
            cp_label = args.labels[idx] if idx < len(args.labels) else f"model{idx}"
            if cp_label != label:
                continue
            p = Path(cp_path)
            candidate = p.parent / f"{p.name}.buffer.npz"
            if candidate.exists():
                buf_path = candidate
                break

        if buf_path is None or not buf_path.exists():
            print(f"\n  [{label}] No replay buffer found, skipping")
            continue

        data = np.load(buf_path)
        policies = data["policies"]
        values = data["values"]
        boards = data["boards"]

        # Sample up to 500
        n = min(500, len(policies))
        indices = np.random.choice(len(policies), n, replace=False)
        policies_s = policies[indices]
        values_s = values[indices]
        boards_s = boards[indices]

        print(f"\n  {'='*60}")
        print(f"  {label} (buffer: {buf_path.name}, {len(policies)} positions, sampled {n})")
        print(f"  {'='*60}")

        # Value distribution
        n_win = int(np.sum(values_s > 0.5))
        n_loss = int(np.sum(values_s < -0.5))
        n_draw = n - n_win - n_loss
        print(f"\n  Value distribution:")
        print(f"    Win (>0.5):   {n_win:4d}  ({n_win/n*100:.1f}%)")
        print(f"    Draw:         {n_draw:4d}  ({n_draw/n*100:.1f}%)")
        print(f"    Loss (<-0.5): {n_loss:4d}  ({n_loss/n*100:.1f}%)")
        print(f"    Mean value:   {values_s.mean():+.4f}")

        # Policy top-1 accuracy: does net's best move match MCTS target's best?
        engine_tmp = _RustMCTSEngine(1, 1.0)
        eval_fn = make_batch_eval_fn(net, batch_size=16)
        correct = 0
        for i in range(n):
            board_planes = boards_s[i]
            target_best = np.argmax(policies_s[i])
            # Get net prediction
            planes_t = torch.tensor(
                board_planes, dtype=torch.float32
            ).reshape(1, BOARD_PLANES, BOARD_SIZE, BOARD_SIZE)
            with torch.no_grad():
                logits, _ = net(planes_t)
            net_best = logits[0].argmax().item()
            if net_best == target_best:
                correct += 1

        print(f"\n  Policy top-1 accuracy: {correct}/{n} ({correct/n*100:.1f}%)")

        # Policy entropy stats
        entropies = np.array([entropy(p) for p in policies_s])
        print(f"\n  MCTS target entropy:")
        print(f"    Mean:   {entropies.mean():.4f}")
        print(f"    Median: {np.median(entropies):.4f}")
        print(f"    Std:    {entropies.std():.4f}")

        # Value calibration: bin net value vs actual outcome
        print(f"\n  Value calibration (net prediction vs actual outcome):")
        print(f"    {'Bin':>12s}  {'Count':>6s}  {'Net Mean':>10s}  {'Actual':>10s}  {'Error':>10s}")
        bins = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]
        for lo, hi in bins:
            mask = (values_s >= lo) & (values_s < hi)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            actual_mean = values_s[mask].mean()
            # Get net predictions for these positions
            net_vals = []
            for idx in np.where(mask)[0]:
                planes_t = torch.tensor(
                    boards_s[idx], dtype=torch.float32
                ).reshape(1, BOARD_PLANES, BOARD_SIZE, BOARD_SIZE)
                with torch.no_grad():
                    _, v = net(planes_t)
                net_vals.append(v.item())
            net_mean = np.mean(net_vals)
            err = abs(net_mean - actual_mean)
            print(f"    [{lo:+.1f},{hi:+.1f})  {cnt:6d}  {net_mean:+10.4f}  "
                  f"{actual_mean:+10.4f}  {err:10.4f}")

        # Search score distribution (if available)
        if "search_scores" in data:
            scores = data["search_scores"][indices]
            print(f"\n  Search score (MCTS root value) distribution:")
            print(f"    Mean:   {scores.mean():+.4f}")
            print(f"    Median: {np.median(scores):+.4f}")
            print(f"    Std:    {scores.std():.4f}")
            print(f"    |score| > 0.5: {int(np.sum(np.abs(scores) > 0.5))}/{n} "
                  f"({np.sum(np.abs(scores) > 0.5)/n*100:.1f}%)")


# ---------------------------------------------------------------------------
# Module 3: Position Deep Dive
# ---------------------------------------------------------------------------

def run_module3(all_games, models, args):
    """Module 3: Deep dive on critical positions from losses."""
    print("\n" + "=" * 70)
    print("  MODULE 3: POSITION DEEP DIVE (CRITICAL TURNING POINTS)")
    print("=" * 70)

    for model_idx, (net, label, sims, gumbel) in enumerate(models):
        if model_idx >= len(all_games):
            continue
        game_label, games = all_games[model_idx]

        losses = [g for g in games if g["result"] == "loss"]
        if not losses:
            print(f"\n  [{label}] No losses to analyze")
            continue

        print(f"\n  {'='*60}")
        print(f"  {label}: CRITICAL POSITIONS FROM LOSSES")
        print(f"  {'='*60}")

        # Find turning points: moves where MCTS value drops significantly
        # or where network disagrees with heuristic
        critical_positions = []
        for g in losses:
            prev_value = 0.0
            for m in g["moves"]:
                if not m["is_net_turn"]:
                    continue
                curr_value = m.get("mcts_value", 0)
                value_drop = prev_value - curr_value
                if value_drop > 0.15:  # Significant value drop
                    critical_positions.append({
                        "game_idx": g["game_idx"],
                        "move_num": m["move_num"],
                        "move": m,
                        "value_drop": value_drop,
                        "net_is_white": g["net_is_white"],
                    })
                prev_value = curr_value

        # Sort by value drop, take top 5
        critical_positions.sort(key=lambda x: -x["value_drop"])
        critical_positions = critical_positions[:5]

        if not critical_positions:
            # Fallback: just show last few net moves from worst loss
            g = losses[0]
            net_moves = [m for m in g["moves"] if m["is_net_turn"]]
            for m in net_moves[-3:]:
                critical_positions.append({
                    "game_idx": g["game_idx"],
                    "move_num": m["move_num"],
                    "move": m,
                    "value_drop": 0,
                    "net_is_white": g["net_is_white"],
                })

        for cp_idx, cp in enumerate(critical_positions):
            m = cp["move"]
            print(f"\n  --- Position #{cp_idx+1}: Game {cp['game_idx']}, "
                  f"Move {m['move_num']} ---")
            print(f"  Value drop: {cp['value_drop']:+.3f}  |  "
                  f"Net={'White' if cp['net_is_white'] else 'Black'}")
            print(f"  Chosen: {m['move_str']}  "
                  f"net_val={m.get('net_value', 0):+.3f}  "
                  f"mcts_val={m.get('mcts_value', 0):+.3f}")
            if m.get("top5"):
                print(f"  Top-5 moves:")
                for name, mcts_p, net_p in m["top5"]:
                    print(f"    {name:20s}  mcts={mcts_p:.3f}  net={net_p:.3f}")
            adv = m["advancement"]
            if adv <= 0:
                print(f"  ** NOT ADVANCING (adv={adv:+d}) **")
            if m["repetition"] >= 2:
                print(f"  ** POSITION REPEATED {m['repetition']} times **")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose AlphaZero training: why can't it beat heuristic?")
    parser.add_argument("checkpoints", nargs="+",
                        help="1 or 2 checkpoint paths")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display names for each model")
    parser.add_argument("--hidden", nargs="+", type=int, default=[128],
                        help="Hidden channels per model")
    parser.add_argument("--simulations", nargs="+", type=int, default=[200],
                        help="Simulations per model")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games vs heuristic per model (default: 20)")
    parser.add_argument("--eval-depth", type=int, default=4,
                        help="Heuristic depth (default: 4)")
    parser.add_argument("--use-gumbel", action="store_true",
                        help="Use Gumbel search for eval")
    parser.add_argument("--module", type=str, default="all",
                        help="Module to run: 1, 2, 3, or all (default: all)")
    args = parser.parse_args()

    n_models = len(args.checkpoints)
    if args.labels is None:
        args.labels = [f"model{i}" for i in range(n_models)]
    # Extend single values to match number of models
    while len(args.labels) < n_models:
        args.labels.append(args.labels[-1])
    while len(args.hidden) < n_models:
        args.hidden.append(args.hidden[-1])
    while len(args.simulations) < n_models:
        args.simulations.append(args.simulations[-1])

    print("=" * 70)
    print("  ALPHAZERO TRAINING DIAGNOSTIC")
    print("=" * 70)

    # Load models
    models = []
    for i in range(n_models):
        print(f"  Loading {args.labels[i]} from {args.checkpoints[i]} "
              f"(hidden={args.hidden[i]}, sims={args.simulations[i]})")
        net = load_model(args.checkpoints[i], hidden=args.hidden[i])
        models.append((net, args.labels[i], args.simulations[i], args.use_gumbel))

    modules = args.module
    all_games = None

    if modules in ("all", "1"):
        all_games = run_module1(models, args)

    if modules in ("all", "2"):
        run_module2(models, args)

    if modules in ("all", "3"):
        if all_games is None:
            # Need to run module 1 first for game data
            print("\n  (Running Module 1 first to get game data for Module 3...)")
            all_games = run_module1(models, args)
        run_module3(all_games, models, args)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
