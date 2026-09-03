#!/usr/bin/env python3
"""How does the engine play? Instrumented self-play + network probes.

Compares the NNUE engine with the handcrafted heuristic on:
  1. Move-type profile: placements / steps / jump chains, forward vs sideways
     vs backward, pail timing and placement heatmap.
  2. Racing vs blocking: per move, change in own race distance (negative =
     progress) and in the opponent's race distance (positive = blocking),
     using the exact single-agent race table.
  3. Decision disagreements: at every position of a game, ask the OTHER
     engine for its move at the same fixed depth; tabulate (kind, kind).
  4. Positional probes: eval of a lone barrel / enemy barrel / pail on each
     square, as 6x6 heatmaps, for both evaluators ("piece-square tables").

Fixed depth for both engines so differences are due to evaluation, not speed.

Usage:
    python scripts/analyze_play.py --games 100 --depth 6 --nnue models/net1b_plain_m_d20_96x16_l05.json
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import numpy as np  # noqa: E402
from tonnesjakk import Board, Engine  # noqa: E402
from tonnesjakk.nnue import board_to_tensor  # noqa: E402

NO_PROGRESS_LIMIT = 60


def make_engine(nnue: str):
    e = Engine()
    e.no_progress_limit = NO_PROGRESS_LIMIT
    if nnue:
        e.load_nnue(nnue)
    return e


def classify(move, white_to_move: bool):
    """(kind, direction, jump_len). kind in pail/place/step/jump."""
    if move.is_pail_only:
        return "pail", "-", 0
    if move.is_barrel_placement:
        return "place", "-", 0
    f, t = move.barrel_from, move.barrel_to
    dr = t.row - f.row
    fwd_sign = -1 if white_to_move else 1
    prog = dr * fwd_sign
    direction = "fwd" if prog > 0 else ("back" if prog < 0 else "side")
    hops = len(move.barrel_path)
    if hops == 1 and max(abs(dr), abs(t.col - f.col)) == 1:
        return "step", direction, 0
    return "jump", direction, hops


def board_race_distance(board, white: bool) -> int:
    """Opponent-aware race distance: sum over the side's on-board barrels of the
    BFS shortest path (steps to empty squares, jumps over any piece to an
    empty square, never over the enemy pail) to the goal row on the CURRENT
    board, plus 1 + 5 per barrel still in hand. Unlike the single-agent race
    table this changes when the opponent blocks a lane or offers a ladder."""
    arr = board.to_array()
    own = 1 if white else -1
    enemy_pail = -2 if white else 2
    goal = 0 if white else 5
    occ = [[arr[r][c] != 0 for c in range(6)] for r in range(6)]
    dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    total = 0
    for r0 in range(6):
        for c0 in range(6):
            if arr[r0][c0] != own:
                continue
            dist = {(r0, c0): 0}
            frontier = [(r0, c0)]
            found = None
            while frontier and found is None:
                nxt = []
                for (r, c) in frontier:
                    d = dist[(r, c)]
                    for dr, dc in dirs:
                        for hop in (1, 2):
                            nr, nc = r + dr * hop, c + dc * hop
                            if not (0 <= nr < 6 and 0 <= nc < 6):
                                continue
                            if hop == 2:
                                mr, mc = r + dr, c + dc
                                if not occ[mr][mc] or arr[mr][mc] == enemy_pail:
                                    continue
                            if occ[nr][nc] and (nr, nc) != (r0, c0):
                                continue
                            if (nr, nc) in dist:
                                continue
                            dist[(nr, nc)] = d + 1
                            if nr == goal:
                                found = d + 1
                                break
                            nxt.append((nr, nc))
                        if found is not None:
                            break
                    if found is not None:
                        break
                frontier = nxt
            total += found if found is not None else 12  # fully boxed in
    in_hand = (board.white_barrels_off_board if white else board.black_barrels_off_board)
    return total + in_hand * 6


def play_instrumented(engine, other, depth, seed, label):
    """Self-play with `engine`; `other` is queried at each position for disagreement."""
    board = Board()
    engine.full_reset(); other.full_reset()
    rng = random.Random(seed)
    for _ in range(6):
        ms = [m for m in board.generate_moves() if not m.is_pail_only]
        if not ms:
            break
        board.make_move(rng.choice(ms))
    recent = [board.get_hash()]; counts = {board.get_hash(): 1}
    records = []
    plies = 0
    while board.check_winner() is None and plies < 400:
        if board.halfmove_clock >= NO_PROGRESS_LIMIT or counts.get(board.get_hash(), 0) >= 3:
            break
        white = "White" in repr(board.current_player)
        engine.set_game_history(recent); other.set_game_history(recent)
        r = engine.search(board, depth)
        if r.best_move is None:
            break
        r2 = other.search(board, depth)
        kind, direction, hops = classify(r.best_move, white)
        okind, odir, ohops = classify(r2.best_move, white) if r2.best_move else ("none", "-", 0)
        own0 = board_race_distance(board, white); opp0 = board_race_distance(board, not white)
        board.make_move(r.best_move)
        own1 = board_race_distance(board, white); opp1 = board_race_distance(board, not white)
        records.append({
            "engine": label, "ply": plies, "phase": board.white_scored + board.black_scored,
            "kind": kind, "dir": direction, "hops": hops,
            "d_own": own1 - own0, "d_opp": opp1 - opp0,
            "pail_sq": (r.best_move.place_pail.row, r.best_move.place_pail.col) if kind == "pail" else None,
            "move_count": board.move_count,
            "agree": (kind, direction) == (okind, odir) and str(r.best_move) == str(r2.best_move),
            "other_kind": okind, "other_dir": odir,
        })
        plies += 1
        h = board.get_hash()
        recent = [h] if board.halfmove_clock == 0 else recent + [h]
        counts[h] = counts.get(h, 0) + 1
    winner = board.check_winner()
    return records, (None if winner is None else ("W" if "White" in str(winner) else "B")), plies


def summarize(records, label):
    barrel = [r for r in records if r["kind"] != "pail"]
    n = len(barrel)
    kinds = Counter(r["kind"] for r in barrel)
    dirs = Counter(r["dir"] for r in barrel if r["kind"] in ("step", "jump"))
    hops = Counter(r["hops"] for r in barrel if r["kind"] == "jump")
    d_own = np.mean([r["d_own"] for r in barrel]); d_opp = np.mean([r["d_opp"] for r in barrel])
    block_frac = np.mean([r["d_opp"] > 0 for r in barrel])
    prog_frac = np.mean([r["d_own"] < 0 for r in barrel])
    pails = [r for r in records if r["kind"] == "pail"]
    pail_ply = [r["move_count"] for r in pails]
    agree = np.mean([r["agree"] for r in barrel])
    lines = [f"── {label} ── {n} barrel moves, {len(pails)} pail placements",
             "  move kinds : " + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in kinds.most_common()),
             "  direction  : " + ", ".join(f"{k} {100*v/max(sum(dirs.values()),1):.0f}%" for k, v in dirs.most_common()),
             "  jump chains: " + ", ".join(f"{k}-hop {v}" for k, v in sorted(hops.items())),
             f"  race effect: mean Δown {d_own:+.2f} (progress moves {100*prog_frac:.0f}%), "
             f"mean Δopp {d_opp:+.2f} (blocking moves {100*block_frac:.0f}%)",
             f"  pail timing: median move {np.median(pail_ply) if pail_ply else float('nan'):.0f}, "
             f"p10/p90 {np.percentile(pail_ply,10) if pail_ply else 0:.0f}/{np.percentile(pail_ply,90) if pail_ply else 0:.0f}",
             f"  agreement with the other engine's move at equal depth: {100*agree:.0f}%"]
    # phase breakdown of blocking
    by_phase = defaultdict(list)
    for r in barrel:
        by_phase[min(r["phase"], 6)].append(r["d_opp"] > 0)
    lines.append("  blocking by phase (scored total): " + ", ".join(
        f"{p}:{100*np.mean(v):.0f}%" for p, v in sorted(by_phase.items())))
    grid = np.zeros((6, 6), dtype=int)
    for r in pails:
        grid[r["pail_sq"][0], r["pail_sq"][1]] += 1
    lines.append("  pail heatmap (row 0 = white's goal):")
    for row in grid:
        lines.append("    " + " ".join(f"{v:3d}" for v in row))
    return "\n".join(lines)


def disagreement_table(records, label_a, label_b):
    """Where A disagreed with B: A's (kind,dir) vs B's (kind,dir)."""
    c = Counter((f"{r['kind']}/{r['dir']}", f"{r['other_kind']}/{r['other_dir']}")
                for r in records if not r["agree"] and r["kind"] != "pail")
    lines = [f"── top disagreements: {label_a} chose → {label_b} would choose ──"]
    for (a, b), v in c.most_common(12):
        lines.append(f"  {a:>12} → {b:<12} {v}")
    return "\n".join(lines)


def probes(nnue_path):
    """Piece-square heatmaps from both evaluators."""
    e_n = make_engine(nnue_path); e_h = Engine()

    def row_for(cells, white_scored=0, black_scored=0, player=1):
        return board_to_tensor(cells, white_scored, black_scored, player).numpy().astype(np.float32)

    def grid(fn):
        rows, coords = [], []
        for r in range(6):
            for c in range(6):
                cells = fn(r, c)
                if cells is None:
                    continue
                rows.append(row_for(cells)); coords.append((r, c))
        flat = np.concatenate(rows).ravel().tolist()
        nn = e_n.nnue_eval_rows(flat); hh = e_h.heuristic_eval_rows(flat)
        gn = np.full((6, 6), np.nan); gh = np.full((6, 6), np.nan)
        for (r, c), a, b in zip(coords, nn, hh):
            gn[r, c] = a; gh[r, c] = b
        return gn, gh

    def show(title, gn, gh):
        print(f"\n{title}\n{'NNUE (cp)':<30}{'heuristic (cp)'}")
        for r in range(6):
            left = " ".join("   ." if np.isnan(v) else f"{v:+5.0f}" for v in gn[r])
            right = " ".join("   ." if np.isnan(v) else f"{v:+5.0f}" for v in gh[r])
            print(f"{left:<30}{right}")

    empty = lambda: [[0] * 6 for _ in range(6)]

    def lone_white(r, c):
        if r == 0: return None
        g = empty(); g[r][c] = 1; return g
    show("Lone white barrel (3 in hand, black all in hand); row 0 = white's goal", *grid(lone_white))

    def lone_black(r, c):
        if r == 5: return None
        g = empty(); g[r][c] = -1; return g
    show("Lone black barrel (row 5 = black's goal)", *grid(lone_black))

    def black_pail_vs_white_runner(r, c):
        g = empty(); g[3][2] = 1; g[2][3] = -1
        if (r, c) in ((3, 2), (2, 3)): return None
        g[r][c] = -2; return g
    base = row_for([[0]*6 for _ in range(6)])  # unused placeholder to keep structure simple
    show("Black pail placed on each square, with white barrel at (3,2) and black barrel at (2,3)",
         *grid(black_pail_vs_white_runner))


def main():
    ap = argparse.ArgumentParser(description="Play-style analysis: NNUE vs heuristic")
    ap.add_argument("--nnue", default="models/net1b_plain_m_d20_96x16_l05.json")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--out", default="runs/analysis")
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    heur, nnue = Engine(), make_engine(args.nnue)
    heur.no_progress_limit = NO_PROGRESS_LIMIT

    t0 = time.time()
    all_records = {"heuristic": [], "nnue": []}
    outcomes = {"heuristic": Counter(), "nnue": Counter()}
    for g in range(args.games):
        rec, res, _ = play_instrumented(heur, nnue, args.depth, 1000 + g, "heuristic")
        all_records["heuristic"] += rec; outcomes["heuristic"][res] += 1
        rec, res, _ = play_instrumented(nnue, heur, args.depth, 1000 + g, "nnue")
        all_records["nnue"] += rec; outcomes["nnue"][res] += 1
    print(f"{args.games} self-play games per engine at depth {args.depth} ({time.time()-t0:.0f}s)")
    print(f"outcomes heuristic self-play {dict(outcomes['heuristic'])}, nnue self-play {dict(outcomes['nnue'])}\n")
    print(summarize(all_records["heuristic"], "heuristic self-play (NNUE queried)"))
    print()
    print(summarize(all_records["nnue"], "NNUE self-play (heuristic queried)"))
    print()
    print(disagreement_table(all_records["nnue"], "NNUE", "heuristic"))
    print()
    print(disagreement_table(all_records["heuristic"], "heuristic", "NNUE"))
    (Path(args.out) / "records.json").write_text(json.dumps(all_records))

    if not args.skip_probes:
        probes(args.nnue)


if __name__ == "__main__":
    main()
