#!/usr/bin/env python3
"""Engine-vs-engine match harness with proper draw rules and statistics.

This is the measurement backbone: every engine change should be gated on a
match run from this script. Design:

- **Paired openings**: each opening (random plies from a seeded RNG) is played
  twice with colors swapped, eliminating opening bias. Elo CI is computed over
  per-pair scores (pentanomial-style), which is tighter than per-game.
- **Real draw rules**: threefold repetition, no-progress clock (60 plies
  without placement/pail/scoring), plus a 400-ply safety cap.
- **Fixed time or fixed depth** per engine side.
- **Parallel**: games distributed over worker processes; engines are created
  once per worker and full_reset() between games.

Usage examples:
    # depth 4 vs depth 6, 200 games (100 pairs), 10 workers
    python scripts/match.py --depth-a 4 --depth-b 6 --games 200

    # equal time 100ms, NNUE vs heuristic
    python scripts/match.py --time-a 100 --time-b 100 --nnue-a path/to/nnue_weights.json

    # eval weight experiment
    python scripts/match.py --depth-a 7 --depth-b 7 --set-a weight_trapped=40
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

NO_PROGRESS_LIMIT = 60   # plies without irreversible event -> draw
MAX_PLIES = 400          # hard safety cap (should never trigger)


@dataclass
class EngineSpec:
    label: str
    depth: int = 0            # fixed-depth search if > 0
    time_ms: int = 0          # fixed-time search if > 0 (takes precedence)
    nnue: str = ""            # path to NNUE weights JSON
    contempt: int = 0
    weights: dict = field(default_factory=dict)  # weight_* overrides
    tablebases: str = ""      # directory of solved tb_*.bin phases (memory-mapped)

    def describe(self) -> str:
        parts = [f"time={self.time_ms}ms" if self.time_ms else f"depth={self.depth}"]
        if self.nnue:
            parts.append(f"nnue={Path(self.nnue).name}")
        if self.tablebases:
            parts.append(f"tb={self.tablebases}")
        if self.contempt:
            parts.append(f"contempt={self.contempt}")
        parts.extend(f"{k}={v}" for k, v in self.weights.items())
        return f"{self.label} ({', '.join(parts)})"


def make_engine(spec: EngineSpec):
    from tonnesjakk import Engine
    e = Engine()
    e.no_progress_limit = NO_PROGRESS_LIMIT
    e.contempt = spec.contempt
    if spec.nnue:
        e.load_nnue(spec.nnue)
    if spec.tablebases:
        e.load_tablebases(spec.tablebases)
    for name, value in spec.weights.items():
        setattr(e, name, value)
    return e


def _search(engine, board, spec: EngineSpec):
    if spec.time_ms:
        return engine.search_timed(board, spec.time_ms)
    return engine.search(board, spec.depth)


def play_game(engine_w, spec_w, engine_b, spec_b, opening_seed: int, opening_plies: int):
    """Play one game. Returns (+1 white win, -1 black win, 0 draw), termination, plies."""
    from tonnesjakk import Board, Player

    board = Board()
    engine_w.full_reset()
    engine_b.full_reset()

    rng = random.Random(opening_seed)
    plies_done = 0
    while plies_done < opening_plies:
        # Openings randomize BARREL moves only. Pail placement is a strategic
        # once-per-game decision — burning it on a random square in the
        # opening would erase exactly the dimension we want engines to play.
        moves = [m for m in board.generate_moves() if not m.is_pail_only]
        if not moves or board.check_winner() is not None:
            break
        board.make_move(rng.choice(moves))
        plies_done += 1

    # Position history: `recent` = hashes since last clock reset (incl. current),
    # passed to the engine so search sees game repetitions. `counts` for 3-fold.
    recent = [board.get_hash()]
    counts = {board.get_hash(): 1}

    plies = 0
    while True:
        winner = board.check_winner()
        if winner is not None:
            return (1 if "White" in str(winner) else -1), "win", plies
        if board.halfmove_clock >= NO_PROGRESS_LIMIT:
            return 0, "no_progress", plies
        if counts.get(board.get_hash(), 0) >= 3:
            return 0, "threefold", plies
        if plies >= MAX_PLIES:
            return 0, "max_plies", plies

        white_to_move = "White" in repr(board.current_player)
        engine, spec = (engine_w, spec_w) if white_to_move else (engine_b, spec_b)
        engine.set_game_history(recent)
        result = _search(engine, board, spec)
        if result.best_move is None:
            return 0, "stalemate", plies

        board.make_move(result.best_move)
        plies += 1

        h = board.get_hash()
        if board.halfmove_clock == 0:
            recent = [h]
        else:
            recent.append(h)
        counts[h] = counts.get(h, 0) + 1


# ─── worker plumbing ────────────────────────────────────────────────────────

_worker_state = {}


def _worker_init(spec_a_dict, spec_b_dict):
    spec_a = EngineSpec(**spec_a_dict)
    spec_b = EngineSpec(**spec_b_dict)
    _worker_state["spec_a"] = spec_a
    _worker_state["spec_b"] = spec_b
    _worker_state["engine_a"] = make_engine(spec_a)
    _worker_state["engine_b"] = make_engine(spec_b)


def _play_pair(args):
    """Play one opening with both color assignments. Returns pair result dict."""
    opening_seed, opening_plies = args
    sa, sb = _worker_state["spec_a"], _worker_state["spec_b"]
    ea, eb = _worker_state["engine_a"], _worker_state["engine_b"]

    results = []
    for a_is_white in (True, False):
        if a_is_white:
            outcome, term, plies = play_game(ea, sa, eb, sb, opening_seed, opening_plies)
            score_a = {1: 1.0, 0: 0.5, -1: 0.0}[outcome]
        else:
            outcome, term, plies = play_game(eb, sb, ea, sa, opening_seed, opening_plies)
            score_a = {1: 0.0, 0: 0.5, -1: 1.0}[outcome]
        results.append({"score_a": score_a, "termination": term, "plies": plies,
                        "a_is_white": a_is_white})
    return {"seed": opening_seed, "games": results,
            "pair_score": sum(g["score_a"] for g in results)}


# ─── statistics ─────────────────────────────────────────────────────────────

def elo_from_score(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def score_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(pair_scores, elo0: float, elo1: float) -> float:
    """GSPRT log-likelihood ratio (normal approximation) over PAIR scores.

    Pair-based statistics are the pentanomial model's point: the two games of
    a pair are correlated, so variance must come from pair scores, not games.
    H0: true score = score_from_elo(elo0), H1: = score_from_elo(elo1).
    LLR = n(p1-p0)(2*mean - p0 - p1) / (2*var), all in per-game score units.
    """
    n = len(pair_scores)
    if n < 2:
        return 0.0
    mean = sum(pair_scores) / n / 2.0  # per-game score in [0,1]
    var = sum((s / 2.0 - mean) ** 2 for s in pair_scores) / (n - 1)
    if var < 1e-9:
        return 0.0
    p0, p1 = score_from_elo(elo0), score_from_elo(elo1)
    return n * (p1 - p0) * (2.0 * mean - p0 - p1) / (2.0 * var)


def summarize(pair_results):
    games = [g for pr in pair_results for g in pr["games"]]
    n = len(games)
    wins = sum(1 for g in games if g["score_a"] == 1.0)
    losses = sum(1 for g in games if g["score_a"] == 0.0)
    draws = n - wins - losses

    pair_scores = [pr["pair_score"] for pr in pair_results]  # 0..2 per pair
    n_pairs = len(pair_scores)
    mean = sum(pair_scores) / n_pairs
    var = sum((s - mean) ** 2 for s in pair_scores) / max(n_pairs - 1, 1)
    se = math.sqrt(var / n_pairs)

    p = mean / 2.0
    elo = elo_from_score(p)
    lo = elo_from_score(max((mean - 1.96 * se) / 2.0, 0.0))
    hi = elo_from_score(min((mean + 1.96 * se) / 2.0, 1.0))

    terms = {}
    for g in games:
        terms[g["termination"]] = terms.get(g["termination"], 0) + 1
    avg_plies = sum(g["plies"] for g in games) / n

    return {
        "games": n, "wins_a": wins, "draws": draws, "losses_a": losses,
        "score_a_pct": 100.0 * p, "elo_a": elo, "elo_ci95": [lo, hi],
        "terminations": terms, "avg_plies": avg_plies,
    }


# ─── main ───────────────────────────────────────────────────────────────────

def parse_sets(items):
    out = {}
    for item in items or []:
        k, v = item.split("=")
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Engine match harness")
    ap.add_argument("--games", type=int, default=200, help="Total games (2 per opening pair)")
    ap.add_argument("--opening-plies", type=int, default=6, help="Random opening plies")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="Base seed for openings")
    ap.add_argument("--out", type=str, default="", help="Write JSON results here")
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"),
                    help="SPRT stop rule (alpha=beta=0.05): stop when LLR hits "
                         "±2.94. Gainer: 0 5. Simplification: -5 0. --games "
                         "becomes the max game budget.")

    for side in ("a", "b"):
        ap.add_argument(f"--label-{side}", type=str, default=side.upper())
        ap.add_argument(f"--depth-{side}", type=int, default=0)
        ap.add_argument(f"--time-{side}", type=int, default=0, help="ms per move")
        ap.add_argument(f"--nnue-{side}", type=str, default="")
        ap.add_argument(f"--tb-{side}", type=str, default="", help="tablebase directory")
        ap.add_argument(f"--contempt-{side}", type=int, default=0)
        ap.add_argument(f"--set-{side}", action="append",
                        help="weight override, e.g. weight_trapped=40 (repeatable)")
    args = ap.parse_args()

    specs = {}
    for side in ("a", "b"):
        g = lambda name: getattr(args, f"{name.replace('-', '_')}_{side}")
        if not g("depth") and not g("time"):
            ap.error(f"engine {side.upper()}: give --depth-{side} or --time-{side}")
        specs[side] = EngineSpec(
            label=g("label"), depth=g("depth"), time_ms=g("time"),
            nnue=g("nnue"), contempt=g("contempt"), weights=parse_sets(g("set")),
            tablebases=g("tb"),
        )

    n_pairs = max(args.games // 2, 1)
    tasks = [(args.seed * 1_000_003 + i, args.opening_plies) for i in range(n_pairs)]

    print(f"Match: {specs['a'].describe()}  vs  {specs['b'].describe()}")
    print(f"  {n_pairs} opening pairs = {2 * n_pairs} games, "
          f"{args.opening_plies} opening plies, {args.workers} workers")
    print(f"  Draw rules: threefold repetition, no-progress {NO_PROGRESS_LIMIT} plies\n", flush=True)

    t0 = time.time()
    pair_results = []
    spec_dicts = ({**specs["a"].__dict__}, {**specs["b"].__dict__})
    LLR_BOUND = math.log(19.0)  # alpha = beta = 0.05
    sprt_verdict = None
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_worker_init, initargs=spec_dicts
    ) as pool:
        for i, pr in enumerate(pool.imap_unordered(_play_pair, tasks), 1):
            pair_results.append(pr)
            llr = None
            if args.sprt:
                llr = sprt_llr([p["pair_score"] for p in pair_results], *args.sprt)
                if llr >= LLR_BOUND:
                    sprt_verdict = "H1 accepted (PASS)"
                elif llr <= -LLR_BOUND:
                    sprt_verdict = "H0 accepted (FAIL)"
            if i % 10 == 0 or i == n_pairs or sprt_verdict:
                s = summarize(pair_results)
                llr_str = f"  LLR {llr:+.2f}/±2.94" if llr is not None else ""
                print(f"  [{i}/{n_pairs} pairs] A: +{s['wins_a']} ={s['draws']} -{s['losses_a']}"
                      f"  score {s['score_a_pct']:.1f}%  elo {s['elo_a']:+.0f} "
                      f"[{s['elo_ci95'][0]:+.0f}, {s['elo_ci95'][1]:+.0f}]{llr_str}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
            if sprt_verdict:
                pool.terminate()
                break

    s = summarize(pair_results)
    elapsed = time.time() - t0
    print(f"\n{'=' * 64}")
    if args.sprt:
        print(f"  SPRT({args.sprt[0]:g}, {args.sprt[1]:g}): "
              f"{sprt_verdict or 'inconclusive (game budget exhausted)'}")
    print(f"  {specs['a'].describe()}")
    print(f"    vs {specs['b'].describe()}")
    print(f"  Games: {s['games']}  W-D-L (A): {s['wins_a']}-{s['draws']}-{s['losses_a']}"
          f"  score {s['score_a_pct']:.1f}%")
    print(f"  Elo (A vs B): {s['elo_a']:+.1f}  95% CI [{s['elo_ci95'][0]:+.1f}, {s['elo_ci95'][1]:+.1f}]")
    print(f"  Terminations: {s['terminations']}")
    print(f"  Avg game length: {s['avg_plies']:.1f} plies   ({elapsed:.0f}s total)")

    if args.out:
        payload = {
            "spec_a": specs["a"].__dict__, "spec_b": specs["b"].__dict__,
            "args": {"games": args.games, "opening_plies": args.opening_plies, "seed": args.seed},
            "sprt": ({"bounds": args.sprt, "verdict": sprt_verdict} if args.sprt else None),
            "summary": s, "pairs": pair_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_sec": elapsed,
        }
        Path(args.out).write_text(json.dumps(payload, indent=1))
        print(f"  Results written to {args.out}")


if __name__ == "__main__":
    main()
