#!/usr/bin/env python3
"""SPSA tuning of engine parameters via self-play: the heuristic eval weights
(--params eval, default) or the search/pruning knobs (--params search, meant
to run with --nnue and --time-ms).

Classic chess-engine SPSA (as used for Stockfish): each iteration perturbs
all weights simultaneously by +/- c_k (Rademacher signs), plays a small match
between theta+ and theta- with paired random openings (common random numbers),
and nudges every weight toward the winning side:

    theta_i += a_k * (score_plus - 0.5) * delta_i * scale_i

Engines are deterministic and openings are shared between theta+ and theta-,
so the match result is a low-variance estimate of which perturbation is better.

Usage:
    python scripts/spsa_tune.py --iterations 150 --pairs 16 --depth 5 --workers 10
    # then validate the result:
    python scripts/match.py --depth-a 7 --depth-b 7 --set-a <tuned...> --games 400
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match import EngineSpec, make_engine, play_game  # noqa: E402

# name: (initial, step_scale, min, max)
# Initials = current engine defaults (SPSA round 1 result, 2026-08-20).
EVAL_PARAMS = {
    "weight_progress":     (77,  10, 10, 400),
    "weight_center_pail":  (15,   5, -50, 100),
    "weight_blocking":     (19,   6, -50, 200),
    "weight_scored":       (700, 40, 100, 2000),
    "weight_threat":       (144, 15, 0, 600),
    "weight_threat2":      (101, 12, 0, 400),
    "weight_adj_blocking": (0,    5, -100, 100),
    "weight_mobility":     (13,   4, -50, 100),
    "weight_passed":       (80,  10, -50, 300),
    "weight_trapped":      (1,    8, -100, 200),
    "weight_score_accel":  (-3,  20, -200, 600),
    "weight_eg_threat":    (-1,  10, -100, 300),
    "weight_jump":         (63,   8, -50, 200),
    "weight_race":         (80,  10, -50, 400),
}

# Search/pruning knobs (--params search). Initials = engine defaults (SPSA
# 2026-08-27 with net-3 @ 100ms, a=6 c=2: +60/+54 Elo vs the hand-tuned
# values). Run with --nnue and a time control: pruning trades depth for accuracy.
SEARCH_PARAMS = {
    "asp_delta":        (28,   8,   5,  200),
    "razor_base":       (190, 40,   0,  800),
    "razor_slope":      (139, 30,   0,  600),
    "nmp_margin":       (48,  15, -200, 400),
    "nmp_boost_margin": (170, 30,   0,  800),
    "fut_scale":        (101, 15,  20,  300),
    "lmr_div":          (95, 12,  40,  300),
    "lmr_hist_good":    (829, 200, 0, 5000),
    "lmr_hist_bad":     (-486, 150, -5000, 0),
    "lmp_base":         (9,   2,   1,   30),
    "rfp_margin":       (77,  30,  0,  400),
    "iir_depth":        (2,   1,   2,   12),
}

PARAMS = EVAL_PARAMS  # selected in main() via --params

_state = {}


def _worker_init(depth, time_ms, nnue):
    spec = EngineSpec(label="w", depth=depth, time_ms=time_ms, nnue=nnue)
    _state["spec"] = spec
    _state["e1"] = make_engine(spec)
    _state["e2"] = make_engine(spec)


def _apply(engine, weights):
    for k, v in weights.items():
        setattr(engine, k, int(v))
    engine.full_reset()  # invalidate eval cache after weight change


def _play_tuning_pair(args):
    """One opening, both colors, theta+ (e1) vs theta- (e2). Returns e1 score in [0,2]."""
    seed, opening_plies, w_plus, w_minus = args
    spec = _state["spec"]
    e1, e2 = _state["e1"], _state["e2"]
    _apply(e1, w_plus)
    _apply(e2, w_minus)

    total = 0.0
    out, _, _ = play_game(e1, spec, e2, spec, seed, opening_plies)
    total += {1: 1.0, 0: 0.5, -1: 0.0}[out]
    out, _, _ = play_game(e2, spec, e1, spec, seed, opening_plies)
    total += {1: 0.0, 0: 0.5, -1: 1.0}[out]
    return total


def main():
    ap = argparse.ArgumentParser(description="SPSA eval-weight tuning")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=16, help="Opening pairs per iteration")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--time-ms", type=int, default=0)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--a", type=float, default=2.0, help="Learning-rate numerator")
    ap.add_argument("--c", type=float, default=1.0, help="Perturbation multiplier on step scales")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="scripts/results/spsa_tune.json")
    ap.add_argument("--resume", type=str, default="", help="Resume theta from a previous out-file")
    ap.add_argument("--nnue", type=str, default="", help="NNUE weights loaded into both engines")
    ap.add_argument("--params", choices=("eval", "search"), default="eval",
                    help="eval = heuristic weights; search = pruning knobs")
    args = ap.parse_args()

    global PARAMS
    PARAMS = SEARCH_PARAMS if args.params == "search" else EVAL_PARAMS

    rng = random.Random(args.seed)
    theta = {k: float(v[0]) for k, v in PARAMS.items()}
    history = []
    if args.resume and Path(args.resume).exists():
        prev = json.loads(Path(args.resume).read_text())
        theta.update({k: float(v) for k, v in prev["theta"].items()})
        history = prev.get("history", [])
        print(f"Resumed theta from {args.resume} ({len(history)} prior iterations)")

    alpha, gamma, A = 0.602, 0.101, max(args.iterations // 10, 1)

    print(f"SPSA: {args.iterations} iters x {args.pairs} pairs "
          f"({'%dms' % args.time_ms if args.time_ms else 'depth %d' % args.depth}, "
          f"{args.workers} workers)")
    print(f"  theta0: { {k: int(v) for k, v in theta.items()} }\n", flush=True)

    t0 = time.time()
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_worker_init, initargs=(args.depth, args.time_ms, args.nnue)
    ) as pool:
        for k in range(1 + len(history), args.iterations + 1 + len(history)):
            ck = args.c / (k ** gamma)
            ak = args.a / ((A + k) ** alpha)

            delta = {name: rng.choice((-1, 1)) for name in PARAMS}

            def perturbed(sign):
                w = {}
                for name, (_, scale, lo, hi) in PARAMS.items():
                    v = theta[name] + sign * ck * scale * delta[name]
                    w[name] = int(round(min(max(v, lo), hi)))
                return w

            w_plus, w_minus = perturbed(+1), perturbed(-1)
            tasks = [(rng.randrange(1 << 30), args.opening_plies, w_plus, w_minus)
                     for _ in range(args.pairs)]
            scores = pool.map(_play_tuning_pair, tasks)
            score_plus = sum(scores) / (2.0 * args.pairs)  # in [0,1]

            for name, (_, scale, lo, hi) in PARAMS.items():
                theta[name] += ak * (score_plus - 0.5) * delta[name] * scale
                theta[name] = min(max(theta[name], lo), hi)

            history.append({"iter": k, "score_plus": score_plus,
                            "theta": {n: round(v, 2) for n, v in theta.items()}})
            if k % 5 == 0 or k == 1:
                el = time.time() - t0
                print(f"  [{k}] score+ {score_plus:.3f}  "
                      f"theta { {n: int(v) for n, v in theta.items()} }  ({el:.0f}s)",
                      flush=True)
            if k % 10 == 0:
                Path(args.out).write_text(json.dumps(
                    {"theta": {n: int(round(v)) for n, v in theta.items()},
                     "history": history, "params": {n: p for n, p in PARAMS.items()},
                     "args": vars(args)}, indent=1))

    final = {n: int(round(v)) for n, v in theta.items()}
    Path(args.out).write_text(json.dumps(
        {"theta": final, "history": history,
         "params": {n: p for n, p in PARAMS.items()}, "args": vars(args)}, indent=1))

    print(f"\nFinal theta: {final}")
    print(f"Saved to {args.out}")
    print("\nValidate with:")
    sets = " ".join(f"--set-a {n}={v}" for n, v in final.items())
    tc = (f"--time-a {args.time_ms} --time-b {args.time_ms}" if args.time_ms
          else "--depth-a 7 --depth-b 7")
    nn = f" --nnue-a {args.nnue} --nnue-b {args.nnue}" if args.nnue else ""
    print(f"  python scripts/match.py {tc}{nn} --games 600 {sets}")


if __name__ == "__main__":
    main()
