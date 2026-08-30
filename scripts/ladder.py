#!/usr/bin/env python3
"""Strength ladder: play a candidate against every frozen reference engine.

Why: strength is not transitive and a net trained on its parent's games can
exploit the parent's blind spots without being generally stronger. Gating
only against the previous version (or only against the heuristic, which has
lost resolution) can promote such a net. Every promotion should show a
monotone ladder: beats the heuristic, beats every previous rung.

Ladder rungs live in models/ladder/*.json (frozen; never retrained). The
heuristic engine is always rung 0.

Usage:
    # candidate vs every rung, equal time
    python scripts/ladder.py --candidate runs/x/nnue_weights.json --time 100 --games 300 --workers 10
    # full round-robin among rungs (+ candidate if given)
    python scripts/ladder.py --round-robin --time 100 --games 200 --workers 10
"""

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ""  # set from --tb
LADDER_DIR = ROOT / "models" / "ladder"


def rungs():
    entries = [("heuristic", "")]
    for f in sorted(LADDER_DIR.glob("*.json")):
        entries.append((f.stem, str(f.relative_to(ROOT))))
    return entries


def play(a_label, a_nnue, b_label, b_nnue, time_ms, games, workers, out_dir):
    out = out_dir / f"{a_label}__vs__{b_label}.json"
    cmd = [PY, "scripts/match.py", "--time-a", str(time_ms), "--time-b", str(time_ms),
           "--label-a", a_label, "--label-b", b_label, "--games", str(games),
           "--workers", str(workers), "--out", str(out)]
    if a_nnue:
        cmd += ["--nnue-a", a_nnue]
    if b_nnue:
        cmd += ["--nnue-b", b_nnue]
    if TB_DIR:
        cmd += ["--tb-a", TB_DIR, "--tb-b", TB_DIR]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    s = json.loads(out.read_text())["summary"]
    return s


def fmt(s):
    return f"{s['elo_a']:+.0f} [{s['elo_ci95'][0]:+.0f},{s['elo_ci95'][1]:+.0f}] ({s['wins_a']}-{s['draws']}-{s['losses_a']})"


def main():
    ap = argparse.ArgumentParser(description="Strength ladder")
    ap.add_argument("--candidate", type=str, default="", help="NNUE JSON to test against every rung")
    ap.add_argument("--label", type=str, default="candidate")
    ap.add_argument("--round-robin", action="store_true", help="All rungs vs all rungs")
    ap.add_argument("--time", type=int, default=100)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=str, default="runs/ladder")
    ap.add_argument("--tb", type=str, default="", help="tablebase dir loaded by BOTH sides of every match")
    args = ap.parse_args()
    global TB_DIR
    TB_DIR = args.tb

    out_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    players = rungs()
    if args.candidate:
        players.append((args.label, args.candidate))

    print(f"Ladder rungs: {[p[0] for p in players]}  ({args.time}ms, {args.games} games each)\n", flush=True)
    results = {}
    if args.round_robin:
        pairs = list(combinations(players, 2))
    else:
        if not args.candidate:
            ap.error("give --candidate or --round-robin")
        cand = players[-1]
        pairs = [(cand, r) for r in players[:-1]]

    for (la, na), (lb, nb) in pairs:
        s = play(la, na, lb, nb, args.time, args.games, args.workers, out_dir)
        results[(la, lb)] = s
        print(f"  {la:>28} vs {lb:<28} {fmt(s)}", flush=True)

    # Matrix (row's Elo vs column)
    names = [p[0] for p in players]
    if args.round_robin:
        print("\nElo matrix (row vs column):")
        w = max(len(n) for n in names)
        print(" " * (w + 2) + "".join(f"{n[:12]:>14}" for n in names))
        for a in names:
            row = f"{a:>{w}}  "
            for b in names:
                if a == b:
                    row += f"{'—':>14}"
                elif (a, b) in results:
                    row += f"{results[(a, b)]['elo_a']:>+14.0f}"
                elif (b, a) in results:
                    row += f"{-results[(b, a)]['elo_a']:>+14.0f}"
            print(row)
        # Average score as a crude ranking
        pts = {n: [] for n in names}
        for (a, b), s in results.items():
            pts[a].append(s["score_a_pct"]); pts[b].append(100 - s["score_a_pct"])
        print("\nAverage score vs the field:")
        for n in sorted(names, key=lambda n: -sum(pts[n]) / max(len(pts[n]), 1)):
            print(f"  {n:>28}: {sum(pts[n]) / max(len(pts[n]), 1):5.1f}%")
    (out_dir / "results.json").write_text(json.dumps(
        {f"{a}__vs__{b}": s for (a, b), s in results.items()}, indent=1))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
