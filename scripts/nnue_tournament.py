#!/usr/bin/env python3
"""NNUE architecture tournament: train candidates on one dataset, gate each at
equal time against the heuristic engine, and print a leaderboard.

Training (GPU) of candidate N+1 overlaps with the match (CPU) of candidate N;
matches run one at a time so they never oversubscribe the CPU budget.

Usage:
    python scripts/nnue_tournament.py --data training_gen1_d8.bin --out runs/gen1 \
        --epochs 20 --games 400 --time 100 --workers 4
    python scripts/nnue_tournament.py ... --only plain_m_d0_256x32,halfpail_d20_128x32
    python scripts/nnue_tournament.py ... --skip-train      # re-run matches only
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Candidate:
    feature_set: str
    mirror: bool
    dense: int
    h1: int
    h2: int
    buckets: int
    lam: float = 0.8
    dedupe: bool = True

    @property
    def tag(self) -> str:
        t = (f"{self.feature_set}{'_m' if self.mirror else ''}_d{self.dense}"
             f"_{self.h1}x{self.h2}{'_b25' if self.buckets == 25 else ''}")
        if self.lam != 0.8:
            t += f"_l{self.lam:g}"
        if not self.dedupe:
            t += "_nodedupe"
        return t


# Ordered so the most informative comparisons resolve first.
DEFAULT_CANDIDATES = [
    Candidate("halfpail", False, 20, 128, 32, 1),        # incumbent architecture
    Candidate("plain", True, 0, 128, 32, 1),             # "dumb dense" + mirror baseline
    Candidate("halfpail", True, 20, 128, 32, 1),         # incumbent + mirror
    Candidate("plain", True, 20, 128, 32, 1),            # does the net want dense hints?
    Candidate("plain", True, 0, 256, 32, 1),             # width
    Candidate("plain", True, 0, 256, 32, 25),            # scored-count output buckets
    Candidate("halfpail", True, 20, 256, 32, 25),        # rich variant
    Candidate("plain", True, 0, 512, 32, 25),            # big
    Candidate("plain", True, 0, 256, 32, 1, lam=0.5),    # label blend sweep
    Candidate("plain", True, 0, 256, 32, 1, lam=1.0),
    Candidate("plain", True, 0, 256, 32, 1, lam=0.2),
    Candidate("plain", True, 0, 128, 32, 1, dedupe=False),  # dedupe A/B vs candidate 2
]


def train(c: Candidate, data: str, out_dir: Path, epochs: int, lr: float, batch: int, log) -> float:
    run_dir = out_dir / c.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "tonnesjakk.nnue", "--load-data", data, "--output", str(run_dir),
           "--feature-set", c.feature_set, "--arch", str(c.h1), str(c.h2),
           "--output-buckets", str(c.buckets), "--epochs", str(epochs),
           "--lr", str(lr), "--batch-size", str(batch), "--lambda", str(c.lam)]
    if c.mirror:
        cmd.append("--mirror")
    if c.dense == 0:
        cmd.append("--no-dense")
    if c.dedupe:
        cmd.append("--dedupe")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (run_dir / "train.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {c.tag}; see {run_dir / 'train.log'}")
    m = re.search(r"Best validation loss: ([0-9.]+)", proc.stdout)
    val = float(m.group(1)) if m else float("nan")
    log(f"  trained {c.tag}: val_loss {val:.4f} ({time.time() - t0:.0f}s)")
    return val


def start_match(c: Candidate, out_dir: Path, games: int, time_ms: int, workers: int):
    run_dir = out_dir / c.tag
    cmd = [PY, "scripts/match.py", "--time-a", str(time_ms), "--time-b", str(time_ms),
           "--nnue-a", str(run_dir / "nnue_weights.json"),
           "--label-a", c.tag, "--label-b", "heuristic",
           "--games", str(games), "--workers", str(workers),
           "--out", str(run_dir / "match_vs_heuristic.json")]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(run_dir / "match.log", "w"),
                            stderr=subprocess.STDOUT)


def leaderboard(out_dir: Path, candidates, val_losses):
    rows = []
    for c in candidates:
        f = out_dir / c.tag / "match_vs_heuristic.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        rows.append((s["elo_a"], c.tag, s, val_losses.get(c.tag, float("nan"))))
    rows.sort(key=lambda r: -r[0])
    lines = ["| # | architecture | Elo vs heuristic (95% CI) | W-D-L | val loss |",
             "|---|---|---|---|---|"]
    for i, (elo, tag, s, val) in enumerate(rows, 1):
        lines.append(f"| {i} | `{tag}` | {elo:+.0f} [{s['elo_ci95'][0]:+.0f}, {s['elo_ci95'][1]:+.0f}] "
                     f"| {s['wins_a']}-{s['draws']}-{s['losses_a']} | {val:.4f} |")
    text = "\n".join(lines)
    (out_dir / "leaderboard.md").write_text(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description="NNUE architecture tournament")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/gen1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--time", type=int, default=100, help="ms per move for the equal-time gate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=str, default="", help="comma-separated candidate tags")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-match", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = DEFAULT_CANDIDATES
    if args.only:
        wanted = set(args.only.split(","))
        cands = [c for c in cands if c.tag in wanted]

    log_f = open(out_dir / "tournament.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"Tournament: {len(cands)} candidates, data={args.data}, epochs={args.epochs}, "
        f"gate={args.games} games @ {args.time}ms, {args.workers} workers")

    val_losses = {}
    vl_path = out_dir / "val_losses.json"
    if vl_path.exists():
        val_losses.update(json.loads(vl_path.read_text()))

    pending_match = None
    for c in cands:
        if not args.skip_train:
            val_losses[c.tag] = train(c, args.data, out_dir, args.epochs, args.lr, args.batch_size, log)
            vl_path.write_text(json.dumps(val_losses, indent=1))
        if args.skip_match:
            continue
        if pending_match is not None:
            pc, proc = pending_match
            proc.wait()
            log(f"  match done {pc.tag}")
            log("\n" + leaderboard(out_dir, cands, val_losses))
        pending_match = (c, start_match(c, out_dir, args.games, args.time, args.workers))
        log(f"  match started {c.tag}")

    if pending_match is not None:
        pc, proc = pending_match
        proc.wait()
        log(f"  match done {pc.tag}")

    log("\nFINAL LEADERBOARD\n" + leaderboard(out_dir, cands, val_losses))


if __name__ == "__main__":
    main()
