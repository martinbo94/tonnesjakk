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
    loss: str = "wdl-ce"
    fraction: float = 1.0
    max_scored: int = -1   # -1 = all phases

    @property
    def tag(self) -> str:
        t = (f"{self.feature_set}{'_m' if self.mirror else ''}_d{self.dense}"
             f"_{self.h1}x{self.h2}{'_b25' if self.buckets == 25 else ''}")
        if self.lam != 0.8:
            t += f"_l{self.lam:g}"
        if not self.dedupe:
            t += "_nodedupe"
        if self.loss != "wdl-ce":
            t += f"_{self.loss}"
        if self.fraction < 1.0:
            t += f"_f{self.fraction:g}"
        if self.max_scored >= 0:
            t += f"_ms{self.max_scored}"
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
    Candidate("plain", True, 0, 128, 32, 1, loss="mse"),    # loss A/B vs candidate 2
]

# Round 2: variations of the round-1 winner (plain + mirror + dense, 128x32).
ROUND2_CANDIDATES = [
    Candidate("plain", True, 20, 128, 32, 1),                 # winner (re-run = noise floor)
    Candidate("plain", True, 20, 128, 32, 25),                # + scored-count buckets
    Candidate("plain", True, 20, 128, 32, 1, lam=0.5),        # label blend sweep
    Candidate("plain", True, 20, 128, 32, 1, lam=1.0),
    Candidate("plain", True, 20, 128, 32, 1, lam=0.65),
    Candidate("plain", True, 20, 128, 32, 1, loss="mse"),     # classic NNUE loss
    Candidate("plain", True, 20, 128, 32, 1, dedupe=False),   # dedupe A/B
    Candidate("plain", True, 20, 64, 32, 1),                  # smaller
    Candidate("plain", True, 20, 192, 32, 1),                 # slightly wider
    Candidate("plain", True, 20, 128, 64, 1),                 # wider second layer
]

# Round 3: gen-1 + gen-1b combined. Data-size curve, width re-test, lambda check.
ROUND3_CANDIDATES = [
    Candidate("plain", True, 20, 128, 32, 1, lam=0.5),                 # net-1 config, 100% data
    Candidate("plain", True, 20, 128, 32, 1, lam=0.5, fraction=0.5),   # data-size curve
    Candidate("plain", True, 20, 128, 32, 1, lam=0.5, fraction=0.25),
    Candidate("plain", True, 20, 256, 32, 1, lam=0.5),                 # width re-test on more data
    Candidate("plain", True, 20, 512, 32, 1, lam=0.5),
    Candidate("plain", True, 20, 256, 64, 1, lam=0.5),
    Candidate("plain", True, 20, 128, 32, 1),                          # lambda 0.8 control
    Candidate("halfpail", True, 20, 256, 32, 1, lam=0.5),              # buckets with more data
]

# Round 4: SPEED. Eval quality plateaued at fixed time (4x data = +16 Elo, ns;
# 512-wide = same loss, -55 Elo). Smaller nets search deeper. Gate vs net-1.
ROUND4_CANDIDATES = [
    Candidate("plain", True, 20, 64, 32, 1, lam=0.5),
    Candidate("plain", True, 20, 64, 16, 1, lam=0.5),
    Candidate("plain", True, 20, 96, 16, 1, lam=0.5),
    Candidate("plain", True, 20, 128, 16, 1, lam=0.5),
    Candidate("plain", True, 20, 32, 16, 1, lam=0.5),
    Candidate("plain", True, 0, 64, 16, 25, lam=0.5),   # no dense compute, buckets carry phase
]

# Round 5: net-2 candidates on gen-1 + gen-1b + gen-2 (gen-2 labeled by net-1b).
# Gate vs net-1b (--opponent-nnue models/net1b_*.json); the winner must ALSO
# beat the heuristic and hold at 200ms before promotion.
ROUND5_CANDIDATES = [
    Candidate("plain", True, 20, 96, 16, 1, lam=0.5),    # net-1b config, more data
    Candidate("plain", True, 20, 64, 16, 1, lam=0.5),
    Candidate("plain", True, 20, 128, 16, 1, lam=0.5),
    Candidate("plain", True, 20, 96, 16, 1, lam=0.65),
    Candidate("plain", True, 20, 96, 16, 25, lam=0.5),   # buckets with 3x data
]

# Round 6: net-3 candidates on gen-1..3 (gen-3 labeled by net-2 + tablebases).
# Gate vs net-2 (--opponent-nnue models/net2_*.json), then the full ladder.
ROUND6_CANDIDATES = [
    Candidate("plain", True, 20, 96, 16, 25, lam=0.5),    # net-2 config, more data
    Candidate("plain", True, 20, 128, 16, 25, lam=0.5),
    Candidate("plain", True, 20, 64, 16, 25, lam=0.5),
    Candidate("plain", True, 20, 96, 16, 25, lam=0.65),
    Candidate("plain", True, 20, 128, 32, 25, lam=0.5),   # wider FC2 now that data is 3x
]

# Round 7: round 6 said "smaller won, data/width flat" — walk the speed curve
# down and vary what the labels weight. Same gen-1..3 data. Gate vs net-3.
ROUND7_CANDIDATES = [
    Candidate("plain", True, 20, 48, 16, 25, lam=0.5),
    Candidate("plain", True, 20, 32, 16, 25, lam=0.5),
    Candidate("plain", True, 20, 64, 16, 25, lam=0.35),   # more outcome weight (6% draws, TB labels)
    Candidate("plain", True, 0, 64, 16, 25, lam=0.5),     # drop the dense 20 (cheaper eval)
    Candidate("plain", True, 20, 64, 16, 25, lam=0.5),    # net-3 config re-run = seed noise floor
]

# Round 8: INPUTS. Round 7's only significant result was -28 without the 20
# engineered features, so richer inputs, not size: dense 28 = the 20 + race
# distances, forward-jump availability, forward mobility, no-progress clock,
# mid-turn flag (computed at decode time; no data regeneration). Gate vs net-3.
ROUND8_CANDIDATES = [
    Candidate("plain", True, 28, 64, 16, 25, lam=0.5),
    Candidate("plain", True, 28, 48, 16, 25, lam=0.5),
    Candidate("plain", True, 28, 96, 16, 25, lam=0.5),
    Candidate("plain", True, 20, 64, 16, 25, lam=0.5),    # net-3 config re-run = seed noise control
]

# Round 9: round-8 candidates on gen-1..4 (gen-4 labelled by net-3 + search
# re-tune + tablebases through 3v3). Gate vs net-3.
ROUND9_CANDIDATES = [
    Candidate("plain", True, 28, 96, 16, 25, lam=0.5),
    Candidate("plain", True, 28, 64, 16, 25, lam=0.5),
    Candidate("plain", True, 20, 64, 16, 25, lam=0.5),    # net-3 config on 4 gens = data-only effect
]

# Round 10: PHASE FILTER. Gen-4 made round 9 worse (d28 96x16: +11 -> -19 vs
# net-3): 36% of its rows are exact +-1 tablebase labels on positions the
# engine never evaluates through the net (tablebases answer <= 6 remaining).
# Train only on >= 7 remaining (max_scored=1). Gate vs net-3.
ROUND10_CANDIDATES = [
    Candidate("plain", True, 28, 96, 16, 25, lam=0.5, max_scored=1),
    Candidate("plain", True, 28, 64, 16, 25, lam=0.5, max_scored=1),
    Candidate("plain", True, 20, 64, 16, 25, lam=0.5, max_scored=1),   # net-3 config, filtered
]

PRESETS = {"round1": DEFAULT_CANDIDATES, "round2": ROUND2_CANDIDATES,
           "round3": ROUND3_CANDIDATES, "round4": ROUND4_CANDIDATES,
           "round5": ROUND5_CANDIDATES, "round6": ROUND6_CANDIDATES,
           "round7": ROUND7_CANDIDATES, "round8": ROUND8_CANDIDATES,
           "round9": ROUND9_CANDIDATES,
           "round10": ROUND10_CANDIDATES}


def train(c: Candidate, data: str, out_dir: Path, epochs: int, lr: float, batch: int, log) -> float:
    run_dir = out_dir / c.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "tonnesjakk.nnue", "--load-data", data, "--output", str(run_dir),
           "--feature-set", c.feature_set, "--arch", str(c.h1), str(c.h2),
           "--output-buckets", str(c.buckets), "--epochs", str(epochs),
           "--lr", str(lr), "--batch-size", str(batch), "--lambda", str(c.lam),
           "--loss", c.loss, "--data-fraction", str(c.fraction)]
    if c.mirror:
        cmd.append("--mirror")
    if c.dense != 20:
        cmd += ["--dense-size", str(c.dense)]
    if c.dedupe:
        cmd.append("--dedupe")
    if c.max_scored >= 0:
        cmd += ["--max-scored", str(c.max_scored)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (run_dir / "train.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {c.tag}; see {run_dir / 'train.log'}")
    m = re.search(r"Best validation loss: ([0-9.]+)", proc.stdout)
    val = float(m.group(1)) if m else float("nan")
    log(f"  trained {c.tag}: val_loss {val:.4f} ({time.time() - t0:.0f}s)")
    return val


def start_match(c: Candidate, out_dir: Path, games: int, time_ms: int, workers: int,
                opponent_nnue: str = "", tb: str = ""):
    """Gate vs the heuristic engine, or vs a reference net (--opponent-nnue):
    once every candidate beats the heuristic by ~200 Elo, a fixed weaker
    opponent no longer resolves differences between nets."""
    run_dir = out_dir / c.tag
    cmd = [PY, "scripts/match.py", "--time-a", str(time_ms), "--time-b", str(time_ms),
           "--nnue-a", str(run_dir / "nnue_weights.json"),
           "--label-a", c.tag, "--label-b", "reference" if opponent_nnue else "heuristic",
           "--games", str(games), "--workers", str(workers),
           "--out", str(run_dir / "match.json")]
    if opponent_nnue:
        cmd += ["--nnue-b", opponent_nnue]
    if tb:
        # Both sides probe the tablebases: required for nets trained only on
        # the phases the tables do not cover (--max-scored), and the real
        # deployment for every net since 2026-08-30.
        cmd += ["--tb-a", tb, "--tb-b", tb]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(run_dir / "match.log", "w"),
                            stderr=subprocess.STDOUT)


def leaderboard(out_dir: Path, candidates, val_losses, opponent_label="heuristic"):
    rows = []
    for c in candidates:
        f = out_dir / c.tag / "match.json"
        if not f.exists():
            f = out_dir / c.tag / "match_vs_heuristic.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        rows.append((s["elo_a"], c.tag, s, val_losses.get(c.tag, float("nan"))))
    rows.sort(key=lambda r: -r[0])
    lines = [f"| # | architecture | Elo vs {opponent_label} (95% CI) | W-D-L | val loss |",
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
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--time", type=int, default=100, help="ms per move for the equal-time gate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=str, default="", help="comma-separated candidate tags")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="round1")
    ap.add_argument("--opponent-nnue", type=str, default="",
                    help="Gate vs this reference net instead of the heuristic engine")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--reuse-trained", action="store_true", help="skip training for candidates whose weights already exist")
    ap.add_argument("--tb", type=str, default="", help="tablebase dir loaded by BOTH sides of every gate")
    ap.add_argument("--skip-match", action="store_true")
    args = ap.parse_args()
    opp_label = "reference net" if args.opponent_nnue else "heuristic"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = PRESETS[args.preset]
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
        already = (out_dir / c.tag / "nnue_weights.json").exists()
        if not args.skip_train and not (args.reuse_trained and already):
            val_losses[c.tag] = train(c, args.data, out_dir, args.epochs, args.lr, args.batch_size, log)
            vl_path.write_text(json.dumps(val_losses, indent=1))
        if args.skip_match:
            continue
        if pending_match is not None:
            pc, proc = pending_match
            proc.wait()
            log(f"  match done {pc.tag}")
            log("\n" + leaderboard(out_dir, cands, val_losses, opp_label))
        pending_match = (c, start_match(c, out_dir, args.games, args.time, args.workers,
                                        args.opponent_nnue, args.tb))
        log(f"  match started {c.tag}")

    if pending_match is not None:
        pc, proc = pending_match
        proc.wait()
        log(f"  match done {pc.tag}")

    log("\nFINAL LEADERBOARD\n" + leaderboard(out_dir, cands, val_losses, opp_label))


if __name__ == "__main__":
    main()
