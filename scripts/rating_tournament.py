#!/usr/bin/env python3
"""Cross-generation rating list: one Elo scale for every engine era.

Plays a round-robin between frozen, reproducible configurations (paired
random openings, both colours) and fits maximum-likelihood Elo
(Bradley-Terry, draws = half a win), anchored at ANCHOR = 1500.

Results accumulate in scripts/results/rating_games.json: re-running plays
only the pairs that are missing games, so new players (a future net-4, new
search eras) can be appended cheaply and the whole list refit.

    python scripts/rating_tournament.py --games 200 --workers 8
    python scripts/rating_tournament.py --refit-only     # just recompute Elo
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match import EngineSpec, make_engine, play_game  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "scripts" / "results" / "rating_games.json"
OUT_MD = ROOT / "RATINGS.md"
ANCHOR = "heur-100ms"
ANCHOR_ELO = 1500.0

# Search constants as they were before the 2026-08-27 SPSA re-tune (the "was"
# values documented in BitBoardEngine::new). Applied via knobs, so old eras
# are reproduced under the current binary (root-search bug fixes included —
# code-level history is not reproducible via knobs and is not attempted).
OLD_SEARCH = {
    "asp_delta": 30, "razor_base": 200, "razor_slope": 150, "nmp_margin": 50,
    "nmp_boost_margin": 150, "fut_scale": 100, "lmr_div": 100,
    "lmr_hist_good": 1000, "lmr_hist_bad": -500, "lmp_base": 6,
    "rfp_margin": 0, "iir_depth": 4,
}

NET1 = "models/ladder/net1_gen1_128x32_l05.json"
NET1B = "models/ladder/net1b_gen1ab_96x16_l05.json"
NET2 = "models/ladder/net2_gen12_96x16_b25_l05.json"
NET3 = "models/ladder/net3_gen123_64x16_b25_l05.json"
TB = "tablebases"


@dataclass
class PlayerDef:
    label: str
    depth: int = 0
    time_ms: int = 100
    nnue: str = ""
    tablebases: str = ""
    weights: dict = field(default_factory=dict)

    def spec(self) -> EngineSpec:
        return EngineSpec(label=self.label, depth=self.depth, time_ms=self.time_ms,
                          nnue=self.nnue, tablebases=self.tablebases,
                          weights=dict(self.weights))


PLAYERS = [
    PlayerDef("heur-d4", depth=4, time_ms=0),                       # the old AlphaZero-era yardstick
    PlayerDef("heur-100ms-oldsearch", weights=OLD_SEARCH),          # heuristic, pre-retune search
    PlayerDef("heur-100ms"),                                        # heuristic, current search (ANCHOR = 1500)
    PlayerDef("net1-100ms", nnue=NET1),                             # first promoted net (2026-08-25)
    PlayerDef("net1b-100ms", nnue=NET1B),
    PlayerDef("net2-100ms", nnue=NET2),
    PlayerDef("net3-100ms-oldsearch", nnue=NET3, weights=OLD_SEARCH),
    PlayerDef("net3-100ms", nnue=NET3),
    PlayerDef("net3-tb-100ms", nnue=NET3, tablebases=TB),           # the deployed engine
]

_state = {}


def _worker_init(specs_json):
    specs = {}
    engines = {}
    for label, d in specs_json.items():
        spec = EngineSpec(**d)
        specs[label] = spec
        engines[label] = make_engine(spec)
    _state["specs"] = specs
    _state["engines"] = engines


def _play_pair(args):
    """One opening played both ways between a and b. Returns a's score in [0, 2]."""
    a, b, seed, opening_plies = args
    ea, eb = _state["engines"][a], _state["engines"][b]
    sa, sb = _state["specs"][a], _state["specs"][b]
    total = 0.0
    out, _, _ = play_game(ea, sa, eb, sb, seed, opening_plies)
    total += {1: 1.0, 0: 0.5, -1: 0.0}[out]
    out, _, _ = play_game(eb, sb, ea, sa, seed, opening_plies)
    total += {1: 0.0, 0: 0.5, -1: 1.0}[out]
    return total


def fit_elo(pair_scores):
    """Maximum-likelihood Elo from pair totals {(a,b): (score_a, games)}.
    Draws entered as half wins; gradient ascent on the logistic likelihood."""
    players = sorted({p for ab in pair_scores for p in ab})
    idx = {p: i for i, p in enumerate(players)}
    r = [0.0] * len(players)
    lr = 30.0
    for _ in range(20000):
        grad = [0.0] * len(players)
        for (a, b), (sa, n) in pair_scores.items():
            ea = 1.0 / (1.0 + 10 ** ((r[idx[b]] - r[idx[a]]) / 400.0))
            g = sa - n * ea
            grad[idx[a]] += g
            grad[idx[b]] -= g
        step = max(abs(g) for g in grad)
        for i in range(len(players)):
            r[i] += lr * grad[i] / max(sum(n for (_, n) in pair_scores.values()), 1)
        if step < 1e-6:
            break
    shift = ANCHOR_ELO - r[idx[ANCHOR]]
    return {p: r[idx[p]] + shift for p in players}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="games per pair (must be even)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--refit-only", action="store_true")
    args = ap.parse_args()

    store = json.loads(STORE.read_text()) if STORE.exists() else {"pairs": {}}
    labels = [p.label for p in PLAYERS]
    specs_json = {p.label: vars(p.spec()) for p in PLAYERS}

    if not args.refit_only:
        todo = []
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                key = f"{a}|{b}"
                have = store["pairs"].get(key, {"score_a": 0.0, "games": 0})
                missing_pairs = (args.games - have["games"]) // 2
                if missing_pairs > 0:
                    todo.append((a, b, key, missing_pairs))
        total_games = sum(2 * m for *_, m in todo)
        print(f"{len(labels)} players, {len(todo)} pairs to play, {total_games} games", flush=True)
        rng = random.Random(20260831)
        with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                          initargs=(specs_json,)) as pool:
            for a, b, key, m in todo:
                t0 = time.time()
                tasks = [(a, b, rng.randrange(1 << 30), args.opening_plies) for _ in range(m)]
                scores = pool.map(_play_pair, tasks)
                have = store["pairs"].setdefault(key, {"score_a": 0.0, "games": 0})
                have["score_a"] += sum(scores)
                have["games"] += 2 * m
                STORE.parent.mkdir(parents=True, exist_ok=True)
                STORE.write_text(json.dumps(store, indent=1))
                print(f"  {a} vs {b}: {sum(scores):.1f}/{2*m} this run "
                      f"(total {have['score_a']:.1f}/{have['games']})  [{time.time()-t0:.0f}s]", flush=True)

    pair_scores = {tuple(k.split("|")): (v["score_a"], v["games"]) for k, v in store["pairs"].items()}
    ratings = fit_elo(pair_scores)
    games_of = {p: 0 for p in ratings}
    for (a, b), (_, n) in pair_scores.items():
        games_of[a] += n
        games_of[b] += n

    lines = [
        "# Tønnesjakk rating list",
        "",
        f"Maximum-likelihood Elo over {sum(n for _, n in pair_scores.values())} round-robin games",
        f"(paired openings, both colours; draws = ½). Anchor: `{ANCHOR}` = {ANCHOR_ELO:.0f}.",
        "All configurations run under the current binary — code-era differences",
        "(e.g. pre-2026-08-27 root-search bugs) are not reproduced, only knob/net/TB eras.",
        "",
        "| # | player | Elo | games |",
        "|---|---|---|---|",
    ]
    for i, (p, e) in enumerate(sorted(ratings.items(), key=lambda kv: -kv[1]), 1):
        lines.append(f"| {i} | `{p}` | **{e:.0f}** | {games_of[p]} |")
    lines.append("")
    lines.append(f"_Updated {time.strftime('%Y-%m-%d %H:%M')}. Extend: add a PlayerDef and re-run "
                 "`python scripts/rating_tournament.py` — only missing pairs are played._")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
