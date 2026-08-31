#!/usr/bin/env python3
"""Time-control scaling: one engine config at doubling time controls.

Round-robin between identical engines (net-3 + tablebases, the deployed
config) at 50/100/200/400/800 ms per move; ML-Elo fit anchored at
100 ms = 1500. Shows what a doubling of thinking time buys, and whether it
saturates. Games accumulate in scripts/results/tc_scaling_games.json.

    python scripts/tc_scaling.py --games 200 --workers 8
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

from match import EngineSpec  # noqa: E402
from rating_tournament import fit_elo, _worker_init, _play_pair  # noqa: E402
import rating_tournament  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "scripts" / "results" / "tc_scaling_games.json"
OUT_MD = ROOT / "TIME_SCALING.md"
NNUE = "models/net3_plain_m_d20_64x16_b25_l05.json"
TB = "tablebases"
TCS = [50, 100, 200, 400, 800]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="games per pair (even)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--opening-plies", type=int, default=6)
    args = ap.parse_args()

    labels = [f"net3-tb-{tc}ms" for tc in TCS]
    specs_json = {f"net3-tb-{tc}ms": vars(EngineSpec(label=f"net3-tb-{tc}ms", time_ms=tc,
                                                     nnue=NNUE, tablebases=TB)) for tc in TCS}
    # anchor the fit at 100 ms
    rating_tournament.ANCHOR = "net3-tb-100ms"

    store = json.loads(STORE.read_text()) if STORE.exists() else {"pairs": {}}
    todo = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            key = f"{a}|{b}"
            have = store["pairs"].get(key, {"score_a": 0.0, "games": 0})
            m = (args.games - have["games"]) // 2
            if m > 0:
                todo.append((a, b, key, m))
    print(f"{len(labels)} time controls, {len(todo)} pairs, "
          f"{sum(2*m for *_, m in todo)} games", flush=True)
    rng = random.Random(1234)
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
            print(f"  {a} vs {b}: {sum(scores):.1f}/{2*m}  [{time.time()-t0:.0f}s]", flush=True)

    pair_scores = {tuple(k.split("|")): (v["score_a"], v["games"]) for k, v in store["pairs"].items()}
    ratings = fit_elo(pair_scores)
    lines = [
        "# Time-control scaling — net-3 + tablebases",
        "",
        f"Same engine at doubling time controls; ML-Elo over "
        f"{sum(n for _, n in pair_scores.values())} round-robin games, anchor 100 ms = 1500.",
        "",
        "| time/move | Elo | Δ per doubling |",
        "|---|---|---|",
    ]
    ordered = [(tc, ratings[f"net3-tb-{tc}ms"]) for tc in TCS]
    prev = None
    for tc, e in ordered:
        delta = f"{e - prev:+.0f}" if prev is not None else "—"
        lines.append(f"| {tc} ms | **{e:.0f}** | {delta} |")
        prev = e
    lines.append("")
    lines.append(f"_Updated {time.strftime('%Y-%m-%d %H:%M')}._")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
