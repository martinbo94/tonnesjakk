#!/bin/bash
# One turn of the self-improvement loop after a new dataset lands, then tablebases.
#   round 6 tournament (gen-1..3, gated vs net-2) -> full ladder on the winner
#   -> solve 4v1 -> solve 3v3 (packed, resumable)
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -x
python scripts/nnue_tournament.py --preset round6 \
  --data training_gen1_d8.bin,training_gen1b_d8.bin,training_gen2_d8.bin,training_gen3_d8.bin \
  --out runs/gen123_r6 --epochs 100 --games 600 --time 100 --workers 10 \
  --opponent-nnue models/net2_plain_m_d20_96x16_b25_l05.json
WINNER=$(python3 -c "
import re
rows=[l for l in open('runs/gen123_r6/leaderboard.md') if l.startswith('| 1 ')]
print(re.search(r'\`([^\`]+)\`', rows[0]).group(1))")
echo "round-6 winner: $WINNER"
python scripts/ladder.py --candidate "runs/gen123_r6/$WINNER/nnue_weights.json" --label "net3_$WINNER" --time 100 --games 600 --workers 10
python3 -c "
import time
from tonnesjakk._core import solve_tablebase, solve_tablebase_packed
t0=time.time(); n,w,b,d = solve_tablebase('tablebases', 4, 1, True); v=w+b+d
print(f'4v1: {n:,} states ({v:,} valid) — white {100*w/v:.2f}%, black {100*b/v:.2f}%, draws {100*d/v:.3f}%  [{time.time()-t0:.0f}s]', flush=True)
t0=time.time(); n,w,b,d = solve_tablebase_packed('tablebases', 3, 5, True); v=w+b+d
print(f'3v3 (packed, white to move): {n:,} states ({v:,} valid) — white {100*w/v:.2f}%, black {100*b/v:.2f}%, draws {100*d/v:.3f}%  [{time.time()-t0:.0f}s]', flush=True)
"
