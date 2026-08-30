#!/bin/bash
# Loop turn: round-8 tournament (28-input nets vs a net-3 re-run control, gen-1..3,
# gated vs net-3) -> full ladder on the winner. WORKERS=4 while gen-4 generates.
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -x
python scripts/nnue_tournament.py --preset round8 \
  --data training_gen1_d8.bin,training_gen1b_d8.bin,training_gen2_d8.bin,training_gen3_d8.bin \
  --out runs/gen123_r8 --epochs 100 --games 600 --time 100 --workers "${WORKERS:-10}" \
  --opponent-nnue models/net3_plain_m_d20_64x16_b25_l05.json
WINNER=$(python3 -c "
import re
rows=[l for l in open('runs/gen123_r8/leaderboard.md') if l.startswith('| 1 ')]
print(re.search(r'\`([^\`]+)\`', rows[0]).group(1))")
echo "round-8 winner: $WINNER"
python scripts/ladder.py --candidate "runs/gen123_r8/$WINNER/nnue_weights.json" --label "net4_$WINNER" --time 100 --games 600 --workers "${WORKERS:-10}"
