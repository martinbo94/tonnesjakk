#!/bin/bash
# A/B: does probing the solved endgame phases in search gain Elo?
# Same net both sides; only side A loads tablebases/.
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -x
NNUE=${NNUE:-models/net3_plain_m_d20_64x16_b25_l05.json}
W=${WORKERS:-10}
for tc_games in "100 600" "200 400"; do
  set -- $tc_games
  python scripts/match.py --time-a "$1" --time-b "$1" --nnue-a "$NNUE" --nnue-b "$NNUE" \
    --tb-a tablebases --label-a "net3+tb" --label-b "net3" \
    --games "$2" --workers "$W" --out "scripts/results/tb_ab_${1}ms.json"
done
