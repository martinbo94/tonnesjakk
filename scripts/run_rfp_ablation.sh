#!/bin/bash
# Which knob carried the +60? Both sides pinned to the PRE-tune values so the
# result does not depend on the compiled defaults; A additionally turns on
# reverse futility pruning at the tuned margin.
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -x
NNUE=${NNUE:-models/net3_plain_m_d20_64x16_b25_l05.json}
W=${WORKERS:-10}
OLD="asp_delta=30 razor_base=200 razor_slope=150 nmp_margin=50 nmp_boost_margin=150 fut_scale=100 lmr_div=100 lmr_hist_good=1000 lmr_hist_bad=-500 lmp_base=6 rfp_margin=0 iir_depth=4"
A=(); B=()
for kv in $OLD; do A+=(--set-a "$kv"); B+=(--set-b "$kv"); done
A+=(--set-a rfp_margin=63)
python scripts/match.py --time-a 100 --time-b 100 --nnue-a "$NNUE" --nnue-b "$NNUE" \
  --label-a "old+rfp63" --label-b "old" --games 600 --workers "$W" \
  --out scripts/results/rfp_ablation_100ms.json "${A[@]}" "${B[@]}"
