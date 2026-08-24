#!/bin/bash
source .venv/bin/activate
set -x
W=4
python scripts/match.py --depth-a 5 --depth-b 5 --label-a race40 --label-b base --set-a weight_race=40 --games 300 --workers $W --out scripts/results/ab_race40_d5.json
python scripts/match.py --depth-a 5 --depth-b 5 --label-a race80 --label-b base --set-a weight_race=80 --games 300 --workers $W --out scripts/results/ab_race80_d5.json
python scripts/match.py --time-a 50 --time-b 50 --label-a race60 --label-b base --set-a weight_race=60 --games 300 --workers $W --out scripts/results/ab_race60_t50.json
python scripts/match.py --time-a 50 --time-b 50 --label-a pailfilter --label-b base --set-a pail_filter=1 --games 300 --workers $W --out scripts/results/ab_pailfilter_t50.json
python scripts/match.py --time-a 50 --time-b 50 --label-a straggler6 --label-b base --set-a weight_straggler=6 --games 300 --workers $W --out scripts/results/ab_straggler_t50.json
