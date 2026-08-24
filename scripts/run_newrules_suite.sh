#!/bin/bash
source .venv/bin/activate
set -x
python scripts/match.py --depth-a 4 --depth-b 6 --label-a d4 --label-b d6 --games 300 --workers 10 --out scripts/results/nr_d4_vs_d6.json
python scripts/match.py --depth-a 6 --depth-b 8 --label-a d6 --label-b d8 --games 300 --workers 10 --out scripts/results/nr_d6_vs_d8.json
python scripts/match.py --depth-a 5 --depth-b 5 --label-a pail30 --label-b pail0 --set-a weight_pail_in_hand=30 --games 400 --workers 10 --out scripts/results/nr_ab_pail30.json
python scripts/match.py --depth-a 5 --depth-b 5 --label-a pail80 --label-b pail0 --set-a weight_pail_in_hand=80 --games 400 --workers 10 --out scripts/results/nr_ab_pail80.json
