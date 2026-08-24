#!/bin/bash
source .venv/bin/activate
set -x
python scripts/match.py --depth-a 4 --depth-b 6 --label-a d4 --label-b d6 --games 400 --workers 10 --out scripts/results/d4_vs_d6.json
python scripts/match.py --depth-a 6 --depth-b 8 --label-a d6 --label-b d8 --games 300 --workers 10 --out scripts/results/d6_vs_d8.json
python scripts/match.py --depth-a 8 --depth-b 10 --label-a d8 --label-b d10 --games 200 --workers 10 --out scripts/results/d8_vs_d10.json
python scripts/match.py --time-a 50 --time-b 100 --label-a t50ms --label-b t100ms --games 300 --workers 10 --out scripts/results/t50_vs_t100.json
python scripts/match.py --time-a 100 --time-b 200 --label-a t100ms --label-b t200ms --games 300 --workers 10 --out scripts/results/t100_vs_t200.json
