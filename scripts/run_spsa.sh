#!/bin/bash
source .venv/bin/activate
set -x
python scripts/spsa_tune.py --iterations 250 --pairs 24 --depth 5 --workers 10 --out scripts/results/spsa_tune.json
# Validation: tuned vs baseline at tuning depth and at deeper depth
python3 - <<'PYEOF'
import json, subprocess
theta = json.load(open('scripts/results/spsa_tune.json'))['theta']
sets = []
for n, v in theta.items():
    sets += ['--set-a', f'{n}={v}']
for depth, games, out in ((5, 600, 'scripts/results/spsa_val_d5.json'),
                          (7, 400, 'scripts/results/spsa_val_d7.json'),
                          (9, 200, 'scripts/results/spsa_val_d9.json')):
    cmd = ['python', 'scripts/match.py', '--depth-a', str(depth), '--depth-b', str(depth),
           '--label-a', 'tuned', '--label-b', 'baseline',
           '--games', str(games), '--workers', '10', '--out', out] + sets
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
PYEOF
