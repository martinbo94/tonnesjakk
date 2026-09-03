#!/bin/bash
# SPSA over the search/pruning knobs WITH the NNUE loaded, at time control
# (pruning trades depth for accuracy, so tune at the TC we play at), then
# validate tuned-vs-default at 100 ms and 200 ms. WORKERS=4 for daytime.
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -x
NNUE=${NNUE:-models/net3_plain_m_d20_64x16_b25_l05.json}
W=${WORKERS:-10}
# a=2/c=1 (the eval-weight run's settings) left the search knobs within a few
# units of their defaults after 100 iterations: too timid for parameters that
# already sit near a hand-tuned optimum. Larger steps + perturbations.
python scripts/spsa_tune.py --params search --nnue "$NNUE" --time-ms 100 \
  --iterations 250 --pairs 24 --workers "$W" --a "${SPSA_A:-6}" --c "${SPSA_C:-2}" \
  --out scripts/results/spsa_search_net3.json
python3 - "$NNUE" "$W" <<'PYEOF'
import json, subprocess, sys
nnue, workers = sys.argv[1], sys.argv[2]
theta = json.load(open('scripts/results/spsa_search_net3.json'))['theta']
sets = []
for n, v in theta.items():
    sets += ['--set-a', f'{n}={v}']
for tc, games, out in ((100, 600, 'scripts/results/spsa_search_val_100ms.json'),
                       (200, 400, 'scripts/results/spsa_search_val_200ms.json')):
    cmd = ['python', 'scripts/match.py', '--time-a', str(tc), '--time-b', str(tc),
           '--nnue-a', nnue, '--nnue-b', nnue,
           '--label-a', 'tuned-search', '--label-b', 'default-search',
           '--games', str(games), '--workers', workers, '--out', out] + sets
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
PYEOF
