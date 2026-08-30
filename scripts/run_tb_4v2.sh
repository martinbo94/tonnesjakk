#!/bin/bash
# Overnight: solve 4v2 as a packed 2-bit WDL table (both sides to move),
# checkpointing every 2 passes (atomic write) so a reboot costs minutes.
# Resumes from tablebases/tb_4v2.p2.partial automatically. caffeinate keeps
# the machine awake for the duration.
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -x
TB_THREADS=${TB_THREADS:-12} caffeinate -i python3 -c "
import time
from tonnesjakk._core import solve_tablebase_packed
t0 = time.time()
n, w, b, d = solve_tablebase_packed('tablebases', 4, 2, 2, True, lowmem=True)
v = w + b + d
print(f'4v2 (packed, white to move): {n:,} states ({v:,} valid) - white {100*w/v:.2f}%, black {100*b/v:.2f}%, draws {100*d/v:.3f}%  [{time.time()-t0:.0f}s]', flush=True)
"
