#!/bin/bash
# Runs ON the GCP VM. Idempotent: safe to re-run after a spot preemption.
set -euxo pipefail
BUCKET=${BUCKET:?set BUCKET=gs://...}
sudo mkfs.ext4 -F /dev/disk/by-id/google-data 2>/dev/null || true
sudo mkdir -p /data && sudo mount /dev/disk/by-id/google-data /data 2>/dev/null || true
sudo chown $USER /data
sudo apt-get update -qq && sudo apt-get install -y -qq git curl zstd build-essential python3-venv python3-pip
command -v cargo >/dev/null || (curl -sSf https://sh.rustup.rs | sh -s -- -y); source "$HOME/.cargo/env"
[ -d /data/tonnesjakk ] || git clone https://github.com/USER/tonnesjakk /data/tonnesjakk
cd /data/tonnesjakk && git pull
python3 -m venv .venv && source .venv/bin/activate && pip install -q maturin
maturin develop --release
mkdir -p tablebases
# prerequisites (skip what already exists / survived preemption)
for f in tb_4v2.p2 tb_3v3.p2 tb_4v1.wdl tb_3v2.wdl tb_2v2.wdl tb_3v1.wdl tb_1v3.wdl tb_2v1.wdl tb_1v2.wdl tb_1v1.wdl; do
  [ -f tablebases/$f ] || (gsutil cp $BUCKET/tb/$f.zst tablebases/ && zstd -d --rm tablebases/$f.zst)
done
solve() {
  local wr=$1 br=$2
  [ -f tablebases/tb_${wr}v${br}.p2 ] && return 0
  python3 -c "
import time
from tonnesjakk._core import solve_tablebase_packed
t0 = time.time()
n, w, b, d = solve_tablebase_packed('tablebases', $wr, $br, 1, True, lowmem=False)
v = w + b + d
print(f'${wr}v${br}: {n:,} states - white {100*w/v:.2f}%, black {100*b/v:.2f}%, draws {100*d/v:.3f}%  [{time.time()-t0:.0f}s]', flush=True)
"
  zstd -3 -T0 -k tablebases/tb_${wr}v${br}.p2
  gsutil cp tablebases/tb_${wr}v${br}.p2.zst $BUCKET/tb/
}
solve 4 3
solve 4 4
# The answer:
python3 -c "
from tonnesjakk import Engine, Board
e = Engine(); print('phases:', e.load_tablebases('tablebases'))
print('INITIAL POSITION:', e.tablebase_probe(Board()))
"
