#!/bin/bash
# Runs ON the GCP VM. Idempotent: safe to re-run after a spot preemption.
set -euxo pipefail
BUCKET=${BUCKET:?set BUCKET=gs://...}
# Format ONLY if the disk has no filesystem yet — after a spot preemption this
# script re-runs and must NOT wipe the checkpoints the data disk exists to keep.
if ! sudo blkid /dev/disk/by-id/google-data >/dev/null 2>&1; then
  sudo mkfs.ext4 -F /dev/disk/by-id/google-data
fi
sudo mkdir -p /data
mountpoint -q /data || sudo mount /dev/disk/by-id/google-data /data
sudo chown $USER /data
sudo apt-get update -qq && sudo apt-get install -y -qq git curl zstd build-essential python3-venv python3-pip
command -v cargo >/dev/null || (curl -sSf https://sh.rustup.rs | sh -s -- -y); source "$HOME/.cargo/env"
# Source: private repo -> a tarball in the bucket (uploaded by the runbook's step 0)
if [ ! -d /data/tonnesjakk ]; then
  gsutil cp $BUCKET/src/tonnesjakk-src.tar.gz /data/ && mkdir -p /data/tonnesjakk \
    && tar xzf /data/tonnesjakk-src.tar.gz -C /data/tonnesjakk
fi
cd /data/tonnesjakk
python3 -m venv .venv && source .venv/bin/activate && pip install -q maturin
maturin develop --release
mkdir -p tablebases
# prerequisites (skip what already exists / survived preemption)
for f in tb_4v2.p2 tb_3v3.p2; do  # 4v3 scores only into 3v3/4v2; 4v4 only into 4v3 (3v4 via mirror)
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
