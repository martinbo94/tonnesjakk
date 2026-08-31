#!/bin/bash
# Runs ON the GCP VM. Idempotent: safe to re-run after a spot preemption.
set -euxo pipefail
BUCKET=${BUCKET:?set BUCKET=gs://...}
# All output to a persistent log (survives ssh drops; heartbeat ships it out).
# NOT under /data: the data disk is mounted over /data later and would shadow
# an already-open log file there.
LOG=$HOME/solve.log
exec > >(tee -a $LOG) 2>&1

# ── Heartbeat: every 2 min push status JSON + log tail to the bucket, so
# progress/stuck/failure is visible from anywhere with `gsutil cat`. ──
heartbeat() {
  while true; do
    pid=$(pgrep -f "solve_tablebase_packed" | head -1 || true)
    cpu=$( [ -n "$pid" ] && ps -o pcpu= -p "$pid" || echo 0 )
    rss=$( [ -n "$pid" ] && ps -o rss= -p "$pid" || echo 0 )
    ckpt=$(ls -la /data/tonnesjakk/tablebases/*.partial 2>/dev/null | awk '{print $5, $9}' | tail -1)
    python3 - <<PY > /tmp/status.json 2>/dev/null || true
import json, time, os
print(json.dumps({
  "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "solver_pid": "${pid}", "solver_cpu_pct": "${cpu}".strip(), "solver_rss_kb": "${rss}".strip(),
  "checkpoint": "${ckpt}",
  "disk_free_gb": round(__import__("shutil").disk_usage("/data").free / 1e9, 1),
  "loadavg": open("/proc/loadavg").read().split()[:3],
}))
PY
    gsutil -q cp /tmp/status.json $BUCKET/status/latest.json 2>/dev/null || true
    tail -40 $LOG > /tmp/log_tail.txt 2>/dev/null || true
    gsutil -q cp /tmp/log_tail.txt $BUCKET/status/log_tail.txt 2>/dev/null || true
    sleep 120
  done
}
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null; echo "EXIT $? at $(date -u +%FT%TZ)" > /tmp/exit.txt; gsutil -q cp /tmp/exit.txt $BUCKET/status/exit.txt 2>/dev/null || true' EXIT
# Format ONLY if the disk has no filesystem yet — after a spot preemption this
# script re-runs and must NOT wipe the checkpoints the data disk exists to keep.
if ! sudo blkid /dev/disk/by-id/google-data >/dev/null 2>&1; then
  sudo mkfs.ext4 -F /dev/disk/by-id/google-data
fi
sudo mkdir -p /data
mountpoint -q /data || sudo mount /dev/disk/by-id/google-data /data
sudo chown $USER /data
sudo apt-get update -qq && sudo apt-get install -y -qq git curl zstd build-essential python3-venv python3-pip pkg-config libssl-dev
command -v cargo >/dev/null || (curl -sSf https://sh.rustup.rs | sh -s -- -y); source "$HOME/.cargo/env"
# Source: private repo -> a tarball in the bucket (uploaded by the runbook's step 0)
if [ ! -d /data/tonnesjakk ]; then
  gsutil cp $BUCKET/src/tonnesjakk-src.tar.gz /data/ && mkdir -p /data/tonnesjakk \
    && tar xzf /data/tonnesjakk-src.tar.gz -C /data/tonnesjakk
fi
cd /data/tonnesjakk
python3 -m venv .venv && source .venv/bin/activate && pip install -q maturin
maturin develop --release --no-default-features  # no mcts/ort: solver only
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
echo "ALL PHASES SOLVED at $(date -u +%FT%TZ)" && gsutil -q cp $LOG $BUCKET/status/solve_complete.log
# The answer:
python3 -c "
from tonnesjakk import Engine, Board
e = Engine(); print('phases:', e.load_tablebases('tablebases'))
print('INITIAL POSITION:', e.tablebase_probe(Board()))
"
