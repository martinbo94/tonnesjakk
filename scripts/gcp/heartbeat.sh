#!/bin/bash
# Standalone status shipper for the solve VM. Independent of vm_solve.sh's
# lifecycle: launch once with nohup; it reports whatever is (not) running.
BUCKET=${BUCKET:-gs://tonnesjakk-tb-solve}
LOG=$HOME/solve.log
while true; do
  pid=$(pgrep -f solve_tablebase_packed | head -1)
  cpu=0; rss=0
  if [ -n "$pid" ]; then
    cpu=$(ps -o pcpu= -p "$pid" | tr -d ' ')
    rss=$(ps -o rss= -p "$pid" | tr -d ' ')
  fi
  ckpt=$(ls -la /data/tonnesjakk/tablebases/*.partial 2>/dev/null | awk '{print $5, $9}' | tail -1)
  free_gb=$(df -BG /data 2>/dev/null | awk 'NR==2{print $4}')
  printf '{"ts":"%s","solver_pid":"%s","solver_cpu_pct":"%s","solver_rss_kb":"%s","checkpoint":"%s","disk_free":"%s","load":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$pid" "$cpu" "$rss" "$ckpt" "$free_gb" "$(cut -d' ' -f1-3 /proc/loadavg)" > /tmp/status.json
  gsutil -q cp /tmp/status.json "$BUCKET/status/latest.json" 2>/dev/null
  tail -40 "$LOG" > /tmp/log_tail.txt 2>/dev/null
  gsutil -q cp /tmp/log_tail.txt "$BUCKET/status/log_tail.txt" 2>/dev/null
  sleep 120
done
