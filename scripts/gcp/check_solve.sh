#!/bin/bash
# Local progress check for the cloud solve. Usage:
#   BUCKET=gs://... ./scripts/gcp/check_solve.sh            # status + log tail
#   BUCKET=gs://... ./scripts/gcp/check_solve.sh --restart  # restart a preempted VM and relaunch
set -euo pipefail
BUCKET=${BUCKET:?set BUCKET=gs://...}
ZONE=${ZONE:-northamerica-northeast2-a}
VM=${VM:-tb-solver}

state=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format="value(status)" 2>/dev/null || echo "ABSENT")
echo "instance: $state"

if gsutil -q stat "$BUCKET/status/exit.txt" 2>/dev/null; then
  echo "exit marker: $(gsutil cat "$BUCKET/status/exit.txt")"
fi
if gsutil -q stat "$BUCKET/status/solve_complete.log" 2>/dev/null; then
  echo "*** SOLVE COMPLETE — full log at $BUCKET/status/solve_complete.log ***"
fi

if gsutil -q stat "$BUCKET/status/latest.json" 2>/dev/null; then
  hb=$(gsutil cat "$BUCKET/status/latest.json")
  echo "heartbeat: $hb"
  ts=$(echo "$hb" | python3 -c "import json,sys,time,calendar; d=json.load(sys.stdin); print(int(time.time()-calendar.timegm(time.strptime(d['ts'],'%Y-%m-%dT%H:%M:%SZ'))))")
  echo "heartbeat age: ${ts}s"
  if [ "$ts" -gt 600 ] && [ "$state" = "RUNNING" ]; then
    echo "WARNING: VM is RUNNING but the heartbeat is stale (>10 min) — ssh in and check:"
    echo "  gcloud compute ssh $VM --zone $ZONE -- tail -40 /data/solve.log"
  fi
  cpu=$(echo "$hb" | python3 -c "import json,sys; print(float(json.load(sys.stdin)['solver_cpu_pct'] or 0))")
  if [ "$state" = "RUNNING" ] && python3 -c "exit(0 if $cpu < 500 else 1)"; then
    echo "NOTE: solver CPU ${cpu}% is low for 128 vCPUs — early pass ramp-up, checkpoint write, or stuck."
  fi
else
  echo "no heartbeat yet"
fi
echo "--- log tail ---"
gsutil cat "$BUCKET/status/log_tail.txt" 2>/dev/null | tail -15 || echo "(none)"

if [ "${1:-}" = "--restart" ] && { [ "$state" = "TERMINATED" ] || [ "$state" = "STOPPED" ]; }; then
  echo "restarting preempted VM and relaunching the solve..."
  gcloud compute instances start "$VM" --zone "$ZONE"
  sleep 45
  gcloud compute scp scripts/gcp/vm_solve.sh "$VM":~ --zone "$ZONE"
  gcloud compute ssh "$VM" --zone "$ZONE" -- "BUCKET=$BUCKET nohup bash vm_solve.sh >/tmp/bootstrap.log 2>&1 & disown; echo relaunched"
fi
