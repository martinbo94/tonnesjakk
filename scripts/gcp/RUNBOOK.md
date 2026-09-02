# Solving 4v3 and 4v4 on GCP

Goal: solve the last two phases. After 4v4, every legal position is solved and
the initial position's value answers whether tønnesjakk is a first-player win.

## Exact sizes (from `packed_phase_stats`, verified against solved phases)

| phase | states | array | pair tables | lookups while solving |
|---|---|---|---|---|
| 4v3 (both sides; 3v4 via mirror) | 366,690,102,916 | 91.7 GB | 2.4 GB | 4v2.p2 11.2 + 3v3.p2 7.5 GB |
| 4v4 (white to move only) | 1,084,932,923,723 | 271.2 GB | 16.1 GB | 4v3.p2 91.7 + 2.4 GB |

Peak RSS: 4v3 ≈ 115 GB; 4v4 ≈ 400 GB.

## Machine

- `n2-highmem-128` (128 vCPU, 864 GB), **spot**. Cheapest spot regions (catalog, 2026-08-31):
  northamerica-northeast2 **$0.76/h**, us-west8 $0.80, asia-south2 $0.83, europe-west8 $0.93
  (on-demand $7.71/h us-central1; the org's negotiated ~26% + SUD apply to on-demand only,
  so spot beats the discounted rate ~7x). Try Toronto first, fall back on stockout.
  The solver checkpoints every pass and resumes from `tb_*.p2.partial`, so
  preemption costs at most one pass. `c3-highmem-88` (704 GB) also fits.
- Boot disk 50 GB + `pd-balanced` data disk 700 GB (~$56/mo, prorated).
- No GPU. Region: whatever has spot capacity (us-central1/us-east1).

## Time & cost (scaled from measured local throughput: 0.77M state-evals/s per M4 thread; assume 0.4M per cloud vCPU)

| step | evals | wall @128 vCPU | spot cost |
|---|---|---|---|
| 4v3 | ~0.73 T | 4–7 h | ~$5 |
| 4v4 | ~2.2 T | 12–24 h | ~$10–20 |
| disk + misc | | | ~$5 |
| egress (zstd -3 gives 16–44x on our tables → ~15–25 GB) | | | ~$2–3 |
| **total** | | | **~$25–35** (on-demand ~$230) |

Budget with margin for a re-run: **$60 spot**.

## Steps

```bash
# -1. once: a bucket in the same region as the VM
gsutil mb -l northamerica-northeast2 gs://BUCKET

# 0. locally: upload prerequisites (~1.2 GB compressed) and the source tarball
#    (use gcloud storage, NOT gsutil: gsutil's legacy reauth path expires under
#     the org session policy and cannot re-auth non-interactively)
zstd -3 -T0 -k tablebases/tb_4v2.p2 tablebases/tb_3v3.p2
gcloud storage cp tablebases/tb_4v2.p2.zst tablebases/tb_3v3.p2.zst gs://BUCKET/tb/
git archive --format=tar.gz -o /tmp/tonnesjakk-src.tar.gz HEAD
gcloud storage cp /tmp/tonnesjakk-src.tar.gz gs://BUCKET/src/

# 1. create the VM (spot). --scopes is REQUIRED (default scopes are storage
#    read-only and the result upload fails after the solve).
gcloud compute instances create tb-solver \
  --machine-type=n2-highmem-128 --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --scopes=cloud-platform \
  --boot-disk-size=50GB \
  --create-disk=size=900GB,type=pd-balanced,auto-delete=yes,device-name=data \
  --image-family=debian-12 --image-project=debian-cloud \
  --zone=northamerica-northeast2-a   # cheapest spot; fall back us-west8-a / us-central1-a
# data-disk sizing: >= 2x the biggest phase array (the atomic checkpoint's .tmp
# and .partial coexist) + lower tables + the final save. A 500 GB disk hit
# 0 free mid-4v4; pd-balanced resizes ONLINE: gcloud compute disks resize +
# sudo resize2fs /dev/disk/by-id/google-data.

# 2. launch DETACHED (survives ssh drops). vm_solve.sh is idempotent (re-run
#    after preemption; refuses to double-start a solver) and starts
#    heartbeat.sh, which ships status JSON + log tail to gs://BUCKET/status/
#    every 2 minutes independent of the solve's lifecycle.
gcloud compute scp scripts/gcp/vm_solve.sh scripts/gcp/heartbeat.sh tb-solver:~ --zone $ZONE
gcloud compute ssh tb-solver --zone $ZONE -- \
  "BUCKET=gs://BUCKET nohup bash vm_solve.sh >/tmp/bootstrap.log 2>&1 & disown"

# 3. poll from anywhere (status JSON age, solver CPU, checkpoint size, log tail;
#    warns on stale heartbeat / low CPU; exit marker on failure):
BUCKET=gs://BUCKET ./scripts/gcp/check_solve.sh
#    after a spot preemption (instance shows TERMINATED/STOPPED):
BUCKET=gs://BUCKET ./scripts/gcp/check_solve.sh --restart   # start + relaunch; resumes from checkpoint

# 4. locally: download the compressed results (~15-25 GB) and verify
gsutil -m cp gs://BUCKET/tb/tb_4v3.p2.zst gs://BUCKET/tb/tb_4v4.p2.zst tablebases/
zstd -d tablebases/tb_4v3.p2.zst tablebases/tb_4v4.p2.zst
# one-ply consistency spot checks (same harness as 3v3/4v2 verification)
```

## The answer

`Tablebase::value(&BitBoard::new())` on the loaded 4v4 phase = the game-theoretic
value of tønnesjakk from the initial position.
