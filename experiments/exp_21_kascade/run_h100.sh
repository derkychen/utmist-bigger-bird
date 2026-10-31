#!/usr/bin/env bash
# Exp 21 on Modal H100 80GB, Nemotron-UltraLong-1M at ~128K tokens. Run the stages in this order:
#   bash experiments/exp_21_kascade/run_h100.sh billing   # spend so far in the team environment (free)
#   bash experiments/exp_21_kascade/run_h100.sh download  # Nemotron weights -> bb-vol (CPU)
#   bash experiments/exp_21_kascade/run_h100.sh upload    # local 16K Phase 0 (anchors + head maps) -> bb-vol
#   bash experiments/exp_21_kascade/run_h100.sh smoke     # ~16K plumbing check, ~5 min H100 (stops itself at 20 min)
#   bash experiments/exp_21_kascade/run_h100.sh plan      # accuracy + long + followup, ~45 min H100 (stops at 90 min)
#   bash experiments/exp_21_kascade/run_h100.sh offload   # exp 21 with a CPU-offloaded cache at ~1M (200 GB container RAM)
# `plan` takes optional stages, e.g. `run_h100.sh plan long,followup`.
# Every command uses the team environment (UTMIST rule: --env engineering_team_bigger_bird for CPU, GPU and volumes).
# Results: /vol/results/exp_21_kascade/<phase>/ on bb-vol; pull with
#   modal volume get --env engineering_team_bigger_bird bb-vol results/exp_21_kascade benchmarks/
# Stop a run early: `modal app list --env engineering_team_bigger_bird`, then `modal app stop --env ... <app id>`.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=$1
ENV=engineering_team_bigger_bird
export MODAL_ENVIRONMENT=$ENV   # also covers lookups inside the app (bb-vol)
export BB_MODEL=nemotron
P0=ul_p0_16k_tail0_pool   # local UltraLong 16K Phase 0; byte-identical checkpoint to Nemotron
mkdir -p modal_rd/logs    # excluded from the Modal image, so logging here does not abort the build
run() {  # run <log name> <args>
  modal run --env $ENV --detach modal_rd/app.py --module experiments.exp_21_kascade.h100_plan --args "$2" \
    2>&1 | tee "modal_rd/logs/exp21_$1.log"
}
case $STAGE in
  billing)  modal environment billing summary --for "this month" $ENV ;;
  download) BB_GPU=none modal run --env $ENV modal_rd/app.py::download ;;
  upload)
    modal volume put --env $ENV bb-vol benchmarks/exp_21_kascade/phase0/$P0 results/exp_21_kascade/phase0/$P0
    modal volume ls --env $ENV bb-vol results/exp_21_kascade/phase0/$P0 ;;   # expect analysis.json, matrices.npz
  smoke)    run smoke "--smoke --map-tag $P0 --max-minutes 20" ;;
  plan)     run plan "--stages ${2:-accuracy,long,followup} --map-tag $P0 --max-minutes 90" ;;
  budget)   run budget "--stages evict_1m --map-tag $P0 --evict-keep ${2:-65536} --fp8 --max-minutes 90" ;;   # bigger eviction budget
  turbo)    BB_MEMORY_MB=120000 run turbo "--stages turbo_1m --map-tag $P0 --kv-bits ${2:-4} --fp8 --tests ${3:-followup_fact,followup_fact_first,long} --max-minutes 120" ;;    # compressed full cache
  offload)  BB_MEMORY_MB=200000 run offload "--stages offload_1m --map-tag $P0 --decode ${2:-dw1024:dv1024} --max-minutes 120" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
