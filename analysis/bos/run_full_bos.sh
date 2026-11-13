#!/usr/bin/env bash
# POST_FREEZE_NOT_CONFIRMATORY: sequential prepend_bos pipeline on one machine.
#
# 1. wait for the 30-base pilot Stage B to finish, then analyze it;
# 2. gate: pilot Stage B exited 0 with no non-finite rows;
# 3. full 390-base Stage A -> Stage B (resumable via checkpoint jsonl) -> analysis.
#
# Run from the repo root:  nohup caffeinate -i analysis/bos/run_full_bos.sh > run_bos/full_run.log 2>&1 &
# Re-running after an interruption skips completed steps; Stage B resumes from its checkpoint.
set -euo pipefail

PILOT_LOG=run_bos/stage_b_pilot30.log
FULL_MANIFEST=run_bos/manifest_bos_n390.json
FULL_RESULTS=run_bos/holonomy_results_bos_n390.parquet

stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "waiting for pilot Stage B ($PILOT_LOG)"
until grep -q '^exit=' "$PILOT_LOG"; do sleep 60; done
grep -q '^exit=0$' "$PILOT_LOG" || { stamp "pilot Stage B failed; stopping"; exit 1; }
grep -q '"nonfinite_rows": 0' "$PILOT_LOG" || { stamp "pilot has non-finite rows; stopping"; exit 1; }

stamp "pilot analysis"
uv run python -m analysis.bos.hsem_bos \
  --results run_bos/holonomy_results_bos_pilot30.parquet \
  --manifest run_bos/manifest_bos_pilot30.json \
  --stem hsem_bos_pilot30
stamp "PILOT_DONE"

if [[ ! -f "$FULL_MANIFEST" ]]; then
  stamp "full Stage A (390 bases, prepend_bos)"
  uv run python -m stage_a --prepend-bos --output "$FULL_MANIFEST" > run_bos/stage_a_n390.log 2>&1
fi
stamp "STAGE_A_DONE"

stamp "full Stage B (resumes from checkpoint if present)"
uv run python -m stage_b --manifest "$FULL_MANIFEST" --results "$FULL_RESULTS" >> run_bos/stage_b_n390.log 2>&1
grep -q '"nonfinite_rows": 0' run_bos/stage_b_n390.log || stamp "WARNING: full run reports non-finite rows"
stamp "STAGE_B_DONE"

stamp "full analysis"
uv run python -m analysis.bos.hsem_bos --results "$FULL_RESULTS" --manifest "$FULL_MANIFEST" --stem hsem_bos_n390
stamp "ALL_DONE"
