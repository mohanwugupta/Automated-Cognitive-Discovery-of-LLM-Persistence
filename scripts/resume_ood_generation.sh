#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OOD_OUTPUT="${OOD_OUTPUT:-artifacts/ood_free_generation_v1}"
GPU_SCRIPT="$PROJECT_DIR/run_ood_generation.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_ood_generation_cpu.slurm"
EVALUATION_CONCURRENCY="${EVALUATION_CONCURRENCY:-2}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs
export OOD_OUTPUT

missing_indices=$(python - <<'PY'
import json
import os
from pathlib import Path

required_engine = "qwen_incremental_single_token_v2"
root = Path(os.environ["OOD_OUTPUT"])
manifest = root / "frozen/evaluation_jobs.json"
if not manifest.exists():
    raise SystemExit(
        f"Frozen OOD manifest is missing: {manifest}. Run the prepare phase first."
    )
jobs = json.loads(manifest.read_text(encoding="utf-8"))
missing = []
for job in jobs:
    index = int(job["job_index"])
    audit_path = root / "shards" / f"job_{index:04d}" / "audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        complete = (
            int(audit["job_index"]) == index
            and (
                job["mode"] != "generation"
                or audit.get("generation_engine_version") == required_engine
            )
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        complete = False
    if not complete:
        missing.append(index)
print(",".join(map(str, missing)))
PY
)

if [ -n "$missing_indices" ]; then
  evaluation_dependency=()
  validation_job=""
  case ",${missing_indices}," in
    *,0,*)
      validation_submission=$(sbatch --parsable --job-name=cog_ood_validate \
        --array=0 --export=ALL,PHASE=validate "$GPU_SCRIPT")
      validation_job="${validation_submission%%;*}"
      evaluation_dependency=(--dependency="afterok:${validation_job}")
      ;;
  esac
  evaluation_submission=$(sbatch --parsable --job-name=cog_ood_resume \
    --array="${missing_indices}%${EVALUATION_CONCURRENCY}" \
    "${evaluation_dependency[@]}" \
    --export=ALL,PHASE=evaluate "$GPU_SCRIPT")
  evaluation_job="${evaluation_submission%%;*}"
  aggregate_submission=$(sbatch --parsable --job-name=cog_ood_aggregate \
    --time=12:00:00 --dependency="afterok:${evaluation_job}" \
    --export=ALL,PHASE=aggregate "$CPU_SCRIPT")
  aggregate_job="${aggregate_submission%%;*}"
  echo "Resubmitted missing or obsolete OOD shards: ${missing_indices}"
  if [ -n "$validation_job" ]; then
    echo "  GPU validation: ${validation_job}"
  fi
  echo "  GPU evaluation: ${evaluation_job}"
  echo "  CPU aggregate:  ${aggregate_job} (afterok:${evaluation_job})"
else
  aggregate_submission=$(sbatch --parsable --job-name=cog_ood_aggregate \
    --time=12:00:00 --export=ALL,PHASE=aggregate "$CPU_SCRIPT")
  aggregate_job="${aggregate_submission%%;*}"
  echo "All OOD shards already have valid audits."
  echo "  CPU aggregate: ${aggregate_job}"
fi
