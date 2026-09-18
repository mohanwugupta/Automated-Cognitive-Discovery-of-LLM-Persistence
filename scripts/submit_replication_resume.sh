#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_gpu.slurm"
CPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_cpu.slurm"
OUTPUT="${OUTPUT:?OUTPUT is required}"
FROM_STAGE="${FROM_STAGE:-counterfactuals}"

if [ "$FROM_STAGE" != "counterfactuals" ]; then
  echo "Only FROM_STAGE=counterfactuals is supported by this repair-resume helper" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if [ ! -d "$OUTPUT" ]; then
  echo "Replication output does not exist: $OUTPUT" >&2
  exit 1
fi

cd "$PROJECT_DIR"
current_commit=$(git rev-parse HEAD)
if ! [[ "$current_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Current repository commit is not an immutable lowercase 40-hex identity" >&2
  exit 2
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing a repair resume from a dirty working tree; commit the repair first" >&2
  exit 2
fi

state_path="$OUTPUT/run_state.json"
provenance_path="$OUTPUT/provenance.json"
if [ ! -f "$state_path" ] || [ ! -f "$provenance_path" ]; then
  echo "The replication state or provenance file is missing from $OUTPUT" >&2
  exit 1
fi

python -c '
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
required = ("interface", "behavior", "model_comparison", "freeze_theory")
incomplete = [name for name in required if state["stages"][name]["status"] != "complete"]
if incomplete:
    raise SystemExit(f"cannot resume counterfactuals; incomplete prerequisites: {incomplete}")
status = state["stages"]["counterfactuals"]["status"]
if status not in {"pending", "blocked"}:
    raise SystemExit(f"counterfactual stage is not resumable: {status}")
' "$state_path"

attempt_id="${RESUME_ATTEMPT_ID:-${current_commit:0:12}}"
if ! [[ "$attempt_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RESUME_ATTEMPT_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 2
fi
submission_root="$OUTPUT/submission"
mkdir -p "$submission_root"
metadata_path="$submission_root/resume_counterfactuals_${attempt_id}.json"
if [ -e "$metadata_path" ]; then
  echo "This resume attempt already exists: $metadata_path" >&2
  echo "Set a new RESUME_ATTEMPT_ID only if an earlier submission genuinely needs retrying" >&2
  exit 2
fi

python -c '
from datetime import datetime, timezone
import json, pathlib, sys
provenance_path, state_path, metadata_path, commit = map(pathlib.Path, sys.argv[1:])
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
state = json.loads(state_path.read_text(encoding="utf-8"))
record = {
    "schema_version": "replication-repair-resume-v1",
    "reason": "counterfactual_sign_ordering_repair",
    "resume_from_stage": "counterfactuals",
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    "initial_git_commit": provenance["git_commit"],
    "resume_git_commit": commit.name,
    "resume_git_dirty": False,
    "mixed_code_run": provenance["git_commit"] != commit.name,
    "completed_stage_snapshot": {
        name: state["stages"][name]
        for name in ("interface", "behavior", "model_comparison", "freeze_theory")
    },
    "submission_status": "prepared",
}
history = provenance.setdefault("code_repair_history", [])
history.append({key: value for key, value in record.items() if key != "completed_stage_snapshot"})
for path, value in ((provenance_path, provenance), (metadata_path, record)):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
' "$provenance_path" "$state_path" "$metadata_path" "$current_commit"

counterfactuals_submission=$(sbatch --parsable --export="ALL,STAGE=counterfactuals,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
counterfactuals="${counterfactuals_submission%%;*}"
mechanism_submission=$(sbatch --parsable --dependency="afterok:${counterfactuals}" --export="ALL,STAGE=mechanism,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
mechanism="${mechanism_submission%%;*}"
generalization_submission=$(sbatch --parsable --dependency="afterok:${mechanism}" --export="ALL,STAGE=generalization,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
generalization="${generalization_submission%%;*}"
specificity_submission=$(sbatch --parsable --dependency="afterok:${generalization}" --export="ALL,STAGE=specificity,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
specificity="${specificity_submission%%;*}"
report_submission=$(sbatch --parsable --dependency="afterok:${specificity}" --export="ALL,STAGE=report,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
report="${report_submission%%;*}"

python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["submission_status"] = "submitted"
value["jobs"] = {
    "counterfactuals": sys.argv[2],
    "mechanism": sys.argv[3],
    "generalization": sys.argv[4],
    "specificity": sys.argv[5],
    "report": sys.argv[6],
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
' "$metadata_path" "$counterfactuals" "$mechanism" "$generalization" "$specificity" "$report"

echo "Replication repair resume submitted from counterfactuals"
echo "counterfactuals=$counterfactuals mechanism=$mechanism"
echo "generalization=$generalization specificity=$specificity report=$report"
echo "Mixed-code provenance: $metadata_path"
