#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_gpu.slurm"
CPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_cpu.slurm"
OUTPUT="${OUTPUT:?OUTPUT is required}"
DISPATCH_JOB_ID="${SLURM_JOB_ID:?dispatcher must run as a Slurm job}"
STATE_PATH="$OUTPUT/run_state.json"

if [ ! -f "$STATE_PATH" ]; then
  echo "Replication state is missing after interface job: $STATE_PATH" >&2
  exit 1
fi

state_fields=$(python -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
interface = value["stages"]["interface"]
print("\t".join([
    str(interface["status"]),
    str(interface.get("outcome") or ""),
    str(value["replication_status"]),
]))
' "$STATE_PATH")
IFS=$'\t' read -r interface_status interface_outcome replication_status <<< "$state_fields"

submission_root="$OUTPUT/submission"
mkdir -p "$submission_root"
if ! mkdir "$submission_root/after_interface_dispatch.lock"; then
  echo "Interface dispatch was already attempted; refusing duplicate submissions" >&2
  exit 2
fi
record="$submission_root/after_interface_jobs.txt"

if [ "$interface_status" != "complete" ]; then
  echo "Interface job did not complete scientifically: status=$interface_status outcome=$interface_outcome" >&2
  exit 1
fi

if [ "$replication_status" == "measurement_failure" ]; then
  report_submission=$(sbatch --parsable --dependency="afterok:${DISPATCH_JOB_ID}" \
    --export="ALL,STAGE=report,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
  report_job="${report_submission%%;*}"
  {
    echo "interface_status=$interface_status"
    echo "interface_outcome=$interface_outcome"
    echo "replication_status=$replication_status"
    echo "report=$report_job"
  } > "$record"
  echo "Submitted measurement-failure report=$report_job; no downstream GPU job submitted"
  exit 0
fi

if [ "$interface_outcome" != "pass" ]; then
  echo "Unexpected completed interface outcome: $interface_outcome" >&2
  exit 1
fi

behavior_submission=$(sbatch --parsable --dependency="afterok:${DISPATCH_JOB_ID}" --export="ALL,STAGE=behavior,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
behavior="${behavior_submission%%;*}"
comparison_submission=$(sbatch --parsable --dependency="afterok:${behavior}" --export="ALL,STAGE=model_comparison,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
comparison="${comparison_submission%%;*}"
freeze_submission=$(sbatch --parsable --dependency="afterok:${comparison}" --export="ALL,STAGE=freeze_theory,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
freeze="${freeze_submission%%;*}"
counterfactuals_submission=$(sbatch --parsable --dependency="afterok:${freeze}" --export="ALL,STAGE=counterfactuals,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
counterfactuals="${counterfactuals_submission%%;*}"
mechanism_submission=$(sbatch --parsable --dependency="afterok:${counterfactuals}" --export="ALL,STAGE=mechanism,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
mechanism="${mechanism_submission%%;*}"
generalization_submission=$(sbatch --parsable --dependency="afterok:${mechanism}" --export="ALL,STAGE=generalization,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
generalization="${generalization_submission%%;*}"
specificity_submission=$(sbatch --parsable --dependency="afterok:${generalization}" --export="ALL,STAGE=specificity,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
specificity="${specificity_submission%%;*}"
report_submission=$(sbatch --parsable --dependency="afterok:${specificity}" --export="ALL,STAGE=report,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
report="${report_submission%%;*}"

{
  echo "interface_status=$interface_status"
  echo "interface_outcome=$interface_outcome"
  echo "replication_status=$replication_status"
  echo "behavior=$behavior"
  echo "model_comparison=$comparison"
  echo "freeze_theory=$freeze"
  echo "counterfactuals=$counterfactuals"
  echo "mechanism=$mechanism"
  echo "generalization=$generalization"
  echo "specificity=$specificity"
  echo "report=$report"
} > "$record"

echo "Interface passed; submitted conditional replication chain"
echo "behavior=$behavior model_comparison=$comparison freeze_theory=$freeze"
echo "counterfactuals=$counterfactuals mechanism=$mechanism"
echo "generalization=$generalization specificity=$specificity report=$report"
