#!/bin/bash
# Prepare a one-controller pilot by default. A full matrix requires an explicit
# budget approval flag because it creates many GPU work units.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPLICATION=${REPLICATION:?export REPLICATION to a completed prospective run}
TRANSFER_OUTPUT=${TRANSFER_OUTPUT:?export TRANSFER_OUTPUT to a new scratch directory}
TRANSFER_CONFIG=${TRANSFER_CONFIG:-$REPO/configs/transfer/v1.yaml}
export REPO REPLICATION TRANSFER_OUTPUT TRANSFER_CONFIG

cd "$REPO"
PREPARE_EXTRA=(--final)
if [[ "${TRANSFER_FULL_MATRIX_APPROVED:-0}" == 1 ]]; then
  TRANSFER_PILOT_REPORT=${TRANSFER_PILOT_REPORT:?full matrix requires resource_projection.json from the one-source pilot}
  TRANSFER_APPROVED_GPU_HOURS=${TRANSFER_APPROVED_GPU_HOURS:?approve a numeric GPU-hour ceiling}
  TRANSFER_APPROVED_STORAGE_GB=${TRANSFER_APPROVED_STORAGE_GB:?approve a numeric storage ceiling}
  python -c '
import hashlib, json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
replication = json.loads((pathlib.Path(sys.argv[2]) / "provenance.json").read_text(encoding="utf-8"))
config_hash = hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
if report.get("schema_version") != "task-transfer-pilot-projection-v1":
    raise SystemExit("invalid pilot resource projection")
if report.get("model") != replication.get("model"):
    raise SystemExit("pilot model identity differs from the requested replication")
if report.get("config_sha256") != config_hash:
    raise SystemExit("pilot and full transfer configurations differ")
gpu, storage = float(sys.argv[4]), float(sys.argv[5])
if report["projected_full_gpu_hours"] > gpu:
    raise SystemExit("pilot projection exceeds approved GPU-hour ceiling")
if report["projected_full_storage_gb"] > storage:
    raise SystemExit("pilot projection exceeds approved storage ceiling")
' "$TRANSFER_PILOT_REPORT" "$REPLICATION" "$TRANSFER_CONFIG" \
  "$TRANSFER_APPROVED_GPU_HOURS" "$TRANSFER_APPROVED_STORAGE_GB"
fi
python scripts/task_transfer.py prepare --replication "$REPLICATION" \
  --output "$TRANSFER_OUTPUT" --config "$TRANSFER_CONFIG" \
  --scope single_task_matrix "${PREPARE_EXTRA[@]}" --execute
COUNT=$(python -c 'import json,os; print(len(json.load(open(os.path.join(os.environ["TRANSFER_OUTPUT"],"work_manifest.json")))["work_units"]))')

if [[ "${TRANSFER_FULL_MATRIX_APPROVED:-0}" == 1 ]]; then
  ARRAY="0-$((COUNT-1))%${TRANSFER_MAX_CONCURRENT:-2}"
  ACTIVE_INDICES=$(seq -s, 0 "$((COUNT-1))")
  ALLOW_INCOMPLETE=0
  SMOKE=0
  RANDOM_COUNT=${TRANSFER_RANDOM_COUNT:-100}
else
  ARRAY="0"
  ACTIVE_INDICES="0"
  ALLOW_INCOMPLETE=1
  SMOKE=1
  RANDOM_COUNT=${TRANSFER_RANDOM_COUNT:-2}
  echo "Submitting the required one-controller GPU-hour pilot only."
  echo "Set TRANSFER_FULL_MATRIX_APPROVED=1 only after reviewing pilot runtime."
fi
export TRANSFER_ACTIVE_INDICES="$ACTIVE_INDICES"
export TRANSFER_ALLOW_INCOMPLETE="$ALLOW_INCOMPLETE"
export TRANSFER_MAX_CONCURRENT="${TRANSFER_MAX_CONCURRENT:-2}"
export TRANSFER_SMOKE="$SMOKE"
export TRANSFER_RANDOM_COUNT="$RANDOM_COUNT"

TRAIN=$(sbatch --parsable --array="$ARRAY" --export=ALL,TRANSFER_MODE=train,TRANSFER_SMOKE="$SMOKE",TRANSFER_RANDOM_COUNT="$RANDOM_COUNT" slurm/run_task_transfer_gpu.slurm)
EVALUATE=$(sbatch --parsable --dependency="afterok:$TRAIN" --array="$ARRAY" --export=ALL,TRANSFER_MODE=evaluate,TRANSFER_EVALUATION_SCOPE=diagonal,TRANSFER_SMOKE="$SMOKE",TRANSFER_RANDOM_COUNT="$RANDOM_COUNT" slurm/run_task_transfer_gpu.slurm)
DISPATCH=$(sbatch --parsable --dependency="afterok:$EVALUATE" --export=ALL,TRANSFER_CPU_MODE=dispatch_diagonal slurm/run_task_transfer_cpu.slurm)
printf 'train=%s\ndiagonal=%s\ndispatch=%s\n' "$TRAIN" "$EVALUATE" "$DISPATCH"
