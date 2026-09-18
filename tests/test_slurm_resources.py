import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_round1_cpu_only_phases_never_request_a_gpu():
    submit = _text("scripts/submit_discovery.sh")
    for phase in ("tests", "pilot_finalize", "discovery_finalize"):
        assert f'PHASE={phase} "$CPU_SCRIPT"' in submit
    for phase in ("pilot_collect", "discovery_collect", "analyze", "validation"):
        assert f"PHASE={phase}" in submit
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$RUN_SCRIPT"' in line


def test_round2_cpu_only_phases_never_request_a_gpu():
    submit = _text("scripts/submit_active_discovery.sh")
    for phase in ("tests", "active_finalize", "update", "final_finalize", "evaluate"):
        assert f'PHASE={phase} "$CPU_SCRIPT"' in submit
    for phase in ("prepare", "active_collect", "final_collect"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$RUN_SCRIPT"' in line


def test_round3_gpu_requests_are_limited_to_neural_audit_and_qwen_collection():
    submit = _text("scripts/submit_theory_resolution.sh")
    for phase in ("tests", "finalize", "evaluate"):
        assert f'PHASE={phase} "$CPU_SCRIPT"' in submit
    for phase in ("prepare", "collect"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$GPU_SCRIPT"' in line


def test_cpu_wrappers_have_no_gpu_directive():
    for script in (
        "run_discovery_cpu.slurm",
        "run_active_cpu.slurm",
        "run_theory_cpu.slurm",
        "run_mechanistic_cpu.slurm",
        "run_action_history_cpu.slurm",
        "run_causal_mechanistic_cpu.slurm",
        "run_causal_specificity_cpu.slurm",
        "run_causal_abstraction_cpu.slurm",
    ):
        text = _text(f"slurm/{script}")
        assert "--gres" not in text
        assert "--gpus" not in text


def test_mechanistic_gpu_is_reserved_only_for_model_forward_phases():
    submit = _text("scripts/submit_mechanistic.sh")
    for phase in ("tests", "prepare", "analyze", "report"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$CPU_SCRIPT"' in line
    for phase in ("scan", "project", "intervene"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$GPU_SCRIPT"' in line


def test_action_history_gpu_is_reserved_only_for_qwen_forward_phases():
    submit = _text("scripts/submit_action_history_disambiguation.sh")
    for phase in ("tests", "prepare", "analyze", "report"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$CPU_SCRIPT"' in line
    for phase in ("fit", "project", "steer", "random", "patch"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$GPU_SCRIPT"' in line


def test_counterfactual_mechanistic_gpu_phases_all_run_qwen_forwards():
    runner = _text("slurm/run_causal_mechanistic.slurm")
    for phase in ("localize", "das", "validate", "circuit"):
        marker = f"\n  {phase})\n"
        assert marker in runner
        block = runner.split(marker, 1)[1].split(";;", 1)[0]
        assert "scripts/causal_mechanistic.py" in block
        assert '"$MODEL_PATH"' in block
    cpu = _text("slurm/run_causal_mechanistic_cpu.slurm")
    assert "--gres" not in cpu
    assert "--gpus" not in cpu


def test_counterfactual_dispatchers_do_not_submit_empty_gpu_arrays():
    das = _text("scripts/dispatch_causal_das.sh")
    validation = _text("scripts/dispatch_causal_validation.sh")
    assert 'if [ "$job_count" -eq 0 ]' in das
    assert 'if [ "$selected_count" -eq 0 ]' in validation


def test_stable_specificity_uses_cpu_preparation_and_exact_gpu_work_array():
    submit = _text("scripts/submit_causal_specificity.sh")
    for phase in ("tests", "prepare", "dispatch"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$CPU_SCRIPT"' in line
    dispatcher = _text("scripts/dispatch_causal_specificity.sh")
    assert 'if [ "$job_count" -eq 0 ]' in dispatcher
    assert 'PHASE=evaluate "$GPU_SCRIPT"' in dispatcher
    assert 'PHASE=aggregate "$CPU_SCRIPT"' in dispatcher
    necessity = _text("scripts/dispatch_causal_necessity.sh")
    assert 'if [ "$job_count" -eq 0 ]' in necessity
    assert 'PHASE=necessity "$GPU_SCRIPT"' in necessity
    assert 'PHASE=finalize "$CPU_SCRIPT"' in necessity
    runner = _text("slurm/run_causal_specificity.slurm")
    block = runner.split("\n  evaluate)\n", 1)[1].split(";;", 1)[0]
    assert "scripts/causal_specificity.py" in block
    assert '"$MODEL_PATH"' in block


def test_causal_abstraction_reserves_gpus_only_for_frozen_qwen_evaluation():
    submit = _text("scripts/submit_causal_abstraction.sh")
    for phase in ("tests", "prepare", "dispatch"):
        line = next(line for line in submit.splitlines() if f"PHASE={phase}" in line)
        assert '"$CPU_SCRIPT"' in line
    dispatcher = _text("scripts/dispatch_causal_abstraction.sh")
    assert 'if [ "$job_count" -eq 0 ]' in dispatcher
    assert 'PHASE=evaluate "$GPU_SCRIPT"' in dispatcher
    assert 'PHASE=aggregate "$CPU_SCRIPT"' in dispatcher
    runner = _text("slurm/run_causal_abstraction.slurm")
    evaluation = runner.rsplit("\n  evaluate)\n", 1)[1].split(";;", 1)[0]
    assert "scripts/causal_abstraction.py" in evaluation
    assert '"$MODEL_PATH"' in evaluation
    cpu_phases = runner.split("tests|prepare|dispatch|aggregate)", 1)[1].split(";;", 1)[
        0
    ]
    assert "CPU-only" in cpu_phases


def test_replication_routes_only_model_forward_stages_to_gpu():
    submit = _text("scripts/submit_replication.sh")
    interface_line = next(
        line for line in submit.splitlines() if "STAGE=interface" in line
    )
    dispatch_line = next(
        line for line in submit.splitlines() if "STAGE=dispatch_interface" in line
    )
    assert '"$GPU_SCRIPT"' in interface_line
    assert '"$CPU_SCRIPT"' in dispatch_line
    assert "afterany:" in dispatch_line
    for phase in ("behavior", "model_comparison", "freeze_theory", "counterfactuals"):
        assert f"STAGE={phase}" not in submit

    dispatcher = _text("scripts/dispatch_replication_after_interface.sh")
    for phase in ("model_comparison", "freeze_theory", "counterfactuals", "report"):
        line = next(line for line in dispatcher.splitlines() if f"STAGE={phase}" in line)
        assert '"$CPU_SCRIPT"' in line
    for phase in ("behavior", "mechanism", "generalization", "specificity"):
        line = next(line for line in dispatcher.splitlines() if f"STAGE={phase}" in line)
        assert '"$GPU_SCRIPT"' in line
    assert 'replication_status" == "measurement_failure"' in dispatcher
    assert "measurement-failure report" in dispatcher
    cpu = _text("slurm/run_replication_cpu.slurm")
    assert "--gres" not in cpu
    assert "--gpus" not in cpu


def test_replication_smoke_routes_only_model_preflight_to_gpu():
    submit = _text("scripts/submit_replication_smoke.sh")
    assert 'SMOKE_PHASE=tests' in submit
    assert '"$CPU_SCRIPT"' in next(
        line for line in submit.splitlines() if "SMOKE_PHASE=tests" in line
    )
    assert '"$GPU_SCRIPT"' in next(
        line for line in submit.splitlines() if "--array=" in line
    )
    summary_line = next(
        line for line in submit.splitlines() if "SMOKE_PHASE=summarize" in line
    )
    assert '"$CPU_SCRIPT"' in summary_line
    assert "afterany:" in summary_line

    gpu = _text("slurm/run_replication_smoke.slurm")
    assert "#SBATCH --gres=gpu:1" in gpu
    assert "#SBATCH --time=01:00:00" in gpu
    assert "torch.cuda.is_available" in gpu
    assert "nvidia-smi" in gpu
    assert "replication_smoke.py" in gpu

    cpu = _text("slurm/run_replication_smoke_cpu.slurm")
    assert "--gres" not in cpu
    assert "--gpus" not in cpu
    assert "SLURM_JOB_GPUS" in cpu

    for wrapper in (gpu, cpu):
        activation = wrapper.index('conda activate "$CONDA_ENV"')
        assert wrapper.rfind("set +u", 0, activation) >= 0
        assert wrapper.find("set -u", activation) > activation


def test_full_replication_conda_activation_is_safe_in_noninteractive_slurm_shells():
    for name in ("run_replication_gpu.slurm", "run_replication_cpu.slurm"):
        wrapper = _text(f"slurm/{name}")
        activation = wrapper.index('conda activate "$CONDA_ENV"')
        assert wrapper.rfind("set +u", 0, activation) >= 0
        assert wrapper.find("set -u", activation) > activation


def _run_replication_dispatch(tmp_path, *, replication_status, interface_outcome):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run_state.json").write_text(
        json.dumps(
            {
                "replication_status": replication_status,
                "stages": {
                    "interface": {
                        "status": "complete",
                        "outcome": interface_outcome,
                        "artifacts": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    executable_root = tmp_path / "bin"
    executable_root.mkdir()
    log = tmp_path / "sbatch.log"
    sbatch = executable_root / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\necho 9001\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{executable_root}:{os.environ['PATH']}",
        "OUTPUT": str(output),
        "SLURM_JOB_ID": "8000",
        "SLURM_SUBMIT_DIR": str(ROOT),
        "FAKE_SBATCH_LOG": str(log),
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/dispatch_replication_after_interface.sh")],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return log.read_text(encoding="utf-8").splitlines()


def test_measurement_failure_dispatches_only_a_cpu_report(tmp_path):
    submissions = _run_replication_dispatch(
        tmp_path,
        replication_status="measurement_failure",
        interface_outcome="measurement_failure",
    )
    assert len(submissions) == 1
    assert "STAGE=report" in submissions[0]
    assert "run_replication_cpu.slurm" in submissions[0]
    assert "run_replication_gpu.slurm" not in submissions[0]


def test_passing_interface_dispatches_resource_separated_pipeline(tmp_path):
    submissions = _run_replication_dispatch(
        tmp_path,
        replication_status="running",
        interface_outcome="pass",
    )
    assert len(submissions) == 8
    stages = {
        line.split("STAGE=", 1)[1].split(",", 1)[0] for line in submissions
    }
    assert stages == {
        "behavior",
        "model_comparison",
        "freeze_theory",
        "counterfactuals",
        "mechanism",
        "generalization",
        "specificity",
        "report",
    }
    for line in submissions:
        stage = line.split("STAGE=", 1)[1].split(",", 1)[0]
        expected = (
            "run_replication_gpu.slurm"
            if stage in {"behavior", "mechanism", "generalization", "specificity"}
            else "run_replication_cpu.slurm"
        )
        assert expected in line
