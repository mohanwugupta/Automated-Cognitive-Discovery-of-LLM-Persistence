from pathlib import Path


ROOT = Path(__file__).parents[1]


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_round1_cpu_only_phases_never_request_a_gpu():
    submit = _text("scripts/submit_discovery.sh")
    for phase in ("tests", "pilot_finalize", "discovery_finalize"):
        assert f'PHASE={phase} "$CPU_SCRIPT"' in submit
    for phase in ("pilot_collect", "discovery_collect", "analyze", "validation"):
        assert f'PHASE={phase}' in submit
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
    ):
        text = _text(script)
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
