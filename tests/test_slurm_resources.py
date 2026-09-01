from pathlib import Path


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


def test_counterfactual_mechanistic_gpu_phases_all_run_qwen_forwards():
    runner = _text("run_causal_mechanistic.slurm")
    for phase in ("localize", "das", "validate", "circuit"):
        marker = f"\n  {phase})\n"
        assert marker in runner
        block = runner.split(marker, 1)[1].split(";;", 1)[0]
        assert "scripts/causal_mechanistic.py" in block
        assert '"$MODEL_PATH"' in block
    cpu = _text("run_causal_mechanistic_cpu.slurm")
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
    runner = _text("run_causal_specificity.slurm")
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
    runner = _text("run_causal_abstraction.slurm")
    evaluation = runner.rsplit("\n  evaluate)\n", 1)[1].split(";;", 1)[0]
    assert "scripts/causal_abstraction.py" in evaluation
    assert '"$MODEL_PATH"' in evaluation
    cpu_phases = runner.split("tests|prepare|dispatch|aggregate)", 1)[1].split(";;", 1)[
        0
    ]
    assert "CPU-only" in cpu_phases
