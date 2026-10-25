"""Submission must enqueue preparation without importing Python on the login node."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def environment(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "sbatch.log"
    python_marker = tmp_path / "python-called"
    model_path = tmp_path / "models" / "Qwen--Qwen3.5-4B"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}")
    for name, text in {
        "sbatch": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_SBATCH_LOG"\n'
        'count=$(wc -l < "$FAKE_SBATCH_LOG")\necho "$((9000 + count))"\n',
        "python": '#!/bin/bash\ntouch "$FAKE_PYTHON_MARKER"\nexit 97\n',
    }.items():
        path = binary / name
        path.write_text(text)
        path.chmod(0o755)
    return (
        {
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "REPO": str(ROOT),
            "DIAGNOSIS_OUTPUT": str(tmp_path / "new-run"),
            "DIAGNOSIS_PILOT": "1",
            "DIAGNOSIS_SHARDS": "8",
            "DIAGNOSIS_MAX_CONCURRENT": "2",
            "DIAGNOSIS_MODEL_PATH": str(model_path),
            "FAKE_SBATCH_LOG": str(log),
            "FAKE_PYTHON_MARKER": str(python_marker),
        },
        log,
        python_marker,
    )


def test_submission_queues_prepare_without_running_python(tmp_path):
    env, log, python_marker = environment(tmp_path)
    completed = subprocess.run(
        ["bash", "scripts/submit_alignment_diagnosis.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert not python_marker.exists(), "Python preparation ran before sbatch"
    assert not Path(env["DIAGNOSIS_OUTPUT"]).exists()
    submissions = log.read_text().splitlines()
    assert len(submissions) == 5
    for index, phase in enumerate(
        ("prepare", "initial", "freeze", "intervention", "analyze")
    ):
        assert f"DIAGNOSIS_PHASE={phase}" in submissions[index]
        wrapper = "gpu" if phase in ("initial", "intervention") else "cpu"
        assert f"run_alignment_diagnosis_{wrapper}.slurm" in submissions[index]
        if wrapper == "gpu":
            assert "--array=0-7%2" in submissions[index]
        else:
            assert "--array" not in submissions[index]
        if index:
            assert f"--dependency=afterok:{9000+index}" in submissions[index]
        assert f"{phase}={9001+index}" in completed.stdout
    assert "Submitting prepare" in completed.stdout


def test_submission_reports_existing_output_before_queueing(tmp_path):
    env, log, python_marker = environment(tmp_path)
    Path(env["DIAGNOSIS_OUTPUT"]).mkdir()
    completed = subprocess.run(
        ["bash", "scripts/submit_alignment_diagnosis.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    assert not log.exists()
    assert not python_marker.exists()


@pytest.mark.parametrize(
    "phase,expected_collectors", [("initial", 2), ("intervention", 1)]
)
def test_gpu_wrapper_passes_local_model_path_offline(
    tmp_path, phase, expected_collectors
):
    env, _, _ = environment(tmp_path)
    binary = tmp_path / "bin"
    calls = tmp_path / "python.log"
    for name, script in {
        "module": "#!/bin/bash\nexit 0\n",
        "conda": "#!/bin/bash\nexit 0\n",
        "python": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_COLLECTOR_LOG"\n'
        '[[ "${HF_HUB_OFFLINE:-}" == 1 && "${TRANSFORMERS_OFFLINE:-}" == 1 ]]\n',
    }.items():
        (binary / name).write_text(script)
        (binary / name).chmod(0o755)
    env.update(DIAGNOSIS_PHASE=phase, FAKE_COLLECTOR_LOG=str(calls))
    completed = subprocess.run(
        ["bash", "slurm/run_alignment_diagnosis_gpu.slurm"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    collectors = [
        line for line in calls.read_text().splitlines() if " collect " in line
    ]
    assert len(collectors) == expected_collectors
    for line in collectors:
        assert f"--model-path {env['DIAGNOSIS_MODEL_PATH']}" in line
        assert "--online" not in line
        assert "--shard-count 8" in line
        assert "--shard-index 0" in line
