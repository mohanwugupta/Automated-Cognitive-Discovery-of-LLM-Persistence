"""Submission must enqueue preparation without importing Python on the login node."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def environment(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "sbatch.log"
    python_marker = tmp_path / "python-called"
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
