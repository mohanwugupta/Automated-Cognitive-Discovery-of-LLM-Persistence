"""Immutable acquisition and structural validation for the Gate-B dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import pandas as pd


EXPECTED_FACTUAL_COLUMNS = (
    "subjectid",
    "type",
    "reward",
    "choice",
    "false_start",
    "rt",
    "key",
)


class BenchmarkAcquisitionError(RuntimeError):
    pass


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, record: dict) -> None:
    if not path.is_file():
        raise BenchmarkAcquisitionError(f"benchmark file is absent: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise BenchmarkAcquisitionError(f"benchmark size mismatch: {path.name}")
    for algorithm in ("md5", "sha256"):
        if _hash(path, algorithm) != record[algorithm]:
            raise BenchmarkAcquisitionError(
                f"benchmark {algorithm} mismatch: {path.name}"
            )


def validate_factual_dataset(path: str | Path) -> dict:
    """Validate the owner-frozen 143 x 2 x 96 factual subset without filtering."""

    path = Path(path)
    frame = pd.read_csv(path)
    if tuple(frame.columns) != EXPECTED_FACTUAL_COLUMNS:
        raise BenchmarkAcquisitionError(
            f"unexpected factual columns: {tuple(frame.columns)}"
        )
    if len(frame) != 27_456:
        raise BenchmarkAcquisitionError(
            f"expected 27,456 factual rows, observed {len(frame):,}"
        )
    participants = sorted(frame.subjectid.astype(int).unique().tolist())
    if len(participants) != 143:
        raise BenchmarkAcquisitionError(
            f"expected 143 retained participants, observed {len(participants)}"
        )
    counts = frame.groupby("subjectid", sort=False).size()
    if set(counts.astype(int)) != {192}:
        raise BenchmarkAcquisitionError(
            "every retained participant must have exactly 192 factual trials"
        )
    trial_index = frame.groupby("subjectid", sort=False).cumcount()
    session = (trial_index // 96) + 1
    session_counts = frame.assign(_session=session).groupby(
        ["subjectid", "_session"], sort=False
    ).size()
    if set(session_counts.astype(int)) != {96} or set(session.unique()) != {1, 2}:
        raise BenchmarkAcquisitionError(
            "factual file does not form two ordered 96-trial sessions"
        )
    return {
        "rows": int(len(frame)),
        "participants": len(participants),
        "participant_ids_sha256": hashlib.sha256(
            json.dumps(participants, separators=(",", ":")).encode()
        ).hexdigest(),
        "sessions_per_participant": 2,
        "trials_per_session": 96,
        "columns": list(EXPECTED_FACTUAL_COLUMNS),
        "false_start_value_counts": {
            str(key): int(value)
            for key, value in frame.false_start.value_counts().sort_index().items()
        },
        "no_response_rows": int((frame.key.astype(int) == 0).sum()),
        "choice_zero_rows": int((frame.choice.astype(int) == 0).sum()),
        "additional_rows_removed": 0,
    }


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "cognitive-discovery-gate-b/1"})
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(request, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)


def acquire_benchmark_dataset(
    *,
    root: str | Path,
    output: str | Path | None = None,
    execute: bool = False,
) -> dict:
    root = Path(root).resolve()
    manifest_path = root / "validation/external_benchmark/DATASET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("article_version") != 3:
        raise BenchmarkAcquisitionError("Gate B requires Figshare article Version 3")
    output = Path(
        output
        or root / "validation/external_benchmark/raw/figshare_10042319_v3"
    ).resolve()
    plan = {
        "schema_version": "external-benchmark-acquisition-v1",
        "mode": "execute" if execute else "dry-run",
        "article_id": manifest["article_id"],
        "article_version": manifest["article_version"],
        "version_doi": manifest["version_doi"],
        "license": manifest["license"],
        "manifest_sha256": _hash(manifest_path, "sha256"),
        "output": str(output),
        "files": [record["name"] for record in manifest["files"]],
    }
    if not execute:
        return plan
    output.mkdir(parents=True, exist_ok=True)
    file_records = []
    for record in manifest["files"]:
        destination = output / record["name"]
        if not destination.exists():
            _download(record["url"], destination)
        validate_file(destination, record)
        file_records.append(
            {
                "name": record["name"],
                "size_bytes": destination.stat().st_size,
                "md5": _hash(destination, "md5"),
                "sha256": _hash(destination, "sha256"),
                "analysis_role": record["analysis_role"],
            }
        )
    factual = validate_factual_dataset(output / "factual.csv")
    result = {
        **plan,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
        "factual_subset": factual,
        "verified": True,
    }
    record_path = output / "acquisition_record.json"
    record_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and hash-verify the frozen Sugawara–Katahira benchmark"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = acquire_benchmark_dataset(
        root=args.root, output=args.output, execute=args.execute
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
