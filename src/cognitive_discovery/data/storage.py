from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def records_frame(records) -> pd.DataFrame:
    return pd.DataFrame(
        [record.to_dict() if hasattr(record, "to_dict") else dict(record) for record in records]
    )


def write_records(records, path: str | Path) -> Path:
    frame = records_frame(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            temporary = path.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            return path
        except (ImportError, ModuleNotFoundError):
            path = path.with_suffix(".csv.gz")
    if path.suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".csv":
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
        temporary.replace(path)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in frame.to_dict("records")),
            encoding="utf-8",
        )
        temporary.replace(path)
    return path


def read_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)
