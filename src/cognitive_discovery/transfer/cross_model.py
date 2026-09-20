"""Cross-model transfer-topology comparison with task-pair resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_transfer_topology(tables: dict[str, pd.DataFrame], *, samples=2000, seed=28001) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    models = sorted(tables)
    for left_index, left_name in enumerate(models):
        left = tables[left_name]
        for right_name in models[left_index + 1:]:
            right = tables[right_name]
            keys = ["theory", "source_task", "target_task"]
            if "response_mapping" in left and "response_mapping" in right:
                keys.append("response_mapping")
            merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
            merged = merged[(merged.source_task != merged.target_task) & merged.global_cfr_left.notna() & merged.global_cfr_right.notna()]
            if len(merged) < 3:
                rows.append({"model_left": left_name, "model_right": right_name, "task_pairs": len(merged), "correlation": np.nan, "bootstrap_low": np.nan, "bootstrap_high": np.nan, "permutation_p": np.nan})
                continue
            x, y = merged.global_cfr_left.to_numpy(), merged.global_cfr_right.to_numpy()
            correlation = float(np.corrcoef(x, y)[0, 1])
            boot = []
            for _ in range(int(samples)):
                indices = rng.integers(0, len(x), len(x))
                value = np.corrcoef(x[indices], y[indices])[0, 1]
                if np.isfinite(value): boot.append(value)
            permutation = [np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(int(samples))]
            rows.append({"model_left": left_name, "model_right": right_name, "task_pairs": len(merged), "correlation": correlation, "bootstrap_low": float(np.quantile(boot, .025)), "bootstrap_high": float(np.quantile(boot, .975)), "permutation_p": float((1 + np.sum(np.abs(permutation) >= abs(correlation))) / (1 + len(permutation)))})
    return pd.DataFrame(rows)
