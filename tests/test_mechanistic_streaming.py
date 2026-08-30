from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from cognitive_discovery.mechanistic.activations.streaming_stats import (
    LayerTemporaryCache,
    StreamingMean,
    StreamingRidge,
)


def test_streaming_mean_equals_full_matrix_mean():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(31, 9))
    streamed = StreamingMean()
    streamed.update(matrix[:3])
    streamed.update(matrix[3:19])
    streamed.update(matrix[19:])
    assert streamed.count == len(matrix)
    assert np.allclose(streamed.mean, matrix.mean(axis=0))
    assert np.allclose(streamed.variance, matrix.var(axis=0, ddof=1))


def test_streaming_ridge_equals_batch_ridge():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(80, 6))
    target = matrix @ rng.normal(size=6) + rng.normal(scale=0.01, size=80)
    streamed = StreamingRidge()
    for start in range(0, len(matrix), 11):
        streamed.update(matrix[start : start + 11], target[start : start + 11])
    solution = streamed.solve(alpha=0.7)
    batch = Ridge(alpha=0.7).fit(matrix, target)
    assert np.allclose(solution.coefficient, batch.coef_)
    assert np.isclose(solution.intercept, batch.intercept_)
    assert np.allclose(solution.predict(matrix), batch.predict(matrix))


def test_layer_temporary_cache_is_deleted_after_layer(tmp_path):
    with LayerTemporaryCache(tmp_path, 3) as directory:
        path = Path(directory)
        (path / "tiny.bin").write_bytes(b"temporary")
        assert path.exists()
    assert not path.exists()
