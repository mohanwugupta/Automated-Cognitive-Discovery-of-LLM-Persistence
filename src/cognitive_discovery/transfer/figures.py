"""Figures rendered only from frozen validation/transfer tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def transfer_matrix_figure(table: str | Path, output: str | Path, *, task_order):
    frame = pd.read_csv(table)
    required = {"source_task", "target_task", "global_cfr"}
    if required - set(frame):
        raise ValueError("transfer figure input has the wrong schema")
    matrix = frame.pivot(index="source_task", columns="target_task", values="global_cfr").reindex(index=task_order, columns=task_order)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(task_order)), task_order, rotation=45, ha="right")
    axis.set_yticks(range(len(task_order)), task_order)
    axis.set_xlabel("target task")
    axis.set_ylabel("source task")
    figure.colorbar(image, ax=axis, label="global CFR v1")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def computational_validation_figure(parameter_table, recovery_table, output):
    parameters, recovery = pd.read_csv(parameter_table), pd.read_csv(recovery_table)
    matrix = recovery.pivot(index="teacher_model", columns="candidate_model", values="recovery_probability")
    summary = parameters.groupby("sample_size").absolute_error.mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    image = axes[0].imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    axes[0].set_yticks(range(len(matrix.index)), matrix.index)
    axes[0].set_title("Synthetic model recovery")
    figure.colorbar(image, ax=axes[0], label="selection probability")
    axes[1].plot(summary.index, summary.values, marker="o")
    axes[1].set(xlabel="sample size", ylabel="mean absolute parameter error", title="Parameter recovery")
    return _save(figure, output)


def qwen_self_replication_figure(comparison_table, output):
    frame = pd.read_csv(comparison_table)
    required = {"quantity", "historical_qwen", "prospective_qwen"}
    if required - set(frame): raise ValueError("Qwen comparison table has the wrong schema")
    numeric = frame.dropna(subset=["historical_qwen", "prospective_qwen"])
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(numeric))
    axis.bar(positions - .18, numeric.historical_qwen, width=.36, label="historical")
    axis.bar(positions + .18, numeric.prospective_qwen, width=.36, label="prospective")
    axis.set_xticks(positions, numeric.quantity, rotation=45, ha="right")
    axis.legend()
    return _save(figure, output)


def cross_model_figure(cross_model_table, output):
    frame = pd.read_csv(cross_model_table)
    required = {"model_left", "model_right", "correlation", "bootstrap_low", "bootstrap_high"}
    if required - set(frame): raise ValueError("cross-model table has the wrong schema")
    labels = frame.model_left.astype(str) + " vs " + frame.model_right.astype(str)
    figure, axis = plt.subplots(figsize=(8, 4))
    errors = np.vstack((frame.correlation - frame.bootstrap_low, frame.bootstrap_high - frame.correlation))
    axis.errorbar(range(len(frame)), frame.correlation, yerr=errors, fmt="o")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(range(len(frame)), labels, rotation=45, ha="right")
    axis.set_ylabel("task-pair transfer correlation")
    return _save(figure, output)


def transfer_prediction_figure(predictions_table, coefficients_table, output):
    predictions, coefficients = pd.read_csv(predictions_table), pd.read_csv(coefficients_table)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(predictions.observed_global_cfr, predictions.predicted_global_cfr)
    limits = [min(predictions.observed_global_cfr.min(), predictions.predicted_global_cfr.min()), max(predictions.observed_global_cfr.max(), predictions.predicted_global_cfr.max())]
    axes[0].plot(limits, limits, linestyle="--", color="black")
    axes[0].set(xlabel="observed held-out transfer", ylabel="predicted transfer")
    summary = coefficients.groupby("predictor").coefficient.mean().sort_values()
    axes[1].barh(summary.index, summary.values)
    axes[1].set_xlabel("mean held-out-fold coefficient")
    return _save(figure, output)


def geometry_transfer_figure(joined_table, output):
    frame = pd.read_csv(joined_table)
    if {"subspace_overlap", "global_cfr"} - set(frame): raise ValueError("geometry/transfer table has the wrong schema")
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(frame.subspace_overlap, frame.global_cfr)
    axis.set(xlabel="controller subspace overlap", ylabel="global CFR v1")
    return _save(figure, output)


def _save(figure, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
