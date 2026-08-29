"""Validated MLP and GRU regression ceilings for the continuous policy logit."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cognitive_discovery.models.features import FeatureEncoder, all_observable_features
from cognitive_discovery.models.fitting import regression_metrics


@dataclass
class FlexibleFit:
    kind: str
    prediction: np.ndarray
    metrics: dict[str, float]
    selected_epochs: int
    feature_names: tuple[str, ...]
    model: object
    encoder: FeatureEncoder


def _sequences(frame, matrix, target, device):
    import torch

    if "episode_id" not in frame:
        frame = frame.copy()
        frame["episode_id"] = np.arange(len(frame)).astype(str)
    groups = list(frame.groupby("episode_id", sort=False).groups.values())
    maximum = max(len(group) for group in groups)
    x = torch.zeros((len(groups), maximum, matrix.shape[1]), device=device)
    y = torch.zeros((len(groups), maximum), device=device)
    mask = torch.zeros((len(groups), maximum), dtype=torch.bool, device=device)
    positions = np.full((len(groups), maximum), -1, dtype=int)
    for group_index, indices in enumerate(groups):
        ordered = sorted(indices, key=lambda index: int(frame.loc[index, "step"]))
        length = len(ordered)
        x[group_index, :length] = torch.tensor(matrix[ordered], dtype=torch.float32, device=device)
        y[group_index, :length] = torch.tensor(target[ordered], dtype=torch.float32, device=device)
        mask[group_index, :length] = True
        positions[group_index, :length] = ordered
    return x, y, mask, positions


def fit_flexible_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    kind: str,
    seed: int = 0,
    hidden_size: int = 64,
    learning_rate: float = 0.003,
    max_epochs: int = 200,
    patience: int = 20,
) -> FlexibleFit:
    import torch

    if kind not in {"mlp", "gru"}:
        raise ValueError("flexible neural model must be 'mlp' or 'gru'")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    features = all_observable_features(train)
    encoder = FeatureEncoder(features)
    x_all, names = encoder.fit_transform(train.reset_index(drop=True))
    x_test, _ = encoder.transform(test.reset_index(drop=True))
    y_all = train.persistence_logit.to_numpy(dtype=np.float32)
    y_mean, y_scale = float(y_all.mean()), float(y_all.std())
    y_scale = y_scale if y_scale > 1e-8 else 1.0
    y_all = (y_all - y_mean) / y_scale
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(train))
    validation_count = max(1, int(0.2 * len(train)))
    validation_indices = order[:validation_count]
    fit_indices = order[validation_count:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if kind == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(x_all.shape[1], int(hidden_size)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_size), max(4, int(hidden_size) // 2)),
            torch.nn.GELU(),
            torch.nn.Linear(max(4, int(hidden_size) // 2), 1),
        ).to(device)
        x_fit = torch.tensor(x_all[fit_indices], dtype=torch.float32, device=device)
        y_fit = torch.tensor(y_all[fit_indices], dtype=torch.float32, device=device)
        x_validation = torch.tensor(
            x_all[validation_indices], dtype=torch.float32, device=device
        )
        y_validation = torch.tensor(
            y_all[validation_indices], dtype=torch.float32, device=device
        )

        def train_loss():
            return torch.nn.functional.mse_loss(model(x_fit).squeeze(-1), y_fit)

        def validation_loss():
            return torch.nn.functional.mse_loss(
                model(x_validation).squeeze(-1), y_validation
            )

        def apply_test():
            return model(torch.tensor(x_test, dtype=torch.float32, device=device)).squeeze(-1)

    else:
        class PolicyGRU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project = torch.nn.Linear(x_all.shape[1], int(hidden_size))
                self.gru = torch.nn.GRU(int(hidden_size), int(hidden_size), batch_first=True)
                self.output = torch.nn.Linear(int(hidden_size), 1)

            def forward(self, values):
                sequence, _ = self.gru(torch.nn.functional.gelu(self.project(values)))
                return self.output(sequence).squeeze(-1)

        model = PolicyGRU().to(device)
        # Sequence-safe split: validation episodes are selected as whole groups.
        train_frame = train.reset_index(drop=True)
        episodes = np.asarray(sorted(train_frame.episode_id.astype(str).unique()))
        rng.shuffle(episodes)
        validation_episodes = set(episodes[: max(1, int(0.2 * len(episodes)))])
        fit_mask = ~train_frame.episode_id.astype(str).isin(validation_episodes)
        validation_mask = ~fit_mask
        fit_frame = train_frame[fit_mask].reset_index(drop=True)
        validation_frame = train_frame[validation_mask].reset_index(drop=True)
        fit_x = x_all[fit_mask.to_numpy()]
        validation_x = x_all[validation_mask.to_numpy()]
        fit_y = y_all[fit_mask.to_numpy()]
        validation_y = y_all[validation_mask.to_numpy()]
        fit_batch = _sequences(fit_frame, fit_x, fit_y, device)
        validation_batch = _sequences(validation_frame, validation_x, validation_y, device)

        def sequence_loss(batch):
            prediction = model(batch[0])
            return torch.nn.functional.mse_loss(prediction[batch[2]], batch[1][batch[2]])

        def train_loss():
            return sequence_loss(fit_batch)

        def validation_loss():
            return sequence_loss(validation_batch)

        test_frame = test.reset_index(drop=True)
        test_batch = _sequences(
            test_frame,
            x_test,
            np.zeros(len(test_frame), dtype=np.float32),
            device,
        )

        def apply_test():
            raw = model(test_batch[0]).detach().cpu().numpy()
            prediction = np.zeros(len(test_frame), dtype=float)
            for group in range(raw.shape[0]):
                for step in range(raw.shape[1]):
                    position = test_batch[3][group, step]
                    if position >= 0:
                        prediction[position] = raw[group, step]
            return torch.tensor(prediction, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad()
        loss = train_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            current = float(validation_loss())
        if current < best_loss - 1e-6:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("flexible model failed before producing a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = apply_test().detach().cpu().numpy() * y_scale + y_mean
    return FlexibleFit(
        kind,
        np.asarray(prediction),
        regression_metrics(test.persistence_logit, prediction),
        best_epoch,
        names,
        model,
        encoder,
    )

