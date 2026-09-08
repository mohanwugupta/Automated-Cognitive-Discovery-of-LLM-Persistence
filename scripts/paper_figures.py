"""Render publication figures from the audited tables, never rerun experiments."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / "paper/generated"
frame = pd.read_csv(root / "controllers.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
y = np.arange(len(frame))
for prefix, shift, color, label in [("test", -.11, "#245a81", "Pair test"), ("holdout", .11, "#b86b30", "Task holdout")]:
    estimate = frame[f"{prefix}_estimate"].to_numpy()
    error = np.array([estimate - frame[f"{prefix}_ci_lower"], frame[f"{prefix}_ci_upper"] - estimate])
    ax.errorbar(estimate, y+shift, xerr=error, fmt="o", color=color, capsize=3, label=label)
ax.set_yticks(y, ["Latent context (L30, r8)", "Dual history (L28, r2)", "Outcome history (L30, r2)"])
ax.invert_yaxis()
ax.set(xlabel="Global counterfactual recovery", xlim=(0.35, 1.02))
ax.grid(axis="x", alpha=.18)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", frameon=False)
fig.savefig(root / "controllers.pdf", metadata={"CreationDate": None})
plt.close(fig)
