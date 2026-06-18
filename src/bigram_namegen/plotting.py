"""Plotting helpers for inspecting training progress."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def plot_loss_curve(
    loss_history: List[float],
    baseline: Optional[float] = None,
    baseline_label: str = "Counting model loss",
    title: str = "Neural network training loss",
    output_path: str | Path = "nn_loss.png",
) -> Path:
    """Plot the neural model's loss curve, optionally against a baseline.

    Saves the figure to ``output_path`` and returns that path. Matplotlib
    is imported lazily so the rest of the package has no hard dependency
    on a display backend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="#4C72B0", linewidth=2)

    if baseline is not None:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"{baseline_label} ({baseline:.4f})",
        )
        plt.legend()

    plt.xlabel("Step")
    plt.ylabel("Loss (NLL)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
