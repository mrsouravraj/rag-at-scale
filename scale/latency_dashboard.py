"""
Latency & cost dashboard: aggregate per-stage latencies and build a headline metrics card.

Extracted from the original H100 notebook (Sec 16). Use it after a batch of RAG answers
to see where the wall-clock time goes (generation and verification usually dominate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


class HasLatencies(Protocol):
    """Anything with a `latencies` dict like {"stage": seconds, ...}."""
    latencies: dict[str, float]


def aggregate_latencies(results: list[HasLatencies]) -> pd.DataFrame:
    """
    Aggregate per-stage latencies across a batch of results.

    Returns a DataFrame with columns: stage, p50_s, p95_s, mean_s.
    """
    stages: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.latencies.items():
            stages.setdefault(k, []).append(v)

    rows = [
        {
            "stage": k,
            "p50_s": round(float(np.percentile(v, 50)), 3),
            "p95_s": round(float(np.percentile(v, 95)), 3),
            "mean_s": round(float(np.mean(v)), 3),
        }
        for k, v in stages.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_s", ascending=False)


def plot_stage_latency(df: pd.DataFrame, save_path: str | None = None):
    """Bar chart of per-stage p95 latency."""
    import matplotlib.pyplot as plt

    d = df[df["stage"] != "total"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(d["stage"], d["p95_s"])
    ax.set_xlabel("p95 latency (s)")
    ax.set_title("Per-stage latency (p95)")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def headline_metrics(
    metrics: dict[str, Any],
    confusion_matrix: np.ndarray | None,
    tau_star: float,
    verifier_eval: dict,
    scale_cost: dict,
    lat_df: pd.DataFrame | None,
) -> dict:
    """
    Build a single headline-metrics dict for a run.

    Args:
        metrics: base metrics dict (coverage, accuracy, etc.).
        confusion_matrix: 2x2 matrix [answerable, unanswerable] x [answered, abstained].
        tau_star: selected abstention threshold.
        verifier_eval: dict with at least an "auroc" key.
        scale_cost: output of cost_at_scale() from scale_lab.py.
        lat_df: latency DataFrame from aggregate_latencies().

    Returns:
        A dict combining all headline numbers.
    """
    out = dict(metrics)

    if confusion_matrix is not None:
        unans = int(confusion_matrix[1].sum())
        if unans:
            out["abstention_on_unanswerable"] = round(int(confusion_matrix[1, 1]) / unans, 3)
            out["hallucination_on_unanswerable"] = round(int(confusion_matrix[1, 0]) / unans, 3)

        ans = int(confusion_matrix[0].sum())
        if ans:
            out["coverage_on_answerable"] = round(int(confusion_matrix[0, 0]) / ans, 3)

    out["tau_star"] = tau_star
    out["verifier_auroc"] = verifier_eval.get("auroc")
    out["scale_projection_100M"] = scale_cost

    if lat_df is not None and len(lat_df):
        tot = lat_df[lat_df["stage"] == "total"]
        if len(tot):
            out["latency_p95_s"] = float(tot["p95_s"].iloc[0])

    return out
