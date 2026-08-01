"""
Scale lab: benchmark LanceDB dense-vector index build time, on-disk size,
query latency (p50/p95), and recall@10 at 100k -> 1M -> 10M vectors.

Extracted from the original H100 notebook (Sec 15). It literally builds the
indexes and measures them, then extrapolates the trend to 100M vectors.

Dependencies: lancedb, pyarrow, numpy, pandas, matplotlib
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa

SCALE_DIM = 1024


def make_synthetic_vectors(n: int, d: int = SCALE_DIM) -> np.ndarray:
    """Generate normalized random vectors for scale benchmarking."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, d)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def _dir_gb(path: Path) -> float:
    """Total on-disk size of a directory in gigabytes."""
    return round(
        sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024**3, 3
    )


def _arrow(vecs: np.ndarray) -> pa.Table:
    """Efficient ingest path for tens of millions of rows."""
    n, d = vecs.shape
    flat = pa.array(vecs.reshape(-1).astype("float32"))
    vcol = pa.FixedSizeListArray.from_arrays(flat, d)
    return pa.table({"id": pa.array(np.arange(n, dtype="int64")), "vector": vcol})


def run_scale_lab(
    sizes: list[int],
    scratch_dir: Path,
    dim: int = SCALE_DIM,
    n_queries: int = 50,
    recall_sample: int = 20,
) -> pd.DataFrame:
    """
    Build LanceDB indexes at the requested sizes and report metrics.

    Args:
        sizes: list of vector counts, e.g. [100_000, 1_000_000, 10_000_000].
        scratch_dir: directory to create benchmark databases under.
        dim: vector dimension.
        n_queries: number of random queries for latency measurement.
        recall_sample: number of queries for exact vs ANN recall comparison.

    Returns:
        DataFrame with columns: n, build_s, disk_gb, p50_ms, p95_ms, recall@10.
    """
    rows = []
    for n in sizes:
        n = int(n)
        uri = str(scratch_dir / f"scale_{n}")
        print(f"[scale] building n={n:,} (dim={dim}) ...")

        vecs = make_synthetic_vectors(n, dim)
        db = lancedb.connect(uri)

        t0 = time.time()
        tbl = db.create_table("v", data=_arrow(vecs), mode="overwrite")

        # Build an ANN index so latency/recall are realistic at scale.
        # For small n (<100k) flat search is usually fine.
        if n >= 100_000:
            nsv = 64
            while dim % nsv and nsv > 1:
                nsv //= 2
            try:
                tbl.create_index(
                    metric="cosine",
                    num_partitions=int(min(4096, max(256, n ** 0.5))),
                    num_sub_vectors=nsv,
                )
            except Exception as e:
                print(f"   (ANN index skipped at n={n}: {e}; using flat search)")
        build_s = time.time() - t0

        # Latency benchmark
        rng = np.random.default_rng(0)
        qs = vecs[rng.integers(0, n, size=n_queries)]
        lats = []
        for q in qs:
            t = time.time()
            tbl.search(q).metric("cosine").limit(10).to_list()
            lats.append((time.time() - t) * 1000)

        # Recall@10 vs exact brute force
        sample = qs[:recall_sample]
        bf = (sample @ vecs.T).argsort(axis=1)[:, -10:]
        ann = [
            [r["id"] for r in tbl.search(q).metric("cosine").limit(10).to_list()]
            for q in sample
        ]
        recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ann, bf)]))

        rows.append(
            {
                "n": n,
                "build_s": round(build_s, 2),
                "disk_gb": _dir_gb(uri),
                "p50_ms": round(float(np.percentile(lats, 50)), 2),
                "p95_ms": round(float(np.percentile(lats, 95)), 2),
                "recall@10": round(recall, 3),
            }
        )
        print("  ->", rows[-1])

        del vecs, qs
        gc.collect()

    return pd.DataFrame(rows)


def fit_and_extrapolate(df: pd.DataFrame, target: int = 100_000_000) -> dict:
    """Linear extrapolation from measured points to a target vector count."""
    n = df["n"].values.astype(float)
    out = {"target": target}
    for col in ["build_s", "disk_gb", "p95_ms"]:
        a, b = np.polyfit(n, df[col].values, 1)
        out[col] = round(float(a * target + b), 2)
    return out


def cost_at_scale(extrap: dict, hourly: float = 1.90) -> dict:
    """Convert build seconds to wall-clock hours and a rough machine cost."""
    build_hr = extrap["build_s"] / 3600.0
    return {
        "projected_disk_gb": extrap["disk_gb"],
        "projected_p95_ms": extrap["p95_ms"],
        "projected_index_build_hours": round(build_hr, 2),
        "projected_index_build_cost_usd": round(build_hr * hourly, 2),
    }


def plot_scale_lab(df: pd.DataFrame, extrap: dict, save_path: str | None = None):
    """Plot measured metrics plus the 100M projection."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, col in zip(axes, ["p95_ms", "disk_gb", "build_s"]):
        ax.plot(df["n"], df[col], "-o", label="measured")
        ax.scatter([extrap["target"]], [extrap[col]], color="red", label="100M (projected)")
        ax.set_xscale("log")
        ax.set_xlabel("n vectors")
        ax.set_title(col)
        ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    scratch = Path("/mnt/scratch")
    sizes = [100_000, 1_000_000, 10_000_000]

    df = run_scale_lab(sizes, scratch)
    extrap = fit_and_extrapolate(df, target=100_000_000)
    cost = cost_at_scale(extrap)

    print("\nMEASURED (up to 10M):\n", df.to_string(index=False))
    print("\nprojection to 100M:", json.dumps(cost, indent=2))
    plot_scale_lab(df, extrap, save_path="scale_lab.png")
