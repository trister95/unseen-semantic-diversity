"""Compute occurrence- and type-based Rao's quadratic entropy per group from a combined NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def cosine_distance_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normed = vecs / safe
    sim = normed @ normed.T
    dm = 1.0 - sim
    np.fill_diagonal(dm, 0.0)
    return np.clip(dm, 0.0, None)


def rao_q(counts: np.ndarray, dm: np.ndarray, uniform: bool) -> float:
    if dm.shape[0] <= 1:
        return 0.0
    if uniform:
        S = dm.shape[0]
        p = np.full(S, 1.0 / S)
    else:
        total = float(counts.sum())
        if total <= 0:
            return float("nan")
        p = counts / total
    return float(p @ dm @ p)


def effective_number(q: float) -> Optional[float]:
    if not np.isfinite(q) or q >= 1.0:
        return None
    return float(1.0 / (1.0 - q))


def load_groups(npz_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"group", "keys", "vecs", "counts"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ file {npz_path} is missing arrays: {sorted(missing)}")
        groups = data["group"].astype(str)
        keys = data["keys"].astype(str)
        vecs = data["vecs"].astype(float)
        counts = data["counts"].astype(float)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for g in np.unique(groups):
        mask = groups == g
        if not mask.any():
            continue
        out[str(g)] = {
            "keys": keys[mask],
            "vecs": vecs[mask],
            "counts": counts[mask],
        }
    return out


def compute_rows(group_data: Dict[str, Dict[str, np.ndarray]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group, payload in group_data.items():
        vecs = payload["vecs"]
        counts = payload["counts"]
        if vecs.shape[0] == 0:
            continue
        dm = cosine_distance_matrix(vecs)
        q_occ = rao_q(counts, dm, uniform=False)
        q_type = rao_q(counts, dm, uniform=True)
        rows.append({
            "group": group,
            "n_types": int(vecs.shape[0]),
            "n_tokens": int(counts.sum()),
            "rao_q_occurrence": q_occ,
            "rao_q_type": q_type,
            "effective_n_occurrence": effective_number(q_occ),
            "effective_n_type": effective_number(q_type),
        })

    def sortable(g: object) -> float:
        try:
            return float(g)  # decades sort numerically; non-numeric labels sink to the end
        except (TypeError, ValueError):
            return float("inf")

    rows.sort(key=lambda r: sortable(r["group"]))
    return rows


def write_jsonl(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def plot_rao_over_time(rows: List[Dict[str, object]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib unavailable, skipping plot ({exc})")
        return

    numeric: List[Tuple[int, float, float]] = []
    for r in rows:
        try:
            g = int(float(r["group"]))
        except (TypeError, ValueError):
            continue
        numeric.append((g, float(r["rao_q_occurrence"]), float(r["rao_q_type"])))
    if not numeric:
        return
    numeric.sort()
    decades = np.array([n[0] for n in numeric])
    q_occ = np.array([n[1] for n in numeric])
    q_type = np.array([n[2] for n in numeric])

    plt.figure(figsize=(10, 6))
    plt.plot(decades, q_occ, marker="o", label="Rao Q (occurrence-weighted)")
    plt.plot(decades, q_type, marker="s", label="Rao Q (type-uniform)")
    plt.xlabel("Decade")
    plt.ylabel("Rao's quadratic entropy")
    plt.title("Rao's Q over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute occurrence- and type-based Rao's Q per group from a combined NPZ")
    p.add_argument("--input", required=True, help="Combined NPZ with group/keys/vecs/counts arrays")
    p.add_argument("--output", required=True, help="Output JSONL path (one row per group)")
    p.add_argument("--plot-output", default=None, help="Optional PNG path for a time-series plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    group_data = load_groups(Path(args.input))
    rows = compute_rows(group_data)
    write_jsonl(rows, Path(args.output))
    if args.plot_output:
        plot_rao_over_time(rows, Path(args.plot_output))


if __name__ == "__main__":
    main()
