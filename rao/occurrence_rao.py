"""Stage 3 of the Rao occurrence pipeline.

Compute Rao's quadratic entropy directly on the cloud of *occurrence*
embeddings within each group (default: decade). Unlike type-level FD,
this preserves the within-type contextual variance produced by the
embedding model -- which is the whole reason to use contextual
embeddings in the first place.

Math
----
For cosine distance on L2-normalized vectors ``v_k``:

    Q = (1/N^2) * sum_{k,l} d(v_k, v_l)
      = (1/N^2) * sum_{k,l} (1 - v_k . v_l)
      = 1 - (1/N^2) * || sum_k v_k ||^2
      = 1 - || centroid ||^2

so we compute one centroid per group and report ``1 - ||centroid||^2``.
O(N*D), no N x N distance matrix.

Output
------
JSONL with one row per group:

    {"group": "1620", "n_occurrences": 4218, "n_unique_types": 312,
     "rao_q_occurrence": 0.234, "centroid_norm": 0.875}

Example
-------
    python occurrence_rao.py \\
        --occurrences /content/work/occurrences.npz \\
        --metadata /content/drive/MyDrive/dbnl/dbnl_metadata.csv \\
        --output /content/work/rao_occurrence.jsonl \\
        --plot-output /content/work/rao_occurrence.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from build_corpus import doc_id_to_ti_id


def load_doc_year_map(metadata_path: str) -> Dict[str, int]:
    """Return ``{ti_id: jaar}`` from dbnl_metadata.csv (skips the 'sep=|' line)."""
    out: Dict[str, int] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first.lstrip().lower().startswith("sep="):
            f.seek(0)
        reader = csv.DictReader(f, delimiter="|", quotechar='"')
        for row in reader:
            ti = (row.get("ti_id") or "").strip()
            jaar_s = (row.get("jaar") or "").strip()
            if not ti or not jaar_s.isdigit():
                continue
            out[ti] = int(jaar_s)
    return out


def group_label(jaar: int, mode: str, threshold: Optional[int]) -> str:
    if mode == "decade":
        return str((jaar // 10) * 10)
    if mode == "half-century":
        return str((jaar // 50) * 50)
    if mode == "century":
        return str((jaar // 100) * 100)
    if mode == "split":
        if threshold is None:
            raise SystemExit("--year-threshold required for --group-by split")
        return f"pre{threshold}" if jaar < threshold else f"post{threshold}"
    raise SystemExit(f"unknown --group-by: {mode}")


def occurrence_rao_q(vecs: np.ndarray) -> float:
    """Rao Q over cosine distance on L2-normalized vectors, closed form.

    Renormalizes defensively, so this is robust to ``--no-normalize`` at embed
    time. Returns 0.0 for groups of size 0 or 1 (Rao Q is undefined / trivially
    zero with a single point).
    """
    if len(vecs) <= 1:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normed = vecs / safe
    centroid = normed.mean(axis=0)
    return float(1.0 - np.dot(centroid, centroid))


def write_jsonl(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def plot_q(rows: List[Dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib unavailable, skipping plot ({exc})")
        return

    numeric: List = []
    for r in rows:
        try:
            g = int(float(r["group"]))
        except (TypeError, ValueError):
            continue
        numeric.append((g, float(r["rao_q_occurrence"]), int(r["n_occurrences"])))
    if not numeric:
        return
    numeric.sort()
    decades = np.array([n[0] for n in numeric])
    q = np.array([n[1] for n in numeric])
    n_occ = np.array([n[2] for n in numeric])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(decades, q, marker="o", color="C0", label="Rao Q (occurrence-level)")
    ax1.set_xlabel("Decade")
    ax1.set_ylabel("Rao's Q (cosine, L2-normalized)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(decades, n_occ, width=8, alpha=0.2, color="C1")
    ax2.set_ylabel("Number of occurrences (bars)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    plt.title("Occurrence-level Rao Q over time")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    z = np.load(args.occurrences, allow_pickle=False)
    vecs = z["vecs"]
    doc_ids = z["doc_id"].astype(str)
    texts = z["text"].astype(str)
    n = len(vecs)
    print(f"loaded {n} occurrences (dim {vecs.shape[1]}) from {args.occurrences}")

    # Sanity: are embeddings actually L2-normalized?
    sample_norms = np.linalg.norm(vecs[: min(1000, n)], axis=1)
    if float(np.median(sample_norms)) < 0.9:
        print(
            f"warning: vectors don't look L2-normalized "
            f"(median ||v||={np.median(sample_norms):.3f}); renormalizing defensively",
            file=sys.stderr,
        )

    ti_to_jaar = load_doc_year_map(args.metadata)
    print(f"metadata: {len(ti_to_jaar)} ti_ids with parseable jaar")

    # Group occurrence indices by group label.
    by_group: Dict[str, List[int]] = defaultdict(list)
    missing_meta = 0
    for i, did in enumerate(doc_ids):
        ti = doc_id_to_ti_id(did)
        jaar = ti_to_jaar.get(ti)
        if jaar is None:
            missing_meta += 1
            continue
        by_group[group_label(jaar, args.group_by, args.year_threshold)].append(i)

    if missing_meta:
        pct = 100 * missing_meta / n
        print(
            f"warning: {missing_meta} occurrences ({pct:.1f}%) have no metadata jaar; skipping",
            file=sys.stderr,
        )

    rows: List[Dict[str, object]] = []
    for g, idxs in by_group.items():
        sub = vecs[idxs]
        sub_texts = texts[idxs]
        q = occurrence_rao_q(sub)
        type_set = {t.lower() for t in sub_texts} if args.lowercase else set(sub_texts)
        rows.append(
            {
                "group": g,
                "n_occurrences": len(idxs),
                "n_unique_types": len(type_set),
                "rao_q_occurrence": q,
                "centroid_norm": float(np.sqrt(max(0.0, 1.0 - q))),
            }
        )

    def sortable(r):
        try:
            return (0, int(r["group"]))
        except (TypeError, ValueError):
            return (1, str(r["group"]))

    rows.sort(key=sortable)

    write_jsonl(rows, Path(args.output))
    print(f"wrote {len(rows)} groups to {args.output}")

    print("\nper-group results:")
    print(f"  {'group':>12s}  {'n_occ':>7s}  {'n_types':>7s}  {'Rao Q':>8s}  {'cent norm':>10s}")
    for r in rows:
        print(
            f"  {str(r['group']):>12s}  {r['n_occurrences']:>7d}  {r['n_unique_types']:>7d}  "
            f"{r['rao_q_occurrence']:>8.4f}  {r['centroid_norm']:>10.4f}"
        )

    if args.plot_output:
        plot_q(rows, Path(args.plot_output))
        print(f"plot saved to {args.plot_output}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--occurrences", required=True, help="Occurrence NPZ from embed_mentions.py")
    p.add_argument("--metadata", required=True, help="dbnl_metadata.csv (provides ti_id -> jaar)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--plot-output", default=None, help="Optional PNG path for the Q-over-time plot")
    p.add_argument(
        "--group-by",
        choices=["decade", "half-century", "century", "split"],
        default="decade",
        help="How to derive the time-axis group from jaar (default: decade)",
    )
    p.add_argument("--year-threshold", type=int, default=None, help="For --group-by split: pre/post boundary year")
    p.add_argument("--lowercase", action="store_true", default=True, help="Lowercase types when counting n_unique_types")
    p.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
