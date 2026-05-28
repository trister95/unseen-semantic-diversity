"""Stage 2 of the Rao occurrence pipeline.

Read animal mentions + their sentences from the SQLite database produced by
``build_corpus.py`` and compute one contextual embedding per *occurrence* with
the GysBERT base model (``emanjavacas/GysBERT``). Each sentence is encoded once;
for every mention in it we mean-pool the model's last hidden states over the
sub-word tokens overlapping the mention's char span ``[start, end)`` (the same
pooling used in ``compute_group_fd.py``). Embeddings are L2-normalized by
default (matching that script) and written to a single NPZ.

NPZ contents (all arrays parallel, length = #mentions):
    vecs        float32 (N, hidden)   the occurrence embeddings
    mention_id  int64   (N,)          PK into the mentions table
    sentence_id int64   (N,)          PK into the sentences table
    doc_id      <U..    (N,)          DBNL file id (filename stem)
    text        <U..    (N,)          animal surface form
    start       int32   (N,)          char offset within the sentence text
    end         int32   (N,)

Example (Colab)
---------------
    python embed_mentions.py \
        --db /content/drive/MyDrive/rao/dbnl_occurrences.sqlite \
        --output /content/drive/MyDrive/rao/occurrence_embeddings.npz \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def _fmt_hms(seconds: float) -> str:
    s = max(int(seconds), 0)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _progress(done: int, total: int, elapsed: float) -> str:
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    return (
        f"elapsed {_fmt_hms(elapsed)}, remaining ~{_fmt_hms(remaining)} "
        f"({rate:.1f}/s, {done}/{total})"
    )


def _device(device: str) -> str:
    import torch

    d = device.strip().lower()
    if d.startswith("cuda") or d.startswith("gpu"):
        return "cuda" + (d[d.index(":"):] if ":" in d else "")
    if d == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModel.from_pretrained(model_name, token=token)
    model.eval()
    if device.startswith("cuda"):
        model = model.half()  # fp16 on GPU
    model.to(device)
    return tok, model


def fetch_mentions(con: sqlite3.Connection) -> List[Tuple]:
    """Rows ordered by sentence so each sentence is encoded once.

    Returns (sentence_id, sentence_text, mention_id, start, end, animal_text, doc_id).
    """
    cur = con.execute(
        """
        SELECT s.sentence_id, s.text, m.mention_id, m.start, m.end, m.text, m.doc_id
        FROM mentions m
        JOIN sentences s ON s.sentence_id = m.sentence_id
        ORDER BY s.sentence_id, m.mention_id
        """
    )
    return cur.fetchall()


def group_by_sentence(rows: List[Tuple]) -> List[Tuple[int, str, List[Tuple]]]:
    """Collapse to [(sentence_id, sentence_text, [mention rows...])]."""
    out: List[Tuple[int, str, List[Tuple]]] = []
    cur_id = None
    for sid, stext, mid, start, end, mtext, doc_id in rows:
        if sid != cur_id:
            out.append((sid, stext, []))
            cur_id = sid
        out[-1][2].append((mid, start, end, mtext, doc_id))
    return out


def embed_batch(
    sentences: List[Tuple[int, str, List[Tuple]]],
    tok,
    model,
    device: str,
    max_length: int,
    normalize: bool,
):
    """Yield (vec, mention_id, sentence_id, doc_id, animal_text, start, end) per mention."""
    import torch

    texts = [s[1] for s in sentences]
    enc = tok(
        texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = enc.pop("offset_mapping").tolist()  # (B, T, 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    last = out["last_hidden_state"]  # (B, T, H)

    results = []
    for b, (sid, _stext, mentions) in enumerate(sentences):
        off = offsets[b]
        hidden = last[b]
        for mid, start, end, mtext, doc_id in mentions:
            idxs = [
                i for i, (s, e) in enumerate(off)
                if not (s == 0 and e == 0) and s < end and e > start
            ]
            if not idxs:
                # span fell past truncation / no overlapping token -> skip
                continue
            vec = hidden[idxs].mean(0).float().cpu().numpy()
            if normalize:
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
            results.append((vec, mid, sid, doc_id, mtext, start, end))
    return results


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    con = sqlite3.connect(args.db)
    rows = fetch_mentions(con)
    con.close()
    if not rows:
        raise SystemExit(f"No mentions found in {args.db}. Run build_corpus.py first.")
    sentences = group_by_sentence(rows)
    # Sort by sentence-text length (descending) so each batch has homogeneous
    # lengths and padding is minimized. Output ordering is recoverable via
    # mention_id/sentence_id in the NPZ.
    sentences.sort(key=lambda s: len(s[1]), reverse=True)
    print(f"{len(rows)} mentions across {len(sentences)} sentences; device={device}")

    tok, model = load_model(args.model, device)

    vecs: List[np.ndarray] = []
    mention_id: List[int] = []
    sentence_id: List[int] = []
    doc_id: List[str] = []
    text: List[str] = []
    start: List[int] = []
    end: List[int] = []

    bs = args.batch_size
    done = 0
    t0 = time.time()
    n_batches = (len(sentences) + bs - 1) // bs
    for i in range(0, len(sentences), bs):
        batch = sentences[i:i + bs]
        for vec, mid, sid, did, mtext, s, e in embed_batch(
            batch, tok, model, device, args.max_length, not args.no_normalize
        ):
            vecs.append(vec)
            mention_id.append(int(mid))
            sentence_id.append(int(sid))
            doc_id.append(str(did))
            text.append(str(mtext))
            start.append(int(s))
            end.append(int(e))
        done += len(batch)
        batch_idx = i // bs
        if batch_idx % args.log_every == 0 or done >= len(sentences):
            print(
                f"  {len(vecs)} embeddings; {_progress(done, len(sentences), time.time() - t0)}",
                flush=True,
            )

    if not vecs:
        raise SystemExit("No embeddings produced (all spans fell outside truncation?).")

    arr = np.stack(vecs).astype(np.float32)
    np.savez(
        args.output,
        vecs=arr,
        mention_id=np.array(mention_id, dtype=np.int64),
        sentence_id=np.array(sentence_id, dtype=np.int64),
        doc_id=np.array(doc_id),
        text=np.array(text),
        start=np.array(start, dtype=np.int32),
        end=np.array(end, dtype=np.int32),
    )
    print(f"Wrote {arr.shape[0]} occurrence embeddings (dim {arr.shape[1]}) to {args.output}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="SQLite DB from build_corpus.py")
    p.add_argument("--output", required=True, help="Output NPZ path")
    p.add_argument("--model", default="emanjavacas/GysBERT", help="HF model for embeddings (base GysBERT)")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0 (auto if unset matches torch)")
    p.add_argument("--batch-size", type=int, default=128, help="Sentences per forward pass")
    p.add_argument("--max-length", type=int, default=512, help="Max tokens per sentence")
    p.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize embeddings")
    p.add_argument("--log-every", type=int, default=20, help="Log every N batches")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
