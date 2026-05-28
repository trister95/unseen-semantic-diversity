"""Stage 1 of the Rao occurrence pipeline.

Walk a directory of DBNL plain-text files (one work per file, ``doc_id`` = the
filename without its extension), split each file into sentences with the Dutch
spaCy model, glue short sentences forward until every stored segment has at
least ``--min-words`` words, run the animal NER model over the segments, and
store the segments + animal mentions (with char offsets relative to the
segment text) in a SQLite database.

The database is the intermediate store consumed by ``embed_mentions.py``.

Schema
------
docs(doc_id TEXT PRIMARY KEY, n_sentences INT, n_mentions INT)   -- resume marker
sentences(sentence_id INTEGER PRIMARY KEY, doc_id TEXT, sent_index INT,
          text TEXT, n_words INT)
mentions(mention_id INTEGER PRIMARY KEY, sentence_id INT, doc_id TEXT,
         start INT, end INT, text TEXT, score REAL)

Only sentences that contain at least one animal mention are stored by default
(pass ``--keep-empty`` to store every segment). A ``docs`` row is written for
*every* processed file so reruns can skip finished files even when they yielded
no animals.

Example (Colab)
---------------
    python build_corpus.py \
        --input-dir /content/drive/MyDrive/dbnl_txt_files \
        --db /content/drive/MyDrive/rao/dbnl_occurrences.sqlite \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
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
        f"({rate:.2f}/s, {done}/{total})"
    )


# --------------------------------------------------------------------------- #
# DBNL filename -> metadata ti_id (strips trailing _NN volume marker).
# --------------------------------------------------------------------------- #
_VOL_SUFFIX = re.compile(r"_\d+$")


def doc_id_to_ti_id(doc_id: str) -> str:
    return _VOL_SUFFIX.sub("", doc_id)


def load_year_filter(
    metadata_path: str, min_year: Optional[int], max_year: Optional[int]
) -> set:
    """Read dbnl_metadata.csv and return the set of ti_ids whose 'jaar' is in
    [min_year, max_year]. The file starts with a 'sep=|' hint line, then a
    pipe-separated table with quoted values."""
    ti_ids: set = set()
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
            jaar = int(jaar_s)
            if min_year is not None and jaar < min_year:
                continue
            if max_year is not None and jaar > max_year:
                continue
            ti_ids.add(ti)
    return ti_ids


# --------------------------------------------------------------------------- #
# spaCy sentence splitting + short-sentence gluing
# --------------------------------------------------------------------------- #
def load_spacy(model: str, max_length: int, sentence_mode: str):
    """Load a minimal spaCy pipeline for sentence splitting.

    sentence_mode:
      * ``sentencizer`` — rule-based, ``spacy.blank("nl") + sentencizer``. No
        model download needed; ~20-100x faster than ``parser``. Sentence
        boundaries follow punctuation; downstream glue logic compensates for
        over-splitting on abbreviations.
      * ``senter`` — statistical sentence segmenter from the named model.
        Faster than ``parser`` while staying model-based; falls back to
        ``sentencizer`` if the model has no ``senter`` component.
      * ``parser`` — full dependency parser (most accurate, slowest).
    """
    import spacy

    if sentence_mode == "sentencizer":
        nlp = spacy.blank("nl")
        nlp.add_pipe("sentencizer")
    elif sentence_mode in ("senter", "parser"):
        if sentence_mode == "senter":
            disable = ["ner", "lemmatizer", "attribute_ruler", "parser", "tagger", "morphologizer"]
        else:
            disable = ["ner", "lemmatizer", "attribute_ruler"]
        try:
            nlp = spacy.load(model, disable=disable)
        except OSError as exc:
            raise SystemExit(
                f"spaCy model '{model}' is not installed. Install it with:\n"
                f"    python -m spacy download {model}\n"
                f"(original error: {exc})"
            )
        if sentence_mode == "senter" and "senter" not in nlp.pipe_names:
            print(f"warning: '{model}' has no 'senter' component; using rule-based sentencizer instead")
            nlp.add_pipe("sentencizer")
    else:
        raise SystemExit(f"unknown --sentence-mode: {sentence_mode!r}")
    nlp.max_length = max_length
    return nlp


def _count_words(span) -> int:
    """Words = tokens that are neither punctuation nor whitespace."""
    return sum(1 for t in span if not (t.is_punct or t.is_space))


def segments_from_doc(doc, min_words: int) -> List[Tuple[str, int]]:
    """Greedy forward-glue of spaCy sentences.

    Returns a list of ``(segment_text, n_words)``. A segment accumulates whole
    sentences (preserving the original spacing via char offsets) until it has at
    least ``min_words`` words. A short trailing remainder is merged back into the
    previous segment so we never emit a dangling sub-threshold tail.
    """
    out: List[Tuple[int, int, int]] = []  # (start_char, end_char, n_words)
    cur_start: Optional[int] = None
    cur_end = 0
    cur_wc = 0
    for sent in doc.sents:
        wc = _count_words(sent)
        if wc == 0:
            continue  # skip whitespace-only / punctuation-only spans
        if cur_start is None:
            cur_start = sent.start_char
        cur_end = sent.end_char
        cur_wc += wc
        if cur_wc >= min_words:
            out.append((cur_start, cur_end, cur_wc))
            cur_start, cur_end, cur_wc = None, 0, 0
    if cur_start is not None:
        if out and cur_wc < min_words:
            ps, _, pw = out[-1]
            out[-1] = (ps, cur_end, pw + cur_wc)
        else:
            out.append((cur_start, cur_end, cur_wc))

    text = doc.text
    return [(text[s:e].strip(), wc) for (s, e, wc) in out if text[s:e].strip()]


# --------------------------------------------------------------------------- #
# Animal NER pipeline
# --------------------------------------------------------------------------- #
_ANIMAL_LABELS = {"animal", "banimal", "ianimal"}


def _device_to_index(device: str) -> int:
    d = device.strip().lower()
    if d in ("cpu", "-1"):
        return -1
    if d.startswith("cuda") or d.startswith("gpu"):
        parts = d.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0
    return int(d) if d.lstrip("-").isdigit() else -1


def load_ner(model_name: str, device: str, batch_size: int, max_length: int, aggregation: str):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(model_name, token=token)
    tok.model_max_length = max_length
    model = AutoModelForTokenClassification.from_pretrained(model_name, token=token)
    device_idx = _device_to_index(device)
    if device_idx >= 0:
        model = model.half()  # fp16 on GPU
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tok,
        aggregation_strategy=aggregation,
        device=device_idx,
        batch_size=batch_size,
    )


def animal_spans(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for ent in entities:
        label = (ent.get("entity_group") or ent.get("entity") or "")
        if str(label).lower().replace("-", "") not in _ANIMAL_LABELS:
            continue
        start, end = int(ent["start"]), int(ent["end"])
        spans.append({
            "start": start,
            "end": end,
            "text": text[start:end],
            "score": float(ent.get("score", 0.0)),
        })
    return spans


# --------------------------------------------------------------------------- #
# SQLite storage
# --------------------------------------------------------------------------- #
def open_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS docs(
            doc_id TEXT PRIMARY KEY,
            n_sentences INTEGER,        -- segments stored in THIS db (mention-bearing, unless --keep-empty)
            n_mentions INTEGER,
            n_sentences_total INTEGER,  -- all segments in the file (incl. mention-less)
            n_words_total INTEGER       -- total words across all segments
        );
        CREATE TABLE IF NOT EXISTS sentences(
            sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            sent_index INTEGER,
            text TEXT,
            n_words INTEGER
        );
        CREATE TABLE IF NOT EXISTS mentions(
            mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id INTEGER,
            doc_id TEXT,
            start INTEGER,
            end INTEGER,
            text TEXT,
            score REAL
        );
        CREATE INDEX IF NOT EXISTS ix_sentences_doc ON sentences(doc_id);
        CREATE INDEX IF NOT EXISTS ix_mentions_sentence ON mentions(sentence_id);
        CREATE INDEX IF NOT EXISTS ix_mentions_doc ON mentions(doc_id);
        """
    )
    con.commit()
    return con


def processed_docs(con: sqlite3.Connection) -> set:
    return {row[0] for row in con.execute("SELECT doc_id FROM docs")}


def store_doc(
    con: sqlite3.Connection,
    doc_id: str,
    segments: List[Tuple[str, int, List[Dict[str, Any]]]],  # (text, n_words, spans)
    keep_empty: bool,
    empty_con: Optional[sqlite3.Connection] = None,
) -> Tuple[int, int]:
    """Store one file's segments.

    Mention-bearing segments (+ their mentions) always go to ``con``. Mention-less
    segments go to ``empty_con`` if given, else to ``con`` when ``keep_empty``,
    else dropped. ``sent_index`` is the position within the whole file, so order
    is preserved even when segments are split across the two databases.
    """
    n_sent = 0
    n_ment = 0
    n_words_total = 0
    for sent_index, (text, n_words, spans) in enumerate(segments):
        n_words_total += n_words
        if spans:
            cur = con.execute(
                "INSERT INTO sentences(doc_id, sent_index, text, n_words) VALUES (?,?,?,?)",
                (doc_id, sent_index, text, n_words),
            )
            sentence_id = cur.lastrowid
            n_sent += 1
            for sp in spans:
                con.execute(
                    "INSERT INTO mentions(sentence_id, doc_id, start, end, text, score) "
                    "VALUES (?,?,?,?,?,?)",
                    (sentence_id, doc_id, sp["start"], sp["end"], sp["text"], sp["score"]),
                )
                n_ment += 1
        elif empty_con is not None:
            empty_con.execute(
                "INSERT INTO sentences(doc_id, sent_index, text, n_words) VALUES (?,?,?,?)",
                (doc_id, sent_index, text, n_words),
            )
        elif keep_empty:
            con.execute(
                "INSERT INTO sentences(doc_id, sent_index, text, n_words) VALUES (?,?,?,?)",
                (doc_id, sent_index, text, n_words),
            )
            n_sent += 1
    con.execute(
        "INSERT OR REPLACE INTO docs(doc_id, n_sentences, n_mentions, n_sentences_total, n_words_total) "
        "VALUES (?,?,?,?,?)",
        (doc_id, n_sent, n_ment, len(segments), n_words_total),
    )
    con.commit()
    if empty_con is not None:
        empty_con.commit()
    return n_sent, n_ment


# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #
def iter_txt_files(input_dir: str) -> List[Tuple[str, str]]:
    """Return sorted ``(doc_id, path)`` for every *.txt under input_dir."""
    found: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                doc_id = os.path.splitext(fn)[0]
                found.append((doc_id, os.path.join(root, fn)))
    found.sort()
    return found


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def split_for_spacy(text: str, max_length: int) -> List[str]:
    """Split oversized files on blank lines so each block fits nlp.max_length."""
    if len(text) <= max_length:
        return [text]
    blocks: List[str] = []
    buf: List[str] = []
    size = 0
    for para in text.split("\n\n"):
        para = para + "\n\n"
        if size + len(para) > max_length and buf:
            blocks.append("".join(buf))
            buf, size = [], 0
        if len(para) > max_length:  # single huge paragraph: hard-split
            for i in range(0, len(para), max_length):
                blocks.append(para[i:i + max_length])
            continue
        buf.append(para)
        size += len(para)
    if buf:
        blocks.append("".join(buf))
    return blocks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    files = iter_txt_files(args.input_dir)
    if not files:
        raise SystemExit(f"No .txt files found under {args.input_dir}")
    if args.metadata and (args.min_year is not None or args.max_year is not None):
        allowed_ti = load_year_filter(args.metadata, args.min_year, args.max_year)
        before = len(files)
        files = [(d, p) for (d, p) in files if doc_id_to_ti_id(d) in allowed_ti]
        yr = f"[{args.min_year or '-inf'}..{args.max_year or 'inf'}]"
        print(f"Year filter {yr}: {len(files)}/{before} files match {len(allowed_ti)} ti_ids")
    if args.limit is not None:
        files = files[: args.limit]

    con = open_db(args.db)
    empty_con = open_db(args.empty_db) if args.empty_db else None
    done = processed_docs(con)
    todo = [(d, p) for (d, p) in files if d not in done]
    print(f"{len(files)} files found; {len(done)} already done; processing {len(todo)}.")
    if not todo:
        return

    nlp = load_spacy(args.spacy_model, args.spacy_max_length, args.sentence_mode)
    ner = load_ner(
        args.ner_model, args.device, args.ner_batch_size, args.ner_max_length, args.aggregation
    )

    # Buffer of spaCy-processed files waiting for NER. We batch NER across
    # multiple files so the GPU stays busy on docs with few segments and the
    # HF pipeline's DataLoader can overlap CPU tokenization with GPU compute.
    buffer: List[Tuple[str, List[Tuple[str, int]]]] = []  # (doc_id, [(text, n_words)])
    buf_segs = 0

    def flush() -> Tuple[int, int]:
        """NER everything in ``buffer`` in one shot, route results back per-file, store."""
        nonlocal buf_segs
        if not buffer:
            return 0, 0
        # Flatten across files.
        flat_texts: List[str] = []
        flat_words: List[int] = []
        owners: List[Tuple[int, int]] = []  # (file_idx_in_buffer, seg_idx_in_file)
        for di, (_, segs) in enumerate(buffer):
            for sj, (text, n_words) in enumerate(segs):
                flat_texts.append(text)
                flat_words.append(n_words)
                owners.append((di, sj))
        # Sort by length desc so HF's internal batches are length-homogeneous.
        order = sorted(range(len(flat_texts)), key=lambda j: len(flat_texts[j]), reverse=True)
        sorted_texts = [flat_texts[j] for j in order]
        sorted_ents = ner(sorted_texts)
        if isinstance(sorted_ents, dict):  # single-segment edge case
            sorted_ents = [sorted_ents]
        ents_per_seg: List[Any] = [None] * len(flat_texts)
        for orig_idx, ent in zip(order, sorted_ents):
            ents_per_seg[orig_idx] = ent
        # Route back to per-file enriched lists.
        per_doc: List[List[Tuple[str, int, List[Dict[str, Any]]]]] = [[] for _ in buffer]
        for j, (di, sj) in enumerate(owners):
            text = flat_texts[j]
            per_doc[di].append((text, flat_words[j], animal_spans(text, ents_per_seg[j])))
        # Persist each file (one docs row, commit per file -> resume granularity).
        sent_added = ment_added = 0
        for (doc_id, _), enriched in zip(buffer, per_doc):
            ns, nm = store_doc(con, doc_id, enriched, args.keep_empty, empty_con)
            sent_added += ns
            ment_added += nm
        buffer.clear()
        buf_segs = 0
        return sent_added, ment_added

    t0 = time.time()
    total_sent = total_ment = 0
    last_log_total_sent = last_log_total_ment = 0
    for i, (doc_id, path) in enumerate(todo, 1):
        text = read_text(path)
        # 1) sentence split + glue (over blocks if the file is huge)
        segments: List[Tuple[str, int]] = []
        for block in split_for_spacy(text, args.spacy_max_length):
            segments.extend(segments_from_doc(nlp(block), args.min_words))
        if not segments:
            # Empty file: write a docs row directly so resume skips it next time.
            store_doc(con, doc_id, [], args.keep_empty, empty_con)
        else:
            buffer.append((doc_id, segments))
            buf_segs += len(segments)
        # 2) flush when the buffer is full, or at the very end.
        if buf_segs >= args.ner_buffer_segments or i == len(todo):
            fs, fm = flush()
            total_sent += fs
            total_ment += fm
        if i % args.log_every == 0 or i == len(todo):
            elapsed = time.time() - t0
            delta_sent = total_sent - last_log_total_sent
            delta_ment = total_ment - last_log_total_ment
            last_log_total_sent, last_log_total_ment = total_sent, total_ment
            print(
                f"[{i}/{len(todo)}] {doc_id}  "
                f"(+{delta_sent} sents, +{delta_ment} mentions since last log; "
                f"buf={buf_segs} segs; cum {total_sent}/{total_ment}; "
                f"{_progress(i, len(todo), elapsed)})",
                flush=True,
            )
    con.close()
    if empty_con is not None:
        empty_con.close()
    print(f"Done. Stored {total_sent} sentences, {total_ment} animal mentions to {args.db}")
    if args.empty_db:
        print(f"Mention-less sentences written to {args.empty_db}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="Directory of DBNL .txt files (doc_id = filename stem)")
    p.add_argument("--db", required=True, help="Output SQLite path")
    p.add_argument("--min-words", type=int, default=10, help="Min words per stored sentence (glue forward below this)")
    p.add_argument("--spacy-model", default="nl_core_news_lg", help="spaCy Dutch model (only used by --sentence-mode senter/parser)")
    p.add_argument("--spacy-max-length", type=int, default=1_000_000, help="spaCy nlp.max_length; files larger are block-split")
    p.add_argument("--sentence-mode", choices=["sentencizer", "senter", "parser"], default="sentencizer",
                   help="How to split sentences: 'sentencizer' (fast rule-based, default), 'senter' (model statistical), 'parser' (slow but accurate)")
    p.add_argument("--ner-model", default="ArjanvD95/animals_ffr_gysbert_512", help="HF token-classification animal model")
    p.add_argument("--aggregation", choices=["simple", "first", "average", "max"], default="first", help="HF aggregation_strategy")
    p.add_argument("--ner-batch-size", type=int, default=64, help="Batch size for the NER pipeline")
    p.add_argument("--ner-max-length", type=int, default=512, help="Max tokens for the NER tokenizer")
    p.add_argument("--ner-buffer-segments", type=int, default=512,
                   help="Buffer this many segments across files before invoking NER (cross-file batching)")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    p.add_argument("--empty-db", default=None, help="Write mention-less sentences to this separate SQLite DB (for per-year sentence/word counts)")
    p.add_argument("--keep-empty", action="store_true", help="Store mention-less sentences in the MAIN db instead of dropping them (ignored if --empty-db is set)")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N files (for testing)")
    p.add_argument("--log-every", type=int, default=50, help="Progress log frequency (files)")
    p.add_argument("--metadata", default=None, help="dbnl_metadata.csv path for year-based filtering (ti_id <-> jaar)")
    p.add_argument("--min-year", type=int, default=None, help="Keep only files whose metadata 'jaar' >= this")
    p.add_argument("--max-year", type=int, default=None, help="Keep only files whose metadata 'jaar' <= this")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
