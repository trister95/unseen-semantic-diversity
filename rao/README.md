# Rao occurrence pipeline

Builds **per-occurrence** animal embeddings over the whole DBNL corpus, so Rao's
Q (and other FD metrics) can be computed from individual mention vectors rather
than the type-mean vectors in `../in/all_types.npz`.

Two stages, with a SQLite database as the intermediate store:

```
dbnl_txt_files/*.txt
      │  build_corpus.py   (spaCy sentence split + glue, animal NER)
      ▼
dbnl_occurrences.sqlite     (sentences + animal mentions with char offsets)
      │  embed_mentions.py  (GysBERT base, mean-pool entity tokens)
      ▼
occurrence_embeddings.npz   (one 768-d vector per mention + metadata)
```

`rao_q.py` (already here) consumes the *combined type NPZ* format and is
unchanged; the new NPZ is occurrence-level (see format below) and is meant for
new occurrence-based Rao/FD analysis.

## Dependencies

`transformers` and `torch` are already in the venv. spaCy is **not** — install
it plus the large Dutch model once:

```bash
pip install spacy
python -m spacy download nl_core_news_lg
```

## Stage 1 — `build_corpus.py`

For every `*.txt` under `--input-dir` (recursively; `doc_id` = filename without
extension):

1. Split into sentences with `nl_core_news_lg`. Files larger than
   `--spacy-max-length` are first split on blank lines so spaCy doesn't choke.
2. **Glue short sentences forward**: a stored segment accumulates whole
   sentences until it has at least `--min-words` (default 10) words; *words* =
   spaCy tokens that are neither punctuation nor whitespace. A sub-threshold
   trailing remainder is merged back into the previous segment.
3. Run the animal NER model (`ArjanvD95/animals_ffr_gysbert_512`,
   `aggregation_strategy=first`) over the segments and keep ANIMAL spans with
   their char `start`/`end` **relative to the segment text**.
4. Store to SQLite. Only segments with ≥1 mention are kept (use `--keep-empty`
   to store all). Every processed file gets a `docs` row, so reruns **resume**
   by skipping finished `doc_id`s.

```bash
python build_corpus.py \
    --input-dir /content/drive/MyDrive/dbnl_txt_files \
    --db        /content/drive/MyDrive/rao/dbnl_occurrences.sqlite \
    --empty-db  /content/drive/MyDrive/rao/dbnl_empty_sentences.sqlite \
    --device    cuda:0
# quick test first:  --limit 20
```

`--empty-db` routes **mention-less** sentences into a separate database (same
`sentences` schema) — useful for later "how many sentences/words in year X"
counts without bloating the main DB. Without it, mention-less sentences are
dropped (or, with `--keep-empty`, stored in the main DB). Either way, the main
`docs` table records total counts per file (`n_sentences_total`,
`n_words_total`), so per-year totals are a `docs ⋈ year-metadata` join even if
you never keep the empty sentences themselves.

### SQLite schema (main DB)

```
docs(doc_id PK, n_sentences, n_mentions, n_sentences_total, n_words_total)  -- one per file; resume marker
sentences(sentence_id PK, doc_id, sent_index, text, n_words)
mentions(mention_id PK, sentence_id, doc_id, start, end, text, score)
```

`start`/`end` index into the matching `sentences.text`. `sent_index` is the
segment's position within the whole file, so order is preserved even when a
file's segments are split across the main and `--empty-db` databases. The empty
DB has the same `docs`/`sentences`/`mentions` tables but only `sentences` is
populated.

## Stage 2 — `embed_mentions.py`

Reads `mentions ⋈ sentences`, encodes each sentence once with the **base**
GysBERT model (`emanjavacas/GysBERT`), and for each mention mean-pools the last
hidden states over the sub-word tokens overlapping `[start, end)` (same pooling
as `../compute_group_fd.py`). Vectors are L2-normalized by default
(`--no-normalize` to disable).

```bash
python embed_mentions.py \
    --db     /content/drive/MyDrive/rao/dbnl_occurrences.sqlite \
    --output /content/drive/MyDrive/rao/occurrence_embeddings.npz \
    --device cuda:0
```

### NPZ format (occurrence-level)

All arrays are parallel, length `N` = number of mentions:

| array         | dtype           | meaning                              |
|---------------|-----------------|--------------------------------------|
| `vecs`        | float32 (N,768) | occurrence embedding                 |
| `mention_id`  | int64           | PK into `mentions`                   |
| `sentence_id` | int64           | PK into `sentences`                  |
| `doc_id`      | str             | DBNL file id (filename stem)         |
| `text`        | str             | animal surface form                  |
| `start`,`end` | int32           | char offsets within the sentence     |

This is **not** the `group/keys/vecs/counts` shape that `rao_q.py` /
`cluster_embeddings.py` expect — those use one row per (group, type). To reuse
them, aggregate these occurrences into type vectors per group (e.g. by decade)
first; `doc_id` is the join key for any per-work metadata (year, genre).

## Notes

- Char offsets are relative to the stored sentence, not the original file, which
  is all that's needed to re-locate and re-embed a mention.
- A mention whose tokens fall entirely past `--max-length` truncation is skipped
  in stage 2 (logged via the running embedding count).
