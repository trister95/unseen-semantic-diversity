# Start here (note to self — 2026-05-27)

Built the new occurrence-level pipeline in `rao/`. Picking up tomorrow:

## What's done
- `build_corpus.py` — DBNL `*.txt` → spaCy sentence split + forward-glue (<10 words
  glued to the next) → animal NER → **SQLite**. Sentence↔file link = `doc_id`.
- `embed_mentions.py` — each animal mention → GysBERT-base occurrence embedding → **NPZ**.
- `README.md` — full CLI usage, schema, NPZ format.
- Separate **empty-sentence DB** via `--empty-db` (your request, for per-year
  sentence/word counts). Main `docs` table also stores `n_sentences_total` /
  `n_words_total` per file.
- Verified: gluing logic, block-splitting, SQLite store/fetch/group, split-DB
  routing + totals. **Not yet run with the real models** (spaCy + weights live on Colab).

## First steps tomorrow
1. Confirm `data/dbnl_txt_files/` finished copying (was still empty when this was written):
   `ls data/dbnl_txt_files | wc -l`
2. Install spaCy + Dutch model (once, on whatever machine runs it):
   ```
   pip install spacy
   python -m spacy download nl_core_news_lg
   ```
3. Smoke test on 20 files:
   ```
   python functional_diversity/rao/build_corpus.py \
       --input-dir data/dbnl_txt_files \
       --db /tmp/test.sqlite \
       --empty-db /tmp/test_empty.sqlite \
       --limit 20 --device cuda:0
   ```
   Sanity-check: `sqlite3 /tmp/test.sqlite "SELECT * FROM docs LIMIT 5;"` and peek at
   `sentences` / `mentions`. Then run `embed_mentions.py` on that DB and inspect the NPZ
   (`vecs.shape`, a few `text`/`doc_id`).
4. Full run with the Colab `/content/drive/...` paths from the README. It resumes, so a
   killed run can just be re-launched.

## Decisions / open questions
- Default keeps **only** mention-bearing sentences in the main DB; empties go to
  `--empty-db` if set. Confirm that split is what you want vs. one combined DB.
- **Years aren't in the .txt files.** To get per-year/decade counts you'll need a
  `doc_id → jaar` mapping. Use `data/dbnl_metadata.csv` (`ti_id` + `jaar`); join
  via `doc_id_to_ti_id(doc_id)` which strips a trailing `_NN` volume marker. The
  `--metadata --min-year --max-year` flags on `build_corpus.py` already use this
  mapping for filtering; the same lookup needs to land in stage 3.
- NPZ is occurrence-level (one row per mention), **not** the `group/keys/vecs/counts`
  shape `rao_q.py` expects. Next real task: aggregate occurrences → per-(decade,type)
  vectors, or write an occurrence-based Rao directly.

## Open: missing mentions under `--sentence-mode sentencizer`
On 2026-05-28 the 50-file 1600-1700 smoke test produced **4848 mentions in
sentencizer mode vs. 5379 in the original `parser` mode** — a ~10% drop. Claude
asserted (without verifying) that this is because sentencizer under-splits long
unpunctuated passages → NER's 512-token truncation drops mentions past the cut.
**That hypothesis is unverified.** Before the paper run:

1. Run the diagnostic in `rao/README.md` (segment-length percentiles + BPE-token
   check on the longest segments) to see how often segments actually exceed 512
   tokens. If they almost never do, the truncation theory is wrong.
2. More likely explanation: segments have *different shapes* across modes →
   different BERT context windows → some mentions fall below the model's
   confidence threshold in one mode but not the other. Neither set is ground
   truth; what matters is agreement.
3. A/B test: same 50 files in `--sentence-mode senter` (delete DB first). Compare
   `(doc_id, mention_text)` set overlap with sentencizer- and parser-mode runs.
   If agreement is >90% across all three, the "drop" is just disagreement in a
   small fraction, not actual recall loss.
4. If real recall loss is confirmed and significant, the cheap fix is a
   `--max-segment-words` cap in `segments_from_doc` that forces a hard split when
   a single sentence exceeds N words — keeps sentencizer speed, eliminates the
   truncation risk.
