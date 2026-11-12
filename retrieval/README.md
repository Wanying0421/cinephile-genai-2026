# Retrieval layer (Level 1, Week 2)

Hybrid retrieval (dense + BM25, fused with a hand-written RRF) over 44 Wikipedia movie/TV documents.

## Setup

```
pip install -r requirements.txt
```

On Windows, if PyTorch fails to load (`WinError 1114`), install the CPU build first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.
If TensorFlow/Keras is installed in the environment, `retrieval/__init__.py` already
stops `transformers` from importing it.

## Build the index (not committed, rebuild locally)

```
python -m retrieval.build_index
```

First run downloads the embedding model (~1.1 GB). Encoding takes about 1.5 minutes on CPU.
Outputs go to `index/`: `chunks_{size}.json`, `embeddings_{size}.npy`, `faiss_{size}.index`
for sizes 100, 200 and 400. The corpus itself can be rebuilt with `python scripts/build_corpus.py`.

## Usage

```python
from retrieval import retrieve

results = retrieve("Who is Saito in Inception?", k=5)
```

Each result is a dict:

| Field | Meaning |
|---|---|
| `chunk_id` | e.g. `inception-2010#3` (already contains the doc id: cite it as `[inception-2010#3]`) |
| `doc_id`, `title`, `type`, `year`, `source_url` | document metadata |
| `text` | chunk text |
| `score` | ranking score of the method (RRF score in hybrid mode, about 0.01 to 0.03) |
| `dense_score` | cosine similarity between query and chunk, computed for every result |

## Design choices

- **Corpus:** 44 Wikipedia pages (lead + Plot + Cast sections). Includes deliberate hard
  negatives: other Nolan films, both versions of *Solaris*, the Matrix sequels, other
  time-travel series. The original 6-document corpus made Recall@5 meaningless.
- **Chunking:** sliding word windows with 20% overlap; a very short final window is merged
  into the previous one. The film title is prepended before embedding, because chunks from
  the middle of a plot rarely mention the title.
- **Embedding model:** `intfloat/multilingual-e5-base` (768 dimensions). Runs on CPU, and is
  multilingual, so French or Chinese questions can match the English corpus. Limit: 512
  tokens, so the end of 400-word chunks is truncated.
- **Search:** dense search with FAISS `IndexFlatIP` (exact cosine) and BM25 (`rank_bm25`),
  top 50 candidates each, fused with Reciprocal Rank Fusion (k=60) written by hand in
  `fusion.py`.
- **Default configuration:** 100-word chunks, hybrid (see results below).

## Results

Reproduce with `python -m retrieval.eval_retrieval` (full table in `results/retrieval_comparison.md`,
per-question results in `results/retrieval_details.jsonl`).

Two test sets: **original** (15 `rag_document` cases from `tests/tests.jsonl`) and
**paraphrased** (20 cases in `data/retrieval_tests.jsonl`, no film titles, written to
target hard negatives).

Paraphrased set, k=5:

| Chunk size | Method | Recall@5 | MRR@5 | Context words / query |
|---|---|---|---|---|
| 100 | dense | 0.900 | 0.875 | 492 |
| **100** | **hybrid** | **0.950** | **0.917** | **502** |
| 200 | dense | 0.900 | 0.808 | 969 |
| 200 | hybrid | 0.950 | 0.900 | 990 |
| 400 | dense | 0.900 | 0.875 | 1945 |
| 400 | hybrid | 0.950 | 0.892 | 1929 |

- On the original set every configuration scores Recall@5 = 1.000: questions that name the
  film are too easy to compare methods, hence the paraphrased set.
- Hybrid beats dense at every chunk size. 100-word hybrid has the best MRR and sends about
  4x fewer words to the LLM than 400-word chunks.
- Caveat: with 20 questions, a 0.05 difference in recall is a single question.

## Notes for the generation layer

- **Use `dense_score` of the top result for the abstention threshold, never `score`**
  (RRF scores only reflect rank).
- Observed top-1 `dense_score`: questions about the corpus about 0.83 to 0.89; an unrelated
  question ("What is the capital of France?") about 0.74 to 0.75; **but a hard answerable
  question (R1) only 0.75 to 0.77.** A threshold cannot separate these cases cleanly: keep it
  conservative and rely on the prompt ("I don't know") for most abstentions.
- Absence questions about films that ARE in the corpus (budget, filming locations) retrieve
  the right film with a high score, so only the prompt can make the model abstain.
- Test case 21 (Inception's budget) is now answerable: the corpus states US$160 million.

## Known retrieval failures

- **R1** (father moves the second hand of his daughter's wristwatch, Interstellar): fails in
  all 6 configurations. Dense retrieval favours semantically similar father/space stories
  (*Ad Astra*, *Mirror*). BM25 matches "wristwatch", but RRF rewards chunks found by both
  methods, so a chunk found by BM25 alone ranks too low. The text also says "Murph", not
  "daughter".
- **R14** (two hitmen at breakfast in a diner held up by robbers, Pulp Fiction): fails with
  dense retrieval at every chunk size, found by hybrid at every size thanks to exact
  keywords (hitmen, diner, breakfast). This single question is the whole recall gap between
  dense and hybrid.