# Level 1 generation handoff

## Run the generator

Use the existing Level 0 LLM settings in `.env`. Once the retrieval module is
available as `retrieval.py`, call the generator from the repository root.

```python
from level1 import rag_answer

result = rag_answer("Who directed Interstellar?")
print(result)
```

The default call uses `retrieval.retrieve(query, k=5)`. If the retrieval module
has a different name, pass its function as `rag_answer(query, retrieve_fn=retrieve)`.
Each retrieved passage must have `chunk_id`, `doc_id`, `title`, `text`, and `score`.
The generation layer expects a numeric score where **higher means more relevant**.
If the index returns distances where smaller values rank higher, the retriever
must convert them before returning `score`. Confirm the convention with the
retrieval implementation before calibrating the cutoff.

| Result field | Meaning |
|---|---|
| `answer` | English answer with inline citations, or `I don't know.` |
| `citations` | Distinct `doc_id#chunk_id` strings in order of appearance |
| `abstained` | Whether the request returned the standard abstention text |
| `abstention_reason` | `no_passages`, `below_threshold`, `model_abstained`, `empty_response`, or `citation_rejected`. `None` for an accepted answer. |
| `rejected_answer` | Original model text for a citation rejection. Use it in evaluation traces for audit. `None` otherwise. |
| `usage` | Prompt, completion, and total token counts. Calls skipped by the retrieval gate use zero tokens. Counts missing from the provider response are `None`. |
| `latency` | Elapsed time in seconds, including retrieval and generation |

## Citation rejection tracking

Record `abstention_reason` in evaluation traces and count `citation_rejected`
separately from explicit model abstention and retrieval gates. Each citation
rejection also emits one INFO log event. The displayed `answer` remains the
standard abstention text. To count factually correct answers lost to validation,
compare each `rejected_answer` against the gold answer and supporting passages.
The rejection count alone measures citation failures, regardless of correctness.

## Prompt history

- **v1:** Answer in English using the supplied passages. Put a citation after
  each factual statement, and abstain when the question is ambiguous, outside
  the scope, or unsupported. The validator checks citation IDs and requires a
  citation at the end of each answer line. Whether a cited passage supports a
  particular claim must still be checked during evaluation.

## Abstention threshold

Set `RAG_MIN_SCORE` in `.env`, or pass `min_score` for a single run. If the highest
retrieved score is below the cutoff, the generator abstains before calling the
LLM. An empty result always triggers abstention. With the cutoff unset, the
prompt and the citation validator handle queries with nonempty results.

After the retrieval module and index are available, run the comparison from the
repository root using the retrieval mode selected for Level 1.

```bash
python calibrate_abstention.py
```

Pass a different JSONL path as the first argument to use a separate calibration
set. The script retrieves once per answerable or out-of-corpus case and prints
the gate-only results for the baseline and the observed score cutoffs. It
suggests a cutoff only when it blocks additional out-of-corpus cases while
preserving the answerable cases that passed with the cutoff off. No LLM calls
are made during calibration.

Record the table and retrieval mode, then verify the suggested setting with the
full Level 1 pipeline. Keep the cutoff unset until real retrieval scores are
available. No empirical cutoff has been selected yet.

## Offline checks

These cases were exercised with a simulated retriever and LLM response. The
passage for Interstellar used the English text in the repository corpus.

| Case | Observed behavior |
|---|---|
| Question about the director of Interstellar, with a valid citation | Returns the cited answer and reports token usage |
| No retrieved passages, or the highest score below a supplied cutoff | Records `no_passages` or `below_threshold` and reports zero tokens |
| The LLM reports that the answer is absent | Records `model_abstained` and retains the LLM token usage |
| The LLM includes an unknown citation or leaves a line uncited | Records `citation_rejected` and preserves the candidate for audit |

## Live LLM checks

On November 10, 2026, both requests used `openai/gpt-oss-120b` via Groq. The
retriever was simulated by passing the Interstellar corpus entry as one passage.
Each row records a single call.

| Question | Observed result | Abstained | Total tokens | Latency (s) |
|---|---|---|---:|---:|
| Who directed Interstellar? | Christopher Nolan directed Interstellar. `[interstellar-2014#001]` | No | 347 | 13.14 |
| Who composed the soundtrack for Interstellar? | I don't know. | Yes | 344 | 1.62 |
