# Cinephile GenAI 2026

A movie and TV assistant built in levels, each one measured on the same test set:
**Level 0** LLM alone → **Level 1** RAG → **Level 2** tool-using agent → **Level 3** extension.

Project: Mineur IA Générative 2026 (Text track). Theme: Cinéphile.

## Required language model

All generation, agent and LLM-judge calls use **`gpt-6-luna`**, as required by the instructor.
The shared `level0.complete()` function applies the same model and request settings
to Level 0, Level 1 and the evaluator; `agent.py` and `eval_agent.py` also check the model.
See [model setup and migration](docs/model-setup.md) for the local setup and checks.

## Team

| Member | Weeks 1–2 | Week 3 (Level 2) |
|---|---|---|
| Member 1 | Corpus and data (movie/TV summaries, TMDB API) | A: TMDB tools, SQLite watchlist, schemas |
| Member 2 | Test protocol (`tests/tests.jsonl`, `METRICS.md`) | B: manual agent loop, validation, confirmation |
| Member 3 | Level 0 baseline, repository, evaluation script | C: Level 2 evaluation, pass³, UI |

## Repository structure

```
data/                 movie and TV corpus (English); local watchlist DB (not committed)
index/                chunk, embedding and FAISS index files
retrieval/            BM25 + dense retrieval with manual RRF fusion
tools/                TMDB tools, SQLite watchlist, JSON schemas (Level 2)
tests/tests.jsonl     44 shared test cases (Levels 0/1)
tests/agent_tests.jsonl  17 Level 2 agent tasks with expected watchlist states
level0.py             Level 0: LLM alone, no documents, no tools
level1.py             Level 1: RAG with citations and abstention
agent.py              Level 2: manual agent loop (no framework)
eval.py               evaluation entry point for Levels 0, 1 and 2
eval_agent.py         Level 2 evaluator (final-state checks, pass^k)
app.py                Streamlit UI for Levels 0/1/2
docs/                 setup, tool contracts, agent failure analysis
reports/              committed evaluation results and traces
results/              raw local results (not committed)
.env.example          environment variables template
```

## Installation

```bash
git clone https://github.com/Wanying0421/cinephile-genai-2026.git
cd cinephile-genai-2026
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
# .\.venv\Scripts\Activate.ps1       # PowerShell
# source .venv/bin/activate          # macOS / Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

A dedicated virtual environment is strongly recommended. Mixing Anaconda packages
with `pip --user` packages caused PyTorch DLL errors (`WinError 1114`) and NumPy 1/2
binary conflicts on one team machine. The CPU build of PyTorch is sufficient.

Create your own `.env` file from the template and fill in your values:

```bash
cp .env.example .env
```

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_own_key_here
LLM_MODEL=gpt-6-luna
RAG_MIN_SCORE=
TMDB_API_KEY=your_tmdb_v3_api_key      # or TMDB_READ_TOKEN=your_v4_read_token
```

**Never commit `.env`.** Credentials are never distributed through Git.
Get a free TMDB key at themoviedb.org → Settings → API. Restart Python after
changing the configuration, then run `python check_llm.py`.

## Usage

### Evaluation

```bash
mkdir -p results
python eval.py --level 0                    # Level 0 on tests/tests.jsonl
python eval.py --level 1                    # Level 1 (RAG)
python eval.py --level 2 --runs 3           # Level 2 agent tasks, pass³
python eval.py --level 2 --runs 1 --only L2-01 L2-02   # quick check
```

Levels 0/1 write `results/level{0,1}_results.jsonl`. Level 2 writes one JSONL trace
line per run to `results/level2_traces.jsonl` and a summary to
`results/level2_summary.json`. Copy results you want to keep into `reports/`.

### User interface

```bash
python -m streamlit run app.py
```

Choose the level in the sidebar. The UI shows the answer, sources (RAG citations and
TMDB links), tokens, latency, stop reason, tool activity and the raw trace, plus the
current watchlist. Level 2 watchlist changes show **Confirm / Cancel** buttons bound to
the exact pending `request_id`. Every interaction is appended to `results/ui_traces.jsonl`.

The UI uses the real local watchlist (`data/watchlist.sqlite3`). To reset it, stop the
application and delete that file (this also deletes pending confirmations).

## Level 2 design

- **Tools** (`tools/`, JSON schemas in `docs/tool-schemas.json`): `search_movies`,
  `get_movie_details` (TMDB) and `update_watchlist` (SQLite), plus `rag_search`,
  which reuses the Level 1 pipeline.
- **Manual loop** (`agent.py`): model request → proposed tool call → validation →
  execution → tool result → next model response, at most 5 iterations. Unknown tools
  and invalid arguments are rejected before execution; tool outputs are treated as data.
- **Confirmation contract** (`docs/week3-watchlist.md`): `update_watchlist` only prepares
  a request (`request_id`, exact action, movie and title). Only an explicit user action
  calls `confirm_watchlist_action(request_id, approved=...)`. Requests are bound to the
  session, expire after 10 minutes and can be resolved once. A changed request needs a
  new confirmation; cancellation leaves storage unchanged.

## Level 2 evaluation

`eval_agent.py` runs each task in a **fresh temporary SQLite watchlist**, simulates the
user's confirm/cancel decision, and checks outcomes **by code**: final watchlist, tool
calls, tool errors, pending action and confirmation behaviour. TMDB failures are simulated
for API-error tasks. A task counts toward pass³ only if all three runs succeed.

Task categories: confirmation, cancellation, duplicate, changed action, invalid IDs,
ambiguous requests, API errors, multi-step (reusing tests 9 and 26), RAG reuse, details.

## Results (gpt-6-luna)

| Level | Accuracy | Abstention | Task success | pass³ | Tokens / request | Latency |
|---|---|---|---|---|---|---|
| 0 · LLM alone | 78% (25/32) | 0% (0/6) | — | — | 186 | 1.13 s |
| 1 · RAG | 91% (29/32) | 100% (6/6) | — | — | 996 | 1.34 s |
| 2 · Agent | — | — | 98% (50/51) | 94% (16/17) | 1550 | 3.46 s |

Levels 0/1: 44 shared cases, 32 scored (multi-step cases are scored at Level 2). RAG raises abstention on absent information from 0/6 to 6/6; on document questions Level 0 answers well-known films from memory, while Level 1 misses 3/20 when retrieved passages are incomplete.
Level 2: 17 tasks × 3 runs. Details and failure analysis:
[docs/agent-failures.md](docs/agent-failures.md).

## Known limitations

- **Agent:** an ambiguous title ("the Batman movie") was resolved without clarification
  in 1 of 3 runs; multi-step requests needing genre/year filters are not reproducible,
  because `search_movies` searches by title only. See [docs/agent-failures.md](docs/agent-failures.md).
- **Agent loop:** fixed `SESSION_ID` in `agent.py` (the UI overrides it per conversation);
  LLM API exceptions are not caught; the agent does not resume a multi-step request
  after confirmation.
- The judge model is the same as the answering model, so accuracy may be optimistic.
  A 20-case human-annotated calibration is planned.
- Abstention detection relies on the `[UNCERTAIN]` marker; polite refusals without the
  marker are scored as failures.
- The first RAG query in a process takes ~18 s while the embedding model loads.

## Historical results (Groq development run)

Retained as development records only; they predate the `gpt-6-luna` requirement and must
not be presented as GPT-6 Luna measurements. Setup: 9 test cases (7 scored), model
`openai/gpt-oss-120b` via Groq, temperature 0.

| Level | Accuracy | Abstention | Tokens / request | Latency |
|---|---|---|---|---|
| 0 · LLM alone | 86% (6/7) | 50% (1/2) | 409 | 0.63 s |
