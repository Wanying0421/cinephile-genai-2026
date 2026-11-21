# Level 2 agent: evaluation notes and failure analysis

Run: `python eval.py --level 2 --runs 3` with `gpt-6-luna`, 17 tasks × 3 runs.
Raw evidence: `reports/level2_traces.jsonl` (one line per run, filter on `task_id`),
`reports/level2_summary.json`, `reports/level2_console.txt`.

## Results

| Metric | Value |
|---|---|
| Task success (51 runs) | 98% (50/51) |
| pass³ | 94% (16/17 tasks succeed in all three runs) |
| Tokens / user request | 1550 |
| Latency / user request | 3.46 s |

Every run uses a fresh temporary SQLite watchlist. A run succeeds only if every
code-checked condition holds: no crash, final watchlist equals the expected state,
confirmation requested exactly when expected, pending action matches the expected
action and movie ID, required tools called, forbidden tools not called, and tool
errors observed where expected.

Checks that held in all runs: confirmation before every write, cancellation leaving
storage unchanged, duplicate additions as no-ops, invalid IDs returning structured
errors without writes, simulated TMDB failures handled without crashing, and a
changed request (Inception → Interstellar) requiring a new confirmation.

## Failure 1: ambiguous title resolved without clarification (L2-10)

- **Input:** "Add the Batman movie to my watchlist."
- **Expected:** a clarifying question; no confirmation proposal; watchlist unchanged.
- **Observed:** run 1 called `search_movies` → `get_movie_details` → `update_watchlist`
  and proposed adding *Batman* (1989, TMDB 268) without asking. Runs 2 and 3 asked
  which Batman film was meant. An earlier run (before the environment fix) also
  picked *The Batman* (TMDB 414906).
- **Why it matters:** identical input at temperature 0 produced different behaviour.
  A single-run evaluation could report this task as solved; pass³ exposes the
  instability.
- **Mitigation already in place:** the write is still gated. The proposal shows the
  exact title and TMDB ID, and nothing is written unless the user confirms.
- **Possible fixes:** a prompt rule ("if a search returns several plausible titles,
  ask before proposing a write"); a code-level rule refusing `update_watchlist` when
  the movie ID was not stated by the user or uniquely resolved.

## Failure 2: non-reproducible multi-step result (L2-14)

- **Input:** "Find sci-fi movies released after 2010 and add the first one to my
  watchlist." (reuses test 26)
- **Observed:** all three runs passed the checks (one movie added after
  confirmation), but the movie differed between runs: TMDB 577922 in runs 1 and 3,
  45612 in run 2.
- **Cause:** `search_movies` searches by title only; it cannot filter by genre or
  release year. The agent improvises by asking `rag_search` for candidates and then
  searching for one of them, so "the first one" is not well defined.
- **Why it matters:** a check on the number of added movies hides the issue. Users
  would get different results for the same request.
- **Possible fixes:** add a TMDB discover tool with `genre` and `year` filters, or
  make the agent present the candidate list and let the user choose.

## Other observations

- **Redundant tool calls:** in L2-13 and L2-17 the agent called `search_movies`,
  then `rag_search`, then `search_movies` again before `get_movie_details`. The
  results are correct, but token use and latency are higher.
- **Cold start:** the first RAG query in a process takes ~18 s (embedding model
  load); later requests take ~3 s.
- **Agent loop robustness:** `agent.py` uses a fixed `SESSION_ID`
  (`"default_session"`); the UI overrides it with a random per-conversation ID.
  Exceptions from the LLM API are not caught inside the loop.
- **Confirmation ends the turn:** after the user confirms, the agent does not resume
  remaining steps of a multi-step request.
- **Environment:** a first full run reached 59% pass³ because PyTorch failed to load
  on one machine (`WinError 1114`), crashing every task that used `rag_search`. Those
  failures were environmental and were excluded by fixing the install and rerunning
  all tasks.
