# Week 3 tools handoff (Laroche -> Wanying -> Nouha)

## Configuration

Use the existing project environment. `requests` and `python-dotenv` are already
in requirements.txt; SQLite uses Python's standard library. Configure either
`TMDB_READ_TOKEN` (preferred read access token) or `TMDB_API_KEY` in the local
ignored `.env`. Never include credentials in traces or commits.

## Interfaces

| Function | Behaviour |
|---|---|
| `tools.tmdb.search_movies(query, year=None)` | Title/keyword search; up to five matches from page one. |
| `tools.tmdb.get_movie_details(movie_id)` | Movie details with overview, genres, runtime and source URL. |
| `tools.watchlist.update_watchlist(action, movie_id, *, session_id, db_path=None)` | Prepare a proposed addition/removal; never apply an entry change immediately. |
| `tools.watchlist.list_watchlist(db_path=None)` | Read helper for UI/evaluation; not a model-facing tool. |
| `tools.watchlist.confirm_watchlist_action(request_id, *, approved, session_id, db_path=None)` | Trusted UI callback, deliberately excluded from the model tool registry. |

All tool results use `{ok, data, error}`. Errors contain a code and a safe message.
Movie results use `movie_id`, `title`, `release_date` and `source_url`; search
also provides `genre_ids`, details provide named `genres`, `runtime` and
`overview`. `year` is an exact primary release year, not a minimum-year filter.

## Schemas and validated execution

```python
from tools.schemas import get_tool_schemas
from tools.registry import execute_tool

chat_tools = get_tool_schemas("chat_completions")
response_tools = get_tool_schemas("responses")

result = execute_tool(
    "search_movies",
    '{"query": "Inception", "year": 2010}',
)
```

`docs/tool-schemas.json` contains the same function definitions in JSON. Strict
schemas require all declared fields; use `year: null` when the year is unknown.
The dispatcher also validates arguments manually. It rejects unknown tools,
extra/missing fields, invalid types/ranges, duplicate JSON keys and malformed
JSON. Validation and dispatch are implemented without a framework.

For watchlist calls, inject a stable, unpredictable application session ID:

```python
proposal = execute_tool(
    "update_watchlist",
    {"action": "add", "movie_id": 27205},
    session_id=conversation_session_id,
)
```

The model cannot supply `session_id`, `db_path`, `approved` or `confirmed`.
Never execute a tool name using eval, dynamic imports or unrestricted getattr.

## Confirmation handoff

When `proposal["data"]["needs_confirmation"]` is true, pause the agent loop.
Save the returned request ID, show the exact action/movie/title, and wait for
an explicit user decision in the UI. Only trusted application code may call
`confirm_watchlist_action`. Return its result to the resumed agent; do not ask
the model whether it has permission. A pending proposal is not a completed task.
See [watchlist confirmation contract](week3-watchlist.md) for expiry, session
binding, persistence and concurrency semantics. The UI must retain session state
across turns; watchlist state is shared across sessions.

## Model integration (B)

These tools themselves make no language-model calls. B must connect the agent
using the instructor-required `gpt-6-luna`. Its Chat Completions function calls
require `reasoning_effort="none"`; Responses can be used for reasoning with
tools. Select the matching schema wrapper, preserve the model's tool-call IDs,
and return the corresponding tool results. The existing Level 0 completion
helper currently accepts messages only; B will add tool support as part of the
agent work while preserving Levels 0/1 and the required model configuration.

The current search is by title/keywords. A request such as "find science-fiction
movies released after 2010" requires discovery support or an explicit candidate
strategy; do not assume title search provides a complete genre/date search.
This limitation must be resolved before using that example as a Level 2 success.

## Verification and observed development checks

On November 17, 2026, live search/details returned Inception (27205), release
date 2010-07-15 and runtime 148 minutes. The console watchlist demonstration
confirmed that `no` cancelled an addition and left the list empty, while `yes`
returned `added`. No agent or OpenAI tool-calling end-to-end run is claimed here.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

TMDB error/argument tests, watchlist tests and registry tests are offline. They
simulate external responses and use temporary storage. For live checks, use the
existing search/details example and `python -m scripts.demo_watchlist add 27205`.
The demo shows state before and after an explicit console decision.

## Evaluation and attribution (C)

Use isolated databases via the internal `db_path` parameter. Check final state,
including cancellation and duplicate handling, rather than trusting the answer.
The agent still needs limits, conversation handling, token/latency tracking and
its own tests. The UI and final evaluation are separate contributions.

External data is from TMDB, not the Wikipedia RAG corpus. Keep source URLs and
add the required TMDB attribution/branding when the UI is integrated, following
the service's current attribution requirements.

Official references:
- https://developer.themoviedb.org/reference/search-movie
- https://developer.themoviedb.org/reference/movie-details
- https://developer.themoviedb.org/docs/authentication-application
- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/api/docs/models/gpt-6-luna
