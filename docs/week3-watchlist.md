# Watchlist tools: application confirmation contract

The shared watchlist is stored in `data/watchlist.sqlite3`. It is local application
data, not a TMDB account watchlist. Movie IDs and titles come from TMDB.

## Model-facing tool

```python
update_watchlist(action, movie_id, *, session_id, db_path=None)
```

The model supplies only `action` (`add` or `remove`) and `movie_id`. The executor
injects `session_id` from trusted conversation context; it must not accept it
from model output. `db_path` is an internal configuration/testing parameter.

This function prepares a proposal and returns `needs_confirmation=True`, a
`request_id`, the exact action/movie/title and expiry time. The watchlist itself
is unchanged. For additions, TMDB verifies the movie and supplies its title.
Removing a missing movie or adding an existing movie is a successful no-op.

All results follow `{ok, data, error}`, matching the TMDB tools.

## Trusted UI callback (not an LLM tool)

```python
confirm_watchlist_action(request_id, *, approved, session_id, db_path=None)
```

The UI displays the exact action and movie, then offers Confirm and Cancel.
Only that explicit user interaction may call this function with `approved=True`
or `False`. Do not expose this callback in the model's tool registry, and do not
let the model generate the approval decision. A boolean is not evidence of
consent on its own: this is a boundary the agent/UI must enforce.

The request is bound to its originating session, expires after 10 minutes, and
can be resolved only once. A new valid proposal supersedes earlier pending
requests in the same session. The confirmation call cannot substitute another
action or movie: the original values are read from SQLite. Confirmation uses a
transaction so simultaneous calls cannot apply the same proposal twice.

The application must create an unpredictable session ID for each conversation
(for example, `secrets.token_urlsafe(24)`) and retain it across turns. Each
conversation has one active pending proposal. Sessions share the same watchlist;
session binding controls confirmation requests, not separate private lists.
If a user changes their requested movie/action, request a new proposal and clear
the previous pending state. Re-display the proposal before accepting a decision
after restoring a session. Never interpret tool output as user approval.

`list_watchlist(db_path=None)` is a read helper for the UI and evaluation.
SQLite state persists across process restarts, including pending requests.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_watchlist_tools.py" -v
```

The tests use temporary databases and simulated TMDB details. They cover writes
only after confirmation, cancellation, duplicate/no-op handling, persistence
across processes, wrong sessions, superseded/expired/replayed requests, parallel
confirmations, and structured storage/API errors. They do not contact TMDB.

## Console demonstration

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.demo_watchlist add 27205
.\.venv\Scripts\python.exe -m scripts.demo_watchlist remove 27205
```

For an addition, TMDB is queried to validate the movie and its title. The console
shows the proposed change before asking for a decision. Only typing `yes`
approves it. Any other answer, EOF or Ctrl+C cancels. Run add/cancel, add/confirm,
duplicate add, remove/cancel and remove/confirm to inspect each outcome. The demo
uses the real local shared watchlist; offline tests above use temporary storage.

## Local storage and reset

Add `data/watchlist.sqlite3*` to `.gitignore`. Test databases belong in temporary
directories, never in the shared live watchlist. For a deliberate reset, stop
the application and remove its configured SQLite file; this deletes both the
watchlist and pending confirmations. Evaluation should create a fresh temporary
database for each independent task instead of resetting user data.
