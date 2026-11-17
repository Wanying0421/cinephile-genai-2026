"""Shared SQLite watchlist with session-bound, single-use confirmations.

Only update_watchlist is a model-facing write tool: it prepares a proposal.
confirm_watchlist_action must be called exclusively by trusted application/UI
code after an explicit user decision. Neither approval nor session_id is a
model-controlled argument.
"""

import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .tmdb import get_movie_details


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "watchlist.sqlite3"
CONFIRMATION_TTL = 600


def _success(data):
    return {"ok": True, "data": data, "error": None}


def _failure(code, message):
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def _valid_session(session_id):
    return isinstance(session_id, str) and 0 < len(session_id) <= 128 and bool(session_id.strip())


@contextmanager
def _database(db_path):
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                movie_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                added_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS watchlist_requests (
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('add', 'remove')),
                movie_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                status TEXT NOT NULL
            );
        """)
        with connection:
            yield connection
    finally:
        connection.close()


def list_watchlist(*, db_path=None):
    """Read the shared watchlist without requiring confirmation."""
    try:
        with _database(db_path) as connection:
            rows = connection.execute(
                "SELECT movie_id, title, added_at FROM watchlist ORDER BY title COLLATE NOCASE, movie_id"
            ).fetchall()
        return _success({"movies": [dict(row) for row in rows]})
    except (OSError, sqlite3.Error):
        return _failure("STORAGE_ERROR", "Could not read the watchlist storage.")


def update_watchlist(action, movie_id, *, session_id, db_path=None):
    """Prepare a change; never add/remove a watchlist entry at this stage.

    The agent supplies action and movie_id. Its executor injects session_id and
    optionally db_path from trusted application context. A new valid proposal
    supersedes previous pending requests in that session.
    """
    if action not in ("add", "remove"):
        return _failure("INVALID_ARGUMENT", "action must be add or remove.")
    if type(movie_id) is not int or not 1 <= movie_id <= 2147483647:
        return _failure("INVALID_ARGUMENT", "movie_id must be a positive 32-bit integer.")
    if not _valid_session(session_id):
        return _failure("INVALID_ARGUMENT", "A valid application session_id is required.")

    title = None
    if action == "add":
        details = get_movie_details(movie_id)
        if not details["ok"]:
            return details
        title = details["data"].get("title")
        if not isinstance(title, str) or not title.strip():
            return _failure("INVALID_RESPONSE", "TMDB returned no valid movie title.")
        title = title.strip()

    try:
        with _database(db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE watchlist_requests SET status='superseded' WHERE session_id=? AND status='pending'",
                (session_id,),
            )
            existing = connection.execute(
                "SELECT title FROM watchlist WHERE movie_id=?", (movie_id,)
            ).fetchone()
            if action == "add" and existing:
                return _success({"status": "already_present", "needs_confirmation": False, "movie_id": movie_id, "title": existing["title"]})
            if action == "remove" and not existing:
                return _success({"status": "not_present", "needs_confirmation": False, "movie_id": movie_id})
            if action == "remove":
                title = existing["title"]
            request_id = secrets.token_urlsafe(24)
            expires_at = int(time.time()) + CONFIRMATION_TTL
            connection.execute(
                "INSERT INTO watchlist_requests VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                (request_id, session_id, action, movie_id, title, expires_at),
            )
        return _success({
            "status": "pending_confirmation", "needs_confirmation": True,
            "request_id": request_id, "action": action, "movie_id": movie_id,
            "title": title, "expires_at": expires_at,
        })
    except (OSError, sqlite3.Error):
        return _failure("STORAGE_ERROR", "Could not prepare the watchlist change.")


def confirm_watchlist_action(request_id, *, approved, session_id, db_path=None):
    """Trusted UI callback, never an LLM tool. Applies the stored proposal once.

    Invoke only after the UI presents the exact action/movie and records the
    user's explicit approval or cancellation. The payload cannot be changed
    through this function: action, movie ID and title come from storage.
    """
    if not isinstance(request_id, str) or not request_id.strip():
        return _failure("INVALID_ARGUMENT", "A request_id is required.")
    if type(approved) is not bool or not _valid_session(session_id):
        return _failure("INVALID_ARGUMENT", "A boolean user decision and valid application session are required.")
    try:
        with _database(db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM watchlist_requests WHERE request_id=? AND session_id=?",
                (request_id, session_id),
            ).fetchone()
            if row is None:
                return _failure("REQUEST_NOT_FOUND", "No confirmation request exists for this session.")
            if row["status"] != "pending":
                return _failure("REQUEST_RESOLVED", "This request is no longer pending.")
            if int(time.time()) >= row["expires_at"]:
                connection.execute("UPDATE watchlist_requests SET status='expired' WHERE request_id=?", (request_id,))
                return _failure("REQUEST_EXPIRED", "The request expired. Prepare a new proposal.")
            if not approved:
                connection.execute("UPDATE watchlist_requests SET status='cancelled' WHERE request_id=?", (request_id,))
                return _success({"status": "cancelled", "needs_confirmation": False, "movie_id": row["movie_id"], "action": row["action"]})

            if row["action"] == "add":
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO watchlist (movie_id, title, added_at) VALUES (?, ?, ?)",
                    (row["movie_id"], row["title"], int(time.time())),
                )
                status = "added" if cursor.rowcount else "already_present"
            else:
                cursor = connection.execute("DELETE FROM watchlist WHERE movie_id=?", (row["movie_id"],))
                status = "removed" if cursor.rowcount else "not_present"
            connection.execute("UPDATE watchlist_requests SET status='confirmed' WHERE request_id=?", (request_id,))
        return _success({"status": status, "needs_confirmation": False, "action": row["action"], "movie_id": row["movie_id"], "title": row["title"]})
    except (OSError, sqlite3.Error):
        return _failure("STORAGE_ERROR", "Could not confirm the watchlist change.")
