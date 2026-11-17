"""Offline watchlist tests: temporary databases and simulated TMDB details."""

import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tools.watchlist import confirm_watchlist_action, list_watchlist, update_watchlist


class WatchlistTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = Path(folder.name) / "watchlist.sqlite3"
        self.session = "test-session-a"
        details = patch("tools.watchlist.get_movie_details", side_effect=lambda movie_id: {
            "ok": True, "data": {"movie_id": movie_id, "title": f"Movie {movie_id}"}, "error": None,
        })
        self.details = details.start()
        self.addCleanup(details.stop)

    def propose(self, action="add", movie_id=27205, session_id=None):
        return update_watchlist(action, movie_id, session_id=session_id or self.session, db_path=self.db)

    def decide(self, request_id, approved=True, session_id=None):
        return confirm_watchlist_action(request_id, approved=approved, session_id=session_id or self.session, db_path=self.db)

    def movies(self):
        result = list_watchlist(db_path=self.db)
        self.assertTrue(result["ok"], result)
        return result["data"]["movies"]

    def add(self, movie_id=27205, session_id=None):
        pending = self.propose(movie_id=movie_id, session_id=session_id)
        self.assertTrue(pending["ok"], pending)
        return self.decide(pending["data"]["request_id"], session_id=session_id)

    def test_new_watchlist_is_empty(self):
        self.assertEqual(self.movies(), [])

    def test_proposal_never_changes_watchlist(self):
        pending = self.propose()
        self.assertTrue(pending["data"]["needs_confirmation"])
        self.assertEqual(pending["data"]["status"], "pending_confirmation")
        self.assertEqual(pending["data"]["title"], "Movie 27205")
        self.assertEqual(self.movies(), [])

    def test_confirmed_add_persists_across_processes(self):
        self.assertEqual(self.add()["data"]["status"], "added")
        script = "import json,sys; from tools.watchlist import list_watchlist; print(json.dumps(list_watchlist(db_path=sys.argv[1])))"
        output = subprocess.check_output([sys.executable, "-c", script, str(self.db)], text=True)
        movies = json.loads(output)["data"]["movies"]
        self.assertEqual([movie["movie_id"] for movie in movies], [27205])

    def test_cancelled_add_leaves_storage_unchanged(self):
        self.add(movie_id=1)
        before = self.movies()
        pending = self.propose(movie_id=2)
        result = self.decide(pending["data"]["request_id"], approved=False)
        self.assertEqual(result["data"]["status"], "cancelled")
        self.assertEqual(self.movies(), before)

    def test_confirmed_remove(self):
        self.add()
        self.details.reset_mock()
        pending = self.propose("remove")
        self.assertEqual(len(self.movies()), 1)
        result = self.decide(pending["data"]["request_id"])
        self.assertEqual(result["data"]["status"], "removed")
        self.assertEqual(self.movies(), [])
        self.details.assert_not_called()

    def test_cancelled_remove_keeps_movie(self):
        self.add()
        before = self.movies()
        pending = self.propose("remove")
        self.decide(pending["data"]["request_id"], approved=False)
        self.assertEqual(self.movies(), before)

    def test_duplicate_add_is_noop(self):
        self.add()
        result = self.propose()
        self.assertEqual(result["data"]["status"], "already_present")
        self.assertFalse(result["data"]["needs_confirmation"])
        self.assertEqual(len(self.movies()), 1)

    def test_missing_remove_is_noop(self):
        result = self.propose("remove")
        self.assertEqual(result["data"]["status"], "not_present")
        self.assertEqual(self.movies(), [])
        self.details.assert_not_called()

    def test_confirmation_is_bound_to_session(self):
        request_id = self.propose()["data"]["request_id"]
        result = self.decide(request_id, session_id="another-session")
        self.assertEqual(result["error"]["code"], "REQUEST_NOT_FOUND")
        self.assertEqual(self.movies(), [])
        self.assertEqual(self.decide(request_id)["data"]["status"], "added")

    def test_approval_must_be_a_boolean(self):
        request_id = self.propose()["data"]["request_id"]
        result = self.decide(request_id, approved="true")
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(self.movies(), [])

    def test_resolved_request_cannot_be_replayed(self):
        request_id = self.propose()["data"]["request_id"]
        self.decide(request_id)
        result = self.decide(request_id)
        self.assertEqual(result["error"]["code"], "REQUEST_RESOLVED")
        self.assertEqual(len(self.movies()), 1)

    def test_cancelled_request_cannot_be_approved_later(self):
        request_id = self.propose()["data"]["request_id"]
        self.decide(request_id, approved=False)
        result = self.decide(request_id)
        self.assertEqual(result["error"]["code"], "REQUEST_RESOLVED")
        self.assertEqual(self.movies(), [])

    def test_new_proposal_supersedes_previous_one(self):
        old = self.propose(movie_id=1)["data"]["request_id"]
        new = self.propose(movie_id=2)["data"]["request_id"]
        self.assertEqual(self.decide(old)["error"]["code"], "REQUEST_RESOLVED")
        self.decide(new)
        self.assertEqual([movie["movie_id"] for movie in self.movies()], [2])

    def test_expired_request_cannot_write(self):
        with patch("tools.watchlist.time.time", return_value=100):
            request_id = self.propose()["data"]["request_id"]
        with patch("tools.watchlist.time.time", return_value=700):
            result = self.decide(request_id)
        self.assertEqual(result["error"]["code"], "REQUEST_EXPIRED")
        self.assertEqual(self.movies(), [])

    def test_parallel_confirmations_apply_once(self):
        request_id = self.propose()["data"]["request_id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.decide(request_id), range(2)))
        self.assertEqual(sum(result["ok"] for result in results), 1)
        self.assertEqual(len(self.movies()), 1)

    def test_two_sessions_cannot_create_duplicates(self):
        first = self.propose()["data"]["request_id"]
        second = self.propose(session_id="session-b")["data"]["request_id"]
        self.decide(first)
        result = self.decide(second, session_id="session-b")
        self.assertEqual(result["data"]["status"], "already_present")
        self.assertEqual(len(self.movies()), 1)

    def test_invalid_arguments_skip_tmdb(self):
        for action, movie_id, session in [("delete", 1, self.session), ("add", True, self.session), ("add", 0, self.session), ("add", 1, "")]:
            result = update_watchlist(action, movie_id, session_id=session, db_path=self.db)
            self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
        self.details.assert_not_called()
        self.assertFalse(self.db.exists())

    def test_tmdb_failure_does_not_prepare_action(self):
        self.details.side_effect = None
        self.details.return_value = {"ok": False, "data": None, "error": {"code": "TIMEOUT", "message": "Timed out."}}
        self.assertEqual(self.propose()["error"]["code"], "TIMEOUT")
        self.assertFalse(self.db.exists())

    def test_storage_error_is_structured(self):
        self.db.write_text("not a SQLite database", encoding="utf-8")
        result = list_watchlist(db_path=self.db)
        self.assertEqual(result["error"]["code"], "STORAGE_ERROR")


if __name__ == "__main__":
    unittest.main()
