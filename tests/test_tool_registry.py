"""Offline schema/dispatcher checks, including the confirmation boundary."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.registry import execute_tool
from tools.schemas import get_tool_schemas
from tools.watchlist import list_watchlist


class ToolRegistryTests(unittest.TestCase):
    def test_schema_names_and_confirmation_boundary(self):
        tools = get_tool_schemas()
        names = {tool["function"]["name"] for tool in tools}
        self.assertEqual(names, {"search_movies", "get_movie_details", "update_watchlist"})
        for tool in tools:
            function = tool["function"]
            parameters = function["parameters"]
            self.assertTrue(function["strict"])
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
            for forbidden in ("approved", "confirmed", "session_id", "db_path", "request_id"):
                self.assertNotIn(forbidden, parameters["properties"])

    def test_response_shape_and_independent_copies(self):
        response_tools = get_tool_schemas("responses")
        self.assertEqual(response_tools[0]["name"], "search_movies")
        self.assertNotIn("function", response_tools[0])
        response_tools[0]["parameters"]["properties"].clear()
        self.assertIn("query", get_tool_schemas()[0]["function"]["parameters"]["properties"])

    def test_search_dispatch_accepts_nullable_year(self):
        expected = {"ok": True, "data": {"movies": []}, "error": None}
        with patch("tools.registry.tmdb.search_movies", return_value=expected) as search:
            result = execute_tool("search_movies", '{"query":"Inception","year":null}')
        self.assertEqual(result, expected)
        search.assert_called_once_with("Inception", year=None)

    def test_detail_dispatch(self):
        with patch("tools.registry.tmdb.get_movie_details", return_value={"ok": True}) as details:
            self.assertTrue(execute_tool("get_movie_details", {"movie_id": 27205})["ok"])
        details.assert_called_once_with(27205)

    def test_unknown_tools_are_rejected(self):
        for name in ("confirm_watchlist_action", "eval", "__import__", "list_watchlist", [], None):
            result = execute_tool(name, {})
            self.assertEqual(result["error"]["code"], "UNKNOWN_TOOL")

    def test_malformed_arguments_skip_execution(self):
        cases = ['{', '[]', 'null', '{"movie_id":1,"movie_id":2}', '{"movie_id":NaN}', 123, []]
        with patch("tools.registry.tmdb.get_movie_details") as details:
            for arguments in cases:
                self.assertEqual(execute_tool("get_movie_details", arguments)["error"]["code"], "INVALID_ARGUMENT")
            details.assert_not_called()

    def test_types_bounds_and_required_fields(self):
        cases = [
            ("get_movie_details", {"movie_id": True}),
            ("get_movie_details", {"movie_id": "27205"}),
            ("get_movie_details", {"movie_id": 0}),
            ("get_movie_details", {"movie_id": 2147483648}),
            ("search_movies", {"query": "Inception"}),
            ("search_movies", {"query": " ", "year": None}),
            ("search_movies", {"query": "x" * 201, "year": None}),
            ("search_movies", {"query": "Inception", "year": "2010"}),
            ("search_movies", {"query": "Inception", "year": 999}),
            ("update_watchlist", {"action": "delete", "movie_id": 27205}),
        ]
        with patch("tools.registry.tmdb.get_movie_details") as details, patch("tools.registry.tmdb.search_movies") as search, patch("tools.registry.watchlist.update_watchlist") as update:
            for name, arguments in cases:
                self.assertEqual(execute_tool(name, arguments)["error"]["code"], "INVALID_ARGUMENT")
            details.assert_not_called(); search.assert_not_called(); update.assert_not_called()

    def test_model_cannot_inject_approval_session_or_storage(self):
        with patch("tools.registry.watchlist.update_watchlist") as update:
            for key, value in [("approved", True), ("confirmed", True), ("request_id", "id"), ("session_id", "other"), ("db_path", "other.sqlite3")]:
                arguments = {"action": "add", "movie_id": 27205, key: value}
                self.assertEqual(execute_tool("update_watchlist", arguments, session_id="trusted")["error"]["code"], "INVALID_ARGUMENT")
            update.assert_not_called()

    def test_watchlist_dispatch_prepares_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "watchlist.sqlite3"
            fake = {"ok": True, "data": {"title": "Inception"}, "error": None}
            with patch("tools.watchlist.get_movie_details", return_value=fake):
                result = execute_tool("update_watchlist", {"action": "add", "movie_id": 27205}, session_id="trusted", db_path=db)
            self.assertTrue(result["data"]["needs_confirmation"])
            self.assertEqual(list_watchlist(db_path=db)["data"]["movies"], [])

    def test_write_requires_trusted_session(self):
        with patch("tools.watchlist.get_movie_details") as details:
            result = execute_tool("update_watchlist", {"action": "add", "movie_id": 27205})
            self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
            details.assert_not_called()

    def test_tool_error_is_preserved(self):
        error = {"ok": False, "data": None, "error": {"code": "TIMEOUT", "message": "Timed out."}}
        with patch("tools.registry.tmdb.search_movies", return_value=error):
            self.assertEqual(execute_tool("search_movies", {"query": "Inception", "year": 2010}), error)


if __name__ == "__main__":
    unittest.main()
