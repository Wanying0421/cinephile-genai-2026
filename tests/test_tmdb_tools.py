"""Offline tests for TMDB argument validation and error handling."""

import os
import unittest
from unittest.mock import Mock, patch

import requests

from tools.tmdb import search_movies, get_movie_details


class TMDBToolsTests(unittest.TestCase):
    def setUp(self):
        credentials = patch.dict(os.environ, {
            "TMDB_READ_TOKEN": "offline-test-token",
            "TMDB_API_KEY": "",
        })
        credentials.start()
        self.addCleanup(credentials.stop)

        http = patch("tools.tmdb.requests.get")
        self.get = http.start()
        self.addCleanup(http.stop)

    def test_invalid_arguments_skip_network(self):
        cases = [
            lambda: search_movies(""),
            lambda: search_movies("   "),
            lambda: search_movies("Inception", year="2010"),
            lambda: search_movies("Inception", year=True),
            lambda: get_movie_details(0),
            lambda: get_movie_details(-1),
            lambda: get_movie_details("27205"),
            lambda: get_movie_details(True),
        ]

        for number, call in enumerate(cases):
            with self.subTest(case=number):
                result = call()
                self.assertFalse(result["ok"])
                self.assertEqual(
                    result["error"]["code"], "INVALID_ARGUMENT"
                )

        self.get.assert_not_called()

    def test_missing_credentials_skip_network(self):
        with patch.dict(os.environ, {
            "TMDB_READ_TOKEN": "",
            "TMDB_API_KEY": "",
        }):
            result = search_movies("Inception")

        self.assertEqual(
            result["error"]["code"], "MISSING_CREDENTIALS"
        )
        self.get.assert_not_called()

    def test_http_errors(self):
        cases = {
            401: "UNAUTHORIZED",
            404: "NOT_FOUND",
            429: "RATE_LIMITED",
            500: "API_ERROR",
        }

        for status, expected in cases.items():
            with self.subTest(status=status):
                self.get.return_value = Mock(
                    status_code=status,
                    ok=False,
                )
                result = get_movie_details(27205)

                self.assertFalse(result["ok"])
                self.assertIsNone(result["data"])
                self.assertEqual(result["error"]["code"], expected)

    def test_network_errors_do_not_expose_exception_text(self):
        cases = [
            (requests.exceptions.Timeout, "TIMEOUT"),
            (requests.exceptions.ConnectionError, "NETWORK_ERROR"),
        ]

        for exception, expected in cases:
            with self.subTest(error=expected):
                self.get.side_effect = exception("sensitive-example")
                result = search_movies("Inception")

                self.assertEqual(result["error"]["code"], expected)
                self.assertNotIn("sensitive-example", str(result))

    def test_invalid_json(self):
        response = Mock(status_code=200, ok=True)
        response.json.side_effect = ValueError("invalid JSON")
        self.get.return_value = response

        result = get_movie_details(27205)

        self.assertEqual(
            result["error"]["code"], "INVALID_RESPONSE"
        )


if __name__ == "__main__":
    unittest.main()