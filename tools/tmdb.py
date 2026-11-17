"""TMDB search and details tools for the Cinephile agent."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.themoviedb.org/3"


def _failure(code, message):
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
    }


def _success(data):
    return {"ok": True, "data": data, "error": None}


def _get(path, params=None):
    """Make a read-only request without exposing credentials in errors."""
    token = os.getenv("TMDB_READ_TOKEN", "").strip()
    api_key = os.getenv("TMDB_API_KEY", "").strip()

    if not token and not api_key:
        return _failure(
            "MISSING_CREDENTIALS",
            "Configure TMDB_READ_TOKEN or TMDB_API_KEY in your local .env.",
        )

    headers = {"accept": "application/json"}
    query_params = {"language": "en-US", **(params or {})}

    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        query_params["api_key"] = api_key

    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            headers=headers,
            params=query_params,
            timeout=(5, 20),
        )
    except requests.exceptions.Timeout:
        return _failure("TIMEOUT", "The TMDB request timed out.")
    except requests.exceptions.RequestException:
        return _failure("NETWORK_ERROR", "Could not connect to TMDB.")

    if response.status_code == 401:
        return _failure("UNAUTHORIZED", "TMDB rejected the credentials.")
    if response.status_code == 404:
        return _failure("NOT_FOUND", "The requested movie was not found.")
    if response.status_code == 429:
        return _failure("RATE_LIMITED", "TMDB rate limit reached. Retry later.")
    if not response.ok:
        return _failure(
            "API_ERROR",
            f"TMDB returned HTTP {response.status_code}.",
        )

    try:
        payload = response.json()
    except ValueError:
        return _failure("INVALID_RESPONSE", "TMDB returned invalid JSON.")

    if not isinstance(payload, dict):
        return _failure("INVALID_RESPONSE", "Unexpected TMDB response format.")

    return _success(payload)


def search_movies(query, year=None):
    """Return up to five matches from the first TMDB results page."""
    if not isinstance(query, str) or not query.strip():
        return _failure("INVALID_ARGUMENT", "query must be a non-empty string.")

    if year is not None:
        if type(year) is not int or not 1000 <= year <= 9999:
            return _failure(
                "INVALID_ARGUMENT",
                "year must be a four-digit integer.",
            )

    params = {
        "query": query.strip(),
        "include_adult": "false",
        "page": 1,
    }
    if year is not None:
        params["primary_release_year"] = year

    result = _get("/search/movie", params)
    if not result["ok"]:
        return result

    payload = result["data"]
    matches = payload.get("results")
    if not isinstance(matches, list):
        return _failure("INVALID_RESPONSE", "TMDB returned no results list.")

    movies = []
    for movie in matches[:5]:
        if not isinstance(movie, dict) or type(movie.get("id")) is not int:
            return _failure("INVALID_RESPONSE", "Invalid movie in TMDB results.")

        movies.append({
            "movie_id": movie["id"],
            "title": movie.get("title"),
            "release_date": movie.get("release_date"),
            "genre_ids": movie.get("genre_ids", []),
            "source_url": f"https://www.themoviedb.org/movie/{movie['id']}",
        })

    return _success({
        "query": query.strip(),
        "year": year,
        "total_results": payload.get("total_results", 0),
        "movies": movies,
    })


def get_movie_details(movie_id):
    """Return the details of one movie identified by its TMDB ID."""
    if type(movie_id) is not int or not 1 <= movie_id <= 2147483647:
        return _failure(
            "INVALID_ARGUMENT",
            "movie_id must be a positive 32-bit integer.",
        )

    result = _get(f"/movie/{movie_id}")
    if not result["ok"]:
        return result

    movie = result["data"]
    if movie.get("id") != movie_id:
        return _failure("INVALID_RESPONSE", "Unexpected movie identifier.")

    return _success({
        "movie_id": movie["id"],
        "title": movie.get("title"),
        "release_date": movie.get("release_date"),
        "overview": movie.get("overview"),
        "runtime": movie.get("runtime"),
        "genres": movie.get("genres", []),
        "source_url": f"https://www.themoviedb.org/movie/{movie_id}",
    })