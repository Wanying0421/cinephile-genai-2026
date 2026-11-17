"""JSON schemas for the three model-facing tools."""
from copy import deepcopy


FUNCTION_SCHEMAS = [
    {
        "name": "search_movies",
        "description": "Search TMDB by a movie title or title keywords. Returns up to five matches from page one. This is not genre discovery. Use year=null when no exact release year is requested.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "year": {"type": ["integer", "null"], "minimum": 1000, "maximum": 9999},
            },
            "required": ["query", "year"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_movie_details",
        "description": "Get a movie's title, release date, overview, runtime and genres using its TMDB movie ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"movie_id": {"type": "integer", "minimum": 1, "maximum": 2147483647}},
            "required": ["movie_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_watchlist",
        "description": "Propose adding or removing a movie in the shared local watchlist. This prepares a pending action, not an executed change. If needs_confirmation is true, show the exact proposal and wait for the user's decision through the application UI. Never claim a pending proposal was executed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "remove"]},
                "movie_id": {"type": "integer", "minimum": 1, "maximum": 2147483647},
            },
            "required": ["action", "movie_id"],
            "additionalProperties": False,
        },
    },
]


def get_tool_schemas(api="chat_completions"):
    """Return independent tool definitions for the chosen OpenAI API shape."""
    functions = deepcopy(FUNCTION_SCHEMAS)
    if api == "chat_completions":
        return [{"type": "function", "function": item} for item in functions]
    if api == "responses":
        return [{"type": "function", **item} for item in functions]
    raise ValueError("api must be chat_completions or responses")
