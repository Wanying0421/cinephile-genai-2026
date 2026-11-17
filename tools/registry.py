"""Validate model tool calls manually before dispatching to approved functions."""
import json

from . import tmdb, watchlist
from .schemas import FUNCTION_SCHEMAS


def _failure(code, message):
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-standard JSON constant")


def _validate(arguments, schema):
    """Validate the flat scalar schemas used here; this is not a general validator."""
    properties = schema["properties"]
    if any(key not in properties for key in arguments):
        return "Unexpected tool arguments are not allowed."
    if any(key not in arguments for key in schema["required"]):
        return "Required tool arguments are missing."
    for key, value in arguments.items():
        rule = properties[key]
        types = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
        valid = (
            ("null" in types and value is None)
            or ("integer" in types and type(value) is int)
            or ("string" in types and isinstance(value, str))
        )
        if not valid:
            return f"Invalid argument type for {key}."
        if value is None:
            continue
        if "enum" in rule and value not in rule["enum"]:
            return f"Invalid value for {key}."
        if isinstance(value, str):
            if not value.strip() or not rule.get("minLength", 0) <= len(value) <= rule.get("maxLength", float("inf")):
                return f"Invalid text length or empty value for {key}."
        if type(value) is int:
            if value < rule.get("minimum", -float("inf")) or value > rule.get("maximum", float("inf")):
                return f"Argument {key} is outside its allowed range."
    return None


def execute_tool(name, arguments, *, session_id=None, db_path=None):
    """Execute a model-facing tool; session and storage path are trusted context.

    arguments is a JSON string or decoded dict. Approval and confirmation
    callbacks are deliberately absent from this registry.
    """
    functions = {item["name"]: item for item in FUNCTION_SCHEMAS}
    if not isinstance(name, str) or name not in functions:
        return _failure("UNKNOWN_TOOL", "The requested tool is not available.")
    if isinstance(arguments, str):
        if len(arguments) > 10000:
            return _failure("INVALID_ARGUMENT", "Tool arguments exceed the size limit.")
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            return _failure("INVALID_ARGUMENT", "Tool arguments must be valid JSON with unique keys.")
    if type(arguments) is not dict:
        return _failure("INVALID_ARGUMENT", "Tool arguments must be a JSON object.")
    error = _validate(arguments, functions[name]["parameters"])
    if error:
        return _failure("INVALID_ARGUMENT", error)

    if name == "search_movies":
        return tmdb.search_movies(arguments["query"], year=arguments["year"])
    if name == "get_movie_details":
        return tmdb.get_movie_details(arguments["movie_id"])
    return watchlist.update_watchlist(arguments["action"], arguments["movie_id"], session_id=session_id, db_path=db_path)
