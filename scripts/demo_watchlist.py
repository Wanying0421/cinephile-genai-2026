"""Console demonstration of the trusted confirmation boundary."""
import argparse
import json
import secrets

from tools.watchlist import confirm_watchlist_action, list_watchlist, update_watchlist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["add", "remove"])
    parser.add_argument("movie_id", type=int)
    args = parser.parse_args()
    session_id = secrets.token_urlsafe(24)

    print("Current watchlist:")
    current = list_watchlist()
    print(json.dumps(current, indent=2, ensure_ascii=False))
    if not current["ok"]:
        raise SystemExit(1)

    proposal = update_watchlist(args.action, args.movie_id, session_id=session_id)
    print("\nProposed action:")
    print(json.dumps(proposal, indent=2, ensure_ascii=False))
    if not proposal["ok"]:
        raise SystemExit(1)
    pending = proposal["data"]
    if not pending["needs_confirmation"]:
        return

    print(f"\n{pending['action'].upper()}: {pending['title']} (TMDB {pending['movie_id']})")
    try:
        approved = input("Confirm this exact change? Type yes to approve, anything else to cancel: ").strip().casefold() == "yes"
    except (EOFError, KeyboardInterrupt):
        approved = False
        print("\nCancelling the proposed change.")
    result = confirm_watchlist_action(pending["request_id"], approved=approved, session_id=session_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nWatchlist after the decision:")
    print(json.dumps(list_watchlist(), indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
