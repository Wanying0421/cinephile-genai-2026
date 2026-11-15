"""Check the configured model with one short request; never print credentials."""
from openai import APIConnectionError, APIStatusError


def main():
    from level0 import MODEL, complete

    print(f"Configured model: {MODEL}")
    try:
        response = complete([{"role": "user", "content": "Reply with exactly OK."}])
    except APIStatusError as error:
        raise SystemExit(
            f"API check failed (HTTP {error.status_code}). "
            "Check the endpoint, key, model access and project quota locally."
        ) from None
    except APIConnectionError:
        raise SystemExit("Connection failed. Check the endpoint and network settings.") from None

    answer = (response.choices[0].message.content or "").strip()
    if answer != "OK":
        raise SystemExit("API connected, but the expected OK response was not returned.")
    print("Connection OK")
    print(f"Response model: {response.model}")
    print("Answer: OK")


if __name__ == "__main__":
    main()
