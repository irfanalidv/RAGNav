from __future__ import annotations


def load_env() -> None:
    """
    Best-effort `.env` loader.

    Examples should call this before reading env vars so `MISTRAL_API_KEY` works
    when stored in a local `.env` file.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()

