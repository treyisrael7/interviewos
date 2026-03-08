"""Backward-compatible wrapper for the retrieval eval entry point."""

from .entrypoint import main


if __name__ == "__main__":
    raise SystemExit(main())
