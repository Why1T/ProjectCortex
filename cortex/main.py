"""Compatibility entry point for the Cortex CLI package."""

from __future__ import annotations

from cortex.cli.app import main


if __name__ == "__main__":
    raise SystemExit(main())