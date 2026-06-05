"""AI sandbox backends.

Selects the appropriate sandbox implementation based on the host platform:
- Linux  -> bwrap (bubblewrap)
- macOS  -> sandbox-exec (Seatbelt)
- other  -> RuntimeError

Both backends expose the same `Sandbox` interface defined in `base.py`.
"""

import sys

from .base import ExecutionSpec, Sandbox


__all__ = ["ExecutionSpec", "Sandbox", "get_sandbox", "get_sandbox_class"]


def get_sandbox_class() -> type[Sandbox]:
    """Return the Sandbox subclass for the current platform.

    Raises a clear `RuntimeError` on unsupported platforms.
    """
    if sys.platform.startswith("linux"):
        from .bwrap import BwrapSandbox

        return BwrapSandbox
    if sys.platform == "darwin":
        from .seatbelt import SeatbeltSandbox

        return SeatbeltSandbox
    raise RuntimeError(
        f"AI sandbox is only supported on Linux and macOS (detected platform: {sys.platform!r})."
    )


def get_sandbox(
    cli: str,
    handler,
    model: str = "auto",
    yolo: bool = False,
    headless: bool = False,
) -> Sandbox:
    """Instantiate the appropriate Sandbox backend for the current platform."""
    return get_sandbox_class()(cli=cli, handler=handler, model=model, yolo=yolo, headless=headless)
