"""Export an Excalidraw board to an SVG the agent can read, on demand.

A board is a live document: there is no url that hands back a picture, so the only way
to get one is to open the board in a browser and drive its own export. A run that builds
a prompt can do that for the agent up front, but the agent itself is behind a sandbox
with a fresh browser, and most boards a task links it never needs to look at. This
command is the agent's way in: handed the url of a board, it writes the SVG and says
where, so the agent exports the architecture of the diagram it actually needs when it
needs it, instead of being handed every board a task links whether it looks or not.

SVG rather than PNG, because the agent reads a diagram far better that way: every label
of the board stays text it can read straight out of the markup, and every box a shape
with coordinates, so it reads the names of the models and the arrows between them rather
than inferring them from pixels. See :mod:`~.common.excalidraw`.

Non-interactive: this runs from inside the sandbox, where there is no developer at the
terminal to hand over a board the browser cannot open. A private Excalidraw+ board,
signed in to nothing, is reported and left out rather than asked for.
"""

from __future__ import annotations

from pathlib import Path

from odev.common import args
from odev.common.commands import Command
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.excalidraw import export_diagrams


logger = logging.getLogger(__name__)


class ExcalidrawExportCommand(Command):
    """Export one or more Excalidraw boards to SVG, and print where each one landed."""

    _name = "excalidraw-export"

    path = args.Path(
        aliases=["--path"],
        description="Directory to write the exported SVG(s) into (default: current directory).",
        default=Path(".").resolve(),
    )
    urls = args.String(
        name="urls",
        nargs="+",
        description="The Excalidraw board URL(s) to export.",
    )

    def run(self) -> None:
        """Export each board of ``--urls`` and print the path of every SVG written."""
        directory = self.args.path.resolve()
        directory.mkdir(parents=True, exist_ok=True)

        paths = export_diagrams(list(self.args.urls), directory, self.odev, interactive=False)

        if not paths:
            raise self.error(
                "Could not export any Excalidraw board. A board that is private and opens only for "
                "the people it was shared with cannot be exported from here, which is signed in to "
                "nothing - ask the developer to export it by hand (Ctrl+Shift+E as SVG) if you need it."
            )

        for path in paths:
            logger.info(f"Exported Excalidraw diagram to {path}")

        if len(paths) < len(self.args.urls):
            logger.warning(
                f"Exported {len(paths)} of {len(self.args.urls)} board(s); "
                "the rest did not open (see the warnings above)."
            )
