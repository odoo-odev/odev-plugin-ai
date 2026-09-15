"""Open the prompt of a run in the developer's editor before the agent sees it.

An agent started with a prompt answers it. There is no way to be handed one and wait:
no CLI odev drives - claude, agy, copilot, opencode - takes a prompt without running
it, and the only way to land text in a composer unsubmitted is to type it into the
terminal after the interface has booted, which races the boot and pastes kilobytes
through a text field.

So the prompt is edited before it is sent, the way a commit message is: odev writes what
it built to a file, opens it in ``$VISUAL`` or ``$EDITOR``, and hands the agent whatever
came back. An emptied buffer aborts the run, and so does an editor that exits in error -
the two ways of saying "not this one" that a developer already knows from ``git commit``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from odev.common.errors import CommandError
from odev.common.logging import logging


logger = logging.getLogger(__name__)

EDITOR_VARIABLES = ("VISUAL", "EDITOR")
"""Environment variables naming the editor to open, in the order they are honoured.

``VISUAL`` first, then ``EDITOR``: the convention every tool that opens one follows, and
the reason a developer who set both gets the full-screen editor rather than the line one.
"""


class PromptEditingAbortedError(Exception):
    """The developer asked for the run not to happen, from inside their editor."""


def edit_prompt(prompt: str) -> str:
    """Return the prompt as the developer saved it, having opened it in their editor.

    :param prompt: The prompt odev assembled, which is what the file opens on.
    :raises PromptEditingAbortedError: The buffer was saved empty, or the editor exited in
        error - ``:cq`` in vim. Both mean the run is off, and neither is a failure to
        report as one.
    :raises CommandError: No editor could be found to open. Deliberately fatal: running
        the prompt unedited is the one thing ``--edit`` exists to prevent, so a run that
        cannot be edited does not quietly become a run that was not.
    """
    editor = _resolve_editor()

    # Markdown so the editor wraps and highlights it, and named so it is recognisable in
    # a list of open buffers. Kept out of the sandbox directories on purpose: this is
    # read and written before any of them exist, and no agent ever sees the file itself.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="odev-ai-prompt-", delete=False, encoding="utf-8"
    ) as prompt_file:
        prompt_file.write(prompt)
        path = Path(prompt_file.name)

    logger.info(f"Opening the prompt in {Path(editor[0]).name}: save and quit to run it, save it empty to abort.")

    try:
        # stdio is inherited rather than captured: the editor is a full-screen program on
        # the developer's terminal, and a captured one draws to a pipe nobody is looking at.
        completed = subprocess.run([*editor, str(path)], check=False)  # noqa: S603 - the editor is the developer's own

        if completed.returncode:
            raise PromptEditingAbortedError(f"{Path(editor[0]).name} exited with code {completed.returncode}.")

        edited = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise CommandError(f"Could not edit the prompt in {' '.join(editor)}: {e}") from e
    finally:
        path.unlink(missing_ok=True)

    if not edited:
        raise PromptEditingAbortedError("The prompt was saved empty.")

    return edited


def _resolve_editor() -> list[str]:
    """Return the editor to open, as a command and its arguments.

    Split rather than handed to a shell: ``EDITOR`` is routinely set to a command with
    arguments - ``code --wait``, ``emacsclient -c`` - and running it through a shell to
    get those would run whatever else the variable happens to contain.
    """
    for variable in EDITOR_VARIABLES:
        if value := os.environ.get(variable, "").strip():
            command = value.split()

            if shutil.which(command[0]):
                return command

            logger.warning(f"${variable} names {command[0]!r}, which is not installed; looking for another editor.")

    # Nano first among the fallbacks: a developer who set neither variable is a developer
    # who has not chosen, and the one editor of the three that says how to leave it.
    for fallback in ("nano", "vim", "vi"):
        if path := shutil.which(fallback):
            logger.debug(f"No editor is configured; opening {fallback}.")
            return [path]

    raise CommandError("No editor to open the prompt in: set $EDITOR to the one you use, e.g. `export EDITOR=vim`.")
