"""Persistent index of the AI sessions launched through odev.

Only sessions started by an odev command land here: the store is written by
:meth:`AgentCLI.run` and nowhere else, so a ``claude`` run started by hand never shows
up. Each record ties a session id to the CLI that holds its transcript and to the odev
command that launched it - which is what ``odev ai --sessions`` groups by.

The heavy, changing part of a session - its title and token counts - is *not* kept here.
It is read back from the agent's own transcript at display time (see
:meth:`BaseAgentHandler.get_session_info`), so it cannot go stale and is never counted
twice.
"""

import json
from datetime import datetime
from pathlib import Path

from odev.common.config import CONFIG_DIR
from odev.common.logging import logging


logger = logging.getLogger(__name__)

SESSIONS_FILE = CONFIG_DIR / "ai-sessions.json"
"""Where the index lives, next to odev's own config files.

Written by the host odev process once the sandboxed agent has exited, so the read-only
view the agent had of ``~/.config/odev`` does not apply: the recording happens outside
the sandbox.
"""

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
"""How timestamps are written, matching odev's other config sections."""


class SessionStore:
    """Read/write access to the on-disk index of odev-launched AI sessions."""

    def __init__(self, path: Path = SESSIONS_FILE):
        self.path = path

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError) as error:
            logger.debug(f"Could not read the AI sessions index {self.path}: {error}")
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2))
        except OSError as error:
            logger.debug(f"Could not write the AI sessions index {self.path}: {error}")

    @staticmethod
    def _key(cli: str, session_id: str) -> str:
        """Key a record by CLI and id, so two agents cannot clash on the same id."""
        return f"{cli}:{session_id}"

    def record(self, session_id: str, cli: str, source: str, cwd: str | None) -> None:
        """Remember that ``source`` launched ``session_id`` on ``cli`` in ``cwd``.

        The origin command is kept as first written: resuming a ``scaffold`` session from
        ``odev ai`` does not turn it into an ``ai`` session - what created it is what it is
        grouped under. Only the last-seen time and the working directory move.
        """
        if not session_id:
            return

        data = self._load()
        key = self._key(cli, session_id)
        now = datetime.now().strftime(DATETIME_FORMAT)

        record = data.get(key, {})
        record.setdefault("id", session_id)
        record.setdefault("cli", cli)
        record.setdefault("source", source)
        record.setdefault("started_at", now)
        record["cwd"] = cwd or record.get("cwd")
        record["updated_at"] = now

        data[key] = record
        self._save(data)

    def list(self) -> list[dict]:
        """Return the recorded sessions, most recently used first."""
        records = list(self._load().values())
        records.sort(key=lambda record: record.get("updated_at", ""), reverse=True)
        return records
