import json
import sqlite3
from pathlib import Path

from odev.common.logging import logging

from .base import BaseAgentHandler


logger = logging.getLogger(__name__)

SESSION_STATE_DIR = Path(".copilot") / "session-state"
"""Where Copilot CLI keeps its conversations, one directory per session.

Each directory is named after the session id and holds an ``events.jsonl`` streaming
log plus a ``workspace.yaml`` of metadata. There is no index file in here: the store
*is* the listing, and the name of the directory is the session id - the same shape
Claude Code's project store has, one level deeper.
"""

SESSION_STORE_DB = Path(".copilot") / "session-store.db"
"""Copilot CLI's SQLite index, next to the per-session directories.

A ``sessions`` table carries an auto-generated summary of each conversation - the best
one-line account of what a session was about, and what the listing prefers as its title.
An internal, unofficial store whose schema is not promised stable between versions, so it
is read defensively - column names are resolved at read time and any failure degrades to
no title rather than an error.
"""

TRANSCRIPT_NAME = "events.jsonl"
"""The streaming event log inside each session directory - Copilot CLI's transcript."""

WORKSPACE_METADATA = "workspace.yaml"
"""The per-session metadata file, read only to tell which directory a session was held in."""

SESSION_ID_COLUMNS = ("id", "session_id", "sessionId", "uuid")
"""Column names the ``sessions`` table might key a session on, tried in order."""

SUMMARY_COLUMNS = ("summary", "title", "description", "auto_summary")
"""Column names the ``sessions`` table might keep its auto-generated summary in."""


class CopilotHandler(BaseAgentHandler):
    def get_config_dirs(self):
        return [".copilot", ".config/github-copilot", ".config/gh"]

    def get_persistent_dirs(self):
        return [".copilot", ".config/github-copilot", ".config/gh"]

    def get_creds_files(self):
        return ["config.json", "hosts.json", "hosts.yml", "config.yml"]

    def get_agent_config_rel_path(self):
        return ".copilot"

    def get_latest_session_id(self, cwd=None):
        """Return the id of the last Copilot CLI conversation, of ``cwd`` for choice.

        Read off the session directories themselves, newest ``events.jsonl`` first: the
        directory name is the session id, and the transcript's mtime is when the session
        was last spoken to. Empty transcripts are skipped - a session that was started and
        never used gives a resumed run nothing and loses the one that had it.

        The sandbox binds a working directory at the same path inside as outside, so a
        session held in the sandbox is findable from the host and the other way round -
        the same guarantee the Claude handler leans on.
        """
        state_dir = self.host_home / SESSION_STATE_DIR

        if not state_dir.is_dir():
            return None

        try:
            sessions = sorted(
                (
                    directory
                    for directory in state_dir.iterdir()
                    if directory.is_dir()
                    and (transcript := directory / TRANSCRIPT_NAME).is_file()
                    and transcript.stat().st_size
                ),
                key=lambda directory: (directory / TRANSCRIPT_NAME).stat().st_mtime,
                reverse=True,
            )
        except OSError as e:
            logger.debug(f"Could not list the Copilot CLI sessions in {state_dir}: {e}")
            return None

        if not sessions:
            return None

        if cwd:
            here = [directory for directory in sessions if self._session_cwd(directory) == str(cwd)]

            if here:
                return here[0].name

            logger.warning(
                f"No previous Copilot CLI session was held in {cwd}; "
                "resuming the most recent one from anywhere instead."
            )

        return sessions[0].name

    def _find_session_transcript(self, session_id, cwd=None):
        """Return the ``events.jsonl`` of ``session_id``, or None.

        Looked up straight by id: the session directory is named after it, so there is no
        scan to do. See :meth:`BaseAgentHandler.get_session_info`.
        """
        if not session_id:
            return None

        transcript = self.host_home / SESSION_STATE_DIR / session_id / TRANSCRIPT_NAME
        return transcript if transcript.is_file() else None

    def _parse_session_transcript(self, transcript):
        """Read a title, a fallback prompt and token counts out of a Copilot transcript.

        The ``events.jsonl`` is a stream of typed events - user messages, assistant turns,
        tool runs. The listing wants three things out of it: the last ``user.message`` as a
        fallback description, the first as a weaker fallback still, and whatever token usage
        the events carry. The best title, though, is the summary Copilot auto-generates and
        keeps in its SQLite store, so that is preferred when :meth:`_session_summary` finds
        one - the id it is keyed on is the name of the directory holding the transcript.

        The event schema is Copilot's own internal detail and not a promised-stable API, so
        every field is read with a fallback and a missing one costs the listing a column,
        never the run.
        """
        first_prompt = last_prompt = model = None
        tokens_in = tokens_out = 0

        with transcript.open() as lines:
            for line in lines:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue

                data = entry.get("data") or {}
                if not isinstance(data, dict):
                    continue

                if entry.get("type") == "user.message":
                    content = (data.get("content") or "").strip()
                    if content:
                        last_prompt = content
                        if first_prompt is None:
                            first_prompt = content

                if data.get("model"):
                    model = data["model"]

                usage = data.get("usage") or {}
                if isinstance(usage, dict):
                    tokens_in += self._usage(usage, "input_tokens", "prompt_tokens")
                    tokens_out += self._usage(usage, "output_tokens", "completion_tokens")

        return {
            "title": self._session_summary(transcript.parent.name) or first_prompt,
            "last_prompt": last_prompt,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": model,
        }

    @staticmethod
    def _usage(usage: dict, *keys: str) -> int:
        """Return the first token count ``usage`` carries under any of ``keys``, or 0."""
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
        return 0

    def _session_summary(self, session_id: str) -> str | None:
        """Return the auto-generated summary Copilot keeps for ``session_id``, or None.

        Read out of the ``sessions`` table of ``session-store.db``, opened read-only. The
        table is an internal detail with no stable schema, so the id and summary columns
        are resolved from ``PRAGMA table_info`` at read time rather than assumed - and any
        of a missing file, a missing table or a locked database degrades to None.
        """
        db = self.host_home / SESSION_STORE_DB

        if not db.is_file():
            return None

        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
                id_column = next((column for column in SESSION_ID_COLUMNS if column in columns), None)
                summary_column = next((column for column in SUMMARY_COLUMNS if column in columns), None)

                if not id_column or not summary_column:
                    return None

                # Column names come from PRAGMA, not from the caller, so this interpolation
                # carries no untrusted input; the id is still bound as a parameter.
                row = connection.execute(
                    f"SELECT {summary_column} FROM sessions WHERE {id_column} = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
        except sqlite3.Error as e:
            logger.debug(f"Could not read the Copilot session summary for {session_id!r}: {e}")
            return None

        if row and row[0]:
            return str(row[0]).strip() or None

        return None

    def _session_cwd(self, session_dir: Path) -> str | None:
        """Return the directory a session was held in, as its ``workspace.yaml`` records it.

        Only used to tell which of this place's sessions "latest" means, so a shape that
        does not carry a path, or a metadata file that cannot be read, costs the choice its
        precision and nothing else - the caller falls back to the newest session anywhere.
        """
        metadata = session_dir / WORKSPACE_METADATA

        if not metadata.is_file():
            return None

        try:
            import yaml  # noqa: PLC0415 - kept off the import path of a run that never lists sessions

            data = yaml.safe_load(metadata.read_text()) or {}
        except Exception as e:  # noqa: BLE001 - a metadata file we cannot read just costs the cwd hint
            logger.debug(f"Could not read the working directory of {session_dir}: {e}")
            return None

        if isinstance(data, dict):
            for key in ("cwd", "workspace", "workingDirectory", "directory", "path", "root"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value

        return None

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_server_names=()):  # noqa: PLR0913 - signature set by BaseAgentHandler
        cmd = ["copilot"]
        if prompt:
            cmd.extend(["-p" if headless else "-i", prompt])
        if resume:
            cmd.append(f"--resume={resume}")
        if yolo:
            cmd.append("--yolo")
        else:
            cmd.extend(
                [
                    "--allow-tool=read",
                    "--allow-tool=write",
                    "--allow-tool=shell(rtk:*)",
                    "--allow-tool=shell(odev:*)",
                    "--allow-tool=shell(git:*)",
                    "--allow-tool=shell(pre-commit:*)",
                ]
            )
        if model and model != "auto":
            cmd.extend(["-m", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
