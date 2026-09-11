"""Launch an AI CLI agent sandboxed with bwrap."""

import re

from odev.common import args
from odev.common.commands import DatabaseCommand
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.mixins import DEFAULT_AI_SESSION_LIMIT, AICommandMixin


logger = logging.getLogger(__name__)

SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
"""What a session id looks like, used to tell one from prompt text.

``--resume`` takes its value optionally, so argparse hands it whatever word follows on
the command line - and before a prompt, that word is the prompt's, not a session's.
"""


class AICommand(DatabaseCommand, AICommandMixin):
    """Launch an AI CLI agent sandboxed with bwrap."""

    _name = "ai"
    _database_arg_required = False
    _database_allowed_platforms = ["local"]

    @property
    def _database_exists_required(self) -> bool:
        """Return True if a database has to exist for the command to work."""
        return False

    prompt = args.String(
        description="The prompt to send to the AI agent. If multiple words are provided, they will be joined.",
        nargs="*",
    )

    sessions = args.String(
        aliases=["-s", "--sessions"],
        description="List the AI sessions started through odev (most recent first) and reopen one by id, "
        f"instead of running a prompt. Lists {DEFAULT_AI_SESSION_LIMIT} by default; pass a number to change it.",
        # Optional value: `--sessions` on its own lists the default count, `--sessions 30`
        # lists thirty. Absent, it stays None - the run is a normal prompt, not a listing.
        nargs="?",
        const="",
        default=None,
    )

    def infer_database_instance(self):
        try:
            return super().infer_database_instance()
        except Exception:
            # The first positional arg may be the start of the prompt (not a real database name).
            # Catching broadly here prevents SQL syntax errors when the "name" contains characters
            # like apostrophes (e.g. French text) that break raw-string SQL in database_exists().
            from odev.common.databases import DummyDatabase

            return DummyDatabase()

    @staticmethod
    def _resolve_sessions_limit(value: str) -> int:
        """Return how many sessions ``--sessions`` lists: the number passed, or the default.

        ``--sessions`` on its own arrives as an empty string, and a value that is not a
        positive number is a typo worth ignoring rather than a count - both fall back to
        the default instead of failing the listing over an argument.
        """
        try:
            count = int(value)
        except (TypeError, ValueError):
            return DEFAULT_AI_SESSION_LIMIT
        return count if count > 0 else DEFAULT_AI_SESSION_LIMIT

    def run(self) -> None:
        # --sessions lists past sessions and reopens one; it never starts a fresh prompt,
        # so it short-circuits before any prompt/database recovery below.
        if self.args.sessions is not None:
            self.show_ai_sessions(limit=self._resolve_sessions_limit(self.args.sessions))
            return

        # If database was provided but doesn't exist, we assume it's the start of the prompt
        # This allows 'odev ai "some prompt"' to work even though argparse puts "some prompt" in 'database'
        database_name = self.database_name
        prompt_parts = list(self.args.prompt)

        if database_name and not self._database.exists:
            prompt_parts.insert(0, database_name)
            database_name = None

        # Inserted after the database recovery above rather than before it, so the words
        # end up in the order they were typed: --resume comes first on the line.
        if resume := self.args.resume:
            if any(character.isspace() for character in resume):
                # No session id carries a space, so this is the prompt argparse took for
                # the value of the flag. Give it back, and read the flag as it was meant.
                prompt_parts.insert(0, resume)
                self.args.resume = "latest"
            elif resume != "latest" and not SESSION_ID.fullmatch(resume):
                logger.warning(
                    f"{resume!r} does not look like a session id; pass `--resume` on its own for the "
                    "latest session, and put the prompt before it."
                )

        self.run_ai_agent(
            prompt=" ".join(prompt_parts),
            database=database_name,
        )
