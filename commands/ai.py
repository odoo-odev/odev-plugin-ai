"""Launch an AI CLI agent sandboxed with bwrap."""


from odev.common import args
from odev.common.commands import DatabaseCommand
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin


logger = logging.getLogger(__name__)


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

    def infer_database_instance(self):
        try:
            return super().infer_database_instance()
        except Exception:
            # The first positional arg may be the start of the prompt (not a real database name).
            # Catching broadly here prevents SQL syntax errors when the "name" contains characters
            # like apostrophes (e.g. French text) that break raw-string SQL in database_exists().
            from odev.common.databases import DummyDatabase

            return DummyDatabase()

    def run(self) -> None:
        # If database was provided but doesn't exist, we assume it's the start of the prompt
        # This allows 'odev ai "some prompt"' to work even though argparse puts "some prompt" in 'database'
        database_name = self.database_name
        prompt_parts = list(self.args.prompt)

        if database_name and not self._database.exists:
            prompt_parts.insert(0, database_name)
            database_name = None

        self.run_ai_agent(
            prompt=" ".join(prompt_parts),
            database=database_name,
        )
