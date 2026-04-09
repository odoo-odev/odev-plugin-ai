"""Launch an AI CLI agent sandboxed with bwrap."""

from pathlib import Path

from odev.common import args
from odev.common.commands import DatabaseCommand
from odev.common.errors.commands import CommandError
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

    dirs = args.List(
        aliases=["-d", "--dirs"],
        description="Comma-separated list of directories to include in the sandbox (read-write). Defaults to the current directory.",
        default=[Path(".").resolve()],
    )

    def infer_database_instance(self):
        try:
            return super().infer_database_instance()
        except CommandError:
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

        self.get_ai_agent().run(
            prompt=" ".join(prompt_parts),
            sandbox_dirs=[str(Path(d).resolve()) for d in self.args.dirs],
            database=database_name,
            resume=self.args.resume,
        )
