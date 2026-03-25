from odev.commands.database.test import TestCommand as BaseTestCommand
from odev.common import args, string
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin


logger = logging.getLogger(__name__)


class TestCommand(BaseTestCommand, AICommandMixin):
    """AI-enhanced test command."""

    _name = "test"

    ai = args.Flag(
        aliases=["--ai"],
        description="Use AI to run tests and fix failures.",
        default=False,
    )

    def run(self):
        if not self.args.ai:
            return super().run()

        logger.info(string.stylize("Launching AI agent to run tests and fix failures...", "color.cyan"))

        # Reconstruct the command string for the AI agent
        args_to_pass = [a for a in self._argv if a not in ("--ai",)]
        cmd_to_run = f"odev test {' '.join(args_to_pass)}"

        prompt = (
            f"I want to run Odoo tests with the following command: `{cmd_to_run}`\n\n"
            "Please:\n"
            "1. Run the command.\n"
            "2. If tests fail, analyze the failures and fix the code or the tests.\n"
            "3. Repeat until all tests pass.\n"
            "4. Provide a summary of your changes."
        )

        agent = self.get_ai_agent()

        # Collect unique directories containing the addons
        paths = set()
        if hasattr(self, "odoobin") and self.odoobin:
            paths.update([p.as_posix() for p in self.odoobin.addons_paths if p.exists()])

        agent.run(prompt, sandbox_dirs=list(paths), database=self.database_name)
