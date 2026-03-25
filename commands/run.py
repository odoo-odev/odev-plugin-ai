from odev.commands.database.run import RunCommand as BaseRunCommand
from odev.common import args, string
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin


logger = logging.getLogger(__name__)


class RunCommand(BaseRunCommand, AICommandMixin):
    """AI-enhanced run command."""

    _name = "run"

    ai = args.Flag(
        aliases=["--ai"],
        description="Use AI to run Odoo and fix issues if it fails.",
        default=False,
    )

    def run(self):
        if not self.args.ai:
            return super().run()

        logger.info(string.stylize("Launching AI agent to run Odoo and fix issues...", "color.cyan"))

        # Reconstruct the command string for the AI agent
        # self._argv contains the arguments passed after 'odev run'
        args_to_pass = [a for a in self._argv if a not in ("--ai",)]
        cmd_to_run = f"odev run {' '.join(args_to_pass)}"

        # Ensure --stop-after-init is there if they want it to terminate after init
        if "--stop-after-init" not in cmd_to_run:
            cmd_to_run += " --stop-after-init"

        prompt = (
            f"I want to run Odoo with the following command: `{cmd_to_run}`\n\n"
            "Please:\n"
            "1. Run the command.\n"
            "2. If it fails (CRITICAL/ERROR/Traceback), analyze the logs and fix the code.\n"
            "3. Repeat until the command runs successfully (at least through initialization).\n"
            "4. Provide a summary of your changes."
        )

        agent = self.get_ai_agent()

        # Collect unique directories containing the addons
        paths = set()
        if self.odoobin:
            paths.update([p.as_posix() for p in self.odoobin.addons_paths if p.exists()])

        agent.run(prompt, sandbox_dirs=list(paths), database=self.database_name)
