from pathlib import Path

from odev.commands.database.run import RunCommand as BaseRunCommand
from odev.common import args, string
from odev.common.errors import CommandError
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run_buffer: list[str] = []

    def odoobin_progress(self, line: str):
        self.run_buffer.append(line)
        super().odoobin_progress(line)

    def run(self):
        if not self.args.ai:
            return super().run()

        logger.info(string.stylize("Running Odoo locally before launching AI agent...", "color.cyan"))

        try:
            super().run()
            logger.info(string.stylize("Odoo ran successfully! No AI intervention needed.", "color.green"))
            return
        except CommandError:
            logger.info(string.stylize("Odoo failed. Capturing logs and launching AI agent...", "color.yellow"))
        except Exception as e:
            logger.error(f"Unexpected error during Odoo execution: {e}")
            raise

        # Reconstruct the command string for the AI agent
        # self._argv contains the arguments passed after 'odev run'
        args_to_pass = [a for a in self._argv if a not in ("--ai",)]
        cmd_to_run = f"odev run {' '.join(args_to_pass)}"

        # Ensure --stop-after-init is there if they want it to terminate after init
        if "--stop-after-init" not in cmd_to_run:
            cmd_to_run += " --stop-after-init"

        # Save failure logs to a file in the current directory
        log_file = Path(".odev-run-failures.log").resolve()
        log_file.write_text("\n".join(self.run_buffer))

        prompt = (
            f"I ran Odoo with the following command: `{cmd_to_run}`\n\n"
            f"The execution FAILED. I have captured the logs in the file: `{log_file.name}`\n\n"
            "Please:\n"
            f"1. Read `{log_file.name}` to understand why it failed (look for CRITICAL/ERROR/Traceback).\n"
            "2. Analyze the logs and fix the code.\n"
            "3. Verify your fixes by running the command again (you can use --stop-after-init).\n"
            "4. Provide a summary of your changes."
        )

        agent = self.get_ai_agent()

        # Collect unique directories containing the addons
        paths = set()
        if self.odoobin:
            paths.update([p.as_posix() for p in self.odoobin.addons_paths if p.exists()])

        # Ensure the current directory (where the log file is) is also included
        paths.add(Path.cwd().as_posix())

        agent.run(prompt, sandbox_dirs=list(paths), database=self.database_name)
