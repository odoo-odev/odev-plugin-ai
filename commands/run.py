import sys
from pathlib import Path

from odev.commands.database.run import RunCommand as BaseRunCommand
from odev.common import args
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

    def _reset_cursor(self):
        """Move cursor to column 0 after PTY raw-mode streaming leaves it mid-line."""
        sys.stdout.write("\r\n")
        sys.stdout.flush()

    def _ensure_stop_after_init(self):
        """Inject --stop-after-init into odoo_args if not already present."""
        odoo_args = list(self.args.odoo_args or [])
        if "--stop-after-init" not in odoo_args:
            self.args.odoo_args = odoo_args + ["--stop-after-init"]

    def _database_has_demo(self) -> bool:
        """Return True if the database has demo data installed."""
        try:
            result = self._database.query("SELECT COUNT(*) FROM ir_module_module_demo")
            return bool(result and result[0][0] > 0)
        except Exception:
            return False

    def _has_installation_errors(self) -> bool:
        """Return True if the buffer contains module installation errors despite a zero exit code."""
        return any("ERROR" in line or "CRITICAL" in line for line in self.run_buffer)

    def _build_cmd_to_run(self) -> str:
        """Build the odev run command string for use in AI prompts."""
        argv = [a for a in self._argv if a not in ("--ai",)]
        cmd = f"odev run {' '.join(argv)}"
        if "--stop-after-init" not in cmd:
            cmd += " --stop-after-init"
        return cmd

    def run(self):
        if not self.args.ai:
            return super().run()

        self._ensure_stop_after_init()

        if self._database_has_demo():
            self._run_ai_with_demo()
        else:
            self._run_ai_without_demo()

    def _run_ai_with_demo(self):
        """Demo data present: LLM can install and fix directly."""
        logger.info("Running Odoo locally before launching AI agent...")

        try:
            try:
                super().run()
            finally:
                self._reset_cursor()

            if self._has_installation_errors():
                logger.warning("Odoo exited cleanly but module errors detected. Launching AI agent...")
            else:
                logger.info("Odoo ran successfully! No AI intervention needed.")
                return
        except CommandError:
            logger.warning("Odoo failed. Capturing logs and launching AI agent...")
        except Exception as e:
            logger.error(f"Unexpected error during Odoo execution: {e}")
            raise

        cmd_to_run = self._build_cmd_to_run()
        log_file = Path(".odev-run-failures.log").resolve()
        log_file.write_text("\n".join(self.run_buffer))

        prompt = (
            f"I ran Odoo with the following command: `{cmd_to_run}`\n\n"
            f"The execution FAILED. I have captured the logs in the file: `{log_file.name}`\n\n"
            "Please:\n"
            f"1. Read `{log_file.name}` to understand why it failed (look for CRITICAL/ERROR/Traceback).\n"
            "2. Analyze the failures.\n"
            "   IMPORTANT: Your goal is to ensure the custom module(s) work correctly and integrate seamlessly with Odoo.\n"
            "   - If a failure is IN the custom module, fix it.\n"
            "   - If a standard Odoo core failure is caused by your changes or overrides in the custom module, "
            "you MUST fix it (e.g., by adapting the custom code or monkey-patching core logic within the custom module).\n"
            "   - If a standard Odoo core failure is UNRELATED to the custom module and NOT caused by your changes, "
            "DO NOT attempt to fix it.\n"
            "3. Fix the code for the custom module.\n"
            "4. Verify your fixes by running the command again (you can use --stop-after-init).\n"
            "5. Provide a summary of your changes."
        )

        self.run_ai_agent(prompt, database=self.database_name)

    def _run_ai_without_demo(self):
        """No demo data (customer data): secure mode — only errors shared with LLM, loop until fixed."""
        logger.warning(
            "No demo data detected (customer data). Running in secure mode: "
            "only error logs will be shared with the AI agent."
        )

        cmd_to_run = self._build_cmd_to_run()

        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Attempt {iteration}: Running Odoo...")
            self.run_buffer.clear()

            try:
                try:
                    super().run()
                finally:
                    self._reset_cursor()

                if self._has_installation_errors():
                    logger.warning(f"Attempt {iteration}: module errors detected despite clean exit.")
                else:
                    logger.info("Odoo ran successfully! No AI intervention needed.")
                    return
            except CommandError:
                logger.warning(f"Attempt {iteration} failed. Extracting errors for AI agent...")
            except Exception as e:
                logger.error(f"Unexpected error during Odoo execution: {e}")
                raise

            log_file = Path(".odev-run-errors.log").resolve()
            log_file.write_text("\n".join(self.run_buffer))

            prompt = (
                f"I ran Odoo with: `{cmd_to_run}` (attempt {iteration})\n\n"
                "SECURITY NOTICE: This database contains customer data (no demo data). "
                "Do NOT run odev or odoo-bin — the system will retry automatically after you finish.\n\n"
                f"Logs are in: `{log_file.name}`\n\n"
                "Please:\n"
                f"1. Read `{log_file.name}` and look for ERROR/CRITICAL/Traceback to understand the failure.\n"
                "2. Fix errors IN the custom module or caused by the custom module.\n"
                "   Do NOT fix unrelated Odoo core failures.\n"
                "3. Fix the code only. Do NOT run odev or odoo-bin.\n"
                "4. Provide a summary of your changes."
            )

            self.run_ai_agent(prompt, database=None, ephemeral_pg=False)

            if not self.console.confirm("Do you want to try the installation again?"):
                break
