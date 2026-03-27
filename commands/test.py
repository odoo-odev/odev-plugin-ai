from pathlib import Path

import requests

from odev.commands.database.test import TestCommand as BaseTestCommand
from odev.common import args
from odev.common.errors import CommandError
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

        # Fetch auto-tags from Runbot
        auto_tags = []
        try:
            logger.info("Fetching auto-tags from Runbot...")
            response = requests.get("https://runbot.odoo.com/runbot/auto-tags", timeout=10)
            auto_tags = [t.strip() for t in response.text.split(",") if t.strip()]

            for tag in auto_tags:
                if tag not in self.test_tags:
                    self.test_tags.append(tag)
        except Exception as error:
            logger.warning(f"Could not fetch auto-tags: {error}")

        logger.info("Running tests locally before launching AI agent...")
        try:
            super().run()
            logger.info("Tests passed! No AI intervention needed.")
            return
        except (RuntimeError, CommandError):
            if not self.test_buffer:
                raise
            logger.info("Tests failed. Capturing logs and launching AI agent...")

        # Reconstruct the command string for the AI agent
        args_to_pass = [a for a in self._argv if a not in ("--ai",)]

        if auto_tags:
            args_to_pass.extend(["--test-tags", ",".join(auto_tags)])

        cmd_to_run = f"odev test --no-pretty {' '.join(args_to_pass)}"

        # Save failure logs to a file in the current directory
        log_file = Path(".odev-test-failures.log").resolve()
        log_file.write_text("\n".join(self.test_buffer))

        prompt = (
            f"I ran Odoo tests with the following command: `{cmd_to_run}`\n\n"
            f"The tests FAILED. I have captured the failure logs in the file: `{log_file.name}`\n\n"
            "Please:\n"
            f"1. Read `{log_file.name}` to understand the failures.\n"
            "2. Analyze the failures and fix the code or the tests.\n"
            "3. Verify your fixes by running the tests again (you can run specific tests to save time).\n"
            "4. Provide a summary of your changes."
        )

        self.run_ai_agent(prompt, database=self.database_name)
