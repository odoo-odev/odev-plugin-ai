from odev.commands.database.test import TestCommand as BaseTestCommand
from odev.common import args
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

        if not self.args.no_auto_tags:
            self.apply_auto_tags()

        _ai_bool = {"--ai", "-y", "--yolo", "-H", "--headless"}
        _ai_valued = {"--cli", "--model", "--resume", "-d", "--dirs"}

        args_to_pass, _skip = [], False
        for _a in self._argv:
            if _skip:
                _skip = False
            elif _a in _ai_bool:
                pass
            elif _a in _ai_valued:
                _skip = True
            else:
                args_to_pass.append(_a)
        cmd_to_run = f"odev test {' '.join(args_to_pass)}"

        custom_modules = [m for m in self.args.modules if m != "base"]
        modules_str = f" related to the following module(s): {', '.join(custom_modules)}" if custom_modules else ""

        prompt = (
            f"Run the following Odoo tests: `{cmd_to_run}`\n\n"
            f"Analyze any failures{modules_str} and resolve them by following the "
            f"**AI TEST FAILURE RESOLUTION RULES** defined in your `test_skill` skill.\n\n"
            f"After applying a fix, re-run using exactly `{cmd_to_run}` and repeat"
            " until all Rule-1 and Rule-2 failures are resolved.\n"
            "Finish with a summary of every change made and which rule justified it."
        )

        self.run_ai_agent(prompt, database=self.database_name)
