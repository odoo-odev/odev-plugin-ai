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
            f"Analyze any failures{modules_str} and apply exactly one of the three rules below — nothing else:\n\n"
            "RULE 1 — Custom test fails:\n"
            "   The custom module code is broken (e.g. upgrade logic did not preserve the expected workflow).\n"
            "   → Fix the custom module CODE so the workflow is correct again. Never weaken or skip the test.\n\n"
            "RULE 2 — Standard Odoo test fails because of our custom code:\n"
            "   Our changes altered a workflow or model that a standard test relied on.\n"
            "   → Monkey-patch the standard test from within the custom module so it aligns with the new workflow.\n"
            "   Never modify standard Odoo files. Never make the test trivially pass by removing assertions.\n\n"
            "RULE 3 — Standard Odoo test fails for an unrelated reason:\n"
            "   → Do nothing. Skip it and move on.\n\n"
            f"After applying the fix, re-run using exactly `{cmd_to_run}` and repeat"
            " until all Rule-1 and Rule-2 failures are resolved.\n"
            "Finish with a summary of every change made and which rule justified it."
        )

        self.run_ai_agent(prompt, database=self.database_name)
