"""Common mixins for AI-related commands."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from odev.common.odev import Odev
    from odev.common.args import Namespace
    from odev.common.config import Config
    from odev.common.console import Console

import shutil
from pathlib import Path

from odev.common import args
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.agent import AgentCLI


logger = logging.getLogger(__name__)


class AICommandMixin:
    if TYPE_CHECKING:
        odev: "Odev"
        args: "Namespace"
        config: "Config"
        console: "Console"

    """Mixin for commands that use AI agents.

    Provides common arguments: cli, model, llm, yolo.
    """

    cli = args.String(
        aliases=["--cli"],
        description="The CLI AI agent to use (claude, gemini, copilot, or opencode-cli).",
        default=None,
        choices=["claude", "gemini", "copilot", "opencode-cli"],
    )

    model = args.String(
        aliases=["--model"],
        description="The specific model to use for the chosen AI agent.",
        default=None,
    )

    llm = args.String(
        aliases=["--llm"],
        description="The specific LLM model to use (alias for --model).",
        default=None,
    )

    yolo = args.Flag(
        aliases=["-y", "--yolo"],
        description="Automatically accept all AI commands (YOLO mode).",
        default=False,
    )

    def get_ai_agent(self) -> AgentCLI:
        """Initialize and return an AgentCLI instance based on command arguments.

        Handles favorite CLI selection if no CLI is specified.
        """
        all_clis = ["claude", "gemini", "copilot", "opencode-cli"]
        favorite = self.config.ai.favorite_cli
        chosen_cli = self.args.cli

        if not favorite:
            available = [c for c in all_clis if shutil.which(c)]
            if not available:
                logger.warning("No AI CLI tools found in PATH.")
                favorite = "claude"
            elif len(available) == 1:
                favorite = available[0]
                self.config.ai.favorite_cli = favorite
                logger.info(f"Setting your only detected AI CLI '{favorite}' as favorite.")
            else:
                favorite = self.console.select(
                    "Which AI CLI tool do you want to use as your favorite?",
                    choices=[(c, c) for c in available],
                )
                self.config.ai.favorite_cli = favorite
                logger.info(f"Setting '{favorite}' as your favorite AI CLI.")

        if chosen_cli and favorite and chosen_cli != favorite:
            if self.console.confirm(
                f"Do you want to set '{chosen_cli}' as your new favorite AI CLI?",
                default=False,
            ):
                self.config.ai.favorite_cli = chosen_cli
                favorite = chosen_cli

        final_cli = chosen_cli or favorite or "claude"
        model = self.args.llm or self.args.model

        return AgentCLI(
            cli=final_cli,
            model=model,
            yolo=self.args.yolo,
        )

    def run_ai_agent(self, prompt: str, database: str | None = None) -> bool:
        """Helper to run the AI agent with common Odoo-related sandbox paths."""
        agent = self.get_ai_agent()

        # Collect unique directories containing the addons
        paths = set()
        if hasattr(self, "odoobin") and self.odoobin:
            paths.update([p.as_posix() for p in self.odoobin.addons_paths if p.exists()])

        # Ensure the current directory (where the log file might be) is also included
        paths.add(Path.cwd().as_posix())

        return agent.run(prompt, sandbox_dirs=list(paths), database=database)
