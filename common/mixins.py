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
    """Mixin for commands that use AI agents.

    Provides common arguments: cli, model, yolo.
    """

    if TYPE_CHECKING:
        odev: "Odev"
        args: "Namespace"
        config: "Config"
        console: "Console"

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
    yolo = args.Flag(
        aliases=["-y", "--yolo"],
        description="Automatically accept all AI commands (YOLO mode).",
        default=False,
    )

    headless = args.Flag(
        aliases=["-H", "--headless"],
        description="Run the AI agent in non-interactive (headless) mode.",
        default=False,
    )

    resume = args.String(
        aliases=["--resume"],
        description="Resume a previous AI session by ID or 'latest'.",
        default=None,
    )

    def get_ai_agent(self) -> AgentCLI:
        """Initialize and return an AgentCLI instance based on command arguments.

        Handles favorite CLI selection and CLI-specific model favorites.
        """
        all_clis = ["claude", "gemini", "copilot", "opencode-cli"]
        chosen_cli = self.args.cli
        favorite_cli = self.config.ai.favorite_cli

        if not favorite_cli:
            available = [c for c in all_clis if shutil.which(c)]
            if not available:
                logger.warning("No AI CLI tools found in PATH.")
                favorite_cli = "claude"
            elif len(available) == 1:
                favorite_cli = available[0]
                self.config.ai.favorite_cli = favorite_cli
                logger.info(f"Setting your only detected AI CLI '{favorite_cli}' as favorite.")
            else:
                favorite_cli = self.console.select(
                    "Which AI CLI tool do you want to use as your favorite?",
                    choices=[(c, c) for c in available],
                )
                self.config.ai.favorite_cli = favorite_cli
                logger.info(f"Setting '{favorite_cli}' as your favorite AI CLI.")

        if chosen_cli and favorite_cli and chosen_cli != favorite_cli:
            if self.console.confirm(
                f"Do you want to set '{chosen_cli}' as your new favorite AI CLI?",
                default=False,
            ):
                self.config.ai.favorite_cli = chosen_cli
                favorite_cli = chosen_cli

        final_cli = chosen_cli or favorite_cli or "claude"

        chosen_model = self.args.model
        favorite_model = self.config.ai.get_favorite_model(final_cli)

        if not chosen_model and not favorite_model:
            choices = [("auto", "auto"), ("other", "Other (type it manually)")]

            favorite_model = self.console.select(
                f"Which model do you want to use for '{final_cli}' as your favorite?",
                choices=choices,
            )
            if favorite_model == "other":
                favorite_model = self.console.input("Please enter the model name:")

            self.config.ai.set_favorite_model(final_cli, favorite_model)
            logger.info(f"Setting '{favorite_model}' as your favorite model for {final_cli}.")

        if chosen_model and favorite_model and chosen_model != favorite_model:
            if self.console.confirm(
                f"Do you want to set '{chosen_model}' as your new favorite model for {final_cli}?",
                default=False,
            ):
                self.config.ai.set_favorite_model(final_cli, chosen_model)
                favorite_model = chosen_model

        final_model = chosen_model or favorite_model or "auto"

        return AgentCLI(
            cli=final_cli,
            model=final_model,
            yolo=self.args.yolo,
            headless=self.args.headless,
        )

    def _prepare_odoo_environment(self, versions: list[str]) -> dict[str, bool]:
        """Ensure required Odoo worktrees are present and up-to-date.

        For each version, clones the worktree if missing then pulls the latest.
        Returns a mapping of version → bool indicating whether the worktree is
        available after the attempt. Callers can use this to conditionally include
        source-code context in their AI prompt.
        """
        available: dict[str, bool] = {}
        for version in versions:
            if not version:
                continue
            logger.info(f"Preparing Odoo {version} environment...")
            worktree_path = self.odev.worktrees_path / version
            try:
                if not worktree_path.exists():
                    self.odev.run_command("worktree", "-C", version, "-V", version)
                self.odev.run_command("pull", "-V", version)
            except Exception as e:
                logger.warning(f"Could not prepare Odoo {version} environment: {e}")
            available[version] = worktree_path.exists()
        return available

    def run_ai_agent(
        self,
        prompt: str,
        database: str | None = None,
        ephemeral_pg: bool = True,
    ) -> bool:
        """Helper to run the AI agent with common Odoo-related sandbox paths."""
        agent = self.get_ai_agent()

        paths = set()
        if hasattr(self, "odoobin") and self.odoobin:
            paths.update([p.as_posix() for p in self.odoobin.addons_paths if p.exists()])

        paths.add(Path.cwd().as_posix())

        return agent.run(
            prompt,
            sandbox_dirs=list(paths),
            database=database,
            resume=self.args.resume,
            ephemeral_pg=ephemeral_pg,
        )
