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

    dirs = args.List(
        aliases=["-d", "--dirs"],
        description="Comma-separated list of extra directories to include in the sandbox (read-only).",
        default=[],
    )

    def get_ai_agent(self) -> AgentCLI:
        """Initialize and return an AgentCLI instance based on command arguments.

        Handles favorite CLI selection and CLI-specific model favorites.
        """
        all_clis = ["claude", "gemini", "copilot", "opencode-cli"]
        chosen_cli = self.args.cli
        favorite_cli = self.config.ai.favorite_cli

        if self.args.headless:
            self.console.bypass_prompt = True

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
                if self.args.headless:
                    favorite_cli = available[0]
                else:
                    favorite_cli = self.console.select(
                        "Which AI CLI tool do you want to use as your favorite?",
                        choices=[(c, c) for c in available],
                    )
                self.config.ai.favorite_cli = favorite_cli
                logger.info(f"Setting '{favorite_cli}' as your favorite AI CLI.")

        if chosen_cli and favorite_cli and chosen_cli != favorite_cli and not self.args.headless:
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
            if self.args.headless:
                favorite_model = "auto"
            else:
                choices = [("auto", "auto"), ("other", "Other (type it manually)")]

                favorite_model = self.console.select(
                    f"Which model do you want to use for '{final_cli}' as your favorite?",
                    choices=choices,
                )
                if favorite_model == "other":
                    favorite_model = self.console.input("Please enter the model name:")

            self.config.ai.set_favorite_model(final_cli, favorite_model)
            logger.info(f"Setting '{favorite_model}' as your favorite model for {final_cli}.")

        if chosen_model and favorite_model and chosen_model != favorite_model and not self.args.headless:
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

    def _database_has_demo(self, database_obj) -> bool:
        """Return True if the database has demo data installed."""
        try:
            # Standard Odoo way to check if demo data is loaded
            result = database_obj.query("SELECT COUNT(*) FROM ir_module_module WHERE demo = true")
            if result and result[0][0] > 0:
                return True

            # Fallback for older versions or specific configurations
            result = database_obj.query("SELECT COUNT(*) FROM ir_module_module_demo")
            return bool(result and result[0][0] > 0)
        except Exception:
            return False

    def _database_is_neutralized(self, database_obj) -> bool:
        """Return True if the database has been neutralized."""
        try:
            result = database_obj.query("SELECT value FROM ir_config_parameter WHERE key = 'database.is_neutralized'")
            return bool(result and result[0][0] == "true")
        except Exception:
            return False

    def _ensure_database_safety(self, database_name: str | None) -> bool:
        """Check if the database contains customer data and ask for confirmation if it does.

        Returns True if the data should be cloned into the sandbox, False otherwise.
        """
        if not database_name:
            return False

        from odev.common.databases import LocalDatabase

        db = LocalDatabase(database_name)
        if not db.exists:
            return False

        has_demo = self._database_has_demo(db)
        is_neutralized = self._database_is_neutralized(db)

        # If it has NO demo data OR it IS neutralized, it likely has customer data.
        has_customer_data = not has_demo or is_neutralized

        if has_customer_data:
            reason = (
                "no demo data detected" if not has_demo else "database is neutralized (indicates a production copy)"
            )
            logger.warning(f"Database '{database_name}' appears to contain customer data ({reason}).")

            if self.args.yolo:
                # YOLO: don't clone sensible data, but proceed
                return False

            if not self.console.confirm(
                f"Are you sure you want to proceed with AI operations on database '{database_name}'?",
                default=False,
            ):
                return False

        return True

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

    def _get_sandbox_dirs(self, database_name: str | None = None, cwd: Path | None = None) -> list[str]:
        """Return the list of directories to include in the sandbox.

        The first directory in the list will be used as the working directory.
        If a database is provided, we try to use its repository path as the
        working directory.
        """
        if database_name:
            from odev.common.databases.local import LocalDatabase

            db = LocalDatabase(database_name)
            if db.exists and db.process:
                # Try to get the repository path linked to the database
                # Using additional_addons_paths is generally safer as it's what Odoo uses
                addons_paths = db.process.additional_addons_paths
                if addons_paths:
                    return [str(p.resolve()) for p in addons_paths]

        target_dir = (cwd or Path.cwd()).resolve()
        home = Path.home().resolve()

        # If we are in the home directory or a non-git subdirectory,
        # use a playground to avoid exposing sensitive folders (Downloads, Documents, etc.)
        if target_dir.as_posix().startswith(home.as_posix()):
            # Check if we are inside a git repository
            is_git = False
            try:
                curr = target_dir
                while curr != curr.parent and curr.as_posix().startswith(home.as_posix()):
                    if (curr / ".git").exists():
                        is_git = True
                        break
                    curr = curr.parent
            except Exception:
                pass

            if not is_git:
                playground = self.odev.home_path / "playground"
                playground.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using AI playground sandbox: {playground}")
                return [str(playground)]

        return [str(target_dir)]

    def run_ai_agent(
        self,
        prompt: str,
        database: str | None = None,
        ephemeral_pg: bool = True,
    ) -> bool:
        """Helper to run the AI agent with common Odoo-related sandbox paths."""
        sandbox_dirs = self._get_sandbox_dirs(database)

        if database:
            should_clone = self._ensure_database_safety(database)
            if not should_clone:
                database = None

        agent = self.get_ai_agent()

        return agent.run(
            prompt,
            sandbox_dirs=sandbox_dirs,
            extra_bind_dirs=[str(d) for d in self.args.dirs] or None,
            database=database,
            resume=self.args.resume,
            ephemeral_pg=ephemeral_pg,
        )
