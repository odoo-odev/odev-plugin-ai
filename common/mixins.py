"""Common mixins for AI-related commands."""

from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    from odev.common.args import Namespace
    from odev.common.config import Config
    from odev.common.console import Console
    from odev.common.odev import Odev

import json
import shutil
import subprocess
from pathlib import Path

from odev.common import args
from odev.common.console import console
from odev.common.databases.local import LocalDatabase
from odev.common.errors import CommandError
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.agent import AgentCLI
from odev.plugins.odev_plugin_ai.common.sandbox import get_sandbox_class


logger = logging.getLogger(__name__)

SKILLS_REPO = "odoo-ps/ps-ai-skills"
GUIDELINES_SKILL = "odoo_coding_guidelines"

# Shared store the skills CLI installs into, whatever the agent.
SKILLS_STORE = Path.home() / ".agents" / "skills"

SKILLS_TIMEOUT = 120
"""How long the skills CLI is given to answer, in seconds.

Generous: installing clones the skills repo, which on a cold run is a network round trip.
It is a ceiling on a hang, not a budget for a normal call."""


def _newest_mtime(directory: Path) -> float:
    """Return the most recent mtime found anywhere under the given directory."""
    return max((p.stat().st_mtime for p in directory.rglob("*")), default=0.0)


class AICommandMixin:
    """Mixin for commands that use AI agents.

    Provides common arguments: cli, model, yolo.
    """

    if TYPE_CHECKING:
        odev: "Odev"
        args: "Namespace"
        config: "Config"
        console: "Console"
        _name: str

    sandbox_repository: Path | None = None
    """A checkout of the code the run is about, to work in ahead of anywhere else.

    Set by the command that knows how to find one - which is not this plugin: what
    links a task, a database and a repository together lives where tasks and hosted
    databases do. Here it is a path, taken on trust and bound into the sandbox.
    """

    sandbox_scope: str | None = None
    """Optional scope used by commands that isolate sandbox playground data."""

    required_skills: ClassVar[list[str]] = ["odev", GUIDELINES_SKILL]
    """Skills the agent needs to run this command, installed before it starts.

    Declared by the command rather than decided from its name: a command knows what
    method its prompt leans on, and a prompt that points at a skill the agent was never
    given is a prompt missing the half that was moved out of it. A command whose skills
    depend on the run appends to this before calling :meth:`run_ai_agent`.
    """

    cli = args.String(
        aliases=["--cli"],
        description="The CLI AI agent to use (claude, agy, copilot, or opencode-cli).",
        default=None,
        choices=["claude", "agy", "copilot", "opencode-cli"],
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
        description="Comma-separated list of extra directories to include in the sandbox (read-write).",
        default=[],
    )

    def _ensure_sandbox_supported(self) -> None:
        """Verify that this host can run the AI sandbox; raise otherwise."""
        try:
            sandbox_cls = get_sandbox_class()
        except RuntimeError as e:
            console.print(f"\n[bold red]Error:[/] {e}")
            raise CommandError(str(e)) from e

        supported, message = sandbox_cls.check_support()
        if not supported:
            console.print(f"\n[bold red]Error:[/] {message}")
            raise CommandError("AI sandbox backend is not available on this host.")

    @staticmethod
    def _ensure_cli_installed(final_cli: str) -> None:
        """Hard-fail with a helpful message if the chosen CLI isn't on PATH."""
        if shutil.which(final_cli):
            return
        install_hints = {
            "claude": "  npm install -g @anthropic-ai/claude-code",
            "agy": "  curl -fsSL https://antigravity.google/cli/install.sh | bash",
            "copilot": "  gh extension install github/gh-copilot",
            "opencode-cli": "  npm install -g opencode-cli",
        }
        hint = install_hints.get(final_cli, "")
        console.print(
            f"\n[bold red]Error:[/] AI CLI '{final_cli}' is not installed (not found in PATH).\n"
            + (f"Install it with:\n{hint}\n" if hint else "")
        )
        raise CommandError(f"AI CLI '{final_cli}' is not installed.")

    def _resolve_cli(self) -> str:
        """Return the AI CLI to run, asking for a favorite the first time around."""
        all_clis = ["claude", "agy", "copilot", "opencode-cli"]
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
                if self.args.headless:
                    favorite_cli = available[0]
                else:
                    favorite_cli = self.console.select(
                        "Which AI CLI tool do you want to use as your favorite?",
                        choices=[(c, c) for c in available],
                    )
                self.config.ai.favorite_cli = favorite_cli
                logger.info(f"Setting '{favorite_cli}' as your favorite AI CLI.")

        if (
            chosen_cli
            and favorite_cli
            and chosen_cli != favorite_cli
            and not self.args.headless
            and self.console.confirm(
                f"Do you want to set '{chosen_cli}' as your new favorite AI CLI?",
                default=False,
            )
        ):
            self.config.ai.favorite_cli = chosen_cli
            favorite_cli = chosen_cli

        return chosen_cli or favorite_cli or "claude"

    def _resolve_model(self, final_cli: str) -> str:
        """Return the model to use for the given CLI, asking for a favorite the first time around."""
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

        if (
            chosen_model
            and favorite_model
            and chosen_model != favorite_model
            and not self.args.headless
            and self.console.confirm(
                f"Do you want to set '{chosen_model}' as your new favorite model for {final_cli}?",
                default=False,
            )
        ):
            self.config.ai.set_favorite_model(final_cli, chosen_model)
            favorite_model = chosen_model

        return chosen_model or favorite_model or "auto"

    def get_ai_agent(self) -> AgentCLI:
        """Initialize and return an AgentCLI instance based on command arguments.

        Handles favorite CLI selection and CLI-specific model favorites.
        """
        self._ensure_sandbox_supported()

        if self.args.headless:
            self.console.bypass_prompt = True

        final_cli = self._resolve_cli()
        self._ensure_cli_installed(final_cli)
        final_model = self._resolve_model(final_cli)

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
        except Exception:  # noqa: BLE001 - any query failure means we cannot tell, assume no demo data
            return False

    def _database_is_neutralized(self, database_obj) -> bool:
        """Return True if the database has been neutralized."""
        try:
            result = database_obj.query("SELECT value FROM ir_config_parameter WHERE key = 'database.is_neutralized'")
            return bool(result and result[0][0] == "true")
        except Exception:  # noqa: BLE001 - any query failure means we cannot tell, assume not neutralized
            return False

    def _ensure_database_safety(self, database_name: str | None) -> bool:
        """Check if the database contains customer data and ask for confirmation if it does.

        Returns True if the data should be cloned into the sandbox, False otherwise.
        """
        if not database_name:
            return False

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
                if self.odev.config.repositories.is_pull_needed(version):
                    self.odev.run_command("pull", "-V", version)
            except Exception as e:  # noqa: BLE001 - a broken version must not abort the other ones
                logger.warning(f"Could not prepare Odoo {version} environment: {e}")
            available[version] = worktree_path.exists()
        return available

    def _get_sandbox_dirs(self, database_name: str | None = None, cwd: Path | None = None) -> list[str]:
        """Return the directories of the sandbox, resolving them once per run.

        Cached because a command asks for them more than once - where to write the
        artifacts of a run, then where to sandbox the agent - and resolving the same
        answer twice announces it twice: a run picking the playground said so on two
        consecutive lines, which reads like it happened twice.

        The repository to work in is read off the command rather than passed here: the
        two callers of a run - the one placing its artifacts and the one sandboxing the
        agent - have to agree, and an argument only one of them passes makes them
        disagree silently, by missing the cache.
        """
        cache = self.__dict__.setdefault("_sandbox_dirs_cache", {})
        key = (database_name, str(cwd) if cwd is not None else None, self.sandbox_repository)

        if key not in cache:
            cache[key] = self._resolve_sandbox_dirs(database_name, cwd)

        return cache[key]

    def _resolve_sandbox_dirs(self, database_name: str | None = None, cwd: Path | None = None) -> list[str]:
        """Work out the list of directories to include in the sandbox.

        The first directory in the list is the working directory of the agent, in
        decreasing order of how much it says about the task:

        1. the addons paths of a local database, which are the checkouts it runs on;
        2. :attr:`sandbox_repository`, the checkout of the code the run is about, which
           a command resolved for itself - the only code a hosted database can offer,
           running nowhere local;
        3. the directory the command was called from, when it is inside a repository;
        4. the playground, which exposes no personal folder to the agent.
        """
        if database_name:
            db = LocalDatabase(database_name)
            if db.exists and db.process:
                # Try to get the repository path linked to the database
                # Using additional_addons_paths is generally safer as it's what Odoo uses
                addons_paths = db.process.additional_addons_paths
                if addons_paths:
                    return [str(p.resolve()) for p in addons_paths]

        if self.sandbox_repository is not None:
            logger.info(f"Using the repository as working directory: {self.sandbox_repository}")
            return [str(self.sandbox_repository)]

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
            except OSError as e:
                logger.debug(f"Could not walk up from {target_dir} looking for a git repository: {e}")

            if not is_git:
                playground = self.odev.home_path / "playground"
                playground.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using AI playground sandbox: {playground}")
                return [str(playground)]

        return [str(target_dir)]

    @staticmethod
    def _run_skills_cli(*arguments: str) -> subprocess.CompletedProcess | None:
        """Run the skills CLI through npx, returning None if it could not be run.

        Its output is captured, which means anything it asks is asked of a terminal that
        cannot show the question: a clone wanting an SSH passphrase or a host key
        confirmed would read from a stdin nobody is watching and never come back, with
        the run stopped on "Loading missing skill(s)..." and no way to tell why. Closed
        stdin turns that into a refusal, and the timeout into a warning - a skill that
        cannot be fetched degrades the run, it does not end it.
        """
        npx = shutil.which("npx")
        if not npx:
            logger.debug("Could not find 'npx' in PATH; skipping the skills check.")
            return None
        try:
            return subprocess.run(  # noqa: S603 - the arguments are built here, never user input
                [npx, "-y", "skills", *arguments],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=SKILLS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"The skills CLI did not answer within {SKILLS_TIMEOUT}s "
                f"(skills {' '.join(arguments)}); continuing without it."
            )
            return None
        except OSError as e:
            logger.debug(f"Could not run the skills CLI: {e}")
            return None

    def _get_loaded_skills(self) -> list[str]:
        """Check loaded skills using npx skills list -g --json."""
        result = self._run_skills_cli("list", "-g", "--json")
        if result is None or result.returncode != 0:
            return []
        try:
            data = json.loads(result.stdout)
        except ValueError as e:
            logger.debug(f"Could not parse the skills listing: {e}")
            return []
        return [s["name"] for s in data if "name" in s]

    def _skills_package(self) -> str:
        """Return the skills repository to install from, on odev's release channel.

        odev tracks a branch - ``main`` or ``beta`` - and checks its plugins out on it;
        the skills those plugins point at live in the same two branches of their own
        repository, and a beta plugin asking a main skill for the method it works by is
        the sort of mismatch that shows up as an agent doing something no code in the
        checkout can explain. Anything else in ``update.release`` names a branch of odev
        rather than of the skills repository, so the default branch is asked for instead.
        """
        release = self.config.update.release
        return f"{SKILLS_REPO}#{release}" if release in ("main", "beta") else SKILLS_REPO

    def _install_skills(self, skills: list[str]) -> list[str]:
        """Install skills globally from the PS skills repo, return those still missing."""
        package = self._skills_package()
        logger.info(f"Loading missing skill(s) from {package}: {', '.join(skills)}...")
        # --skill, singular: --skills is not an option the CLI knows, and an unknown
        # option is ignored rather than refused, which quietly installed every skill of
        # the repository on every call instead of the one that was missing.
        result = self._run_skills_cli("add", package, "--skill", ",".join(skills), "-g")
        if result is None:
            return skills

        # The installer exits 0 even when it fails for individual agent targets
        # (e.g. agents that do not support global installs), so the only reliable
        # check is to ask for the list again.
        still_missing = [s for s in skills if s not in self._get_loaded_skills()]
        if still_missing:
            logger.debug(f"skills add output:\n{result.stdout or result.stderr}")
        return still_missing

    def _mirror_skills(self, skills: list[str], skills_dir: Path) -> list[str]:
        """Copy skills from the shared store into an agent-specific directory.

        The skills CLI only maintains ~/.agents/skills and symlinks it into
        ~/.claude/skills; agents reading from their own directory see nothing.
        Files are copied rather than symlinked because that is what the CLI
        itself does for those agents, and agy does not follow symlinks.
        """
        failed = []
        for skill in skills:
            source = SKILLS_STORE / skill
            target = skills_dir / skill
            if not source.is_dir():
                failed.append(skill)
                continue
            try:
                if target.is_dir() and _newest_mtime(target) >= _newest_mtime(source):
                    continue
                skills_dir.mkdir(parents=True, exist_ok=True)
                # Overwrite in place rather than replacing the directory, so any
                # file the user added next to the skill survives the refresh.
                shutil.copytree(source, target, dirs_exist_ok=True)
                logger.debug(f"Mirrored the {skill!r} skill into {skills_dir}.")
            except (OSError, shutil.Error) as e:
                logger.debug(f"Could not mirror the {skill!r} skill into {skills_dir}: {e}")
                failed.append(skill)
        return failed

    def _ensure_skills(self, required: list[str], handler=None) -> None:
        """Make sure the given skills are loaded, installing the missing ones."""
        disabled = self.config.skills.disabled
        wanted = [s for s in required if s not in disabled]
        if not wanted:
            return

        missing = [s for s in wanted if s not in self._get_loaded_skills()]
        still_missing = self._install_skills(missing) if missing else []

        # Agents the skills CLI does not install to need the files copied over.
        skills_dir = handler.get_global_skills_dir() if handler else None
        if skills_dir:
            still_missing += self._mirror_skills([s for s in wanted if s not in still_missing], skills_dir)

        if still_missing:
            logger.warning(
                f"The following skill(s) are missing: {', '.join(still_missing)}. "
                "For a better experience, you can load them by running:\n"
                f"npx -y skills add {self._skills_package()} --skill {','.join(still_missing)} -g"
            )
        elif missing:
            logger.info(f"Loaded skill(s): {', '.join(missing)}")

    def run_ai_agent(
        self,
        prompt: str,
        database: str | None = None,
        ephemeral_pg: bool = True,
        extra_ro_bind_dirs: list[str] | None = None,
        mcp_servers: dict | None = None,
    ) -> bool:
        """Helper to run the AI agent with common Odoo-related sandbox paths."""
        sandbox_dirs = self._get_sandbox_dirs(database)

        if database:
            should_clone = self._ensure_database_safety(database)
            if not should_clone:
                database = None

        agent = self.get_ai_agent()

        self._ensure_skills(self.required_skills, handler=agent.handler)

        return agent.run(
            prompt,
            sandbox_dirs=sandbox_dirs,
            extra_bind_dirs=[str(d) for d in self.args.dirs] or None,
            extra_ro_bind_dirs=extra_ro_bind_dirs,
            mcp_servers=mcp_servers,
            database=database,
            resume=self.args.resume,
            ephemeral_pg=ephemeral_pg,
        )
