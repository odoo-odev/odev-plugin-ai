"""Common mixins for AI-related commands."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from odev.common.args import Namespace
    from odev.common.config import Config
    from odev.common.console import Console
    from odev.common.odev import Odev

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from odev.common import args
from odev.common.console import console
from odev.common.errors import CommandError
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.agent import AgentCLI
from odev.plugins.odev_plugin_ai.common.sandbox import get_sandbox_class


logger = logging.getLogger(__name__)

<<<<<<< Updated upstream
=======
SKILLS_REPO = "odoo-ps/ps-ai-skills"

GUIDELINES_SKILL = "odoo_coding_guidelines"
"""Skill carrying how Odoo code is written: module layout, per-language conventions, and
what a change is allowed to touch.

Needed by every command here, not only the ones that write code: each of them puts an
agent in front of a client's checkout, and the rule that a dev reformats nothing it was
not asked to reformat - and runs pre-commit only where the repository configures it -
holds whether the agent is scaffolding a module, fixing a test or reading the code to
estimate it."""

# Shared store the skills CLI installs into, whatever the agent.
SKILLS_STORE = Path.home() / ".agents" / "skills"

SKILLS_TIMEOUT = 120
"""How long the skills CLI is given to answer, in seconds.

Generous: installing clones the skills repo, which on a cold run is a network round trip.
It is a ceiling on a hang, not a budget for a normal call."""


def _newest_mtime(directory: Path) -> float:
    """Return the most recent mtime found anywhere under the given directory."""
    return max((p.stat().st_mtime for p in directory.rglob("*")), default=0.0)

>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
    sandbox_repository: Path | None = None
    """A checkout of the code the run is about, to work in ahead of anywhere else.

    Set by the command that knows how to find one - which is not this plugin: what
    links a task, a database and a repository together lives where tasks and hosted
    databases do. Here it is a path, taken on trust and bound into the sandbox.
    """

    sandbox_scope: str | None = None
    """What this run is about, to give it a playground of its own.

    An analysis is written from the artifacts of one task - its diagrams, the images of
    its description - and the playground is where they land when the run has no
    repository to work in. Shared between runs, the second analysis overwrites the
    diagrams of the first and leaves behind whatever it has no file of its own to
    overwrite: an ``embedded-image-4.png`` of a task that had four screenshots, sitting
    in the working directory of one that has two, is a picture the agent can open and
    has no way to tell is not its own.

    Only the playground is scoped. A run working in a repository stays where the code
    is - that is the point of resolving one - and a task-named directory there would
    only leave untracked files in someone's checkout.

    Set by the command that knows what its run is about, and read off it rather than
    passed to :meth:`_get_sandbox_dirs`: the two callers of a run - the one placing its
    artifacts and the one sandboxing the agent - have to agree on the answer.
    """

    required_skills: ClassVar[list[str]] = ["odev", GUIDELINES_SKILL]
    """Skills the agent needs to run this command, installed before it starts.

    Declared by the command rather than decided from its name: a command knows what
    method its prompt leans on, and a prompt that points at a skill the agent was never
    given is a prompt missing the half that was moved out of it. A command whose skills
    depend on the run appends to this before calling :meth:`run_ai_agent`.
    """

>>>>>>> Stashed changes
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

    edit = args.Flag(
        # -E rather than the -e this reads like: `scaffold` spends -e on --no-excalidraw,
        # and a second command registering the same letter is a parser that refuses to
        # build - the command stops existing rather than the flag being ignored.
        aliases=["-E", "--edit"],
        description="Open the prompt in $EDITOR before sending it; save it empty to abort the run.",
        default=False,
    )

    resume = args.String(
        aliases=["--resume"],
        description="Resume a previous AI session, by id or 'latest'. Defaults to the latest session, "
        "which is the last one held in the directory the run works in.",
        default=None,
        # A bare --resume means the latest session: it is the only one a developer can
        # name without going to look it up, and asking to resume without saying which
        # session has no other reading.
        nargs="?",
        const="latest",
    )

    dirs = args.List(
        aliases=["-d", "--dirs"],
        description="Comma-separated list of extra directories to include in the sandbox (read-write).",
        default=[],
    )

    version = args.String(
        aliases=["-V", "--version"],
        description="""Launch the AI agent in the worktree directory of a specific Odoo version
        (e.g. 17.0, saas-17.2, master) instead of a database's linked repository. The worktree is
        created and updated automatically if needed. Takes precedence over a database's repository
        path if both a database and a version are given.""",
        default=None,
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

    def get_ai_agent(self) -> AgentCLI:
        """Initialize and return an AgentCLI instance based on command arguments.

        Handles favorite CLI selection and CLI-specific model favorites.
        """
        self._ensure_sandbox_supported()

        all_clis = ["claude", "agy", "copilot", "opencode-cli"]
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
        self._ensure_cli_installed(final_cli)

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
            edit=self.args.edit,
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
                if self.odev.config.repositories.is_pull_needed(version):
                    self.odev.run_command("pull", "-V", version)
            except Exception as e:
                logger.warning(f"Could not prepare Odoo {version} environment: {e}")
            available[version] = worktree_path.exists()
        return available

    def _get_sandbox_dirs(
        self, database_name: str | None = None, version: str | None = None, cwd: Path | None = None
    ) -> list[str]:
        """Return the list of directories to include in the sandbox.

        The first directory in the list will be used as the working directory.
        If a version is provided, we use the Odoo worktree for that version.
        Otherwise, if a database is provided, we try to use its repository path
        as the working directory.
        """
<<<<<<< Updated upstream
        if version:
            from odev.common.version import OdooVersion
=======
        cache = self.__dict__.setdefault("_sandbox_dirs_cache", {})
        key = (database_name, str(cwd) if cwd is not None else None, self.sandbox_repository, self.sandbox_scope)
>>>>>>> Stashed changes

            normalized_version = str(OdooVersion(version))
            available = self._prepare_odoo_environment([normalized_version])
            if available.get(normalized_version):
                return [str((self.odev.worktrees_path / normalized_version).resolve())]
            logger.warning(f"Could not prepare a worktree for Odoo {normalized_version}, falling back.")

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

                if self.sandbox_scope:
                    playground /= self.sandbox_scope

                playground.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using AI playground sandbox: {playground}")
                return [str(playground)]

        return [str(target_dir)]

    #: Maps odev's `--cli` values to the agent names the `skills` npm package expects
    #: for its `-a/--agent` flag.
    _SKILLS_PACKAGE_AGENT_NAMES = {
        "claude": "claude-code",
        "agy": "antigravity",
        "copilot": "github-copilot",
        "opencode-cli": "opencode",
    }

    def _get_loaded_skills(self) -> list[str]:
        """Check loaded skills using npx skills list -g --json."""
        try:
            import json

            result = subprocess.run(
                ["npx", "-y", "skills", "list", "-g", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return [s["name"] for s in data if "name" in s]
        except Exception:
            pass
        return []

    def _ensure_skills(self, agent: AgentCLI, required_skills: list[str]) -> None:
        """Warn about any of ``required_skills`` not yet installed globally for ``agent``.

        Also gives the agent's handler a chance to reconcile where `skills` installs
        with where the agent actually looks (see ``BaseAgentHandler.ensure_skills_discoverable``)
        so a suggested install actually gets discovered.
        """
        agent.handler.ensure_skills_discoverable()

<<<<<<< Updated upstream
        loaded_skills = self._get_loaded_skills()
        missing_skills = [s for s in required_skills if s not in loaded_skills]
        if not missing_skills:
            return

        skills_agent = self._SKILLS_PACKAGE_AGENT_NAMES.get(agent.cli, agent.cli)
        logger.warning(
            f"The following skill(s) are missing: {', '.join(missing_skills)}. "
            "For a better experience, you can load them by running:\n"
            f"npx -y skills add odoo-ps/ps-ai-skills --skills {','.join(missing_skills)} -g -a {skills_agent}"
        )
=======
    def _refresh_skills(self, skills: list[str]) -> None:
        """Fetch the given skills again, overwriting the copies already installed.

        Installing only what is missing leaves a store that never changes: a skill
        installed once stayed at the revision it was installed at, so a rule added to it
        reached the agents that had never loaded it and no one else. The store is the
        agents' only copy - they read ~/.agents/skills, not the checkout - so it has to
        be brought forward on its own.

        Rate-limited by ``skills.interval`` rather than done on every run: a refresh
        clones the skills repository, and paying a network round trip to start every
        agent is how a check like this ends up being turned off. Failures are silent by
        design - the skills already installed still work, and a run is not worth losing
        over a fetch that did not answer.

        Note that this overwrites a skill edited in place: the store is a checkout of the
        repository, not somewhere to keep local changes. Edit the repository and let the
        refresh bring them down.
        """
        if not self.config.skills.is_refresh_needed():
            return

        logger.debug(f"Refreshing the installed skill(s): {', '.join(skills)}...")

        # `update`, not `add`: adding a skill that is already installed is a no-op, which
        # is why the store never moved. Update compares the hash of the upstream skill
        # folder against the one recorded at install time and refetches on a mismatch -
        # so it costs nothing when nothing changed, and overwrites when something did.
        if self._run_skills_cli("update", *skills, "-g", "-y") is None:
            return

        # Recorded on the attempt rather than on a verified result: what the CLI
        # overwrote cannot be told apart from what it left alone, and a store that
        # cannot be refreshed should still not be retried on every command.
        self.config.skills.date = datetime.now()

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

        installed = self._get_loaded_skills()
        missing = [s for s in wanted if s not in installed]
        still_missing = self._install_skills(missing) if missing else []

        # The ones that were already there, which installing would not have touched.
        if already_installed := [s for s in wanted if s in installed]:
            self._refresh_skills(already_installed)

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
>>>>>>> Stashed changes

    def run_ai_agent(
        self,
        prompt: str,
        database: str | None = None,
        version: str | None = None,
        ephemeral_pg: bool = True,
        mcp_servers: dict[str, dict] | None = None,
        extra_ro_bind_dirs: list[str] | None = None,
    ) -> bool:
        """Helper to run the AI agent with common Odoo-related sandbox paths.

        ``extra_ro_bind_dirs`` are mounted readable but not writable, on top of the
        read-write dirs the caller passes through -d/--dirs.
        """
        sandbox_dirs = self._get_sandbox_dirs(database, version or self.args.version)

        if database:
            should_clone = self._ensure_database_safety(database)
            if not should_clone:
                database = None

        agent = self.get_ai_agent()

        required_skills = ["odev"]
        if self._name == "test":
            required_skills.append("test_skill")
        self._ensure_skills(agent, required_skills)

        return agent.run(
            prompt,
            sandbox_dirs=sandbox_dirs,
            extra_bind_dirs=[str(d) for d in self.args.dirs] or None,
            extra_ro_bind_dirs=extra_ro_bind_dirs or None,
            database=database,
            resume=self.args.resume,
            ephemeral_pg=ephemeral_pg,
            mcp_servers=mcp_servers,
        )
