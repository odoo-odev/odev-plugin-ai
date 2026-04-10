from pathlib import Path

from odev.common import args, bash, progress
from odev.common.databases import DummyDatabase
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin
from odev.plugins.odev_plugin_project.commands.pre_commit import PreCommit as BasePreCommit


logger = logging.getLogger(__name__)


class PreCommit(BasePreCommit, AICommandMixin):
    """AI-enhanced pre-commit command."""

    ai = args.Flag(
        aliases=["--ai"],
        description="Run pre-commit checks and use AI to fix issues if it fails.",
        default=False,
    )

    run_checks = args.Flag(
        aliases=["--run"],
        description="Run pre-commit checks.",
        default=False,
    )

    def infer_database_instance(self):
        if not self.database_name and not self.args.repository and (Path.cwd() / ".git").exists():
            self.args.repository = str(Path.cwd().resolve())
            return DummyDatabase()
        return super().infer_database_instance()

    def run(self) -> None:
        super().run()

        if not self.args.ai and not self.args.run:
            return

        result = self._run_checks()

        if self._repository.is_dirty:
            if self._repository.is_protected_branch:
                logger.warning(
                    f"Changes detected in protected branch {self._repository.branch!r}, won't auto-commit fixes."
                )
            elif self.console.confirm("Do you want to commit auto fixed files?", default=True):
                self._commit_auto_fixes()

        if not self.args.ai:
            return

        if result and result.returncode == 0:
            logger.info("Pre-commit passed!")
            return

        logger.info("Launching AI agent to fix pre-commit issues...")

        prompt = """
I ran `pre-commit run --all-files` and it failed.

Please:
1. Run `pre-commit run --all-files` to see the failures.
2. Analyze and fix the reported issues.
3. Run `pre-commit run --all-files` again to verify your fixes.
4. Once everything passes, commit the fixes with a meaningful message (e.g. "[FIX] module: description").
5. Provide a summary of your changes.
"""
        self.run_ai_agent(prompt, database=self.database_name)

    def _run_checks(self) -> bash.CompletedProcess | None:
        """Run pre-commit checks on all files and display output in real-time."""
        logger.info(f"Running pre-commit checks in {self._repository.path}...")

        output = []
        try:
            for line in bash.stream(f"cd {self._repository.path} && pre-commit run --all-files"):
                print(line)
                output.append(line)

            return bash.CompletedProcess(0, "pre-commit", "\n".join(output).encode(), b"")
        except bash.CalledProcessError as error:
            logger.warning("Pre-commit checks failed.")
            return bash.CompletedProcess(error.returncode, error.cmd, "\n".join(output).encode(), b"")

    def _commit_auto_fixes(self) -> None:
        """Commit changes made by pre-commit auto-fixes."""
        try:
            repo = self._repository.repository
            if not repo:
                return

            with progress.spinner("Committing auto-fixed files"):
                repo.git.add(".")
                repo.git.commit("-m", "[REF] pre-commit: auto fix")
                logger.info(f"Auto-fixed files committed with [REF] prefix in branch {repo.active_branch.name!r}")
        except Exception as error:
            logger.error(f"Failed to commit auto-fixed files: {error}")
