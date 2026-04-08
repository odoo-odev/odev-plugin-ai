from pathlib import Path

from odev.common import args, bash
from odev.common.databases import DummyDatabase
from odev.common.logging import logging

try:
    from odev.plugins.odev_plugin_project.commands.pre_commit import (
        PreCommit as BasePreCommit,
    )
except ImportError:
    from odev.common.commands import DatabaseOrRepositoryCommand as BasePreCommit

from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin


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
        # Fallback to current repository if neither database nor repository is specified
        if (
            not self.database_name
            and not self.args.repository
            and (Path.cwd() / ".git").exists()
        ):
            self.args.repository = str(Path.cwd().resolve())
            return DummyDatabase()
        return super().infer_database_instance()

    def run(self):
        # Setup the repository first using the base class logic
        super().run()

        if not self.args.ai:
            return

        # Run pre-commit checks to find issues
        import tempfile

        repo_path = Path(self._repository.path)

        # We create a temporary file INSIDE the repo to ensure the sandbox can read it
        with tempfile.NamedTemporaryFile(
            mode="w+", dir=repo_path, suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            try:
                result = self._run_checks()

                if result and result.returncode == 0:
                    logger.info("Pre-commit passed!")
                    return

                # Pre-commit failed
                tmp.write(result.stdout.decode() if result else "No output captured.")
                tmp.flush()

                logger.info("Launching AI agent to fix pre-commit issues...")

                prompt = f"""
I ran `pre-commit run --all-files` and it failed. 
The full output has been saved to `{tmp_path.name}`.

Please:
1. Read that file to understand the failures.
2. Analyze and fix the reported issues. 
3. Run `pre-commit run --all-files` again to verify your fixes.
4. Once everything passes, commit the fixes with a meaningful message (e.g. "[FIX] module: description") and THEN delete the `{tmp_path.name}` file.
5. Provide a summary of your changes.
"""
                self.run_ai_agent(prompt, database=self.database_name)
            finally:
                # Ensure the file is deleted on the host even if the agent fails
                if tmp_path.exists():
                    tmp_path.unlink()

    def _run_checks(self) -> bash.CompletedProcess | None:
        """Run pre-commit checks on all files and display output in real-time."""
        logger.info(f"Running pre-commit checks in {self._repository.path}...")

        output = []
        try:
            # We use stream to have real-time visibility for the user
            for line in bash.stream(
                f"cd {self._repository.path} && pre-commit run --all-files"
            ):
                print(line)
                output.append(line)

            return bash.CompletedProcess(
                0, "pre-commit", "\n".join(output).encode(), b""
            )
        except bash.CalledProcessError as error:
            logger.warning("Pre-commit checks failed.")
            return bash.CompletedProcess(
                error.returncode, error.cmd, "\n".join(output).encode(), b""
            )
