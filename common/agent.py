import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin

from .handlers import get_agent_handler
from .postgres import PostgresSandbox
from .sandbox import ExecutionSpec, get_sandbox


logger = logging.getLogger(__name__)


class AgentCLI(OdevFrameworkMixin):
    """An execution wrapper for CLI AI agents (claude, agy, copilot).

    Composes a platform-specific sandbox backend (bwrap on Linux,
    sandbox-exec/Seatbelt on macOS) and an ephemeral PostgreSQL sandbox.
    """

    def __init__(
        self,
        cli: str,
        model: str = "auto",
        yolo: bool = False,
        headless: bool = False,
    ):
        super().__init__()
        host_home = Path.home().resolve()

        from odev.common.odev import Odev

        self.cli = cli
        self.model = model
        self.headless = headless
        self.yolo = yolo or headless
        self.handler = get_agent_handler(cli, host_home, Odev())
        self.sandbox = get_sandbox(
            cli=cli,
            handler=self.handler,
            model=model,
            yolo=self.yolo,
            headless=headless,
        )

    def _get_agent_setup(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
        host_home: Path,
    ) -> tuple[list[str], list[Path], list[Path]]:
        """Determine agent-specific command, directories, and files to mount."""
        agent_dirs = [
            host_home / ".cache",
            host_home / ".config" / "rtk",
            host_home / ".claude",
            host_home / ".agents",
            host_home / ".antigravity",
            # Skills are symlinked into the agents' config dirs from this clone
            self.odev.home_path / "skills",
        ]
        agent_files = [
            host_home / ".gitconfig",
        ]

        for d in self.handler.get_config_dirs():
            agent_dirs.append(host_home / d)
        for f in self.handler.get_config_files():
            agent_files.append(host_home / f)

        # Node.js managers (NVM, n, asdf)
        for node_dir in [".nvm", ".n", ".asdf"]:
            p = host_home / node_dir
            if p.exists():
                agent_dirs.append(p)

        # Include .env if it exists in the current directory
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            agent_files.append(env_file)

        agent_cmd = self.handler.get_command(
            prompt=prompt,
            resume=resume,
            all_candidate_paths=all_candidate_paths,
            model=self.model,
            headless=self.headless,
            yolo=self.yolo,
        )

        # Deduplicate directories and files while preserving order
        unique_dirs = list(dict.fromkeys(agent_dirs))
        unique_files = list(dict.fromkeys(agent_files))

        return agent_cmd, unique_dirs, unique_files

    def _build_env(
        self,
        host_home: Path,
        sandbox_path: str,
        database: str | None,
    ) -> dict[str, str]:
        """Build the platform-agnostic env dict to inject into the sandbox."""
        env: dict[str, str] = {
            "HOME": str(host_home),
            "USER": host_home.name,
            "SHELL": "/bin/bash",
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "PYTHONPATH": str(self.odev.path),
            "PATH": sandbox_path,
            "ODEV_NO_SSH_AGENT": "1",
            "ODEV_SKIP_GIT_UPDATE": "1",
            "AI_SANDBOX": "1",
        }
        if database:
            env["PGDATABASE"] = database
        return env

    def _build_sandbox_path(self, active_venv_path: Path | None, host_home: Path) -> str:
        """Compose the PATH used inside the sandbox so the agent finds tools."""
        items = [str(Path(sys.prefix) / "bin")]
        if active_venv_path:
            items.append(str(active_venv_path / "bin"))

        # Make 'node' inside the sandbox match 'node' on the host (NVM-aware).
        host_node = shutil.which("node")
        if host_node:
            node_bin_dir = str(Path(host_node).parent)
            if node_bin_dir not in items:
                items.append(node_bin_dir)

        items.extend(
            [
                str(host_home / ".npm-global" / "bin"),
                str(host_home / ".local" / "bin"),
            ]
        )
        if sys.platform == "darwin":
            items.extend(["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin", "/usr/local/sbin"])
        items.extend(["/usr/local/bin", "/usr/bin", "/bin"])
        # Deduplicate while preserving order
        return ":".join(dict.fromkeys(items))

    def run(
        self,
        prompt: str,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None = None,
        database: str | None = None,
        db_user: str | None = None,
        version: str | None = None,
        resume: str | None = None,
        ephemeral_pg: bool = True,
        cwd: str | None = None,
    ) -> bool:
        """Run the AI agent within the platform-appropriate sandbox."""
        # Reap any leftover ephemeral postgres clusters / sandbox tmp dirs
        # from previous Ctrl+C'd or crashed runs before we start fresh ones.
        PostgresSandbox.cleanup_orphans()

        if resume == "latest":
            if self.cli == "agy":
                # Agy CLI natively supports --resume latest inside the sandbox
                pass
            else:
                latest_id = self.get_latest_session_id()
                if latest_id:
                    logger.info(f"Resuming latest session: {latest_id}")
                    resume = latest_id
                else:
                    logger.warning("No previous session found to resume.")
                    resume = None

        host_home = Path.home().resolve()
        playground = Path(tempfile.mkdtemp(prefix=f"odev-ai-{self.cli}-"))
        sandbox_tmp = Path(tempfile.mkdtemp(prefix=f"odev-ai-tmp-{self.cli}-"))
        proxy_dir = Path(tempfile.mkdtemp(prefix="odev-ai-pg-"))
        pg_data_dir = Path(tempfile.mkdtemp(prefix="odev-ai-pgdata-"))

        sandbox_data = self.sandbox.prepare_sandbox_config(
            sandbox_dirs=sandbox_dirs,
            extra_bind_dirs=extra_bind_dirs,
            database=database,
            version=version,
        )
        final_binds = sandbox_data["binds"]
        active_venv_path = sandbox_data["active_venv_path"]

        if not cwd:
            # Prioritize the first directory in sandbox_dirs (the main project path)
            if sandbox_dirs:
                main_path = Path(sandbox_dirs[0]).resolve()
                primary_bind = next((b for b in final_binds if b[0] == main_path), None)
                cwd = str(primary_bind[1]) if primary_bind else str(main_path)
            else:
                # Fallback to the first primary bind or home
                primary_bind = next((b for b in final_binds if b[3]), None)
                cwd = str(primary_bind[1]) if primary_bind else str(host_home)

        # Candidate paths for trustedDirectories and --add-dir inclusion
        all_candidate_paths = [str(dst) for src, dst, _, primary in final_binds if src != host_home and primary]

        agent_cmd, agent_dirs, agent_files = self._get_agent_setup(prompt, resume, all_candidate_paths, host_home)

        if not agent_cmd:
            return False

        if database:
            db_info = (
                f"(Environment: You have been granted access to a private, isolated PostgreSQL database named '{database}'. "
                f"If the host database '{database}' exists, it has been cloned into this ephemeral cluster. "
                "Otherwise it is an empty database. You can safely modify it as it "
                "does not affect the live host data. Use 'psql' to work directly with it.)\n\n"
            )
        else:
            db_info = (
                "(Environment: You have been granted access to a private, isolated filesystem sandbox. "
                "No database access has been provided for this session.)\n\n"
            )
        prompt = db_info + prompt

        sandbox_path = self._build_sandbox_path(active_venv_path, host_home)
        env = self._build_env(host_home, sandbox_path, database)
        secrets = self._setup_github_token()

        # Set up ephemeral PG (or pass-through to host PG) before the agent runs
        pg_sandbox = PostgresSandbox(headless=self.headless)
        pg_process = pg_sandbox.setup(
            database=database,
            proxy_dir=proxy_dir,
            pg_data_dir=pg_data_dir,
            ephemeral=ephemeral_pg,
        )

        spec = ExecutionSpec(
            agent_cmd=agent_cmd,
            final_binds=final_binds,
            agent_dirs=agent_dirs,
            agent_files=agent_files,
            env=env,
            secrets=secrets,
            cwd=cwd,
            playground=playground,
            sandbox_tmp=sandbox_tmp,
            proxy_dir=proxy_dir,
            pg_data_dir=pg_data_dir,
            database=database,
            db_user=db_user,
            pg_process=pg_process,
            active_venv_path=active_venv_path,
            odoo_filestore=host_home / ".local" / "share" / "Odoo",
            primary_dirs=[Path(d) for d in sandbox_dirs],
        )

        return self.sandbox.execute(spec)

    def get_latest_session_id(self) -> str | None:
        """Return the ID of the most recent session for this agent CLI."""
        try:
            home = Path.home()
            if self.cli == "agy":
                sessions_file = home / ".antigravity" / "sessions.json"
            elif self.cli in ("claude", "opencode-cli"):
                sessions_file = home / ".claude" / "sessions.json"
            else:
                return None

            if sessions_file and sessions_file.exists():
                data = json.loads(sessions_file.read_text())
                sessions = data.get("sessions", [])
                if sessions:
                    return sessions[-1].get("id")
        except Exception as e:
            logger.debug(f"Could not read latest session id for {self.cli!r}: {e}")
        return None

    def _setup_github_token(self) -> list[tuple[str, str]]:
        """Retrieve GITHUB_TOKEN for PR creation and other GitHub operations."""
        try:
            from odev.common.connectors.git import GITHUB_DOMAIN

            token = self.store.secrets.get(
                GITHUB_DOMAIN,
                scope="api",
                fields=["password"],
                ask_missing=True,
            ).password

            if token:
                return [("GITHUB_TOKEN", token), ("GH_TOKEN", token)]
        except Exception as e:
            logger.debug(f"Could not retrieve GitHub secret: {e}")

        return []
