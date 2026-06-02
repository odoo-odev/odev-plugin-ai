import json
import os
import sys
import tempfile
from pathlib import Path

from odev.common.logging import logging

from .bwrap import BwrapSandbox
from .handlers import get_agent_handler
from .postgres import PostgresSandbox


logger = logging.getLogger(__name__)


class AgentCLI(BwrapSandbox):
    """An execution wrapper for CLI AI agents (claude, gemini, copilot)."""

    def __init__(
        self,
        cli: str,
        model: str = "auto",
        yolo: bool = False,
        headless: bool = False,
    ):
        host_home = Path.home().resolve()
        # Initialize the strategy handler for the specific agent
        from odev.common.odev import Odev

        self.handler = get_agent_handler(cli, host_home, Odev())

        super().__init__(
            cli=cli,
            handler=self.handler,
            model=model,
            yolo=yolo,
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
        ]
        agent_files = [
            host_home / ".gitconfig",
        ]

        # Add agent-specific directories/files from the handler
        for d in self.handler.get_config_dirs():
            agent_dirs.append(host_home / d)
        for f in self.handler.get_config_files():
            agent_files.append(host_home / f)

        # Node.js managers (NVM, n, asdf) and npm
        for node_dir in [".nvm", ".n", ".asdf", ".npm"]:
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
        """Run the AI agent within a bwrap sandbox."""
        if resume == "latest":
            if self.cli == "gemini":
                # Gemini CLI natively supports --resume latest inside the sandbox
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

        sandbox_data = self._prepare_sandbox_config(
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
        all_candidate_paths = [str(dst) for src, dst, _, _ in final_binds if src != host_home]

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

        sandbox_path_items = [str(Path(sys.prefix) / "bin")]
        if active_venv_path:
            sandbox_path_items.append(str(active_venv_path / "bin"))

        # Add the directory containing the current host node to the sandbox path.
        # This ensures that if the user is using NVM, 'node' inside the sandbox
        # matches 'node' outside.
        import shutil

        host_node = shutil.which("node")
        if host_node:
            node_bin_dir = str(Path(host_node).parent)
            if node_bin_dir not in sandbox_path_items:
                sandbox_path_items.append(node_bin_dir)

        sandbox_path_items.extend(
            [
                str(host_home / ".npm-global" / "bin"),
                str(host_home / ".local" / "bin"),
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            ]
        )
        sandbox_path = ":".join(sandbox_path_items)

        odev_path = self.odev.path
        cmd = [
            "bwrap",
            "--dir",
            "/home",
            "--dir",
            str(host_home),
            "--bind",
            str(playground),
            str(host_home),
            "--setenv",
            "HOME",
            str(host_home),
            "--setenv",
            "USER",
            host_home.name,
            "--setenv",
            "XDG_RUNTIME_DIR",
            f"/run/user/{os.getuid()}",
            "--setenv",
            "SHELL",
            "/bin/bash",
            "--setenv",
            "LANG",
            "en_US.UTF-8",
            "--setenv",
            "PYTHONPATH",
            str(odev_path),
            "--setenv",
            "PATH",
            sandbox_path,
            "--setenv",
            "ODEV_NO_SSH_AGENT",
            "1",
            "--setenv",
            "ODEV_SKIP_GIT_UPDATE",
            "1",
            "--setenv",
            "AI_SANDBOX",
            "1",
        ]

        secrets_to_set = self._setup_github_token()
        if database:
            cmd.extend(["--setenv", "PGDATABASE", database])

        top_dirs = {
            "/home",
            "/tmp",
            "/dev",
            "/proc",
            "/sys",
            "/run",
            "/etc",
            "/var",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
        }
        for _, dst, _, _ in final_binds:
            if dst.is_absolute():
                td = f"/{dst.parts[1]}"
                if td not in top_dirs:
                    cmd.extend(["--dir", td])
                    top_dirs.add(td)

        self._prepare_odev_config(playground, host_home)
        self._add_system_binds(cmd, host_home, sandbox_tmp, cwd)

        pg_sandbox = PostgresSandbox(headless=self.headless)
        pg_process = pg_sandbox.setup(
            cmd,
            database,
            proxy_dir,
            pg_data_dir,
            ephemeral=ephemeral_pg,
        )

        self._apply_final_bindings(cmd, agent_dirs, agent_files, final_binds, host_home, playground)
        self._prepare_agent_config(playground, all_candidate_paths, host_home)

        cmd.extend(["--", *agent_cmd])
        return self._execute_sandbox(
            cmd=cmd,
            agent_dirs=agent_dirs,
            agent_files=agent_files,
            final_binds=final_binds,
            database=database,
            db_user=db_user,
            secrets_to_set=secrets_to_set,
            pg_process=pg_process,
            playground=playground,
            sandbox_tmp=sandbox_tmp,
            proxy_dir=proxy_dir,
            pg_data_dir=pg_data_dir,
            active_venv_path=active_venv_path,
            odoo_filestore=host_home / ".local" / "share" / "Odoo",
            primary_dirs=[Path(d) for d in sandbox_dirs],
        )

    def get_latest_session_id(self) -> str | None:
        """Return the ID of the most recent session for this agent CLI."""
        try:
            home = Path.home()
            if self.cli == "gemini":
                sessions_file = home / ".gemini" / "sessions.json"
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
