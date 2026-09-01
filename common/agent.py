import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin

from .editor import PromptEditingAbortedError, edit_prompt
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
        edit: bool = False,
    ):
        super().__init__()
        host_home = Path.home().resolve()

        from odev.common.odev import Odev

        self.cli = cli
        self.model = model
        self.headless = headless
        self.yolo = yolo or headless
        self.edit = edit and not headless

        if edit and headless:
            # Nothing to open an editor on, and nobody to close it: --headless exists to
            # be run without a terminal in front of it.
            logger.warning("Ignoring --edit: there is no terminal to edit the prompt in when running headless.")
        self.handler = get_agent_handler(cli, host_home, Odev())
        self.sandbox = get_sandbox(
            cli=cli,
            handler=self.handler,
            model=model,
            yolo=self.yolo,
            headless=headless,
        )

    def _write_mcp_config(
        self,
        mcp_servers: dict[str, dict],
        playground: Path,
        host_home: Path,
    ) -> str | None:
        """Write the MCP server config and return the path the agent will see.

        The playground is bound over the home directory inside the sandbox, so the file
        is written here but read there — the returned path only resolves in the guest.
        """
        if not self.handler.supports_mcp:
            logger.warning(f"{self.cli} does not support MCP servers, {len(mcp_servers)} server(s) ignored.")
            return None

        config_name = ".odev-mcp-config.json"

        try:
            (playground / config_name).write_text(json.dumps({"mcpServers": mcp_servers}, indent=2))
        except OSError as e:
            logger.warning(f"Could not write the MCP configuration, the agent will run without it: {e}")
            return None

        logger.debug(f"Declared MCP server(s) to {self.cli}: {', '.join(mcp_servers)}")
        return str(host_home / config_name)

    def _get_agent_setup(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
        host_home: Path,
        mcp_config: str | None = None,
        mcp_server_names: tuple[str, ...] = (),
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
            mcp_config=mcp_config,
            mcp_server_names=mcp_server_names,
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

<<<<<<< Updated upstream
    def run(
=======
    def run(  # noqa: PLR0913 - carries the full sandbox invocation context
>>>>>>> Stashed changes
        self,
        prompt: str,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None = None,
        extra_ro_bind_dirs: list[str] | None = None,
        database: str | None = None,
        db_user: str | None = None,
        version: str | None = None,
        resume: str | None = None,
        ephemeral_pg: bool = True,
        cwd: str | None = None,
        mcp_servers: dict[str, dict] | None = None,
    ) -> bool:
<<<<<<< Updated upstream
        """Run the AI agent within the platform-appropriate sandbox.

        ``mcp_servers`` maps a server name to its stdio launch definition, exposing its
        tools to the agent. Each definition carries its own environment: a stdio MCP
        server inherits only an allowlist from the agent, so the sandbox secrets do not
        reach it.

        ``extra_ro_bind_dirs`` are mounted readable but not writable, for source the
        agent is meant to consult rather than edit.
        """
=======
        """Run the AI agent within the platform-appropriate sandbox."""
        # Before the sandbox, the ephemeral cluster and the temporary directories of the
        # run: a prompt the developer decides against is a run that never starts, and
        # nothing it would have to clean up again has been made yet.
        if self.edit and prompt:
            try:
                prompt = edit_prompt(prompt)
            except PromptEditingAbortedError as e:
                logger.info(f"{e} Nothing was run.")
                return False

>>>>>>> Stashed changes
        # Reap any leftover ephemeral postgres clusters / sandbox tmp dirs
        # from previous Ctrl+C'd or crashed runs before we start fresh ones.
        PostgresSandbox.cleanup_orphans()

        host_home = Path.home().resolve()
        playground = Path(tempfile.mkdtemp(prefix=f"odev-ai-{self.cli}-"))
        sandbox_tmp = Path(tempfile.mkdtemp(prefix=f"odev-ai-tmp-{self.cli}-"))
        proxy_dir = Path(tempfile.mkdtemp(prefix="odev-ai-pg-"))
        pg_data_dir = Path(tempfile.mkdtemp(prefix="odev-ai-pgdata-"))

        sandbox_data = self.sandbox.prepare_sandbox_config(
            sandbox_dirs=sandbox_dirs,
            extra_bind_dirs=extra_bind_dirs,
            extra_ro_bind_dirs=extra_ro_bind_dirs,
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

<<<<<<< Updated upstream
        mcp_config = self._write_mcp_config(mcp_servers, playground, host_home) if mcp_servers else None
        agent_cmd, agent_dirs, agent_files = self._get_agent_setup(
            prompt,
            resume,
            all_candidate_paths,
            host_home,
            mcp_config=mcp_config,
            mcp_server_names=tuple(mcp_servers) if mcp_config else (),
        )
=======
        resume = self._resolve_resume(resume, cwd)

        # Only ever prefixed to a prompt there already is: handed to an agent started
        # with nothing to do, the note becomes the first thing it is asked, and a run
        # meant to open a conversation opens with the agent answering its own sandbox
        # description instead of waiting. It used to be assembled after the command that
        # carries the prompt was built, so it reached no agent at all.
        if prompt:
            prompt = self._environment_note(database) + prompt

        agent_cmd, agent_dirs, agent_files = self._get_agent_setup(prompt, resume, all_candidate_paths, host_home)
>>>>>>> Stashed changes

        if not agent_cmd:
            return False

<<<<<<< Updated upstream
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
=======
        agent_cmd.extend(self.handler.get_mcp_config_args(mcp_config_path))
>>>>>>> Stashed changes

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
            mcp_servers=mcp_servers,
        )

        return self.sandbox.execute(spec)

    @staticmethod
    def _environment_note(database: str | None) -> str:
        """Return what the agent has to know about the sandbox it was started in."""
        if database:
            return (
                f"(Environment: You have been granted access to a private, isolated PostgreSQL database "
                f"named '{database}'. If the host database '{database}' exists, it has been cloned into "
                "this ephemeral cluster. Otherwise it is an empty database. You can safely modify it as "
                "it does not affect the live host data. Use 'psql' to work directly with it.)\n\n"
            )

<<<<<<< Updated upstream
            if sessions_file and sessions_file.exists():
                data = json.loads(sessions_file.read_text())
                sessions = data.get("sessions", [])
                if sessions:
                    return sessions[-1].get("id")
        except Exception as e:
            logger.debug(f"Could not read latest session id for {self.cli!r}: {e}")
        return None
=======
        return (
            "(Environment: You have been granted access to a private, isolated filesystem sandbox. "
            "No database access has been provided for this session.)\n\n"
        )

    def _resolve_resume(self, resume: str | None, cwd: str | None) -> str | None:
        """Turn a request to resume "latest" into the id of an actual session.

        Done here rather than in the argument, and once the working directory of the run
        is known: sessions are per-directory, and the one meant by "latest" is the last
        one held where this run works - which is what makes ``odev scaffold`` in a
        folder and ``odev ai --resume`` in that same folder the same conversation.
        """
        if resume != "latest":
            return resume

        if self.handler.resolves_latest_natively:
            return resume

        latest_id = self.get_latest_session_id(cwd)

        if not latest_id:
            logger.warning(f"No previous {self.cli} session was found to resume; starting a new one.")
            return None

        return latest_id

    def get_latest_session_id(self, cwd: str | None = None) -> str | None:
        """Return the id of the most recent session of this agent CLI, if it keeps one.

        Where that is depends on the agent, so the answer is the handler's: see
        :meth:`BaseAgentHandler.get_latest_session_id`.
        """
        try:
            return self.handler.get_latest_session_id(cwd)
        except (OSError, ValueError, AttributeError) as e:
            logger.debug(f"Could not read the latest session id for {self.cli!r}: {e}")
            return None
>>>>>>> Stashed changes

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
