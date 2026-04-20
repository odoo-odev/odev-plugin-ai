import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from odev.common.logging import logging

from .bwrap import BwrapSandbox
from .postgres import PostgresSandbox


logger = logging.getLogger(__name__)


class AgentCLI(BwrapSandbox):
    """An execution wrapper for CLI AI agents (claude, gemini, copilot)."""

    @staticmethod
    def _guest_paths(all_candidate_paths: list[str]) -> list[str]:
        """Return the guest-side path from each 'host:guest' entry."""
        return [d.split(":", 1)[1] if ":" in d else d for d in all_candidate_paths]

    def _get_gemini_agent_cmd(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
        host_home: Path,
    ) -> list[str]:
        """Build the gemini agent command."""
        cmd = ["gemini"]
        if prompt:
            cmd.extend(["-p" if self.headless else "-i", prompt])
        if resume:
            cmd.extend(["--resume", resume])
        cmd.append("--approval-mode")
        cmd.append("yolo" if self.yolo else "auto_edit")
        if self.model and self.model != "auto":
            cmd.extend(["-m", self.model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--include-directories", path])
        return cmd

    def _get_copilot_agent_cmd(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
    ) -> list[str]:
        """Build the copilot agent command."""
        cmd = ["copilot"]
        if prompt:
            cmd.extend(["-p" if self.headless else "-i", prompt])
        if resume:
            cmd.append(f"--resume={resume}")
        if self.yolo:
            cmd.append("--yolo")
        else:
            cmd.extend(
                [
                    "--allow-tool=read",
                    "--allow-tool=write",
                    "--allow-tool=shell(rtk:*)",
                    "--allow-tool=shell(odev:*)",
                    "--allow-tool=shell(git:*)",
                    "--allow-tool=shell(pre-commit:*)",
                ]
            )
        if self.model and self.model != "auto":
            cmd.extend(["-m", self.model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd

    def _get_opencode_agent_cmd(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
        host_home: Path,
    ) -> list[str]:
        """Build the opencode-cli agent command."""
        opencode_bin = host_home / ".opencode/bin/opencode"
        if not opencode_bin.exists():
            logger.error(f"opencode binary not found at {opencode_bin}")
            return []
        cmd = [str(opencode_bin), "run"]
        if prompt:
            cmd.append(prompt)
        if resume:
            cmd.extend(["--session", resume])
        if self.model and self.model != "auto":
            cmd.extend(["-m", self.model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd

    def _get_claude_agent_cmd(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
    ) -> list[str]:
        """Build the claude agent command."""
        cmd = ["claude"]
        if prompt:
            if self.headless:
                cmd.extend(["-p", prompt])
            else:
                cmd.append(prompt)
        if resume:
            cmd.extend(["--session-id", resume])
        if self.yolo:
            cmd.append("--dangerously-skip-permissions")
        else:
            cmd.extend(
                [
                    "--permission-mode",
                    "acceptEdits",
                    "--allowedTools",
                    "Bash(rtk:*),Bash(odev:*),Bash(git:*),Bash(pre-commit:*),Read,Edit",
                ]
            )
        if self.model and self.model != "auto":
            cmd.extend(["--model", self.model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd

    def _get_agent_setup(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
        host_home: Path,
    ) -> tuple[list[str], list[Path], list[Path]]:
        """Determine agent-specific command, directories, and files to mount."""
        # We don't bind-mount host configs directly anymore to prevent
        # 'Workspace Trust' issues inside the sandbox. Instead, we rely on
        # the sanitized configs in the playground.
        agent_dirs = [
            host_home / ".cache",
            host_home / ".local",
            host_home / ".claude",
            host_home / ".gemini",
            host_home / ".opencode",
        ]
        agent_files = [
            host_home / ".gitconfig",
            host_home / ".claude.json",
        ]
        # Include .env if it exists in the current directory
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            agent_files.append(env_file)

        if self.cli == "gemini":
            agent_cmd = self._get_gemini_agent_cmd(prompt, resume, all_candidate_paths, host_home)
        elif self.cli == "copilot":
            agent_cmd = self._get_copilot_agent_cmd(prompt, resume, all_candidate_paths)
        elif self.cli == "opencode-cli":
            agent_cmd = self._get_opencode_agent_cmd(prompt, resume, all_candidate_paths, host_home)
            if not agent_cmd:
                return [], [], []
        else:
            agent_cmd = self._get_claude_agent_cmd(prompt, resume, all_candidate_paths)

        return agent_cmd, agent_dirs, agent_files

    def run(
        self,
        prompt: str,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None = None,
        database: str | None = None,
        db_user: str | None = None,
        version: str | None = None,
        resume: str | None = None,
        path_mapping: dict[str, str] | None = None,
        ephemeral_pg: bool = True,
        cwd: str | None = None,
    ) -> bool:
        """Run the AI agent within a bwrap sandbox."""
        if path_mapping is None:
            path_mapping = {}

        host_home = Path.home()
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
            cwd = "/custom" if any(b[1] == Path("/custom") for b in final_binds) else str(host_home)

        # Binds where src != dst are Type B shortcuts (/custom, /upgrade, /worktrees, ...)
        all_candidate_paths = [f"{src}:{dst}" for src, dst, _, _ in final_binds if src != dst]
        # Sync path_mapping with Type B binds so _prepare_odev_config rewrites paths correctly
        for src, dst, _, _ in final_binds:
            if src != dst:
                path_mapping.setdefault(str(src), str(dst))
        agent_cmd, agent_dirs, agent_files = self._get_agent_setup(prompt, resume, all_candidate_paths, host_home)

        if database:
            db_info = (
                f"(Environment: You have been granted access to a private, isolated CLONE of the host database '{database}'. "
                "This is a separate, ephemeral PostgreSQL cluster. You can safely modify it as it "
                "does not affect the live host data. Use 'psql' to work directly with it.)\n\n"
            )
        else:
            db_info = (
                "(Environment: You have been granted access to a private, isolated filesystem sandbox. "
                "No database access has been provided for this session.)\n\n"
            )
        prompt = db_info + prompt

        sandbox_path_items = []
        if active_venv_path:
            sandbox_path_items.append(str(active_venv_path / "bin"))

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
            "/run/user/1000",
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

        secrets_to_set = self._collect_secrets()
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

        self._prepare_odev_config(playground, path_mapping, host_home)
        self._add_system_binds(cmd, host_home, sandbox_tmp, cwd)

        pg_sandbox = PostgresSandbox(headless=self.headless)
        pg_process = pg_sandbox.setup(cmd, database, proxy_dir, pg_data_dir, ephemeral=ephemeral_pg)

        self._setup_rtk_sandbox(cmd, playground, host_home)

        self._apply_final_bindings(cmd, agent_dirs, agent_files, final_binds, host_home, playground, path_mapping)
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
        )

    def get_latest_session_id(self) -> str | None:
        """Return the ID of the most recent session for this agent CLI."""
        try:
            home = Path.home()
            sessions_file = None
            if self.cli == "gemini":
                sessions_file = home / ".gemini" / "sessions.json"
            elif self.cli in ("claude", "opencode-cli"):
                sessions_file = home / ".claude" / "sessions.json"
            else:
                logger.debug(f"Session resumption not supported for '{self.cli}'")
                return None

            if sessions_file and sessions_file.exists():
                data = json.loads(sessions_file.read_text())
                sessions = data.get("sessions", [])
                if sessions:
                    return sessions[-1].get("id")
        except Exception as e:
            logger.debug(f"Could not read latest session id for {self.cli!r}: {e}")
        return None

    def _collect_secrets(self) -> list[tuple[str, str]]:  # noqa: C901
        """Retrieve and interactively prompt for AI API secrets."""
        relevant = {
            "claude": ["ANTHROPIC_API_KEY", "GITHUB_TOKEN"],
            "gemini": ["GEMINI_API_KEY", "GITHUB_TOKEN"],
            "copilot": [
                "GITHUB_TOKEN",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ],
            "openai": ["OPENAI_API_KEY", "GITHUB_TOKEN"],
        }
        to_process = relevant.get(self.cli, [])
        found_secrets: dict[str, str] = {}

        for key in to_process:
            val = os.environ.get(key)
            if val:
                found_secrets[key] = val

        env_file = Path.cwd() / ".env"
        if env_file.exists():
            try:
                content = env_file.read_text()
                for key in to_process:
                    if key not in found_secrets:
                        # Simple regex to find KEY=VAL or KEY="VAL"
                        pattern = rf"^{key}=[\"']?(.*?)[\"']?$"
                        match = re.search(pattern, content, re.MULTILINE)
                        if match:
                            found_secrets[key] = match.group(1).strip()
            except Exception:
                pass

        if "GITHUB_TOKEN" not in found_secrets and "GH_TOKEN" not in found_secrets:
            try:
                token = subprocess.check_output(
                    ["gh", "auth", "token"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if token:
                    found_secrets["GITHUB_TOKEN"] = token
            except Exception:
                pass

        # Copilot checks env vars in order: COPILOT_GITHUB_TOKEN > GH_TOKEN > GITHUB_TOKEN.
        # The keyring stores the same gho_ OAuth token that gh uses, so we alias it to
        # COPILOT_GITHUB_TOKEN (highest precedence) so copilot uses it without keyring access.
        if self.cli == "copilot" and "GITHUB_TOKEN" in found_secrets:
            found_secrets.setdefault("COPILOT_GITHUB_TOKEN", found_secrets["GITHUB_TOKEN"])

        from odev.common.store.datastore import DataStore

        ds = DataStore().secrets

        for key in to_process:
            if key in found_secrets:
                continue

            try:
                is_opt = key in (
                    ["GH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
                    if self.cli == "copilot"
                    # Claude Code authenticates via OAuth (hosts.json), API key is optional
                    else ["ANTHROPIC_API_KEY", "GH_TOKEN"]
                    if self.cli == "claude"
                    else []
                )
                # If headless, we NEVER prompt. We fail later if required.
                ask = not is_opt and not self.headless

                p_fmt = "GitHub Token:" if "GITHUB" in key else f"{key}:"
                secret_obj = ds.get(
                    key,
                    ask_missing=ask,
                    fields=["password"],
                    prompt_format=p_fmt,
                )
                if secret_obj.password:
                    found_secrets[key] = secret_obj.password
            except Exception as e:
                logger.debug(f"Could not retrieve secret {key!r}: {e}")

        # Mirror canonical keys to their common aliases for underlying tool compatibility
        if "GITHUB_TOKEN" in found_secrets:
            found_secrets["GH_TOKEN"] = found_secrets["GITHUB_TOKEN"]
        if "GEMINI_API_KEY" in found_secrets:
            found_secrets["GOOGLE_API_KEY"] = found_secrets["GEMINI_API_KEY"]

        return list(found_secrets.items())
