import os
import re
import subprocess
from pathlib import Path

from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin
from odev.common.version import OdooVersion


logger = logging.getLogger(__name__)


class AgentCLI(OdevFrameworkMixin):
    """An execution wrapper for CLI AI agents (claude, gemini, copilot)."""

    def __init__(self, cli: str = "claude", model: str | None = None, yolo: bool = False):
        super().__init__()
        self.cli = cli
        self.model = model
        self.yolo = yolo

    def _display_sandbox_warning(
        self,
        sandbox_dirs: list[str],
        agent_dirs: list[Path],
        agent_files: list[Path],
        dynamic_binds: list[tuple[Path, Path, bool]],
        database: str | None,
        db_user: str | None,
    ):
        """Display a warning message about the sandbox access and security risks."""
        from odev.common import string
        from odev.common.console import console

        console.rule(
            string.stylize("AI SANDBOX SECURITY WARNING", "bold color.red"),
            style="color.red",
        )
        console.print(
            "\n[bold color.yellow]ATTENTION:[/bold color.yellow] You are running an AI agent in a sandboxed environment."
        )
        console.print("The agent can read/write files and access the database within this sandbox.")

        console.print("\n[bold color.cyan]PATH ACCESS (Read-Write):[/bold color.cyan]")
        for d in sorted(set(sandbox_dirs)):
            console.print(f" • {d}")

        console.print("\n[bold color.cyan]DATABASE ACCESS:[/bold color.cyan]")
        if database:
            console.print(f" • Database: [color.purple]{database}[/color.purple]")
            console.print(f" • User:     [color.purple]{db_user or 'default'}[/color.purple]")
        else:
            console.print(" • None")

        console.print("\n[bold color.cyan]BINDINGS (System/Config):[/bold color.cyan]")
        # Group similar binds
        important_binds = []
        for d in agent_dirs:
            important_binds.append(f"{d} (RW)")
        for f in agent_files:
            if f.exists():
                important_binds.append(f"{f} (RW)")

        for src, _, ro in dynamic_binds:
            mode = "RO" if ro else "RW"
            important_binds.append(f"{src} ({mode})")

        # Deduplicate and sort
        for bind in sorted(set(important_binds)):
            console.print(f" • {bind}")

        console.print("\n" + "─" * 80 + "\n")

        if not self.yolo and not console.bypass_prompt:
            return console.confirm("Do you want to proceed with this AI agent execution?", default=True)
        return True

    def run(
        self,
        prompt: str,
        sandbox_dirs: list[str],
        database: str | None = None,
        db_user: str | None = None,
        version: str | None = None,
    ) -> bool:
        """Run the AI agent within a bwrap sandbox."""
        host_home = Path.home()
        import shutil
        import tempfile

        playground = Path(tempfile.mkdtemp(prefix=f"odev-ai-{self.cli}-"))

        agent_cmd = []
        # agent_dirs and agent_files will be bound read-write from the host
        agent_dirs = [
            host_home / f".{self.cli}",
            host_home / f".config/{self.cli}",
            host_home / f".cache/{self.cli}",
        ]
        agent_files = [
            host_home / f".{self.cli}.json",
        ]

        if self.cli == "gemini":
            agent_cmd = ["gemini"]
            if prompt:
                agent_cmd.extend(["-i", prompt])
            agent_cmd.append("--approval-mode")
            agent_cmd.append("yolo" if self.yolo else "auto_edit")
            if self.model:
                agent_cmd.extend(["-m", self.model])
            for d in sandbox_dirs:
                agent_cmd.extend(["--include-directories", str(d)])
            agent_dirs.append(host_home / ".config/configstore")  # Specific to gemini
            agent_dirs.extend(
                [
                    host_home / ".gemini",
                    host_home / ".cache/gemini",
                ]
            )

        elif self.cli == "copilot":
            agent_cmd = ["copilot"]
            if prompt:
                agent_cmd.extend(["-i", prompt])
            agent_cmd.extend(
                [
                    "--allow-tool=read",
                    "--allow-tool=write",
                    "--allow-tool=shell(ruff:*)",
                    "--allow-tool=shell(xmllint:*)",
                    "--allow-tool=shell(odev:*)",
                    "--allow-tool=shell(git:*)",
                ]
            )
            if self.model:
                agent_cmd.extend(["-m", self.model])
            for d in sandbox_dirs:
                agent_cmd.extend(["--add-dir", str(d)])
            agent_dirs.extend(
                [
                    host_home / ".config/github-copilot",
                    host_home / ".config/gh-copilot",
                ]
            )

        elif self.cli == "opencode-cli":
            opencode_bin = host_home / ".opencode/bin/opencode"
            if not opencode_bin.exists():
                logger.error(f"opencode binary not found at {opencode_bin}")
                return False

            agent_cmd.extend([str(opencode_bin), "run"])
            if prompt:
                agent_cmd.append(prompt)
            if self.model:
                agent_cmd.extend(["-m", self.model])

            agent_dirs.extend(
                [
                    host_home / ".cache/opencode",
                    host_home / ".config/opencode",
                    host_home / ".local/share/opencode",
                    host_home / ".opencode",
                ]
            )
        else:  # claude
            agent_cmd = ["claude"]
            if prompt:
                agent_cmd.append(prompt)
            agent_cmd.extend(
                [
                    "--permission-mode",
                    "acceptEdits",
                    "--allowedTools",
                    "Bash(ruff:*),Bash(xmllint:*),Bash(odev:*),Bash(git:*),Read,Edit",
                ]
            )
            if self.model:
                agent_cmd.extend(["-m", self.model])
            for d in sandbox_dirs:
                agent_cmd.extend(["--add-dir", str(d)])
            agent_dirs.extend([host_home / ".claude", host_home / ".anthropic"])

        from odev.common.databases.local import LocalDatabase
        from odev.common.odoobin import OdoobinProcess

        # Dynamic binds for odev, odoo-bin, and config
        odev_path = self.odev.path
        dynamic_binds = [
            (odev_path, odev_path, True),  # Bind odev source
        ]
        # Detection logic
        seen_paths = {odev_path}

        # Bind odev home, SSH and venv
        config_dir = host_home / ".config" / "odev"
        gitconfig = host_home / ".gitconfig"

        if config_dir.exists():
            # Create a writable copy of the config for the sandbox
            sandbox_config_dir = playground / ".config" / "odev"
            sandbox_config_dir.mkdir(parents=True, exist_ok=True)
            for item in config_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(
                        item,
                        sandbox_config_dir / item.name,
                        symlinks=True,
                        dirs_exist_ok=True,
                    )
                else:
                    shutil.copy2(item, sandbox_config_dir / item.name)

            # No need for a separate dynamic bind here, because the playground
            # is already bound to host_home in the bwrap command.
            # sandbox_config_dir/.config/odev will appear at host_home/.config/odev.
            seen_paths.add(config_dir)

        if gitconfig.exists() and gitconfig not in seen_paths:
            # Bind to host_home/.gitconfig because sandboxed HOME is host_home
            dynamic_binds.append((gitconfig, host_home / ".gitconfig", True))
            seen_paths.add(gitconfig)

        from odev.common.python import PythonEnv

        odev_venv_path = PythonEnv().path.resolve()
        if odev_venv_path.exists() and odev_venv_path not in seen_paths:
            dynamic_binds.append((odev_venv_path, odev_venv_path, True))
            seen_paths.add(odev_venv_path)

        # Detect Odoo version from database
        if database:
            db = LocalDatabase(database)
            if db.exists:
                db_version = db.version
                if db_version:
                    logger.debug(f"Detected Odoo version {db_version} for database {database}")
                    # Bind venv from database
                    venv_path = db.venv.path.resolve()
                    if venv_path.exists() and venv_path not in seen_paths:
                        dynamic_binds.append((venv_path, venv_path, True))  # RO
                        seen_paths.add(venv_path)

                    # Bind all Odoo worktrees for this version/database
                    for worktree in db.worktrees:
                        w_path = worktree.path.resolve()
                        if w_path.exists() and w_path not in seen_paths:
                            dynamic_binds.append((w_path, w_path, True))  # RO
                            seen_paths.add(w_path)

                    # If no explicit version was given, use the one from database
                    if not version:
                        version = str(db_version)

        # Explicit version binding — bind Odoo repos/worktrees and venv for a given version
        if version:
            from odev.common.odoobin import odoo_repositories
            from odev.common.python import PythonEnv

            ver_obj = OdooVersion(version)
            ver_str = str(ver_obj)  # e.g. "19.0"
            # Bind venv for this version
            venv_path = PythonEnv(ver_str).path.resolve()
            if venv_path.exists() and venv_path not in seen_paths:
                dynamic_binds.append((venv_path, venv_path, True))  # RO
                seen_paths.add(venv_path)

            # Bind all Odoo worktrees for this version
            for repo in odoo_repositories(enterprise=True):
                for worktree in repo.worktrees():
                    if worktree.name == ver_str:
                        w_path = worktree.path.resolve()
                        if w_path.exists() and w_path not in seen_paths:
                            dynamic_binds.append((w_path, w_path, True))  # RO
                            seen_paths.add(w_path)

        # Detect Odoo version and bind venv/worktrees for each sandbox dir
        # This block handles sandbox_dirs which might not be related to the database.
        for d in sandbox_dirs:
            path = Path(d).resolve()
            if not path.exists():
                continue
            try:
                # version_from_addons can be slow on deep directory trees
                version_detected = OdoobinProcess.version_from_addons(path)

                # If not an addons path, check if it's an Odoo root
                if not version_detected:
                    release_py = path / "odoo" / "release.py"
                    if release_py.exists():
                        content = release_py.read_text()
                        v_match = re.search(r"version\s*=\s*['\"]([\d\.]+)['\"]", content)
                        if v_match:
                            version_detected = OdooVersion(v_match.group(1))

                if version_detected:
                    logger.debug(f"Detected Odoo version {version_detected} for directory {path}")
                    ver_str_d = str(version_detected)
                    from odev.common.odoobin import odoo_repositories
                    from odev.common.python import PythonEnv

                    # Bind venv
                    venv_path = PythonEnv(ver_str_d).path.resolve()
                    if venv_path.exists() and venv_path not in seen_paths:
                        dynamic_binds.append((venv_path, venv_path, True))  # RO
                        seen_paths.add(venv_path)

                    # Bind all Odoo worktrees for this version
                    for repo in odoo_repositories(enterprise=True):
                        for worktree in repo.worktrees():
                            if worktree.name == ver_str_d:
                                w_path = worktree.path.resolve()
                                if w_path.exists() and w_path not in seen_paths:
                                    dynamic_binds.append((w_path, w_path, True))  # RO
                                    seen_paths.add(w_path)
            except Exception as e:
                logger.debug(f"Failed to detect Odoo version for {path}: {e}")

        # These paths need to be read-write (e.g. pip install writes to the venv)
        rw_common_paths = [
            self.odev.home_path / "virtualenvs",
        ]

        # These paths are bound read-only
        ro_common_paths = [
            self.odev.home_path / "worktrees",
            self.odev.home_path / "plugins",
            # The main Odoo git repos must be bound so that git worktrees (stored under
            # ~/odev/worktrees) can resolve their `.git` files which point back here.
            # Without this, odev can't see the repos and triggers a fresh clone.
            self.config.paths.repositories / "odoo",
        ]

        # Bind all enabled plugins. Since these are symlinked, we need to bind the
        # resolved paths to ensure the repositories are accessible in the sandbox.
        for plugin in self.odev.plugins:
            ro_common_paths.append(plugin.path)

        for p in rw_common_paths:
            p = p.resolve()
            if p.exists() and p not in seen_paths:
                dynamic_binds.append((p, p, False))  # RW
                seen_paths.add(p)

        for p in ro_common_paths:
            p = p.resolve()
            if p.exists() and p not in seen_paths:
                dynamic_binds.append((p, p, True))  # RO
                seen_paths.add(p)

        for d in agent_dirs:
            d.mkdir(parents=True, exist_ok=True)

        cmd = ["bwrap"]

        # Environment variables

        # Build sandbox PATH
        sandbox_path = ":".join(
            [
                str(host_home / ".npm-global" / "bin"),
                str(host_home / ".local" / "bin"),
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            ]
        )

        # Database access: we restrict only by setting PGDATABASE.
        # We do NOT set PGUSER to a temp role because PostgreSQL peer auth
        # requires a matching OS user, and synthetic roles cannot authenticate
        # via Unix socket.
        # The odev process inside the sandbox runs as the real OS user (crupuk)
        # which already has the correct privileges.

        cmd.extend(
            [
                "--dir",
                "/home",
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
            ]
        )

        # Pass through relevant AI API keys
        relevant_keys = {
            "claude": ["ANTHROPIC_API_KEY"],
            "gemini": ["GOOGLE_API_KEY"],
            "copilot": [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ],  # Copilot sometimes uses these
            "openai": ["OPENAI_API_KEY"],
        }

        keys_to_process = relevant_keys.get(self.cli, [])

        from odev.common.store.datastore import DataStore

        secrets = DataStore().secrets

        for key in keys_to_process:
            val = os.environ.get(key)
            if not val:
                try:
                    # SecretStore.get(key, ask_missing=True) will prompt if missing
                    # We specify fields=["password"] to ensure it's masked as a secret
                    secret_obj = secrets.get(
                        key,
                        ask_missing=True,
                        fields=["password"],
                        prompt_format="{key}:",
                    )
                    val = secret_obj.password
                except Exception:
                    pass

            if val:
                cmd.extend(["--setenv", key, val])

        if database:
            cmd.extend(["--setenv", "PGDATABASE", database])

        # Network-related files
        cmd.extend(
            [
                "--ro-bind",
                "/run/systemd/resolve",
                "/run/systemd/resolve",
                "--ro-bind",
                "/etc/hosts",
                "/etc/hosts",
                "--ro-bind",
                "/etc/ssl",
                "/etc/ssl",
                "--ro-bind-try",
                "/etc/pki",
                "/etc/pki",
                "--ro-bind-try",
                "/etc/ca-certificates",
                "/etc/ca-certificates",
                "--ro-bind-try",
                "/etc/crypto-policies",
                "/etc/crypto-policies",
                "--symlink",
                "../run/systemd/resolve/stub-resolv.conf",
                "/etc/resolv.conf",
            ]
        )

        # System binaries and libraries
        cmd.extend(
            [
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/bin",
                "/bin",
                "--symlink",
                "usr/sbin",
                "/sbin",
                "--symlink",
                "usr/lib",
                "/lib",
                "--symlink",
                "usr/lib64",
                "/lib64",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--ro-bind-try",
                "/etc/machine-id",
                "/etc/machine-id",
            ]
        )

        # Desktop and user specific binds

        # Common user binaries and configs (Read-only)
        cmd.extend(
            [
                "--ro-bind-try",
                str(host_home / ".npm-global"),
                str(host_home / ".npm-global"),
                "--ro-bind-try",
                str(host_home / ".local/bin"),
                str(host_home / ".local/bin"),
                "--ro-bind-try",
                str(host_home / ".local/share/claude"),
                str(host_home / ".local/share/claude"),
            ]
        )

        # Unshare namespaces
        # Set chdir to the first sandbox dir or current working dir
        chdir_path = sandbox_dirs[0] if sandbox_dirs else str(Path.cwd())
        cmd.extend(
            [
                "--chdir",
                str(chdir_path),
                "--unshare-all",
                "--share-net",
                "--die-with-parent",
                "--new-session",
            ]
        )

        # Odoo and system specific binds
        cmd.extend(
            [
                "--bind-try",
                str(host_home / ".local/share/Odoo"),
                str(host_home / ".local/share/Odoo"),
                "--bind-try",
                "/var/run/postgresql",
                "/var/run/postgresql",
                "--ro-bind",
                "/etc/passwd",
                "/etc/passwd",
            ]
        )

        # Bind agent config and cache directories (Read-write)
        for path in agent_dirs:
            # Bind host path to sandbox home path
            cmd.extend(["--bind-try", str(path), str(path)])

        # Bind agent config files (Read-write)
        for path in agent_files:
            # Bind host path only if it exists
            if path.exists():
                cmd.extend(["--bind-try", str(path), str(path)])

        # Bind odev and odoo-bin related paths
        for src, dst, read_only in dynamic_binds:
            # Ensure destination parent directory exists in the playground
            relative_dst = dst.relative_to(host_home) if dst.is_relative_to(host_home) else None
            if relative_dst:
                sandbox_dst_parent = (playground / relative_dst).parent
                sandbox_dst_parent.mkdir(parents=True, exist_ok=True)

            flag = "--ro-bind-try" if read_only else "--bind-try"
            cmd.extend([flag, str(src), str(dst)])

        # Bind allowed directories (sandbox)
        for path in sandbox_dirs:
            path_obj = Path(path).resolve()
            # Ensure destination parent directory exists in the playground
            relative_dst = path_obj.relative_to(host_home) if path_obj.is_relative_to(host_home) else None
            if relative_dst:
                sandbox_dst_parent = (playground / relative_dst).parent
                sandbox_dst_parent.mkdir(parents=True, exist_ok=True)

            cmd.extend(["--bind-try", str(path), str(path)])

        # Execute the respective agent
        cmd.append("--")
        cmd.extend(agent_cmd)

        try:
            if not self._display_sandbox_warning(
                sandbox_dirs=sandbox_dirs,
                agent_dirs=agent_dirs,
                agent_files=agent_files,
                dynamic_binds=dynamic_binds,
                database=database,
                db_user=db_user,
            ):
                return False
            logger.info(f"Starting Project-wide AI execution using {self.cli}")
            logger.debug(f"Running sandbox command: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to run {self.cli}: {e}")
            return False
        finally:
            # Cleanup playground
            try:
                shutil.rmtree(playground)
            except Exception:
                pass
