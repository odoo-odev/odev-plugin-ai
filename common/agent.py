import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from odev.common.console import console
from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin
from odev.common.odoobin import odoo_repositories
from odev.common.python import PythonEnv
from odev.common.version import OdooVersion


logger = logging.getLogger(__name__)


class AgentCLI(OdevFrameworkMixin):
    """An execution wrapper for CLI AI agents (claude, gemini, copilot)."""

    def __init__(
        self,
        cli: str,
        model: str = "auto",
        yolo: bool = False,
        headless: bool = False,
    ):
        super().__init__()
        self.cli = cli
        self.model = model
        self.headless = headless
        self.yolo = yolo or headless  # Headless mode implies YOLO

    def _map_path(self, p: Path | str, path_mapping: dict[str, str] | None) -> Path:
        """Apply path mapping to a path if it matches any host prefix."""
        p = Path(p).resolve()
        dst = p
        if path_mapping:
            p_str = str(p)
            for host, guest in sorted(path_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                if p_str.startswith(host):
                    dst = Path(p_str.replace(host, guest))
                    break
        return dst

    def _bind_paths(
        self,
        paths: list[Path],
        seen_paths: set[Path],
        dynamic_binds: list[tuple[Path, Path, bool]],
        read_only: bool = True,
        path_mapping: dict[str, str] | None = None,
    ):
        """Helper to bind multiple paths to the sandbox while avoiding duplicates."""
        for p in paths:
            p = Path(p).resolve()
            if p.exists() and p not in seen_paths:
                dst = self._map_path(p, path_mapping)
                dynamic_binds.append((p, dst, read_only))
                seen_paths.add(p)

    def _display_sandbox_warning(
        self,
        sandbox_dirs: list[str],
        agent_dirs: list[Path],
        agent_files: list[Path],
        dynamic_binds: list[tuple[Path, Path, bool]],
        extra_bind_dirs: list[str] | None = None,
        database: str | None = None,
        db_user: str | None = None,
        ephemeral_pg: bool = True,
    ) -> bool:
        """Display a warning message about the sandbox access and security risks."""
        if self.headless:
            return True

        from odev.common import string

        console.rule(
            string.stylize("AI SANDBOX SECURITY WARNING", "bold color.red"),
            style="color.red",
        )
        console.print(
            "\n[bold color.yellow]ATTENTION:[/bold color.yellow] You are running an AI agent in a sandboxed environment."
        )
        console.print("The agent can read/write files and access the database within this sandbox.")

        console.print("\n[bold color.cyan]PRIMARY WORKSPACES (Read-Write Access):[/bold color.cyan]")
        for d in sorted(set(sandbox_dirs)):
            if ":" in d:
                src, dst = d.split(":", 1)
                if src == dst:
                    console.print(f" • {src}")
                else:
                    console.print(f" • {src} [bold color.green]-> {dst}[/bold color.green]")
            else:
                console.print(f" • {d}")

        console.print("\n[bold color.cyan]DATABASE ACCESS:[/bold color.cyan]")
        if database:
            console.print(f" • Database: [color.purple]{database}[/color.purple]")
            console.print(f" • User:     [color.purple]{db_user or 'default'}[/color.purple]")
        else:
            console.print(" • [color.green]Isolating (No database access)[/color.green]")

        if database and not ephemeral_pg:
            console.print(
                "\n[bold color.red]WARNING:[/bold color.red] You are granting access to your [bold]HOST[/bold] PostgreSQL cluster."
            )
            console.print("The agent will be able to see and potentially access [bold]ALL[/bold] your local databases.")

        console.print("\n[bold color.cyan]INFRASTRUCTURE & REFERENCE (System/Source/Config):[/bold color.cyan]")
        # Group similar binds
        important_binds = []
        for adir in agent_dirs:
            important_binds.append(f"{adir} (RW)")
        for f in agent_files:
            if f.exists():
                important_binds.append(f"{f} (RW)")

        for src, dst, ro in dynamic_binds:
            mode = "RO" if ro else "RW"
            if str(src) == str(dst):
                important_binds.append(f"{src} ({mode})")
            else:
                important_binds.append(f"{src} [bold color.green]-> {dst}[/bold color.green] ({mode})")

        if extra_bind_dirs:
            for edir in extra_bind_dirs:
                if ":" in edir:
                    src, dst = edir.split(":", 1)
                    if src == dst:
                        important_binds.append(f"{src} (RO)")
                    else:
                        important_binds.append(f"{src} [bold color.green]-> {dst}[/bold color.green] (RO)")
                else:
                    important_binds.append(f"{edir} (RO)")

        # Deduplicate and sort
        for bind in sorted(set(important_binds)):
            console.print(f" • {bind}")

        if not self.yolo and not console.bypass_prompt:
            return console.confirm("Do you want to proceed with this AI agent execution?", default=True)
        return True

    def _get_agent_setup(
        self,
        prompt: str | None,
        resume: str | None,
        all_candidate_paths: list[str],
        host_home: Path,
    ) -> tuple[list[str], list[Path], list[Path]]:
        """Determine agent-specific command, directories, and files to mount."""
        # Sanitize config directories - we don't bind-mount host configs directly
        # anymore to prevent 'Workspace Trust' issues inside the sandbox.
        # Instead, we rely on the sanitized configs in the playground.
        agent_dirs = [
            host_home / ".cache",
            host_home / ".local",
        ]
        agent_files = [
            host_home / ".gitconfig",
        ]
        # Include .env if it exists in the current directory
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            agent_files.append(env_file)

        agent_cmd = []
        if self.cli == "gemini":
            agent_cmd = ["gemini"]
            if prompt:
                agent_cmd.extend(["-p" if self.headless else "-i", prompt])
            if resume:
                agent_cmd.extend(["--resume", resume])
            agent_cmd.append("--approval-mode")
            agent_cmd.append("yolo" if self.yolo else "auto_edit")
            if self.model and self.model != "auto":
                agent_cmd.extend(["-m", self.model])

            # Use specific guest paths for indexing
            indexing_whitelist = ["/skills", "/knowledge", "/custom", str(host_home / ".odev")]
            for d in all_candidate_paths:
                agent_path = d.split(":", 1)[1] if ":" in d else d
                if any(agent_path.startswith(w) for w in indexing_whitelist):
                    agent_cmd.extend(["--include-directories", agent_path])

        elif self.cli == "copilot":
            agent_cmd = ["copilot"]
            if prompt:
                agent_cmd.extend(["-p" if self.headless else "-i", prompt])
            if resume:
                agent_cmd.append(f"--resume={resume}")
            agent_cmd.extend(
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
                agent_cmd.extend(["-m", self.model])
            for d in all_candidate_paths:
                agent_path = d.split(":", 1)[1] if ":" in d else d
                agent_cmd.extend(["--add-dir", agent_path])

        elif self.cli == "opencode-cli":
            opencode_bin = host_home / ".opencode/bin/opencode"
            if not opencode_bin.exists():
                logger.error(f"opencode binary not found at {opencode_bin}")
                return [], [], []

            agent_cmd.extend([str(opencode_bin), "run"])
            if prompt:
                agent_cmd.append(prompt)
            if resume:
                agent_cmd.extend(["--session", resume])
            if self.model and self.model != "auto":
                agent_cmd.extend(["-m", self.model])
        else:  # claude
            agent_cmd = ["claude"]
            if prompt:
                if self.headless:
                    agent_cmd.extend(["-p", prompt])
                else:
                    agent_cmd.append(prompt)
            if resume:
                agent_cmd.extend(["--session-id", resume])
            agent_cmd.extend(
                [
                    "--permission-mode",
                    "acceptEdits",
                    "--allowedTools",
                    "Bash(rtk:*),Bash(odev:*),Bash(git:*),Bash(pre-commit:*),Read,Edit",
                ]
            )
            if self.model and self.model != "auto":
                agent_cmd.extend(["-m", self.model])
            for d in all_candidate_paths:
                agent_path = d.split(":", 1)[1] if ":" in d else d
                agent_cmd.extend(["--add-dir", agent_path])

        return agent_cmd, agent_dirs, agent_files

    def _prepare_odev_config(self, playground, path_mapping, host_home):
        """Create a sandboxed copy of the odev configuration."""
        config_dir = host_home / ".config" / "odev"
        if config_dir.exists():
            sandbox_config_dir = playground / ".config" / "odev"
            sandbox_config_dir.mkdir(parents=True, exist_ok=True)
            for f in config_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, sandbox_config_dir / f.name)

        # Update paths in the sandboxed config if mapping is provided
        if path_mapping:
            config_file = playground / ".config" / "odev" / "odev.cfg"
            if config_file.exists():
                try:
                    import configparser

                    cp = configparser.ConfigParser()
                    cp.read(config_file)

                    # Update all paths to guest paths
                    if cp.has_section("paths"):
                        for option in cp.options("paths"):
                            val = cp.get("paths", option)
                            for host, guest in sorted(path_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                                if val.startswith(host):
                                    cp.set("paths", option, val.replace(host, guest))
                                    break

                        # Explicitly ensure worktrees path is set correctly if we have a mapping for it
                        worktrees_host = str(self.odev.home_path / "worktrees")
                        if worktrees_host in path_mapping:
                            cp.set("paths", "worktrees", path_mapping[worktrees_host])

                    # Add mappings section
                    if not cp.has_section("mappings"):
                        cp.add_section("mappings")
                    mapping_str = ",".join(f"{h}:{g}" for h, g in path_mapping.items())
                    cp.set("mappings", "path_mapping", mapping_str)

                    with open(config_file, "w") as f:
                        cp.write(f)
                except Exception as e:
                    logger.debug(f"Failed to update sandboxed odev config: {e}")

    def _setup_postgresql_sandbox(
        self,
        cmd: list[str],
        database: str | None,
        ephemeral_pg: bool,
        proxy_dir: Path,
        pg_data_dir: Path,
        pg_socket_dir: Path | str | None,
    ) -> subprocess.Popen | None:
        """Initialize PostgreSQL cluster or proxy for the sandbox."""
        if not database:
            return None

        pg_process = None
        host_socket_dir = None
        if pg_socket_dir:
            host_socket_dir = Path(pg_socket_dir)
        else:
            for path in [Path("/var/run/postgresql"), Path("/tmp")]:
                if any(path.glob(".s.PGSQL.*")):
                    host_socket_dir = path
                    break
            else:
                host_socket_dir = Path("/var/run/postgresql")

        if ephemeral_pg:
            pg_process = self._start_ephemeral_postgres(proxy_dir, pg_data_dir)
            if pg_process:
                user = Path.home().name
                if database:
                    try:
                        subprocess.run(
                            [
                                "psql",
                                "-h",
                                str(proxy_dir),
                                "-p",
                                "5432",
                                "-U",
                                user,
                                "-d",
                                "postgres",
                                "-c",
                                f'CREATE DATABASE "{database}";',
                            ],
                            check=True,
                            capture_output=True,
                        )
                    except Exception as e:
                        logger.debug(f"Database {database!r} setup failed/skipped: {e}")

                if database and host_socket_dir and host_socket_dir.exists():
                    self._clone_host_database(database, host_socket_dir, proxy_dir)

                cmd.extend(
                    [
                        "--bind",
                        str(proxy_dir),
                        "/var/run/postgresql",
                        "--symlink",
                        "/var/run/postgresql/.s.PGSQL.5432",
                        "/tmp/.s.PGSQL.5432",
                    ]
                )
        else:
            if host_socket_dir and host_socket_dir.exists():
                cmd.extend(
                    [
                        "--bind",
                        str(host_socket_dir),
                        "/var/run/postgresql",
                        "--symlink",
                        "/var/run/postgresql/.s.PGSQL.5432",
                        "/tmp/.s.PGSQL.5432",
                    ]
                )
        return pg_process

    def _setup_rtk_sandbox(
        self,
        cmd: list[str],
        rtk_path: str | None,
        playground: Path,
        host_home: Path,
    ):
        """Perform automatic RTK initialization and bind hooks to the sandbox."""
        if not rtk_path:
            return

        rtk_flags = ["init", "-g", "--auto-patch"]
        if self.cli == "gemini":
            rtk_flags.append("--gemini")
        elif self.cli == "copilot":
            rtk_flags.append("--copilot")
        elif self.cli == "opencode-cli":
            rtk_flags.append("--opencode")
        else:
            rtk_flags.extend(["--agent", "claude"])

        agent_config_dirs = {
            "gemini": host_home / ".gemini",
            "copilot": host_home / ".config" / "github-copilot",
            "opencode-cli": host_home / ".claude",
            "claude": host_home / ".claude",
        }
        host_config_dir = agent_config_dirs.get(self.cli)
        if host_config_dir and host_config_dir.exists():
            sandbox_config_path = playground / host_config_dir.relative_to(host_home)
            sandbox_config_path.mkdir(parents=True, exist_ok=True)
            s_json = host_config_dir / "settings.json"
            if s_json.exists():
                shutil.copy2(s_json, sandbox_config_path / "settings.json")

        try:
            subprocess.run(
                [rtk_path] + rtk_flags,
                env={**os.environ, "HOME": str(playground)},
                capture_output=True,
                check=False,
            )
            logger.debug(f"Successfully initialized RTK for {self.cli} in sandbox")

            for rel_path in [
                ".claude/settings.json",
                ".claude/CLAUDE.md",
                ".claude/RTK.md",
                ".claude/hooks",
                ".gemini/settings.json",
                ".gemini/GEMINI.md",
                ".gemini/hooks",
                ".config/rtk",
                ".local/share/rtk",
            ]:
                p = playground / rel_path
                if p.exists():
                    cmd.extend(["--bind", str(p), str(host_home / rel_path)])
        except Exception as e:
            logger.warning(f"Failed to auto-initialize RTK in sandbox: {e}")

    def run(
        self,
        prompt: str,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None = None,
        database: str | None = None,
        db_user: str | None = None,
        version: str | None = None,
        resume: str | None = None,
        pg_socket_dir: Path | str | None = None,
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

        # Default CWD to the first sandbox dir's guest path if it's mapped
        if not cwd and sandbox_dirs:
            first = sandbox_dirs[0]
            if ":" in first:
                cwd = first.split(":", 1)[1]
            else:
                cwd = first
        if not cwd:
            cwd = str(host_home)

        # 1. Agent Setup
        all_candidate_paths = sandbox_dirs + (extra_bind_dirs or [])
        agent_cmd, agent_dirs, agent_files = self._get_agent_setup(prompt, resume, all_candidate_paths, host_home)

        # 2. Environmental Context Info
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

        # 3. Dynamic Binds Discovery
        dynamic_binds, seen_paths, active_venv_path = self._discover_binds(
            database, version, sandbox_dirs, path_mapping, host_home
        )

        # 4. Global Deduplication
        # Combine all sources into a unified list of (src, dst, ro)
        pool = []
        for sdir in sandbox_dirs:
            src, dst = sdir.split(":", 1) if ":" in sdir else (sdir, sdir)
            pool.append((Path(src), Path(dst), False))
        for edir in extra_bind_dirs or []:
            src, dst = edir.split(":", 1) if ":" in edir else (edir, edir)
            pool.append((Path(src), Path(dst), True))
        pool.extend(dynamic_binds)

        deduplicated_pool = []
        # Sort by guest path depth so parents come first
        sorted_pool = sorted(pool, key=lambda b: len(b[1].parts))
        for src, dst, ro in sorted_pool:
            is_redundant = False
            for e_src, e_dst, e_ro in deduplicated_pool:
                try:
                    rel_src = src.relative_to(e_src)
                    rel_dst = dst.relative_to(e_dst)
                    if rel_src == rel_dst:
                        is_redundant = True
                        break
                except (ValueError, AttributeError):
                    continue
            if not is_redundant:
                deduplicated_pool.append((src, dst, ro))

        # Re-partition into sandbox_dirs, extra_bind_dirs, and dynamic_binds for display compatibility
        # We use the deduplicated results to filter the originals
        final_sandbox_dirs = []
        final_extra_bind_dirs = []
        final_dynamic_binds = []

        for s, d, r in deduplicated_pool:
            # Reconstruct strings for warning display
            s_str = f"{s}:{d}"
            # Check if this mount was originally requested as a Primary Sandbox (RW)
            original_sandbox = next((sd for sd in sandbox_dirs if sd.startswith(str(s))), None)

            if original_sandbox and not r:
                final_sandbox_dirs.append(original_sandbox)
            elif r:
                # If it's Read-Only, we put it in extra/infrastructure
                final_extra_bind_dirs.append(s_str)
            else:
                final_dynamic_binds.append((s, d, r))

        # Update core variables to use deduplicated versions for the rest of the execution
        sandbox_dirs = final_sandbox_dirs
        extra_bind_dirs = final_extra_bind_dirs
        dynamic_binds = final_dynamic_binds

        # 5. Prepare Odev Config
        self._prepare_odev_config(playground, path_mapping, host_home)

        # 5. Base Bwrap Command
        odev_path = self.odev.path
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

        # 6. Secret Handling
        secrets_to_set = self._collect_secrets()
        if database:
            cmd.extend(["--setenv", "PGDATABASE", database])

        # 7. Collect all guest paths and ensure top-level directories exist in the guest root
        all_guest_paths: set[str] = set()
        for _, dst, _ in dynamic_binds:
            all_guest_paths.add(str(dst))
        for sdir in sandbox_dirs + (extra_bind_dirs or []):
            dst = sdir.split(":", 1)[1] if ":" in sdir else sdir
            all_guest_paths.add(dst)

        top_dirs = set()
        for gp in sorted(all_guest_paths):
            p = Path(gp)
            if p.is_absolute():
                top_dirs.add(f"/{p.parts[1]}")

        for td in sorted(top_dirs):
            if td not in [
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
            ]:
                cmd.extend(["--dir", td])

        # 8. System Binds
        self._add_system_binds(cmd, host_home, sandbox_tmp, cwd)

        # 9. PostgreSQL Setup
        pg_process = self._setup_postgresql_sandbox(cmd, database, ephemeral_pg, proxy_dir, pg_data_dir, pg_socket_dir)

        # 10. RTK Setup
        rtk_path = shutil.which("rtk")
        self._setup_rtk_sandbox(cmd, rtk_path, playground, host_home)

        # 11. Final Bindings loop
        self._apply_final_bindings(
            cmd,
            agent_dirs,
            agent_files,
            dynamic_binds,
            sandbox_dirs,
            extra_bind_dirs,
            host_home,
            playground,
            path_mapping=path_mapping,
        )

        # 12. Run
        cmd.extend(["--", *agent_cmd])

        # 13. Agent Configuration Sanitization (Workspace Trust)
        self._prepare_agent_config(playground, all_candidate_paths, host_home)

        # 14. Execute Sandbox
        return self._execute_sandbox(
            cmd,
            agent_dirs,
            agent_files,
            dynamic_binds,
            extra_bind_dirs,
            database,
            db_user,
            secrets_to_set,
            pg_process,
            sandbox_dirs,
            playground,
            sandbox_tmp,
            proxy_dir,
            pg_data_dir,
        )

    def _discover_binds(
        self, database, version, sandbox_dirs, path_mapping, host_home
    ) -> tuple[list[tuple[Path, Path, bool]], set[Path], Path | None]:
        """Discover all necessary filesystem paths to bind into the sandbox."""
        from odev.common.databases.local import LocalDatabase

        odev_path = self.odev.path
        dynamic_binds = [(odev_path, odev_path, True)]
        seen_paths = {odev_path}
        active_venv_path: Path | None = None

        # Venv
        venv = Path(PythonEnv().path).resolve()
        if venv.exists():
            dst = self._map_path(venv, path_mapping)
            dynamic_binds.append((venv, dst, True))
            seen_paths.add(venv)
            active_venv_path = dst

        # Database specific
        if database:
            db = LocalDatabase(database)
            if db.exists:
                d_ver = db.version
                if d_ver:
                    v_path = Path(db.venv.path).resolve()
                    if v_path.exists() and v_path not in seen_paths:
                        dst = self._map_path(v_path, path_mapping)
                        dynamic_binds.append((v_path, dst, True))
                        seen_paths.add(v_path)
                        # Override default venv with database-specific one
                        active_venv_path = dst

                    for wt in db.worktrees:
                        wp = Path(wt.path).resolve()
                        if wp.exists() and wp not in seen_paths:
                            dst = self._map_path(wp, path_mapping)
                            dynamic_binds.append((wp, dst, True))
                            seen_paths.add(wp)
                    if not version:
                        version = str(d_ver)

        # RTK binary bind
        rtk = shutil.which("rtk")
        if rtk:
            rp = Path(rtk).resolve()
            std = [
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
                str(host_home / ".local/bin"),
            ]
            if not any(str(rp).startswith(p) for p in std):
                dst = self._map_path(rp, path_mapping)
                dynamic_binds.append((rp, dst, True))
                seen_paths.add(rp)

        # Explicit version
        if version:
            ver_obj = OdooVersion(version)
            ver_str = str(ver_obj)
            v_p = Path(PythonEnv(ver_str).path).resolve()
            if v_p.exists() and v_p not in seen_paths:
                dst = self._map_path(v_p, path_mapping)
                dynamic_binds.append((v_p, dst, True))
                seen_paths.add(v_p)
                # Version-specific venv takes precedence
                active_venv_path = dst

            for repo in odoo_repositories(enterprise=True):
                r_p = Path(repo.path).resolve()
                if r_p.exists() and r_p not in seen_paths:
                    dst = self._map_path(r_p, path_mapping)
                    dynamic_binds.append((r_p, dst, True))
                    seen_paths.add(r_p)
                for wt in repo.worktrees():
                    if wt.name == ver_str:
                        wp = Path(wt.path).resolve()
                        if wp.exists() and wp not in seen_paths:
                            dst = self._map_path(wp, path_mapping)
                            dynamic_binds.append((wp, dst, True))
                            seen_paths.add(wp)

        # Common odev paths
        ro_paths = [self.odev.home_path / "worktrees", self.odev.home_path / "plugins"]
        rw_paths = [self.odev.home_path / "virtualenvs"]
        repo_paths = [self.config.paths.repositories / "odoo"]

        for plugin in self.odev.plugins:
            try:
                res = plugin.path.resolve()
                if res.exists():
                    ro_paths.append(res.parent)
            except Exception:
                pass

        if path_mapping:
            seen_paths.update({Path(h).resolve() for h in path_mapping})

        self._bind_paths(
            rw_paths,
            seen_paths,
            dynamic_binds,
            read_only=False,
            path_mapping=path_mapping,
        )
        self._bind_paths(
            ro_paths,
            seen_paths,
            dynamic_binds,
            read_only=True,
            path_mapping=path_mapping,
        )
        self._bind_paths(
            repo_paths,
            seen_paths,
            dynamic_binds,
            read_only=True,
            path_mapping=path_mapping,
        )

        return dynamic_binds, seen_paths, active_venv_path

    def _add_system_binds(self, cmd, host_home, sandbox_tmp, cwd):
        """Add standard system and network-related binds to the command."""
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
                "--bind",
                str(sandbox_tmp),
                "/tmp",
                "--ro-bind-try",
                "/etc/machine-id",
                "/etc/machine-id",
                "--ro-bind-try",
                str(host_home / ".npm-global"),
                str(host_home / ".npm-global"),
                "--ro-bind-try",
                str(host_home / ".local/bin"),
                str(host_home / ".local/bin"),
                "--ro-bind-try",
                str(host_home / ".local/share/claude"),
                str(host_home / ".local/share/claude"),
                "--ro-bind-try",
                str(host_home / ".config/gh"),
                str(host_home / ".config/gh"),
                "--chdir",
                cwd,  # Use the specified or detected CWD in the sandbox
                "--unshare-all",
                "--share-net",
                "--die-with-parent",
                "--bind-try",
                str(host_home / ".local/share/Odoo"),
                str(host_home / ".local/share/Odoo"),
                "--ro-bind",
                "/etc/passwd",
                "/etc/passwd",
            ]
        )

    def _collect_secrets(self) -> list[tuple[str, str]]:
        """Retrieve and interactively prompt for AI API secrets."""
        relevant = {
            "claude": ["ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"],
            "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"],
            "copilot": [
                "GITHUB_TOKEN",
                "GH_TOKEN",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ],
            "openai": ["OPENAI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"],
        }
        to_process = relevant.get(self.cli, [])
        found_secrets: dict[str, str] = {}

        # 1. Host Environment Check
        for key in to_process:
            val = os.environ.get(key)
            if val:
                found_secrets[key] = val

        # 2. Local .env Check (if key missing from env)
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

        # 3. GitHub Token Fallback
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

        # 4. DataStore Check & Prompt
        from odev.common.store.datastore import DataStore

        ds = DataStore().secrets

        if self.yolo and not self.headless:
            logger.info("YOLO mode is enabled. All tool calls will be automatically approved.")

        for key in to_process:
            if key in found_secrets:
                continue

            try:
                is_opt = self.cli == "copilot" and key in [
                    "GH_TOKEN",
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                ]
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
            except Exception:
                pass

        # 5. Smart Aliasing (internal mappings for agent compatibility)
        aliased = False
        if "GOOGLE_API_KEY" in found_secrets and "GEMINI_API_KEY" not in found_secrets:
            found_secrets["GEMINI_API_KEY"] = found_secrets["GOOGLE_API_KEY"]
            aliased = "GEMINI_API_KEY"
        if "GEMINI_API_KEY" in found_secrets and "GOOGLE_API_KEY" not in found_secrets:
            found_secrets["GOOGLE_API_KEY"] = found_secrets["GEMINI_API_KEY"]
            aliased = "GOOGLE_API_KEY"

        if aliased and not self.headless:
            logger.info(
                f"Using {found_secrets.get('GOOGLE_API_KEY' if aliased == 'GEMINI_API_KEY' else 'GEMINI_API_KEY')[:10]}... (aliased to {aliased})"
            )

        if "GITHUB_TOKEN" in found_secrets and "GH_TOKEN" not in found_secrets:
            found_secrets["GH_TOKEN"] = found_secrets["GITHUB_TOKEN"]
        if "GH_TOKEN" in found_secrets and "GITHUB_TOKEN" not in found_secrets:
            found_secrets["GITHUB_TOKEN"] = found_secrets["GH_TOKEN"]

        return list(found_secrets.items())

    def _prepare_agent_config(
        self,
        playground: Path,
        all_candidate_paths: list[str],
        host_home: Path,
    ):
        """Create a sanitized agent configuration inside the sandbox playground."""
        # The playground becomes 'host_home' in the bwrap sandbox
        # We need to create .gemini, .config/gemini, .claude, etc.
        configs = {
            "gemini": [".gemini", ".config/gemini"],
            "claude": [".claude", ".config/claude"],
            "copilot": [".config/github-copilot"],
        }

        relevant_dirs = configs.get(self.cli, [])
        for rel_dir in relevant_dirs:
            target_dir = playground / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            # 1. Copy Credentials
            creds_files = [
                "gemini-credentials.json",
                "google_accounts.json",
                "claude-credentials.json",
                "hosts.json",
            ]
            for cf in creds_files:
                hcf = host_home / rel_dir / cf
                if hcf.exists():
                    shutil.copy2(hcf, target_dir / cf)

            # 2. Forge Workspace Trust
            # We explicitly trust all virtualized sandbox paths
            trusted_paths = [
                "/home/crupuk",
                "/skills",
                "/knowledge",
                "/custom",
                "/worktree",
                "/venvs",
                "/repositories",
                "/upgrade",
                "/dumps",
            ]
            # Add any other dynamic guest paths
            for d in all_candidate_paths:
                if ":" in d:
                    trusted_paths.append(d.split(":")[1])

            trust_data = {p: "TRUST_FOLDER" for p in sorted(set(trusted_paths))}

            try:
                (target_dir / "trustedFolders.json").write_text(json.dumps(trust_data, indent=2))

                # Also clear state/projects to prevent host-path leakage
                # Use correct minimal structures to prevent Node.js crashes
                structures = {
                    "projects.json": {"projects": {}},
                    "state.json": {},
                    "sessions.json": {"sessions": []},
                    "config.json": {},
                }
                for junk, structure in structures.items():
                    junk_file = target_dir / junk
                    if junk_file.exists():
                        junk_file.unlink()
                    # Pre-emptively create minimal valid ones
                    junk_file.write_text(json.dumps(structure))

            except Exception as e:
                logger.debug(f"Failed to write sanitized agent config: {e}")

    def _apply_final_bindings(
        self,
        cmd,
        agent_dirs,
        agent_files,
        dynamic_binds,
        sandbox_dirs,
        extra_bind_dirs,
        host_home,
        playground,
        path_mapping=None,
    ):
        """Apply all final agent-specific and workspace bindings."""
        # 1. Agent dirs & files
        for d in agent_dirs:
            cmd.extend(["--bind-try", str(d), str(d)])
        for f in agent_files:
            if f.exists():
                cmd.extend(["--bind-try", str(f), str(f)])

        # 2. Dynamic binds
        for src, dst, ro in dynamic_binds:
            try:
                # Only attempt to create directories in playground if dst is under host_home
                if str(dst).startswith(str(host_home)):
                    rel = dst.relative_to(host_home)
                    (playground / rel).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            cmd.extend(["--ro-bind-try" if ro else "--bind-try", str(src), str(dst)])

        # 3. Sandbox & Extra dirs
        for sdir in sorted(set(sandbox_dirs + (extra_bind_dirs or []))):
            if ":" in sdir:
                src, dst = sdir.split(":", 1)
                src, dst = Path(src).resolve(), Path(dst)
            else:
                src = Path(sdir).resolve()
                dst = src
            if not src.exists():
                continue
            try:
                # Only attempt to create directories in playground if dst is under host_home
                if str(dst).startswith(str(host_home)):
                    rel = dst.relative_to(host_home)
                    (playground / rel).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            cmd.extend(
                [
                    "--bind-try" if sdir in sandbox_dirs else "--ro-bind-try",
                    str(src),
                    str(dst),
                ]
            )

        # 4. Create compatibility symlinks for host paths in playground
        # This fixes 'gitdir' issues where Git metadata stores absolute host paths
        # that must be resolvable in the sandbox guest.
        if path_mapping:
            for host, guest in sorted(path_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                try:
                    host_p = Path(host).resolve()
                    if str(host_p).startswith(str(host_home)) and str(host_p) != str(guest):
                        rel_host = host_p.relative_to(host_home)
                        sym_path = playground / rel_host
                        if not sym_path.exists():
                            sym_path.parent.mkdir(parents=True, exist_ok=True)
                            sym_path.symlink_to(Path(guest))
                            logger.debug(f"Created compatibility symlink: {sym_path} -> {guest}")
                except Exception as e:
                    logger.debug(f"Failed to create compatibility symlink for {host}: {e}")

    def _execute_sandbox(
        self,
        cmd: list[str],
        agent_dirs: list[Path],
        agent_files: list[Path],
        dynamic_binds: list[tuple[Path, Path, bool]],
        extra_bind_dirs: list[str] | None,
        database: str | None,
        db_user: str | None,
        secrets_to_set: list[tuple[str, str]],
        pg_process: subprocess.Popen | None,
        sandbox_dirs: list[str],
        playground: Path,
        sandbox_tmp: Path,
        proxy_dir: Path,
        pg_data_dir: Path,
    ) -> bool:
        """Final execution logic for the bwrap sandbox."""
        if not self._display_sandbox_warning(
            sandbox_dirs=sandbox_dirs,
            agent_dirs=agent_dirs,
            agent_files=agent_files,
            dynamic_binds=dynamic_binds,
            extra_bind_dirs=extra_bind_dirs,
            database=database,
            db_user=db_user,
            ephemeral_pg=pg_process is not None or not database,
        ):
            return False

        if not self.headless:
            logger.info(f"Starting Project-wide AI execution ({self.cli})")

        from odev.common import bash

        # We use bash.run to grant the command raw TTY access, supporting TUIs natively
        # and ensuring output is correctly displayed.
        returncode = 0
        try:
            # To prevent secrets (like API keys) from appearing in process listings (ps),
            # we write the bwrap arguments to a temporary protected file and pass them
            # to bwrap via a file descriptor using shell redirection.
            with tempfile.NamedTemporaryFile(mode="wb", prefix="odev-ai-args-", delete=True) as f:
                os.chmod(f.name, 0o600)
                # bwrap --args expects null-separated arguments.
                # We ONLY put secrets in the FD to avoid "usage: bwrap" errors
                # on complex option sets (e.g. including --), while still keeping
                # secrets out of 'ps' listings.
                for key, val in secrets_to_set:
                    f.write(f"--setenv\0{key}\0{val}\0".encode())
                f.flush()

                # The rest of the arguments are passed on the command line.
                cmd_str = " ".join(shlex.quote(str(x)) for x in cmd[1:])
                # We use FD 3 for the arguments.
                # This indirection ensures that bwrap's argv only contains '--args 3'
                # followed by the quoted options/command.
                full_cmd = f"bwrap --args 3 {cmd_str} 3<{shlex.quote(f.name)}"
                logger.debug(f"Running sandbox command: {full_cmd}")

                from odev.common.console import console

                # Push content to scrollback to prevent loss if TUI clears screen
                console.print("\n" * (console.height or 20))

                # Use bash.run to grant the command raw TTY access, supporting TUIs natively.
                bash.run(full_cmd)
        except subprocess.CalledProcessError as error:
            returncode = error.returncode
        except Exception as e:
            logger.error(f"Failed to run {self.cli}: {e}")
            return False
        finally:
            # Kill ephemeral postgres if running
            if pg_process:
                try:
                    pg_process.terminate()
                    pg_process.wait(timeout=5)
                except Exception:
                    try:
                        pg_process.kill()
                    except Exception:
                        pass

            # Cleanup playground, sandbox_tmp and proxy_dir
            for path_to_clean in [playground, sandbox_tmp, proxy_dir, pg_data_dir]:
                try:
                    shutil.rmtree(path_to_clean)
                except Exception:
                    pass

        return returncode == 0

    def _rewrite_sandbox_config(self, sandbox_config_dir: Path, path_mapping: dict[str, str]):
        """Rewrite the odev.cfg inside the sandbox to use virtualized paths."""
        config_file = sandbox_config_dir / "odev.cfg"
        if not config_file.exists():
            return

        content = config_file.read_text()
        for host_path, guest_path in path_mapping.items():
            # Robustly handle trailing slashes and absolute paths
            h = str(Path(host_path).resolve())
            content = content.replace(h, guest_path)

        config_file.write_text(content)

        # 2. Rewrite symlinks in plugins directory
        # Host symlinks typically point to absolute host paths which are broken in the virtualized sandbox.
        plugins_dir = sandbox_config_dir / "plugins"
        if plugins_dir.exists():
            for plugin_link in plugins_dir.iterdir():
                if plugin_link.is_symlink():
                    try:
                        target = os.readlink(plugin_link)
                        # Rewriting targets if they match a host path in our mapping
                        for host_path, guest_path in sorted(
                            path_mapping.items(), key=lambda x: len(x[0]), reverse=True
                        ):
                            h = str(Path(host_path).resolve())
                            if target.startswith(h):
                                new_target = target.replace(h, guest_path)
                                # Re-create the symlink to point to the virtualized path
                                plugin_link.unlink()
                                plugin_link.symlink_to(new_target)
                                logger.debug(
                                    f"Virtualized plugin symlink {plugin_link.name!r}: {target} -> {new_target}"
                                )
                                break
                    except Exception as e:
                        logger.warning(f"Failed to virtualize plugin symlink {plugin_link.name}: {e}")

    def _clone_host_database(self, database: str, host_socket_dir: Path, ephemeral_socket_dir: Path) -> bool:
        """Clone host database data into the ephemeral cluster's existing database."""
        # Only log if the database name doesn't suggest a fresh upgrade DB to avoid confusion
        is_upgrade_db = database.endswith("_upgrade")
        if not is_upgrade_db:
            logger.info(f"Cloning host database data for {database!r} into ephemeral cluster...")
        else:
            logger.debug(f"Syncing fresh upgrade database {database!r} to sandbox")

        user = Path.home().name
        try:
            # 1. Pipe pg_dump from host to psql in ephemeral
            # We assume the database has already been created in the caller
            dump_cmd = [
                "pg_dump",
                "-h",
                str(host_socket_dir),
                "-d",
                database,
                "--no-owner",
                "--no-privileges",
            ]
            restore_cmd = [
                "psql",
                "-h",
                str(ephemeral_socket_dir),
                "-p",
                "5432",
                "-U",
                user,
                "-d",
                database,
            ]

            p1 = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p2 = subprocess.Popen(
                restore_cmd,
                stdin=p1.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            p1.stdout.close()
            _, stderr = p2.communicate()

            if p2.returncode != 0:
                if not is_upgrade_db:
                    logger.warning(
                        f"Failed to clone data for {database!r} (it might not exist on host): {stderr.decode()}"
                    )
                return False
            else:
                if not is_upgrade_db:
                    logger.info(f"Database {database!r} data successfully cloned.")
                return True

        except Exception as e:
            logger.debug(f"Failed to clone host database data: {e}")
            return False

    def _start_ephemeral_postgres(self, socket_dir: Path, data_dir: Path) -> subprocess.Popen | None:
        """Initialize and start an ephemeral PostgreSQL cluster."""
        import time

        try:
            if not self.headless:
                logger.info(f"Initializing ephemeral PostgreSQL cluster in {data_dir}")
            # Use current host user as superuser so psql works without -U
            user = Path.home().name
            subprocess.run(
                ["initdb", "-D", str(data_dir), "--nosync", "-U", user, "--auth=trust"],
                check=True,
                capture_output=True,
            )

            # Start postgres listening ONLY on unix socket in socket_dir
            # We use port 5432 so client tools work by default
            process = subprocess.Popen(
                [
                    "postgres",
                    "-D",
                    str(data_dir),
                    "-k",
                    str(socket_dir),
                    "-h",
                    "",
                    "-p",
                    "5432",
                    "-c",
                    "fsync=off",
                    "-c",
                    "full_page_writes=off",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for socket to be ready
            socket_path = socket_dir / ".s.PGSQL.5432"
            retries = 20
            while not socket_path.exists() and retries > 0:
                time.sleep(0.2)
                retries -= 1
                if process.poll() is not None:
                    logger.error("Ephemeral PostgreSQL failed to start")
                    return None

            if not socket_path.exists():
                logger.error("Timed out waiting for ephemeral PostgreSQL socket")
                process.terminate()
                return None

            # Create default database matching the user name so psql works without arguments
            try:
                subprocess.run(
                    [
                        "psql",
                        "-h",
                        str(socket_dir),
                        "-p",
                        "5432",
                        "-U",
                        user,
                        "-d",
                        "postgres",
                        "-c",
                        f'CREATE DATABASE "{user}";',
                    ],
                    check=True,
                    capture_output=True,
                )
                # Also create the 'odev' database requested by the user
                subprocess.run(
                    [
                        "psql",
                        "-h",
                        str(socket_dir),
                        "-p",
                        "5432",
                        "-U",
                        user,
                        "-d",
                        "postgres",
                        "-c",
                        'CREATE DATABASE "odev";',
                    ],
                    check=True,
                    capture_output=True,
                )
            except Exception as e:
                logger.debug(f"Failed to create default user database: {e}")

            if not self.headless:
                logger.info("Ephemeral PostgreSQL cluster is ready")
            return process
        except Exception as e:
            logger.error(f"Failed to start ephemeral PostgreSQL: {e}")
            return None

    def get_latest_session_id(self) -> str | None:
        """Attempt to find the latest session ID for the current CLI."""
        try:
            if self.cli == "gemini":
                # Use execute instead of run to capture output
                from odev.common import bash

                result = bash.execute("gemini --list-sessions")
                if result and result.stdout:
                    output = result.stdout.decode()
                    match = re.search(r"\[([a-f0-9-]{36})\]", output)
                    if match:
                        return match.group(1)
            elif self.cli == "claude":
                history_path = Path.home() / ".claude" / "history.jsonl"
                if history_path.exists():
                    import json

                    with open(history_path, "rb") as f:
                        # Read the last few lines to find the latest session
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        lines = f.read().decode(errors="ignore").splitlines()
                        for line in reversed(lines):
                            try:
                                data = json.loads(line)
                                if "sessionId" in data:
                                    return data["sessionId"]
                            except Exception:
                                continue
            elif self.cli == "copilot":
                # For now, return 'latest'
                return "latest"
            elif self.cli == "opencode-cli":
                return "latest"
        except Exception as e:
            logger.debug(f"Failed to get latest session ID for {self.cli}: {e}")
        return None
