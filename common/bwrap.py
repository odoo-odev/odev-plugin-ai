import json
import os
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


class BwrapSandbox(OdevFrameworkMixin):
    """Manages a bwrap execution environment."""

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
        self.yolo = yolo or headless

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

    def _add_bind(
        self,
        bind_pool: dict[Path, tuple[Path, bool, bool]],
        src: Path | str,
        dst: Path | str | None = None,
        ro: bool = True,
        is_primary: bool = False,
    ):
        """Smartly add a bind to the pool, handling overrides and duplicates.

        Preference: Read-Write (ro=False) over Read-Only (ro=True).
        """
        src = Path(src).resolve()
        if not src.exists():
            return
        dst = Path(dst) if dst else src

        if dst in bind_pool:
            e_src, e_ro, e_primary = bind_pool[dst]
            # Respect higher permission (RW) and Primary status
            new_ro = ro and e_ro
            new_primary = is_primary or e_primary

            # Only log if something actually changes
            if (new_ro != e_ro) or (new_primary != e_primary) or (src != e_src):
                mode = "RW" if not new_ro else "RO"
                logger.debug(f"Updating binding {dst} (Mode: {mode}, Primary: {new_primary})")

            bind_pool[dst] = (src, new_ro, new_primary)
        else:
            bind_pool[dst] = (src, ro, is_primary)

    def _display_sandbox_warning(
        self,
        sandbox_dirs: list[str],
        agent_dirs: list[Path],
        agent_files: list[Path],
        dynamic_binds: list[tuple[Path, Path, bool]],
        extra_bind_dirs: list[str] | None = None,
        database: str | None = None,
        db_user: str | None = None,
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
            console.print(" • [color.green]Isolating (Empty ephemeral cluster, no database copied)[/color.green]")

        console.print("\n[bold color.cyan]INFRASTRUCTURE & REFERENCE (System/Source/Config):[/bold color.cyan]")
        # Group similar binds
        important_binds = []
        for adir in agent_dirs:
            important_binds.append(f"{adir} (RW)")
        for f in agent_files:
            if f.exists():
                important_binds.append(f"{f} (RW)")

        dynamic_binds_typed: list[tuple[Path, Path, bool, bool]] = dynamic_binds  # type: ignore
        for src_path, dst_path, ro_bool, primary_bool in dynamic_binds_typed:
            mode = "RW" if not ro_bool else "RO"
            if str(src_path) == str(dst_path):
                important_binds.append(f"{src_path} ({mode})")
            else:
                important_binds.append(f"{src_path} [bold color.green]-> {dst_path}[/bold color.green] ({mode})")

        # Deduplicate and sort
        for bind in sorted(set(important_binds)):
            console.print(f" • {bind}")

        if not self.yolo and not console.bypass_prompt:
            return console.confirm("Do you want to proceed with this AI agent execution?", default=True)
        return True

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
                            for host, guest in sorted(
                                path_mapping.items(),
                                key=lambda x: len(x[0]),
                                reverse=True,
                            ):
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

    def _prepare_sandbox_config(
        self,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None,
        database: str | None,
        version: str | None,
        path_mapping: dict[str, str],
        host_home: Path,
    ) -> dict:
        """Centralized and 'smart' sandbox binding discovery."""
        from odev.common.databases.local import LocalDatabase

        # 1. Collect all candidates
        # We use maps to track the best intent for each destination: dst -> (src, ro, is_primary)
        candidates: dict[Path, tuple[Path, bool, bool]] = {}

        def _add_candidate(src, dst=None, ro=True, is_primary=False):
            src_p = Path(src).resolve()
            if not src_p.exists():
                return
            dst_p = Path(dst) if dst else src_p
            if dst_p in candidates:
                e_src, e_ro, e_primary = candidates[dst_p]
                # Prioritize: RW > RO, Primary > Non-Primary
                ro = ro and e_ro
                is_primary = is_primary or e_primary
            candidates[dst_p] = (src_p, ro, is_primary)

        # A. Explicit Binds (User/Command)
        for i, sdir in enumerate(sandbox_dirs):
            if ":" in sdir:
                src, dst = sdir.split(":", 1)
            else:
                src = sdir
                dst = sdir
                # Automatically map the primary working directory to /custom if not explicitly mapped
                if i == 0:
                    dst = "/custom"
                    path_mapping[str(Path(src).resolve())] = dst
            _add_candidate(src, dst, ro=False, is_primary=True)
            if dst == "/custom":
                _add_candidate(src, src, ro=False, is_primary=True)

        for edir in extra_bind_dirs or []:
            src, dst = edir.split(":", 1) if ":" in edir else (edir, edir)
            _add_candidate(src, dst, ro=True, is_primary=False)

        # B. Common Odev Containers (Base for hierarchy checks)
        ro_containers = {
            self.odev.home_path / "worktrees",
            self.odev.home_path / "plugins",
        }
        rw_containers = {self.odev.home_path / "virtualenvs"}
        repo_containers = {Path(r.path).resolve() for r in odoo_repositories(enterprise=True)}

        def _is_covered(p: Path) -> bool:
            p = p.resolve()
            for container in ro_containers | rw_containers | repo_containers:
                c_p = container.resolve()
                if p == c_p:
                    return True
                try:
                    p.relative_to(c_p)
                    return True
                except ValueError:
                    continue
            return False

        # C. Plugin Containers & Skills Mapping
        for plugin in self.odev.plugins:
            try:
                res = plugin.path.resolve()
                if res.exists():
                    ro_containers.add(res.parent)
                # Automatically mount all skill subdirectories to /skills/*
                sp = res / "skills"
                if sp.exists() and sp.is_dir():
                    for skill_pkg in sp.iterdir():
                        if skill_pkg.is_dir():
                            dest = f"/skills/{skill_pkg.name}"
                            if Path(dest) not in candidates:
                                _add_candidate(skill_pkg, dest, ro=True)
                                path_mapping[str(skill_pkg)] = dest
            except Exception:
                pass

        # D. Dynamic Discovery
        try:
            up = self.odev.config.paths.upgrade.resolve()
            if up.exists() and Path("/upgrade") not in candidates:
                _add_candidate(up, "/upgrade", ro=True)
                path_mapping[str(up)] = "/upgrade"
        except Exception:
            pass

        odev_path = self.odev.path
        _add_candidate(odev_path, odev_path, ro=True)

        active_venv_path: Path | None = None

        rtk = shutil.which("rtk")
        if rtk:
            rp = Path(rtk).resolve()
            _add_candidate(rp, self._map_path(rp, path_mapping), ro=True)

        # E. Add the Infrastructure Containers
        for p in ro_containers:
            _add_candidate(p, self._map_path(p, path_mapping), ro=True)
        for p in rw_containers:
            _add_candidate(p, self._map_path(p, path_mapping), ro=False)
        for p in repo_containers:
            _add_candidate(p, self._map_path(p, path_mapping), ro=True)

        # 2. Binding Phase (Call _add_bind exactly once per path)
        pool: dict[Path, tuple[Path, bool, bool]] = {}
        for dst_key, (src_val, ro_val, primary_val) in candidates.items():
            self._add_bind(pool, src_val, dst_key, ro=ro_val, is_primary=primary_val)

        # 3. Final Nesting Deduction
        final_pool: list[tuple[Path, Path, bool, bool]] = []
        sorted_dsts = sorted(pool.keys(), key=lambda d: len(d.parts))
        for dst_path in sorted_dsts:
            src_path, ro_bool, primary_bool = pool[dst_path]
            is_redundant = False
            for e_src, e_dst, e_ro, e_primary in final_pool:
                try:
                    rel_src = src_path.relative_to(e_src)
                    rel_dst = dst_path.relative_to(e_dst)
                    if rel_src == rel_dst:
                        if not e_ro or ro_bool:
                            is_redundant = True
                            break
                except (ValueError, AttributeError):
                    continue
            if not is_redundant:
                final_pool.append((src_path, dst_path, ro_bool, primary_bool))
            else:
                logger.debug(f"Skipping redundant binding: {dst_path} (already covered by parent mount)")

        return {
            "binds": final_pool,
            "active_venv_path": active_venv_path,
        }

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
        final_binds,
        host_home,
        playground,
        path_mapping=None,
    ):
        """Apply all final agent-specific and workspace bindings."""
        # 1. Agent dirs & files first (usually ~/.local ~/.cache)
        for d in agent_dirs:
            cmd.extend(["--bind-try", str(d), str(d)])
        for f in agent_files:
            if f.exists():
                cmd.extend(["--bind-try", str(f), str(f)])

        # 2. Apply the sorted binds (parents before children)
        # final_binds is already sorted by depth from _prepare_sandbox_config
        for src, dst, ro, is_primary in final_binds:
            try:
                # Only attempt to create directories in playground if dst is under host_home
                if str(dst).startswith(str(host_home)):
                    rel = dst.relative_to(host_home)
                    (playground / rel).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            cmd.extend(["--ro-bind-try" if ro else "--bind-try", str(src), str(dst)])

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
        final_binds: list[tuple[Path, Path, bool, bool]],
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
        timeout: int | None = None,
    ) -> bool:
        """Final execution logic for the bwrap sandbox."""
        if not self._display_sandbox_warning(
            sandbox_dirs=sandbox_dirs,
            agent_dirs=agent_dirs,
            agent_files=agent_files,
            dynamic_binds=[(s, d, r) for s, d, r, p in final_binds],
            extra_bind_dirs=extra_bind_dirs,
            database=database,
            db_user=db_user,
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

                # Push content to scrollback to prevent loss if TUI clears screen
                if not self.headless:
                    from odev.common.console import console

                    console.print("\n" * 20)

                # Use bash.run to grant the command raw TTY access, supporting TUIs natively.
                bash.run(full_cmd, timeout=timeout)
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
