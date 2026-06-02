import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from odev.common.console import console
from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin
from odev.common.odoobin import odoo_repositories
from odev.common.python import PythonEnv
from odev.common.version import OdooVersion


logger = logging.getLogger(__name__)


def _check_bwrap_support():
    """Check if bwrap can run with unprivileged user namespaces.
    Specifically targets Ubuntu 24.04+ restrictions.
    """
    try:
        # Try a minimal bwrap command that requires user namespaces
        subprocess.run(
            ["bwrap", "--unshare-user", "--version"],
            capture_output=True,
            check=True,
            timeout=2,
        )
        return True, ""
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        stderr = getattr(e, "stderr", b"").decode()
        if "Permission denied" in stderr or "setting up uid map" in stderr:
            # Check for Ubuntu-specific sysctl
            apparmor_restrict_path = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
            if apparmor_restrict_path.exists() and apparmor_restrict_path.read_text().strip() == "1":
                return False, (
                    "Your system (Ubuntu 24.04+) is restricting unprivileged user namespaces via AppArmor.\n"
                    "This prevents the AI sandbox from starting.\n\n"
                    "To fix this, run:\n"
                    "  sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0\n\n"
                    "To make it permanent:\n"
                    "  echo 'kernel.apparmor_restrict_unprivileged_userns = 0' | sudo tee /etc/sysctl.d/60-apparmor-namespace.conf"
                )
        return False, f"Sandbox initialization failed: {stderr or str(e)}"


class BwrapSandbox(OdevFrameworkMixin):
    """Manages a bwrap execution environment."""

    @staticmethod
    def _group_infra_items(items: list[tuple["Path", str]]) -> list[tuple[str, str]]:
        """Sort and group infra paths by parent+mode for a compact display.

        Paths that share the same parent directory *and* the same access mode
        are collapsed into a single line:

            /some/parent/{child_a, child_b} (RO)

        Singletons are shown as-is. The final list is sorted alphabetically.
        """
        from collections import defaultdict

        home = str(Path.home())

        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for path, mode in items:
            parent_str = str(path.parent)
            if parent_str == home:
                parent_str = "~"
            elif parent_str.startswith(home + "/"):
                parent_str = "~" + parent_str[len(home) :]

            groups[(parent_str, mode)].append(path.name)

        result: list[tuple[str, str]] = []
        for (parent, mode), names in groups.items():
            names_sorted = sorted(set(names))
            if len(names_sorted) == 1:
                path_str = f"{parent}/{names_sorted[0]}"
            else:
                path_str = f"{parent}/{{{', '.join(names_sorted)}}}"
            result.append((path_str, mode))

        return sorted(result, key=lambda x: x[0])

    def _build_infra_items(
        self,
        binds: list[tuple[Path, Path, bool, bool]],
        agent_dirs: list[Path],
        agent_files: list[Path],
        active_venv_path: Path | None,
        odoo_filestore: Path | None,
        string,
    ) -> tuple[list[tuple[Path, str]], list[str]]:
        """Build (infra_items, mapping_lines) for the infrastructure display section."""
        infra_items: list[tuple[Path, str]] = []
        mapping_lines: list[str] = []

        for d in agent_dirs:
            infra_items.append((d, "RW"))
        for f in agent_files:
            if f.exists():
                infra_items.append((f, "RW"))
        if odoo_filestore and odoo_filestore.exists():
            infra_items.append((odoo_filestore, "RW"))
        if active_venv_path:
            infra_items.append((active_venv_path, "RO"))

        for src, dst, ro, primary in binds:
            if not primary:
                mode = "RO" if ro else "RW"
                if src == dst:
                    infra_items.append((src, mode))
                else:
                    mapping_lines.append(f"{src} {string.stylize(f'-> {dst}', 'bold color.green')} ({mode})")

        return infra_items, mapping_lines

    def __init__(
        self,
        cli: str,
        handler,
        model: str = "auto",
        yolo: bool = False,
        headless: bool = False,
    ):
        super().__init__()
        self.cli = cli
        self.handler = handler
        self.model = model
        self.headless = headless
        self.yolo = yolo or headless

    def _display_sandbox_warning(
        self,
        binds: list[tuple[Path, Path, bool, bool]],
        agent_dirs: list[Path],
        agent_files: list[Path],
        database: str | None = None,
        db_user: str | None = None,
        ephemeral_pg: bool = True,
        active_venv_path: Path | None = None,
        odoo_filestore: Path | None = None,
        primary_dirs: list[Path] | None = None,
    ) -> bool:
        """Display a warning message about the sandbox access and security risks."""
        if self.headless:
            return True

        from odev.common import string

        console.rule(string.stylize("AI SANDBOX SECURITY WARNING", "bold color.red"), style="color.red")

        console.print(
            f"\n{string.stylize('WARNING:', 'bold color.yellow')} You are running an AI agent in a sandboxed environment."
        )
        console.print("The agent can read/write files and access the database within this sandbox.")

        console.print(f"\n{string.stylize('PRIMARY WORKSPACES (Read-Write Access):', 'bold color.cyan')}")
        primary_binds = [b for b in binds if b[3]]
        if primary_dirs:
            # Sort primary binds so that the ones in primary_dirs come first, in their original order
            primary_paths = [p.resolve() for p in primary_dirs]

            def sort_key(b):
                try:
                    return primary_paths.index(b[0])
                except ValueError:
                    return len(primary_paths)

            primary_binds.sort(key=sort_key)

        for src, dst, _ro, _primary in primary_binds:
            label = f"{src} {string.stylize(f'-> {dst}', 'bold color.green')}" if src != dst else str(src)
            console.print(f" • {label}")

        console.print(f"\n{string.stylize('DATABASE ACCESS:', 'bold color.cyan')}")
        if database:
            console.print(f" • Database: {string.stylize(database, 'color.purple')}")
            console.print(f" • User:     {string.stylize(db_user or 'default', 'color.purple')}")
        else:
            console.print(
                f" • {string.stylize('Isolating (Empty ephemeral cluster, no database copied)', 'color.green')}"
            )

        if database and not ephemeral_pg:
            warning = string.stylize("WARNING:", "bold color.red")
            host = string.stylize("HOST", "bold")
            console.print(f"\n{warning} You are granting access to your {host} PostgreSQL cluster.")
            console.print(
                f"The agent will be able to see and potentially access {string.stylize('ALL', 'bold')} your local databases."
            )

        infra_items, mapping_lines = self._build_infra_items(
            binds, agent_dirs, agent_files, active_venv_path, odoo_filestore, string
        )

        console.print(f"\n{string.stylize('INFRASTRUCTURE & REFERENCE (System/Source/Config):', 'bold color.cyan')}")
        for path, mode in self._group_infra_items(infra_items):
            console.print(f" • {string.stylize(path, 'color.purple')} ({mode})")

        home = str(Path.home())
        for line in sorted(mapping_lines):
            # Also replace home in mapping lines if present
            if line.startswith(home):
                line = "~" + line[len(home) :]
            console.print(f" • {string.stylize(line, 'color.purple')}")

        if not self.yolo and not console.bypass_prompt:
            return console.confirm("Do you want to proceed with this AI agent execution?", default=True)
        return True

    def _prepare_odev_config(self, playground, host_home):
        """Create a sandboxed copy of the odev configuration."""
        config_dir = host_home / ".config" / "odev"
        if config_dir.exists():
            sandbox_config_dir = playground / ".config" / "odev"
            sandbox_config_dir.mkdir(parents=True, exist_ok=True)
            for f in config_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, sandbox_config_dir / f.name)

    def _resolve_sandbox_dirs(self, sandbox_dirs: list[str]) -> list[tuple[Path, Path]]:
        """Parse sandbox_dirs entries into (host, guest) pairs."""
        return [(Path(s).resolve(), Path(s).resolve()) for s in sandbox_dirs]

    def _prepare_sandbox_config(
        self,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None,
        database: str | None,
        version: str | None,
    ) -> dict:
        """Build the flat list of sandbox bindings across the 3 binding categories."""

        def bind(src, dst=None, ro=True, primary=False):
            p = Path(src).resolve()
            if not p.exists():
                return None
            return (p, Path(dst) if dst else p, ro, primary)

        effective_sandbox_binds = self._resolve_sandbox_dirs(sandbox_dirs)

        binds = list(
            filter(
                None,
                [
                    # Type B — user workspace (primary, RW)
                    *[bind(host, guest, ro=False, primary=True) for host, guest in effective_sandbox_binds],
                    # Type B — extra dirs provided by the caller (RW, now Primary)
                    *[bind(e, ro=False, primary=True) for e in (extra_bind_dirs or [])],
                    # Type A — odev infrastructure (parents are mounted before children, no dedup)
                    bind(self.odev.path),
                    bind(self.odev.plugins_path),
                    # Resolve plugin symlink targets so Python can import them inside bwrap.
                    # plugins_path contains symlinks (e.g. odev_plugin_ai -> /path/to/repo);
                    # the symlink itself is visible via the plugins_path mount, but the target
                    # directory must be separately mounted for Python imports to work.
                    *[bind(p) for p in self.odev.plugins_path.iterdir() if p.is_symlink()],
                    # RW access is required for odev to perform git operations/worktree management
                    bind(self.odev.home_path / "worktrees", ro=False),
                    bind(self.odev.home_path / "virtualenvs", ro=False),
                    bind(sys.prefix),
                    *[bind(r.path, ro=False) for r in odoo_repositories(enterprise=True)],
                ],
            )
        )

        # Sort binds by destination path length (ascending) to ensure parents are mounted before children.
        # For same length (e.g. same path), we ensure RW (ro=False) comes after RO (ro=True)
        # so that the RW mount wins in bwrap.
        binds.sort(key=lambda b: (len(b[1].parts), not b[2]))

        return {"binds": binds, "active_venv_path": self._resolve_active_venv(database, version)}

    def _resolve_active_venv(self, database: str | None, version: str | None) -> Path | None:
        """Return the active virtualenv path (used to prepend $PATH), or None."""
        from odev.common.databases.local import LocalDatabase

        if database:
            db = LocalDatabase(database)
            if db.exists:
                p = Path(db.venv.path).resolve()
                if p.exists():
                    return p
                if not version:
                    version = str(db.version)

        if version:
            p = Path(PythonEnv(str(OdooVersion(version))).path).resolve()
            if p.exists():
                return p

        return None

    def _setup_chrome_wrapper(self, cmd, sandbox_tmp):
        """Create a Chrome wrapper script in /tmp that injects rendering-consistency
        flags, then point ODOO_BROWSER_BIN at it so Odoo picks it up."""
        wrapper = sandbox_tmp / "odoo-chrome-wrapper"
        wrapper.write_text(
            "#!/bin/bash\n"
            "for bin in google-chrome chromium chromium-browser google-chrome-stable; do\n"
            '    real=$(command -v "$bin" 2>/dev/null)\n'
            '    if [ -n "$real" ]; then\n'
            '        exec "$real" \\\n'
            "            --font-render-hinting=none \\\n"
            "            --force-device-scale-factor=1 \\\n"
            "            --disable-font-subpixel-positioning \\\n"
            "            --hide-scrollbars \\\n"
            "            --window-size=1366,768 \\\n"
            '            "$@"\n'
            "    fi\n"
            "done\n"
            'echo "Chrome not found" >&2\n'
            "exit 1\n"
        )
        wrapper.chmod(0o755)
        cmd.extend(["--setenv", "ODOO_BROWSER_BIN", "/tmp/odoo-chrome-wrapper"])

    def _add_system_binds(self, cmd, host_home, sandbox_tmp, cwd):
        """Add standard system and network-related binds to the command."""
        self._setup_chrome_wrapper(cmd, sandbox_tmp)
        self._add_runtime_binds(cmd)
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
                "--ro-bind",
                "/etc/alternatives",
                "/etc/alternatives",
                "--ro-bind-try",
                "/opt/google/chrome",
                "/opt/google/chrome",
                "--ro-bind-try",
                "/snap",
                "/snap",
                "--ro-bind-try",
                "/etc/fonts",
                "/etc/fonts",
                "--ro-bind-try",
                str(host_home / ".fonts"),
                str(host_home / ".fonts"),
                "--ro-bind-try",
                str(host_home / ".local/share/fonts"),
                str(host_home / ".local/share/fonts"),
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
            ]
        )
        if "DISPLAY" in os.environ:
            cmd.extend(["--bind-try", "/tmp/.X11-unix", "/tmp/.X11-unix"])
        cmd.extend(
            [
                "--ro-bind-try",
                "/etc/machine-id",
                "/etc/machine-id",
                "--ro-bind-try",
                "/etc/ld.so.cache",
                "/etc/ld.so.cache",
                "--ro-bind-try",
                "/etc/mime.types",
                "/etc/mime.types",
                "--ro-bind-try",
                str(host_home / ".npm-global"),
                str(host_home / ".npm-global"),
                "--ro-bind-try",
                str(host_home / ".local/bin"),
                str(host_home / ".local/bin"),
                "--chdir",
                cwd,  # Use the specified or detected CWD in the sandbox
                "--unshare-all",
                "--share-net",
                "--die-with-parent",
                "--bind-try",
                str(host_home / ".local/share/Odoo"),
                str(host_home / ".local/share/Odoo"),
                "--ro-bind-try",
                str(host_home / ".local/share/claude"),
                str(host_home / ".local/share/claude"),
                "--ro-bind",
                "/etc/passwd",
                "/etc/passwd",
            ]
        )

    def _add_runtime_binds(self, cmd):
        """Bind only necessary runtime sockets (IDE IPC, Display servers) to the sandbox."""
        uid = os.getuid()
        runtime_dir = Path(f"/run/user/{uid}")
        host_home = Path.home().resolve()
        if runtime_dir.exists():
            # Create the runtime directory in the sandbox
            cmd.extend(["--dir", str(runtime_dir)])
            # Bind IDE sockets (vscode, cursor, antigravity)
            # Antigravity often uses vscode-*.sock for compatibility
            for socket in runtime_dir.glob("vscode-*.sock"):
                cmd.extend(["--bind-try", str(socket), str(socket)])
            for socket in runtime_dir.glob("antigravity-*.sock"):
                cmd.extend(["--bind-try", str(socket), str(socket)])
            for socket in runtime_dir.glob("cursor-*.sock"):
                cmd.extend(["--bind-try", str(socket), str(socket)])

            # Bind Wayland display socket if available
            wayland_display = os.environ.get("WAYLAND_DISPLAY")
            if wayland_display:
                wayland_socket = runtime_dir / wayland_display
                if wayland_socket.exists():
                    cmd.extend(["--bind-try", str(wayland_socket), str(wayland_socket)])
                    cmd.extend(["--setenv", "WAYLAND_DISPLAY", wayland_display])

        # Bind X11 authority and environment variables if DISPLAY is available
        if "DISPLAY" in os.environ:
            cmd.extend(["--setenv", "DISPLAY", os.environ["DISPLAY"]])

            xauth = os.environ.get("XAUTHORITY")
            if not xauth:
                default_xauth = host_home / ".Xauthority"
                if default_xauth.exists():
                    xauth = str(default_xauth)

            if xauth:
                xauth_path = Path(xauth)
                if xauth_path.exists():
                    parent_dir = xauth_path.parent
                    cmd.extend(["--dir", str(parent_dir)])
                    cmd.extend(["--ro-bind-try", str(xauth_path), str(xauth_path)])
                    cmd.extend(["--setenv", "XAUTHORITY", str(xauth_path)])

    def _prepare_agent_config(  # noqa: C901
        self,
        playground: Path,
        all_candidate_paths: list[str],
        host_home: Path,
    ):
        """Create a sanitized agent configuration inside the sandbox playground."""
        relevant_dirs = self.handler.get_config_dirs()
        persistent_dirs = self.handler.get_persistent_dirs()

        for rel_dir in relevant_dirs:
            is_persistent = rel_dir in persistent_dirs
            target_dir = (host_home / rel_dir) if is_persistent else (playground / rel_dir)
            if not is_persistent:
                target_dir.mkdir(parents=True, exist_ok=True)
            self._copy_agent_credentials(target_dir, rel_dir, is_persistent, host_home)

        # Copy global CLI config file (e.g. ~/.claude.json)
        if gcn := self.handler.get_global_config_name():
            is_covered = any(gcn == pd or gcn.startswith(pd + "/") for pd in persistent_dirs)
            if not is_covered:
                src = host_home / gcn
                if src.exists():
                    shutil.copy2(src, playground / gcn)

        trusted_paths = [str(host_home), "/knowledge", str(self.odev.home_path / "worktrees"), "/upgrade"]
        for d in all_candidate_paths:
            if ":" in d:
                trusted_paths.append(d.split(":")[1])

        if rel_dir := self.handler.get_agent_config_rel_path():
            is_persistent = rel_dir in persistent_dirs
            target_dir = (host_home / rel_dir) if is_persistent else (playground / rel_dir)
            if not is_persistent:
                target_dir.mkdir(parents=True, exist_ok=True)

            self.handler.inject_trust(target_dir, trusted_paths)
            self.handler.cleanup_junk(target_dir)

    def _copy_agent_credentials(self, target_dir, rel_dir, is_persistent, host_home):
        """Copy credentials from host to playground if not persistent."""
        if is_persistent:
            return
        creds_files = self.handler.get_creds_files()
        for cf in creds_files:
            hcf = host_home / rel_dir / cf
            if hcf.exists():
                shutil.copy2(hcf, target_dir / cf)

    def _apply_final_bindings(
        self,
        cmd,
        agent_dirs,
        agent_files,
        final_binds,
        host_home,
        playground,
    ):
        """Apply all final agent-specific and workspace bindings."""

        def ensure_dst(dst, is_dir=True):
            try:
                if str(dst).startswith(str(host_home)):
                    rel = dst.relative_to(host_home)
                    dst_in_playground = playground / rel
                    if is_dir:
                        dst_in_playground.mkdir(parents=True, exist_ok=True)
                    else:
                        dst_in_playground.parent.mkdir(parents=True, exist_ok=True)
                        dst_in_playground.touch(exist_ok=True)
            except Exception as e:
                logger.debug(f"Could not pre-create destination path for {dst}: {e}")

        for d in agent_dirs:
            ensure_dst(d, is_dir=True)
            cmd.extend(["--bind-try", str(d), str(d)])
        for f in agent_files:
            if f.exists():
                ensure_dst(f, is_dir=False)
                cmd.extend(["--bind-try", str(f), str(f)])

        # final_binds is sorted by depth in _prepare_sandbox_config to ensure correct mount order.
        for src, dst, ro, _is_primary in final_binds:
            ensure_dst(dst, is_dir=src.is_dir())
            cmd.extend(["--ro-bind-try" if ro else "--bind-try", str(src), str(dst)])

    def _execute_sandbox(
        self,
        cmd: list[str],
        agent_dirs: list[Path],
        agent_files: list[Path],
        final_binds: list[tuple[Path, Path, bool, bool]],
        database: str | None,
        db_user: str | None,
        secrets_to_set: list[tuple[str, str]],
        pg_process: subprocess.Popen | None,
        playground: Path,
        sandbox_tmp: Path,
        proxy_dir: Path,
        pg_data_dir: Path,
        active_venv_path: Path | None = None,
        odoo_filestore: Path | None = None,
        primary_dirs: list[Path] | None = None,
    ) -> bool:
        """Final execution logic for the bwrap sandbox."""
        if not self._display_sandbox_warning(
            binds=final_binds,
            agent_dirs=agent_dirs,
            agent_files=agent_files,
            database=database,
            db_user=db_user,
            ephemeral_pg=pg_process is not None,
            active_venv_path=active_venv_path,
            odoo_filestore=odoo_filestore,
            primary_dirs=primary_dirs,
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

                # Use bash.run to grant the command raw TTY access, supporting TUIs natively.
                bash.run(full_cmd)
        except subprocess.CalledProcessError as error:
            returncode = error.returncode
            # If bwrap failed, run diagnostic
            supported, message = _check_bwrap_support()
            if not supported:
                console.print(f"\n[bold red]Error:[/] {message}")
                return False
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
