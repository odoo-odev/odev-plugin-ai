"""Linux sandbox backend backed by bwrap (bubblewrap).

Translates an `ExecutionSpec` into a bwrap argv (with bind mounts, namespace
flags, env, and secrets injected via FD 3 to keep them out of `ps`), then
hands off to the configured shell via `bash.run` for raw-TTY support.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from odev.common.console import console
from odev.common.logging import logging

from .base import ExecutionSpec, Sandbox


logger = logging.getLogger(__name__)


def _check_bwrap_support() -> tuple[bool, str]:
    """Check if bwrap can run with unprivileged user namespaces.
    Specifically targets Ubuntu 24.04+ restrictions.
    """
    if not shutil.which("bwrap"):
        return False, (
            "bwrap is not installed. Install it via your package manager:\n"
            "  Debian/Ubuntu: sudo apt install bubblewrap\n"
            "  Fedora/RHEL:   sudo dnf install bubblewrap\n"
            "  Arch:          sudo pacman -S bubblewrap"
        )
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


class BwrapSandbox(Sandbox):
    """Linux sandbox backend using bwrap (bubblewrap)."""

    @classmethod
    def check_support(cls) -> tuple[bool, str]:
        return _check_bwrap_support()

    # --- profile / argv builders --------------------------------------------

    def _setup_chrome_wrapper(self, cmd: list[str], sandbox_tmp: Path) -> None:
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

    def _add_runtime_binds(self, cmd: list[str]) -> None:
        """Bind only necessary runtime sockets (IDE IPC) to the sandbox."""
        uid = os.getuid()
        runtime_dir = Path(f"/run/user/{uid}")
        if runtime_dir.exists():
            cmd.extend(["--dir", str(runtime_dir)])
            for socket in runtime_dir.glob("vscode-*.sock"):
                cmd.extend(["--bind-try", str(socket), str(socket)])
            for socket in runtime_dir.glob("antigravity-*.sock"):
                cmd.extend(["--bind-try", str(socket), str(socket)])
            for socket in runtime_dir.glob("cursor-*.sock"):
                cmd.extend(["--bind-try", str(socket), str(socket)])

    def _add_system_binds(self, cmd: list[str], host_home: Path, sandbox_tmp: Path, cwd: str) -> None:
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
                cwd,
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

    def _apply_final_bindings(
        self,
        cmd: list[str],
        agent_dirs: list[Path],
        agent_files: list[Path],
        final_binds: list[tuple[Path, Path, bool, bool]],
        host_home: Path,
        playground: Path,
    ) -> None:
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

        for src, dst, ro, _is_primary in final_binds:
            ensure_dst(dst, is_dir=src.is_dir())
            cmd.extend(["--ro-bind-try" if ro else "--bind-try", str(src), str(dst)])

    def _add_postgres_socket_binds(self, cmd: list[str], proxy_dir: Path) -> None:
        """Expose the ephemeral PG socket dir at /var/run/postgresql + /tmp symlink."""
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

    # --- main entry ----------------------------------------------------------

    def execute(self, spec: ExecutionSpec) -> bool:  # noqa: C901
        """Translate the spec into a bwrap argv and execute the agent."""
        host_home = Path.home().resolve()

        cmd: list[str] = [
            "bwrap",
            "--dir",
            "/home",
            "--dir",
            str(host_home),
            "--bind",
            str(spec.playground),
            str(host_home),
        ]

        # Env (Linux gets XDG_RUNTIME_DIR for IDE IPC)
        env = dict(spec.env)
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        for key, val in env.items():
            cmd.extend(["--setenv", key, val])

        # Pre-create top-level dirs needed for arbitrary binds (e.g. /opt, /srv)
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
        for _, dst, _, _ in spec.final_binds:
            if dst.is_absolute():
                td = f"/{dst.parts[1]}"
                if td not in top_dirs:
                    cmd.extend(["--dir", td])
                    top_dirs.add(td)

        self._prepare_odev_config(spec.playground, host_home)
        self._add_system_binds(cmd, host_home, spec.sandbox_tmp, spec.cwd)

        # Postgres socket exposure (proxy_dir is created and populated by PostgresSandbox)
        if spec.pg_process is not None:
            self._add_postgres_socket_binds(cmd, spec.proxy_dir)

        self._apply_final_bindings(cmd, spec.agent_dirs, spec.agent_files, spec.final_binds, host_home, spec.playground)
        self._prepare_agent_config(
            spec.playground,
            [str(dst) for src, dst, _, _ in spec.final_binds if src != host_home],
            host_home,
        )

        cmd.extend(["--", *spec.agent_cmd])

        if not self._display_sandbox_warning(
            binds=spec.final_binds,
            agent_dirs=spec.agent_dirs,
            agent_files=spec.agent_files,
            database=spec.database,
            db_user=spec.db_user,
            ephemeral_pg=spec.pg_process is not None,
        ):
            self._terminate_pg(spec.pg_process)
            self._cleanup_paths([spec.playground, spec.sandbox_tmp, spec.proxy_dir, spec.pg_data_dir])
            return False

        if not self.headless:
            logger.info(f"Starting Project-wide AI execution ({self.cli})")

        from odev.common import bash

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
                for key, val in spec.secrets:
                    f.write(f"--setenv\0{key}\0{val}\0".encode())
                f.flush()

                cmd_str = " ".join(shlex.quote(str(x)) for x in cmd[1:])
                # We use FD 3 for the arguments.
                full_cmd = f"bwrap --args 3 {cmd_str} 3<{shlex.quote(f.name)}"
                logger.debug(f"Running sandbox command: {full_cmd}")

                bash.run(full_cmd)
        except subprocess.CalledProcessError as error:
            returncode = error.returncode
            supported, message = _check_bwrap_support()
            if not supported:
                console.print(f"\n[bold red]Error:[/] {message}")
                returncode = 1
        except Exception as e:
            logger.error(f"Failed to run {self.cli}: {e}")
            returncode = 1
        finally:
            self._terminate_pg(spec.pg_process)
            self._cleanup_paths([spec.playground, spec.sandbox_tmp, spec.proxy_dir, spec.pg_data_dir])

        return returncode == 0


__all__ = ["BwrapSandbox"]
