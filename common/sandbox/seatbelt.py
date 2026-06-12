"""macOS sandbox backend backed by Seatbelt (`sandbox-exec`).

Generates an SBPL profile from the `ExecutionSpec` binds and runs the agent
through `sandbox-exec -f profile.sb`. Secrets are passed via a 0600 env file
sourced by a tiny launcher script, mirroring how the bwrap backend keeps
secrets out of `ps`.

Profile model: PERMISSIVE BASELINE WITH TARGETED DENIES (matches the model
used by Codex CLI / Claude Code on macOS). See `_build_profile` for the
exact policy.

Important caveats vs. bwrap:
- Seatbelt cannot HIDE files; it can only deny operations. Non-allowed paths
  return EPERM, but their existence/metadata may still be observable through
  parent-directory listings.
- There is no namespace unsharing: there's a single host filesystem with
  per-process MAC checks.
- A strict (deny default) allowlist for non-file primitives is impractical:
  Cocoa/Foundation apps deadlock silently on `mach_msg` to system daemons
  rather than failing fast. Non-file ops are therefore allowed broadly; the
  security boundary is enforced via file-write* deny rules.
"""

import os
import shlex
import stat
import subprocess
from pathlib import Path

from odev.common.console import console
from odev.common.logging import logging

from .base import ExecutionSpec, Sandbox


logger = logging.getLogger(__name__)


SANDBOX_EXEC = "/usr/bin/sandbox-exec"


# ---------------------------------------------------------------------------
# Sandbox policy — keep these lists self-explanatory; they ARE the policy.
# ---------------------------------------------------------------------------

#: Absolute paths the agent must never modify. Modifying any of these would
#: compromise the host OS (system binaries, frameworks, package managers).
DENY_SYSTEM_SUBPATHS: tuple[str, ...] = (
    "/System",
    "/Library",
    "/usr",
    "/bin",
    "/sbin",
    "/Applications",
    "/private/etc",
    "/private/var/db",
    "/private/var/log",
    "/opt",
)

#: Paths under $HOME that hold credentials, browsing history, or private
#: communications. Resolved relative to the running user's home dir.
#:
#: NOTE: ~/Library/Keychains is INTENTIONALLY NOT here. macOS apps (claude,
#: agy, …) authenticate to their LLM provider via Keychain; blocking
#: writes silently breaks /login flows. The Keychain has its own per-app
#: ACL system anyway.
DENY_USER_SECRETS_RELATIVE: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".docker",
    ".kube",
    ".netrc",
    ".pypirc",
    ".npmrc",
    "Library/Cookies",
    "Library/Mail",
    "Library/Messages",
    "Library/Safari",
)

#: Subpaths the agent ALWAYS gets RW access to, regardless of the spec.
ALWAYS_RW_SUBPATHS: tuple[str, ...] = (
    "/private/tmp",
    "/tmp",
)

#: Literal device nodes / pipes the agent ALWAYS gets RW access to. Standard
#: POSIX tools redirect to these and break loudly if they're denied.
ALWAYS_RW_LITERALS: tuple[str, ...] = (
    "/dev/null",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
    "/dev/dtracehelper",
    "/dev/tty",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
)


def _sbpl_quote(path: str) -> str:
    """Quote a path for safe inclusion in an SBPL string literal."""
    # SBPL strings are double-quoted; escape backslashes and quotes.
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _expand_paths(paths: list[str | Path]) -> list[str]:
    """Expand each path into both its literal form AND its realpath.

    macOS `sandbox-exec` evaluates `subpath` rules against the canonicalized
    path, so a rule on `/var/folders/...` does NOT cover `/private/var/folders/...`
    (which is what the kernel actually sees). Emitting both forms keeps the
    rule effective regardless of which symlink form the agent uses at access
    time.
    """
    expanded: list[str] = []
    for p in paths:
        path_obj = Path(p)
        expanded.append(str(path_obj))
        try:
            resolved = str(path_obj.resolve())
            if resolved != str(path_obj):
                expanded.append(resolved)
        except OSError:
            pass
    return list(dict.fromkeys(expanded))


def _check_seatbelt_support() -> tuple[bool, str]:
    """Verify that we can actually call sandbox-exec on this macOS host."""
    if not Path(SANDBOX_EXEC).exists():
        return False, (
            f"`sandbox-exec` not found at {SANDBOX_EXEC}. The AI sandbox requires the macOS\n"
            "Seatbelt sandbox tool, which ships with the OS."
        )
    try:
        # Run a no-op profile to verify sandbox-exec is functional.
        subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            check=True,
            capture_output=True,
            timeout=2,
        )
        return True, ""
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        stderr = getattr(e, "stderr", b"").decode() if getattr(e, "stderr", None) else str(e)
        return False, f"sandbox-exec self-test failed: {stderr.strip()}"


class SeatbeltSandbox(Sandbox):
    """macOS sandbox backend using Apple's `sandbox-exec` (Seatbelt)."""

    @classmethod
    def check_support(cls) -> tuple[bool, str]:
        return _check_seatbelt_support()

    def _platform_note(self) -> str:
        return (
            "On macOS, the Seatbelt backend uses a permissive baseline with targeted denies "
            "(matches Codex CLI's model). The agent CAN read your files and CAN write to "
            "most of $HOME. The agent CANNOT write to: the OS, /Applications, /opt/homebrew, "
            "~/.ssh, ~/.aws, ~/.gnupg, ~/.docker, ~/.kube, "
            "~/Library/{Cookies,Mail,Messages,Safari}, or .netrc/.pypirc/.npmrc."
        )

    # --- helpers -------------------------------------------------------------

    def _setup_chrome_wrapper(self, sandbox_tmp: Path) -> Path:
        """Create a Chrome wrapper script that probes common macOS browser locations."""
        wrapper = sandbox_tmp / "odoo-chrome-wrapper"
        wrapper.write_text(
            "#!/bin/bash\n"
            "candidates=(\n"
            '    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"\n'
            '    "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"\n'
            '    "/Applications/Chromium.app/Contents/MacOS/Chromium"\n'
            '    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"\n'
            ")\n"
            'for real in "${candidates[@]}"; do\n'
            '    if [ -x "$real" ]; then\n'
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
        return wrapper

    def _discover_ide_sockets(self) -> list[Path]:
        """Find IDE IPC sockets on macOS (Cursor, VS Code, Antigravity)."""
        sockets: list[Path] = []
        candidates = []

        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            candidates.append(Path(tmpdir))
        candidates.append(Path("/tmp"))
        candidates.append(Path("/private/tmp"))

        seen: set[Path] = set()
        for d in candidates:
            try:
                resolved = d.resolve()
            except OSError:
                continue
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            for pattern in ("vscode-*.sock", "cursor-*.sock", "antigravity-*.sock", "*-ipc-*.sock"):
                try:
                    sockets.extend(resolved.glob(pattern))
                except OSError:
                    pass
        return sockets

    def _collect_rw_paths(
        self,
        spec: ExecutionSpec,
    ) -> tuple[list[str], list[str]]:
        """Compute the (rw_subpaths, rw_literals) the agent legitimately needs.

        Sourced from:
        - the always-RW constants (sandbox tmp scratch, devnodes);
        - the spec's RW binds (workspace + extra_bind_dirs the user passed);
        - agent state dirs/files (~/.claude, ~/.cache, ~/.gitconfig, …);
        - the ephemeral postgres socket + data dir, when applicable;
        - any IDE IPC sockets discovered in $TMPDIR.
        """
        rw_raw: list[str | Path] = [
            spec.sandbox_tmp,
            spec.playground,
            *ALWAYS_RW_SUBPATHS,
        ]
        rw_literals: list[str | Path] = list(ALWAYS_RW_LITERALS)

        for src, dst, ro, _primary in spec.final_binds:
            if not ro:
                rw_raw.extend([dst, src])

        rw_raw.extend(spec.agent_dirs)
        rw_raw.extend(f for f in spec.agent_files if f.exists())

        if spec.pg_process is not None:
            rw_raw.extend([spec.proxy_dir, spec.pg_data_dir])

        rw_literals.extend(self._discover_ide_sockets())

        return _expand_paths(rw_raw), _expand_paths(rw_literals)

    def _build_profile(
        self,
        spec: ExecutionSpec,
        host_home: Path,
        chrome_wrapper: Path,
        secrets_file: Path | None,
        launcher: Path,
    ) -> str:
        """Generate the SBPL profile for this execution.

        See module docstring for the policy rationale and the module-level
        DENY_*/ALWAYS_RW_* constants for the actual policy lists.
        """
        rw_subpaths, rw_lit = self._collect_rw_paths(spec)
        deny_system = _expand_paths(list(DENY_SYSTEM_SUBPATHS))
        deny_secrets = _expand_paths([host_home / rel for rel in DENY_USER_SECRETS_RELATIVE])

        lines: list[str] = [
            "(version 1)",
            ";; Permissive baseline — everything allowed by default. Strictness",
            ";; comes from targeted (deny file-write* ...) rules below. This",
            ";; matches the model used by Codex CLI / Claude Code on macOS.",
            "(allow default)",
            "",
            ";; --- Block writes to the OS itself --------------------------",
        ]
        for p in deny_system:
            lines.append(f"(deny file-write* (subpath {_sbpl_quote(p)}))")

        lines.append("")
        lines.append(";; --- Block writes to user credentials & private comms -------")
        for p in deny_secrets:
            lines.append(f"(deny file-write* (subpath {_sbpl_quote(p)}))")

        lines.append("")
        lines.append(";; --- Re-allow writes the agent legitimately needs -----------")
        lines.append(";; (these override the broader denies above for paths inside them)")
        for lit in rw_lit:
            lines.append(f"(allow file-write* (literal {_sbpl_quote(lit)}))")
        for p in rw_subpaths:
            lines.append(f"(allow file-write* (subpath {_sbpl_quote(p)}))")

        return "\n".join(lines) + "\n"

    def _write_secrets(self, sandbox_tmp: Path, secrets: list[tuple[str, str]]) -> Path | None:
        """Write secrets to a 0600 env file. Returns the path or None if no secrets."""
        if not secrets:
            return None
        env_file = sandbox_tmp / "secrets.env"
        # Build content: KEY=VAL with single-quoted values (escape embedded single quotes)
        body_lines: list[str] = []
        for key, val in secrets:
            quoted = "'" + val.replace("'", "'\\''") + "'"
            body_lines.append(f"export {key}={quoted}")
        env_file.write_text("\n".join(body_lines) + "\n")
        env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return env_file

    def _write_launcher(
        self,
        sandbox_tmp: Path,
        secrets_file: Path | None,
        env: dict[str, str],
        cwd: str,
    ) -> Path:
        """Write a small POSIX shell launcher that loads secrets/env then execs the agent.

        Using a launcher keeps secrets off the sandbox-exec argv (which would be
        visible in `ps`). When `ODEV_AI_SANDBOX_DEBUG=1` is set, the launcher
        echoes each step so a hang can be located precisely.
        """
        launcher = sandbox_tmp / "launcher.sh"
        debug = os.environ.get("ODEV_AI_SANDBOX_DEBUG") == "1"
        lines: list[str] = ["#!/bin/bash", "set -e"]
        if debug:
            lines.append("set -x")

        for key, val in env.items():
            quoted = "'" + val.replace("'", "'\\''") + "'"
            lines.append(f"export {key}={quoted}")

        if secrets_file is not None:
            quoted_secrets = shlex.quote(str(secrets_file))
            # Disable xtrace around the secrets source so debug mode never
            # echoes API/GitHub tokens to the terminal.
            if debug:
                lines.append("{ set +x; } 2>/dev/null")
            lines.extend(
                [
                    f". {quoted_secrets}",
                    f"rm -f {quoted_secrets}",
                ]
            )
            if debug:
                lines.append("set -x")

        lines.append(f"cd {shlex.quote(cwd)}")
        if debug:
            lines.append('echo "[odev-sandbox] launching: $*" >&2')
        lines.append('exec "$@"')

        launcher.write_text("\n".join(lines) + "\n")
        launcher.chmod(0o700)
        return launcher

    # --- main entry ----------------------------------------------------------

    def execute(self, spec: ExecutionSpec) -> bool:  # noqa: C901
        """Generate a Seatbelt profile, then run the agent under sandbox-exec."""
        host_home = Path.home().resolve()

        # Set up wrappers and env
        chrome_wrapper = self._setup_chrome_wrapper(spec.sandbox_tmp)

        env = dict(spec.env)
        env["ODOO_BROWSER_BIN"] = str(chrome_wrapper)
        # Postgres ephemeral cluster: tell client tools where the socket lives
        if spec.pg_process is not None:
            env["PGHOST"] = str(spec.proxy_dir)

        self._prepare_odev_config(spec.playground, host_home)
        self._prepare_agent_config(
            spec.playground,
            [str(dst) for src, dst, _, _ in spec.final_binds if src != host_home],
            host_home,
        )

        secrets_file = self._write_secrets(spec.sandbox_tmp, spec.secrets)
        launcher = self._write_launcher(spec.sandbox_tmp, secrets_file, env, spec.cwd)
        profile = self._build_profile(spec, host_home, chrome_wrapper, secrets_file, launcher)
        profile_path = spec.sandbox_tmp / "profile.sb"
        profile_path.write_text(profile)
        profile_path.chmod(0o600)

        if not self._display_sandbox_warning(
            binds=spec.final_binds,
            agent_dirs=spec.agent_dirs,
            agent_files=spec.agent_files,
            database=spec.database,
            db_user=spec.db_user,
            ephemeral_pg=spec.pg_process is not None,
            active_venv_path=spec.active_venv_path,
            odoo_filestore=spec.odoo_filestore,
            primary_dirs=spec.primary_dirs,
        ):
            self._terminate_pg(spec.pg_process)
            self._cleanup_paths([spec.playground, spec.sandbox_tmp, spec.proxy_dir, spec.pg_data_dir])
            return False

        if not self.headless:
            logger.info(f"Starting Project-wide AI execution ({self.cli})")

        cmd = [
            SANDBOX_EXEC,
            "-f",
            str(profile_path),
            "/bin/bash",
            str(launcher),
            *[str(x) for x in spec.agent_cmd],
        ]
        full_cmd = " ".join(shlex.quote(str(x)) for x in cmd)
        logger.debug("Running sandbox command: %s", full_cmd)
        logger.debug("Profile: %s", profile_path)
        logger.debug("Launcher: %s", launcher)

        # Hand off to odev's bash.run helper, which already sets up a raw TTY
        # the way bwrap does on Linux — interactive TUIs (claude, agy) need
        # that to render correctly.
        from odev.common import bash

        returncode = 0
        keep_artifacts = False
        try:
            bash.run(full_cmd)
        except subprocess.CalledProcessError as error:
            returncode = error.returncode
        except KeyboardInterrupt:
            returncode = 130
        except Exception as e:
            logger.error(f"Failed to run {self.cli}: {e}")
            returncode = 1

        if returncode != 0:
            # SIGABRT (134) usually means the SBPL profile is too restrictive
            # and the dynamic linker aborted before main(). Keep artifacts so
            # the user can re-run sandbox-exec manually for diagnostics.
            keep_artifacts = True
            console.print(
                f"\n[bold red]AI agent exited with code {returncode}.[/]\n"
                f"  Profile:  {profile_path}\n"
                f"  Launcher: {launcher}\n"
                f"Re-run manually to inspect: "
                f"sandbox-exec -f {shlex.quote(str(profile_path))} "
                f"/bin/bash {shlex.quote(str(launcher))} {shlex.quote(str(spec.agent_cmd[0])) if spec.agent_cmd else ''}"
            )
            if returncode == 134:
                console.print(
                    "[yellow]Exit 134 (SIGABRT) typically indicates the Seatbelt profile is too "
                    "strict for the dynamic linker. Try expanding the read-only allowlist.[/]"
                )

        # Best-effort: ensure secrets file is gone even if launcher didn't run.
        if secrets_file is not None:
            try:
                secrets_file.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._terminate_pg(spec.pg_process)
        if keep_artifacts:
            # Drop everything except sandbox_tmp (where profile/launcher live).
            self._cleanup_paths([spec.playground, spec.proxy_dir, spec.pg_data_dir])
        else:
            self._cleanup_paths([spec.playground, spec.sandbox_tmp, spec.proxy_dir, spec.pg_data_dir])

        return returncode == 0


__all__ = ["SeatbeltSandbox"]
