"""Base sandbox abstraction shared by all platform-specific backends.

Holds platform-agnostic logic (bind resolution, agent config sanitization,
warning rendering) and defines the contract that backends must implement.
"""

import os
import shutil
import signal
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from odev.common.console import console
from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin
from odev.common.odoobin import odoo_repositories
from odev.common.python import PythonEnv
from odev.common.version import OdooVersion


logger = logging.getLogger(__name__)


@dataclass
class ExecutionSpec:
    """Backend-agnostic description of a sandboxed execution.

    Backends translate this into their own primitives:
    - bwrap turns binds into --bind/--ro-bind flags;
    - seatbelt turns binds into SBPL allow rules.
    """

    agent_cmd: list[str]
    final_binds: list[tuple[Path, Path, bool, bool]]
    agent_dirs: list[Path]
    agent_files: list[Path]
    env: dict[str, str]
    secrets: list[tuple[str, str]]
    cwd: str
    playground: Path
    sandbox_tmp: Path
    proxy_dir: Path
    pg_data_dir: Path
    database: str | None = None
    db_user: str | None = None
    pg_process: subprocess.Popen | None = None
    active_venv_path: Path | None = None
    odoo_filestore: Path | None = None
    primary_dirs: list[Path] | None = None


class Sandbox(OdevFrameworkMixin, ABC):
    """Common base for AI sandbox backends.

    Subclasses implement `execute()` and `check_support()` to translate an
    `ExecutionSpec` into platform-specific isolation primitives.
    """

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

    # --- contract ------------------------------------------------------------

    @abstractmethod
    def execute(self, spec: ExecutionSpec) -> bool:
        """Run the agent inside the sandbox using the given spec.

        Must perform cleanup of `spec.playground`, `spec.sandbox_tmp`,
        `spec.proxy_dir`, `spec.pg_data_dir`, and terminate `spec.pg_process`.
        """

    @classmethod
    @abstractmethod
    def check_support(cls) -> tuple[bool, str]:
        """Return (supported, message). Message is shown when not supported."""

    def _platform_note(self) -> str | None:
        """Optional backend-specific note shown alongside the security warning.

        Backends override this to surface platform-specific isolation
        guarantees (or lack thereof) before the user confirms execution.
        Default: no extra note.
        """
        return None

    # --- shared helpers ------------------------------------------------------

    @staticmethod
    def _group_infra_items(items: list[tuple[Path, str]]) -> list[tuple[str, str]]:
        """Sort and group infra paths by parent+mode for a compact display.

        Paths that share the same parent directory *and* the same access mode
        are collapsed into a single line:

            /some/parent/{child_a, child_b} (RO)

        Singletons are shown as-is. The final list is sorted alphabetically.
        """
        from collections import defaultdict

        home = str(Path.home().resolve())

        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for path, mode in items:
            parent_str = str(path.parent.resolve())
            if parent_str == home:
                parent_str = "~"
            elif parent_str.startswith(home + "/"):
                parent_str = "~" + parent_str[len(home) :]

            groups[(parent_str, mode)].append(path.name)

        result: list[tuple[str, str]] = []
        for (parent, mode), names in groups.items():
            names_sorted = sorted(names)
            if len(names_sorted) == 1:
                path_str = f"{parent}/{names_sorted[0]}"
            else:
                path_str = f"{parent}/{{{', '.join(names_sorted)}}}"
            result.append((path_str, mode))

        return sorted(result, key=lambda x: x[0])

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

        if note := self._platform_note():
            console.print(f"{string.stylize('NOTE:', 'bold color.yellow')} {note}")

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

        home = str(Path.home().resolve())

        def clean_path(p: Path) -> str:
            p_str = str(p.resolve())
            if p_str == home:
                return "~"
            if p_str.startswith(home + "/"):
                return "~" + p_str[len(home) :]
            return p_str

        for src, dst, _ro, _primary in primary_binds:
            label = (
                f"{clean_path(src)} {string.stylize(f'-> {clean_path(dst)}', 'bold color.green')}"
                if src != dst
                else clean_path(src)
            )
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

        # Build list of infrastructure/reference paths
        infra_items: list[tuple[Path, str]] = []
        mapping_lines: list[tuple[str, str]] = []

        for d in agent_dirs:
            infra_items.append((d, "RW"))
        for f in agent_files:
            if f.exists():
                infra_items.append((f, "RW"))
        if odoo_filestore and odoo_filestore.exists():
            infra_items.append((odoo_filestore, "RW"))
        if active_venv_path and active_venv_path.exists():
            infra_items.append((active_venv_path, "RO"))

        for src, dst, ro, primary in binds:
            if not primary:
                mode = "RO" if ro else "RW"
                if src == dst:
                    infra_items.append((src, mode))
                else:
                    mapping_lines.append(
                        (f"{clean_path(src)} {string.stylize(f'-> {clean_path(dst)}', 'bold color.green')}", mode)
                    )

        console.print(f"\n{string.stylize('INFRASTRUCTURE & REFERENCE (System/Source/Config):', 'bold color.cyan')}")

        # Deduplicate infra_items while preserving order
        seen_items = set()
        unique_infra_items = []
        for path, mode in infra_items:
            key = (path.resolve(), mode)
            if key not in seen_items:
                seen_items.add(key)
                unique_infra_items.append((path, mode))

        # Group and print plain paths
        for path_str, mode in self._group_infra_items(unique_infra_items):
            console.print(f" • {string.stylize(path_str, 'color.purple')} ({mode})")

        # Deduplicate and print mapping lines
        seen_mappings = set()
        for line, mode in sorted(mapping_lines, key=lambda x: x[0]):
            if line not in seen_mappings:
                seen_mappings.add(line)
                console.print(f" • {string.stylize(line, 'color.purple')} ({mode})")

        if not self.yolo and not console.bypass_prompt:
            agent_names = {
                "claude": "Claude",
                "agy": "Agy",
                "copilot": "Copilot",
                "opencode": "OpenCode",
            }
            name = agent_names.get(self.cli, self.cli)

            if self.model and self.model != "auto":
                display_name = f"{name} ({self.model})"
            else:
                display_name = name

            return console.confirm(
                f"Do you want to proceed with the {display_name} AI agent execution?",
                default=True,
            )
        return True

    def _prepare_odev_config(self, playground: Path, host_home: Path) -> None:
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

    def prepare_sandbox_config(
        self,
        sandbox_dirs: list[str],
        extra_bind_dirs: list[str] | None,
        database: str | None,
        version: str | None,
        extra_ro_bind_dirs: list[str] | None = None,
    ) -> dict:
        """Build the flat list of sandbox bindings across the 3 binding categories.

        Public entry point used by `AgentCLI` to assemble an `ExecutionSpec`.
        """

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
                    # Type B — extra dirs provided by the caller (RO, still Primary so the
                    # agent gets them via --add-dir/trustedDirectories, just can't edit them)
                    *[bind(e, ro=True, primary=True) for e in (extra_ro_bind_dirs or [])],
                    # Type A — odev infrastructure (parents are mounted before children, no dedup)
                    bind(self.odev.path),
                    bind(self.odev.plugins_path),
                    # Resolve plugin symlink targets so Python can import them inside the sandbox.
                    *[bind(p) for p in self.odev.plugins_path.iterdir() if p.is_symlink()],
                    # RW access is required for odev to perform git operations/worktree management
                    bind(self.odev.home_path / "worktrees", ro=False),
                    bind(self.odev.home_path / "virtualenvs", ro=False),
                    bind(self.odev.home_path / "browsers"),
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

    def _prepare_agent_config(
        self,
        playground: Path,
        all_candidate_paths: list[str],
        host_home: Path,
    ) -> None:
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

    def _copy_agent_credentials(
        self,
        target_dir: Path,
        rel_dir: str,
        is_persistent: bool,
        host_home: Path,
    ) -> None:
        """Copy credentials from host to playground if not persistent."""
        if is_persistent:
            return
        creds_files = self.handler.get_creds_files()
        for cf in creds_files:
            hcf = host_home / rel_dir / cf
            if hcf.exists():
                shutil.copy2(hcf, target_dir / cf)

    @staticmethod
    def _cleanup_paths(paths: list[Path]) -> None:
        """Best-effort recursive removal of temp dirs after execution."""
        for path_to_clean in paths:
            try:
                shutil.rmtree(path_to_clean)
            except Exception:
                pass

    @staticmethod
    def _terminate_pg(pg_process: subprocess.Popen | None) -> None:
        """Best-effort termination of the ephemeral postgres process tree.

        Postgres forks worker backends that survive a plain SIGTERM to the
        leader. The leader is spawned with `start_new_session=True` (see
        `PostgresSandbox._start_ephemeral_postgres`), so we can kill the
        whole tree via `os.killpg`.
        """
        if not pg_process:
            return

        try:
            pgid = os.getpgid(pg_process.pid)
        except (ProcessLookupError, OSError):
            pgid = None

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                pg_process.terminate()
            pg_process.wait(timeout=5)
            return
        except (subprocess.TimeoutExpired, Exception):
            pass

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                pg_process.kill()
        except Exception:
            pass
        try:
            pg_process.wait(timeout=2)
        except Exception:
            pass
