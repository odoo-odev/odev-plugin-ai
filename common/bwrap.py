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

    def _display_sandbox_warning(
        self,
        binds: list[tuple[Path, Path, bool, bool]],
        agent_dirs: list[Path],
        agent_files: list[Path],
        database: str | None = None,
        db_user: str | None = None,
        ephemeral_pg: bool = True,
    ) -> bool:
        """Display a warning message about the sandbox access and security risks."""
        if self.headless:
            return True

        from odev.common import string

        console.rule(string.stylize("AI SANDBOX SECURITY WARNING", "bold color.red"), style="color.red")
        console.print(
            f"\n{string.stylize('ATTENTION:', 'bold color.yellow')} You are running an AI agent in a sandboxed environment."
        )
        console.print("The agent can read/write files and access the database within this sandbox.")

        console.print(f"\n{string.stylize('PRIMARY WORKSPACES (Read-Write Access):', 'bold color.cyan')}")
        for src, dst, _ro, primary in binds:
            if primary:
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

        console.print(f"\n{string.stylize('INFRASTRUCTURE & REFERENCE (System/Source/Config):', 'bold color.cyan')}")
        for d in agent_dirs:
            console.print(f" • {d} (RW)")
        for f in agent_files:
            if f.exists():
                console.print(f" • {f} (RW)")
        for src, dst, ro, primary in binds:
            if not primary:
                mode = "RO" if ro else "RW"
                label = (
                    f"{src} {string.stylize(f'-> {dst}', 'bold color.green')} ({mode})"
                    if src != dst
                    else f"{src} ({mode})"
                )
                console.print(f" • {label}")

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
        playground: Path,
        host_home: Path,
    ):
        """Copy the host RTK configuration into the sandbox playground.

        RTK is assumed to be pre-configured as a hook in the agent's settings
        on the host. We copy settings.json and the hooks directory into the
        playground (which is mounted as HOME inside the sandbox) so that those
        hooks are active without needing to re-run ``rtk init``.
        ~/.local (and therefore the rtk binary + ~/.local/share/rtk) is already
        covered by agent_dirs. Only ~/.config/rtk needs an explicit bind.
        """
        if not shutil.which("rtk"):
            return

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
            for name in ("settings.json", "hooks"):
                src = host_config_dir / name
                dst = sandbox_config_path / name
                if src.exists() and not dst.exists():
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

        rtk_config = host_home / ".config" / "rtk"
        if rtk_config.exists():
            cmd.extend(["--bind", str(rtk_config), str(host_home / ".config" / "rtk")])

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

        binds = list(
            filter(
                None,
                [
                    # Type B — workspace utilisateur (primary, RW)
                    *[
                        bind(s.split(":", 1)[0], s.split(":", 1)[1] if ":" in s else "/custom", ro=False, primary=True)
                        for s in sandbox_dirs
                    ],
                    # Type B — extra dirs fournis par l'appelant (RO)
                    *[bind(e.split(":", 1)[0], e.split(":", 1)[1] if ":" in e else e) for e in (extra_bind_dirs or [])],
                    # Type A — infrastructure odev (parents montent les enfants, pas de dedup)
                    bind(self.odev.path),
                    bind(self.odev.home_path / "plugins"),
                    bind(self.odev.home_path / "worktrees", "/worktrees"),
                    bind(self.odev.home_path / "virtualenvs", ro=False),
                    *[bind(r.path) for r in odoo_repositories(enterprise=True)],
                ],
            )
        )

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

    def _prepare_agent_config(  # noqa: C901
        self,
        playground: Path,
        all_candidate_paths: list[str],
        host_home: Path,
    ):
        """Create a sanitized agent configuration inside the sandbox playground."""
        # The playground becomes 'host_home' in the bwrap sandbox
        # We need to create .gemini, .config/gemini, .claude, etc.
        # Primary config dir + MD file that supports @imports (skills injection)
        skill_targets = {
            "gemini": (".gemini", "GEMINI.md"),
            "claude": (".claude", "CLAUDE.md"),
            "opencode-cli": (".claude", "CLAUDE.md"),
        }

        configs = {
            "gemini": [".gemini", ".config/gemini"],
            "claude": [".claude", ".config/claude"],
            "copilot": [".config/github-copilot"],
        }

        relevant_dirs = configs.get(self.cli, [])
        for rel_dir in relevant_dirs:
            target_dir = playground / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)

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

            trusted_paths = [
                "/knowledge",
                "/custom",
                "/worktree",
                "/venvs",
                "/repositories",
                "/upgrade",
                "/dumps",
            ]
            for d in all_candidate_paths:
                if ":" in d:
                    trusted_paths.append(d.split(":")[1])

        # Inject skills into the CLI's native context
        if primary := skill_targets.get(self.cli):
            rel_dir, md_filename = primary
            target_dir = playground / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            skills_dest = target_dir / "skills"
            skill_refs = []
            for plugin in self.odev.plugins:
                skills_dir = Path(plugin.path) / "skills"
                if not skills_dir.is_dir():
                    continue
                disabled_skills = self.odev.config.skills.disabled
                for skill_pkg in skills_dir.iterdir():
                    if not skill_pkg.is_dir() or not (skill_pkg / "SKILL.md").exists():
                        continue
                    if skill_pkg.name in disabled_skills:
                        continue
                    skills_dest.mkdir(exist_ok=True)
                    if self.cli == "gemini":
                        # Gemini native skills: copy the entire package dir so
                        # ~/.gemini/skills/{name}/SKILL.md is discovered by /skills list
                        dest = skills_dest / skill_pkg.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(skill_pkg, dest)
                    else:
                        # Claude / opencode-cli: inject via @import in the MD file
                        shutil.copy2(skill_pkg / "SKILL.md", skills_dest / f"{skill_pkg.name}.md")
                        skill_refs.append(f"@skills/{skill_pkg.name}.md")
            # Only Claude/opencode-cli use the @import mechanism
            if skill_refs:
                md_file = target_dir / md_filename
                existing = md_file.read_text() if md_file.exists() else ""
                with open(md_file, "a") as f:
                    if existing and not existing.endswith("\n"):
                        f.write("\n")
                    f.write("\n".join(skill_refs) + "\n")

            trust_data = dict.fromkeys(sorted(set(trusted_paths)), "TRUST_FOLDER")
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
        for d in agent_dirs:
            cmd.extend(["--bind-try", str(d), str(d)])
        for f in agent_files:
            if f.exists():
                cmd.extend(["--bind-try", str(f), str(f)])

        # final_binds is already sorted by depth from _prepare_sandbox_config
        for src, dst, ro, _is_primary in final_binds:
            try:
                # Create the destination path inside the playground so bwrap can mount on top of it.
                # (--bind-try only skips a missing SOURCE; missing DESTINATION still errors.)
                if str(dst).startswith(str(host_home)):
                    rel = dst.relative_to(host_home)
                    dst_in_playground = playground / rel
                    if src.is_dir():
                        dst_in_playground.mkdir(parents=True, exist_ok=True)
                    else:
                        dst_in_playground.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.debug(f"Could not pre-create destination path for {dst}: {e}")
            cmd.extend(["--ro-bind-try" if ro else "--bind-try", str(src), str(dst)])

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
        database: str | None,
        db_user: str | None,
        secrets_to_set: list[tuple[str, str]],
        pg_process: subprocess.Popen | None,
        playground: Path,
        sandbox_tmp: Path,
        proxy_dir: Path,
        pg_data_dir: Path,
    ) -> bool:
        """Final execution logic for the bwrap sandbox."""
        if not self._display_sandbox_warning(
            binds=final_binds,
            agent_dirs=agent_dirs,
            agent_files=agent_files,
            database=database,
            db_user=db_user,
            ephemeral_pg=pg_process is not None,
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
                    console.print("\n" * 20)

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
