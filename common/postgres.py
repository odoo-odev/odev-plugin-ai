import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin


logger = logging.getLogger(__name__)


# Anything we mkdtemp() in agent.run() / postgres setup starts with this:
_ORPHAN_DIR_PREFIX = "odev-ai-"

# Skip dirs younger than this many seconds — protects concurrent `odev ai` runs
# from killing each other. A real launch passes this threshold within a minute.
_ORPHAN_MIN_AGE_SECONDS = 5 * 60


class PostgresSandbox(OdevFrameworkMixin):
    """Manages an ephemeral PostgreSQL sandbox database."""

    @classmethod
    def cleanup_orphans(cls) -> None:
        """Reap leftover ephemeral postgres clusters and tmp dirs from previous
        crashed `odev ai` runs.

        Each ephemeral cluster writes a `postmaster.pid` file into its data
        dir. If a previous run was Ctrl+C'd or otherwise terminated abnormally,
        the postgres backend may still be running and several `odev-ai-*`
        directories may linger in `/tmp` / `$TMPDIR`. This routine kills the
        postgres process group (when applicable) and removes the dirs. Only
        dirs older than `_ORPHAN_MIN_AGE_SECONDS` are touched to avoid racing
        with a concurrently-running `odev ai`.
        """
        scan_dirs: list[Path] = []
        if v := os.environ.get("TMPDIR"):
            scan_dirs.append(Path(v))
        scan_dirs.extend([Path("/tmp"), Path(tempfile.gettempdir())])

        cutoff = time.time() - _ORPHAN_MIN_AGE_SECONDS
        seen: set[Path] = set()
        for d in scan_dirs:
            try:
                resolved = d.resolve()
            except OSError:
                continue
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            try:
                entries = list(resolved.iterdir())
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir() or not entry.name.startswith(_ORPHAN_DIR_PREFIX):
                    continue
                try:
                    if entry.stat().st_mtime > cutoff:
                        continue
                except OSError:
                    continue
                cls._reap_orphan_dir(entry)

    @staticmethod
    def _reap_orphan_dir(orphan: Path) -> None:
        """Kill the postgres process group anchored in `orphan` and rm the dir."""
        pid_file = orphan / "postmaster.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().splitlines()[0].strip())
                # postgres normally heads its own process group via -k.
                # SIGTERM the group; SIGKILL on timeout.
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                for _ in range(25):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                else:
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                logger.debug(f"Reaped orphan postgres at {orphan} (pid {pid})")
            except (ValueError, OSError) as e:
                logger.debug(f"Could not parse {pid_file}: {e}")
        try:
            shutil.rmtree(orphan)
        except OSError as e:
            logger.debug(f"Could not remove orphan dir {orphan}: {e}")

    def __init__(self, headless: bool = False):
        super().__init__()
        self.headless = headless

    def _get_pg_bin(self, name: str) -> str:
        """Find a PostgreSQL binary by name, trying standard paths if not in PATH."""
        bin_path = shutil.which(name)
        if bin_path:
            return bin_path

        for finder in (self._find_pg_bin_linux, self._find_pg_bin_macos):
            found = finder(name)
            if found:
                return found
        return name

    def _find_pg_bin_linux(self, name: str) -> str | None:
        """Look up `name` under standard Linux PostgreSQL install layouts."""
        psql_path = shutil.which("psql")
        if psql_path:
            try:
                out = subprocess.check_output([psql_path, "--version"], text=True)
                match = re.search(r"\(PostgreSQL\)\s+(\d+)", out)
                if match:
                    candidate = Path(f"/usr/lib/postgresql/{match.group(1)}/bin/{name}")
                    if candidate.exists():
                        return str(candidate)
            except Exception:
                pass

        base = Path("/usr/lib/postgresql")
        if not base.exists():
            return None
        versions: list[tuple[float, Path]] = []
        for d in base.iterdir():
            if d.is_dir():
                try:
                    versions.append((float(d.name), d))
                except ValueError:
                    continue
        for _, v_dir in sorted(versions, reverse=True):
            candidate = v_dir / "bin" / name
            if candidate.exists():
                return str(candidate)
        return None

    def _find_pg_bin_macos(self, name: str) -> str | None:
        """Look up `name` under standard macOS PostgreSQL install layouts."""
        for brew_base in (Path("/opt/homebrew/opt"), Path("/usr/local/opt")):
            if not brew_base.exists():
                continue
            try:
                pg_dirs = sorted(brew_base.glob("postgresql@*"), key=lambda p: p.name, reverse=True)
            except OSError:
                pg_dirs = []
            for d in pg_dirs:
                candidate = d / "bin" / name
                if candidate.exists():
                    return str(candidate)
            generic = brew_base / "postgresql" / "bin" / name
            if generic.exists():
                return str(generic)

        pgapp = Path("/Applications/Postgres.app/Contents/Versions")
        if pgapp.exists():
            try:
                versions = sorted(pgapp.iterdir(), key=lambda p: p.name, reverse=True)
            except OSError:
                versions = []
            for v in versions:
                candidate = v / "bin" / name
                if candidate.exists():
                    return str(candidate)
        return None

    def setup(
        self,
        database: str | None,
        proxy_dir: Path,
        pg_data_dir: Path,
        ephemeral: bool = True,
    ) -> subprocess.Popen | None:
        """Initialize an ephemeral PostgreSQL cluster (and optionally clone a host DB).

        The caller (an `AgentCLI` / `Sandbox` backend) is responsible for exposing
        `proxy_dir` to the sandboxed process — for bwrap that means a `--bind`
        onto `/var/run/postgresql`; for sandbox-exec it means allowlisting
        `proxy_dir` and pointing `PGHOST` at it.
        """
        if not ephemeral:
            return None

        # Discover host socket directory
        candidates = [
            Path("/var/run/postgresql"),  # Linux package default
            Path("/tmp"),  # Homebrew default on macOS
            Path("/private/tmp"),
        ]
        host_socket_dir = next((p for p in candidates if p.exists() and any(p.glob(".s.PGSQL.*"))), None)

        pg_process = self._start_ephemeral_postgres(proxy_dir, pg_data_dir)
        if pg_process and database and host_socket_dir is not None:
            cloned = self._clone_host_database(database, host_socket_dir, proxy_dir)
            if not cloned:
                logger.info(
                    f"Host database {database!r} not found or could not be cloned; "
                    "the AI agent will initialize it as needed."
                )
        return pg_process

    def _clone_host_database(self, database: str, host_socket_dir: Path, ephemeral_socket_dir: Path) -> bool:
        """Clone a host database into the ephemeral cluster.

        Uses odev's LocalDatabase to verify the source exists and is an Odoo
        database before cloning. If not, the cluster is left empty so the AI
        agent can initialize the database itself via its Odoo skill.
        """
        from odev.common.databases.local import LocalDatabase

        db = LocalDatabase(database)
        if not db.exists:
            logger.debug(f"Host database {database!r} does not exist, skipping clone.")
            return False
        if not db.is_odoo:
            logger.debug(f"Host database {database!r} is not an Odoo database, skipping clone.")
            return False

        user = Path.home().name
        logger.info(f"Cloning host database {database!r} into ephemeral cluster...")
        self._create_database(ephemeral_socket_dir, user, database)
        try:
            p1 = subprocess.Popen(
                [
                    self._get_pg_bin("pg_dump"),
                    "-h",
                    str(host_socket_dir),
                    "-d",
                    database,
                    "--no-owner",
                    "--no-privileges",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p2 = subprocess.Popen(
                [self._get_pg_bin("psql"), "-h", str(ephemeral_socket_dir), "-p", "5432", "-U", user, "-d", database],
                stdin=p1.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            p1.stdout.close()
            _, stderr = p2.communicate()

            if p2.returncode != 0:
                logger.warning(f"Failed to restore {database!r} into ephemeral cluster: {stderr.decode().strip()}")
                return False

            logger.info(f"Database {database!r} successfully cloned.")
            return True

        except Exception as e:
            logger.debug(f"Failed to clone host database {database!r}: {e}")
            return False

    def _create_database(self, socket_dir: Path, user: str, database: str) -> bool:
        """Create a database in the ephemeral cluster."""
        try:
            subprocess.run(
                [
                    self._get_pg_bin("psql"),
                    "-h",
                    str(socket_dir),
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
            return True
        except Exception as e:
            logger.warning(f"Database {database!r} creation failed: {e}")
            return False

    def _start_ephemeral_postgres(self, socket_dir: Path, data_dir: Path) -> subprocess.Popen | None:
        """Initialize and start an ephemeral PostgreSQL cluster."""
        try:
            if not self.headless:
                logger.info(f"Initializing ephemeral PostgreSQL cluster in {data_dir}")
            # Use current host user as superuser so psql works without -U
            user = Path.home().name
            subprocess.run(
                [self._get_pg_bin("initdb"), "-D", str(data_dir), "--nosync", "-U", user, "--auth=trust"],
                check=True,
                capture_output=True,
            )

            # Start postgres listening ONLY on unix socket in socket_dir.
            # `start_new_session=True` makes the postgres backend its own
            # process group leader so we can kill the whole tree on cleanup
            # via os.killpg() (postgres forks worker backends that survive a
            # plain SIGTERM to the leader otherwise).
            process = subprocess.Popen(
                [
                    self._get_pg_bin("postgres"),
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
                start_new_session=True,
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

            self._create_database(socket_dir, user, user)
            self._create_database(socket_dir, user, "odev")

            if not self.headless:
                logger.info("Ephemeral PostgreSQL cluster is ready")
            return process
        except Exception as e:
            logger.error(f"Failed to start ephemeral PostgreSQL: {e}")
            return None
