import subprocess
import time
from pathlib import Path

from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin


logger = logging.getLogger(__name__)


class PostgresSandbox(OdevFrameworkMixin):
    """Manages an ephemeral PostgreSQL sandbox database."""

    def __init__(self, headless: bool = False):
        super().__init__()
        self.headless = headless

    def setup(
        self,
        cmd: list[str],
        database: str | None,
        proxy_dir: Path,
        pg_data_dir: Path,
        ephemeral: bool = True,
    ) -> subprocess.Popen | None:
        """Initialize PostgreSQL cluster or proxy for the sandbox."""
        # Discover host socket directory
        for path in [Path("/var/run/postgresql"), Path("/tmp")]:
            if any(path.glob(".s.PGSQL.*")):
                host_socket_dir = path
                break
        else:
            host_socket_dir = Path("/var/run/postgresql")

        if ephemeral:
            pg_process = self._start_ephemeral_postgres(proxy_dir, pg_data_dir)
            if pg_process:
                if database and host_socket_dir and host_socket_dir.exists():
                    cloned = self._clone_host_database(database, host_socket_dir, proxy_dir)
                    if not cloned:
                        logger.info(
                            f"Host database {database!r} not found or could not be cloned; "
                            "the AI agent will initialize it as needed."
                        )

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
            return pg_process

        return None

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
                ["pg_dump", "-h", str(host_socket_dir), "-d", database, "--no-owner", "--no-privileges"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p2 = subprocess.Popen(
                ["psql", "-h", str(ephemeral_socket_dir), "-p", "5432", "-U", user, "-d", database],
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

            self._create_database(socket_dir, user, user)
            self._create_database(socket_dir, user, "odev")

            if not self.headless:
                logger.info("Ephemeral PostgreSQL cluster is ready")
            return process
        except Exception as e:
            logger.error(f"Failed to start ephemeral PostgreSQL: {e}")
            return None
