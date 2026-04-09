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
        pg_socket_dir: Path | str | None,
        ephemeral: bool = True,
    ) -> subprocess.Popen | None:
        """Initialize PostgreSQL cluster or proxy for the sandbox."""
        if pg_socket_dir:
            host_socket_dir = Path(pg_socket_dir)
        else:
            for path in [Path("/var/run/postgresql"), Path("/tmp")]:
                if any(path.glob(".s.PGSQL.*")):
                    host_socket_dir = path
                    break
            else:
                host_socket_dir = Path("/var/run/postgresql")

        # 1. Ephemeral Cluster Setup
        if ephemeral:
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
            return pg_process

        return None

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
