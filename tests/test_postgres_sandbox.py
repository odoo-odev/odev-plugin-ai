"""The ephemeral PostgreSQL cluster must never hand Odoo a superuser role.

Recent Odoo versions refuse to run when their own PostgreSQL role is a superuser:
a superuser can run OS commands and read/write files as the postgres system user,
which would let the agent escape the sandbox. So the cluster is bootstrapped with
``postgres`` as its lone superuser, kept for administration, and the host user gets
a separate ``NOSUPERUSER`` / ``CREATEDB`` login role — the one Odoo connects as.

These are integration tests: they boot a real cluster through
:meth:`PostgresSandbox._start_ephemeral_postgres` and inspect its roles, so they
skip where PostgreSQL's server binaries are absent. One assertion per behaviour, so
the first failure never hides the rest.
"""

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from odev.plugins.odev_plugin_ai.common.postgres import _BOOTSTRAP_SUPERUSER, PostgresSandbox


# The role the cluster creates for the host user, as the code derives it.
HOST_USER = Path.home().name

# The host user being literally `postgres` collapses the two roles into one: there is
# then no separate non-superuser identity to assert on, so those tests do not apply.
host_user_is_distinct = pytest.mark.skipif(
    HOST_USER == _BOOTSTRAP_SUPERUSER,
    reason=f"host user is {_BOOTSTRAP_SUPERUSER!r}, so it is the bootstrap superuser itself",
)


def _sandbox() -> PostgresSandbox:
    """Build a :class:`PostgresSandbox` without the odev framework init its PG methods never use."""
    sandbox = object.__new__(PostgresSandbox)
    sandbox.headless = True
    return sandbox


def _resolved(sandbox: PostgresSandbox, name: str) -> str | None:
    """Return a usable path to a PostgreSQL binary, or ``None`` if it cannot be found."""
    found = sandbox._get_pg_bin(name)
    return found if (shutil.which(found) or Path(found).exists()) else None


def _psql(cluster: SimpleNamespace, sql: str, user: str = _BOOTSTRAP_SUPERUSER, database: str = "postgres") -> str:
    """Run a one-shot query against the cluster and return its unaligned, tuples-only output."""
    result = subprocess.run(
        [
            cluster.sandbox._get_pg_bin("psql"),
            "-h",
            str(cluster.socket_dir),
            "-p",
            "5432",
            "-U",
            user,
            "-d",
            database,
            "-At",
            "-c",
            sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture(scope="module")
def cluster() -> SimpleNamespace:
    """Boot one ephemeral cluster for the module and tear it (and its socket dir) down after."""
    sandbox = _sandbox()
    if any(_resolved(sandbox, name) is None for name in ("initdb", "postgres", "psql")):
        pytest.skip("PostgreSQL server binaries are not available")

    data_dir = Path(tempfile.mkdtemp(prefix="odev-ai-pgdata-"))
    # Unix socket paths cap at ~107 chars, so the socket dir lives under the short system
    # temp root rather than a deep pytest path; the `odev-ai-` prefix also lets the
    # sandbox's own orphan reaper clean it up if a test crashes mid-run.
    socket_dir = Path(tempfile.mkdtemp(prefix="odev-ai-pg-"))

    process = sandbox._start_ephemeral_postgres(socket_dir, data_dir)
    if process is None:
        shutil.rmtree(socket_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)
        pytest.skip("the ephemeral PostgreSQL cluster did not start")

    try:
        yield SimpleNamespace(sandbox=sandbox, socket_dir=socket_dir, process=process)
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(socket_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)


# --- The cluster comes up ---------------------------------------------------------------------


def test_cluster_is_running(cluster: SimpleNamespace) -> None:
    """The ephemeral postgres backend is up and accepting connections."""
    assert cluster.process.poll() is None


# --- The role Odoo connects as is not a superuser (the regression) ----------------------------


@host_user_is_distinct
def test_host_role_is_not_a_superuser(cluster: SimpleNamespace) -> None:
    """The host user's role — the one Odoo connects as — is NOSUPERUSER, so Odoo agrees to run."""
    assert _psql(cluster, f"SELECT rolsuper FROM pg_roles WHERE rolname = '{HOST_USER}'") == "f"


@host_user_is_distinct
def test_host_role_may_create_databases(cluster: SimpleNamespace) -> None:
    """Dropping superuser must not cost the agent CREATEDB: it still creates databases."""
    assert _psql(cluster, f"SELECT rolcreatedb FROM pg_roles WHERE rolname = '{HOST_USER}'") == "t"


@host_user_is_distinct
def test_host_role_can_log_in(cluster: SimpleNamespace) -> None:
    """The host role is a LOGIN role, otherwise nothing could connect as it."""
    assert _psql(cluster, f"SELECT rolcanlogin FROM pg_roles WHERE rolname = '{HOST_USER}'") == "t"


def test_bootstrap_superuser_is_the_only_superuser(cluster: SimpleNamespace) -> None:
    """`postgres` is the cluster's lone superuser, reserved for administration."""
    supers = _psql(cluster, "SELECT rolname FROM pg_roles WHERE rolsuper ORDER BY rolname").split("\n")
    assert supers == [_BOOTSTRAP_SUPERUSER]


# --- The default databases are there ----------------------------------------------------------


def test_host_database_exists(cluster: SimpleNamespace) -> None:
    """The cluster ships the per-user database the tooling expects."""
    assert _psql(cluster, f"SELECT 1 FROM pg_database WHERE datname = '{HOST_USER}'") == "1"


def test_odev_database_exists(cluster: SimpleNamespace) -> None:
    """The cluster ships odev's own bookkeeping database."""
    assert _psql(cluster, "SELECT 1 FROM pg_database WHERE datname = 'odev'") == "1"


# --- End to end: the agent's role can do what the sandbox needs --------------------------------


@host_user_is_distinct
def test_agent_role_can_create_a_database(cluster: SimpleNamespace) -> None:
    """Connecting as the host user — exactly as odev does in the sandbox — creates a database."""
    _psql(cluster, 'CREATE DATABASE "probe_db" ENCODING \'UTF8\' TEMPLATE template0', user=HOST_USER)
    try:
        assert _psql(cluster, "SELECT 1 FROM pg_database WHERE datname = 'probe_db'") == "1"
    finally:
        _psql(cluster, 'DROP DATABASE IF EXISTS "probe_db"')


# --- The host-is-postgres edge is handled without a cluster ------------------------------------


def test_app_role_not_created_when_host_user_is_postgres() -> None:
    """When the host user is `postgres` there is no separate role to create, and it is reported."""
    assert _sandbox()._create_app_role(Path("/nonexistent-socket"), _BOOTSTRAP_SUPERUSER) is False
