"""Where what the agent makes in the sandbox ends up, and what it is told about it.

On Linux the home of the agent is a temporary folder, deleted at the end of the session.
An agent that saves a module under a path that looks permanent, such as
``~/odoo/custom-addons``, loses it with the session. These tests cover the three guards:

- the working directory is never odev's own home, which with the legacy layout is the
  clone of odev itself;
- the note given to the agent says which folders are kept and which are not;
- what the agent still leaves in its temporary home is moved to the playground.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from odev.plugins.odev_plugin_ai.common.agent import AgentCLI
from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin
from odev.plugins.odev_plugin_ai.common.sandbox.base import Sandbox


# --- working directory -----------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use a fresh home directory, holding odev's home as a git clone (the legacy layout)."""
    home = tmp_path / "home"
    (home / "odev" / ".git").mkdir(parents=True)
    (home / "project" / ".git").mkdir(parents=True)
    # Localized XDG user dirs, as a French desktop writes them: the folder names live
    # only in this config, never hardcoded in the plugin.
    (home / "Téléchargements").mkdir()
    (home / "Documents" / "scratch").mkdir(parents=True)
    config = home / ".config"
    config.mkdir()
    (config / "user-dirs.dirs").write_text(
        'XDG_DOWNLOAD_DIR="$HOME/Téléchargements"\n'
        'XDG_DOCUMENTS_DIR="$HOME/Documents"\n'
        'XDG_DESKTOP_DIR="$HOME"\n',  # disabled dir: points at home, must be ignored
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home


def _sandbox_dirs(home: Path, cwd: Path, *, sandbox: bool = False) -> list[str]:
    """Resolve the sandbox directories of a run started from ``cwd``."""
    command = SimpleNamespace(
        odev=SimpleNamespace(home_path=home / "odev"),
        sandbox_repository=None,
        args=SimpleNamespace(sandbox=sandbox),
    )
    return AICommandMixin._resolve_sandbox_dirs(command, cwd=cwd)


def test_odev_home_repository_uses_playground(home: Path) -> None:
    assert _sandbox_dirs(home, home / "odev") == [str(home / "odev" / "playground")]


def test_playground_itself_uses_playground(home: Path) -> None:
    (home / "odev" / "playground").mkdir()
    assert _sandbox_dirs(home, home / "odev" / "playground") == [str(home / "odev" / "playground")]


def test_other_repository_is_used(home: Path) -> None:
    assert _sandbox_dirs(home, home / "project") == [str(home / "project")]


def test_home_root_folder_uses_playground(home: Path) -> None:
    # A top-level personal folder, named only in user-dirs.dirs (here, localized).
    assert _sandbox_dirs(home, home / "Téléchargements") == [str(home / "odev" / "playground")]


def test_home_itself_uses_playground(home: Path) -> None:
    assert _sandbox_dirs(home, home) == [str(home / "odev" / "playground")]


def test_plain_folder_inside_a_root_folder_is_used(home: Path) -> None:
    # The change: a non-git folder deeper than a top-level home folder opens in place,
    # rather than being redirected to the playground.
    assert _sandbox_dirs(home, home / "Documents" / "scratch") == [str(home / "Documents" / "scratch")]


def test_sandbox_flag_forces_the_playground(home: Path) -> None:
    assert _sandbox_dirs(home, home / "project", sandbox=True) == [str(home / "odev" / "playground")]


def test_without_xdg_config_only_home_is_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bare = tmp_path / "bare"
    (bare / "odev").mkdir(parents=True)
    (bare / "Downloads").mkdir()
    monkeypatch.setenv("HOME", str(bare))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # No user-dirs.dirs: the names are unknown, so a top-level folder opens in place...
    assert _sandbox_dirs(bare, bare / "Downloads") == [str(bare / "Downloads")]
    # ...but the home directory itself is still guarded.
    assert _sandbox_dirs(bare, bare) == [str(bare / "odev" / "playground")]


# --- note to the agent -----------------------------------------------------------


def _note(cwd: str, binds: list[tuple[Path, Path, bool, bool]], *, temporary: bool, database: str | None = None) -> str:
    agent = SimpleNamespace(sandbox=SimpleNamespace(home_is_temporary=temporary))
    return AgentCLI._environment_note(agent, database, cwd, binds, Path("/home/user"))


BINDS = [
    (Path("/home/user/odev/playground"), Path("/home/user/odev/playground"), False, True),
    (Path("/home/user/odoo/custom-addons"), Path("/home/user/odoo/custom-addons"), False, True),
    (Path("/home/user/odev/worktrees"), Path("/home/user/odev/worktrees"), False, False),
    (Path("/home/user/docs"), Path("/home/user/docs"), True, True),
]


def test_note_names_working_directory_as_kept() -> None:
    assert "Your working directory, /home/user/odev/playground, is a folder" in _note(
        "/home/user/odev/playground", BINDS, temporary=True
    )


def test_note_lists_extra_writable_folders_only() -> None:
    note = _note("/home/user/odev/playground", BINDS, temporary=True)
    assert "kept as well: /home/user/odoo/custom-addons." in note


def test_note_warns_of_temporary_home() -> None:
    assert "Your home directory (/home/user) and /tmp are temporary" in _note(
        "/home/user/odev/playground", BINDS, temporary=True
    )


def test_note_does_not_warn_of_a_real_home() -> None:
    assert "temporary" not in _note("/home/user/odev/playground", BINDS, temporary=False)


def test_note_does_not_call_an_unbound_working_directory_kept() -> None:
    assert "Your working directory" not in _note("/home/user", BINDS[3:], temporary=True)


def test_note_names_the_database() -> None:
    assert "database named 'odoo20'" in _note("/home/user/odev/playground", BINDS, temporary=True, database="odoo20")


# --- rescue of the temporary home ------------------------------------------------


@pytest.fixture
def rescued(tmp_path: Path) -> SimpleNamespace:
    """Leave a representative set of files in a temporary home, then rescue it."""
    home = tmp_path / "sandbox-home"
    odev_home = tmp_path / "odev"

    def write(rel: str) -> None:
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(rel)

    # Before the agent starts: what the sandbox set up itself.
    write("mcp/mcp-config.json")
    write(".claude.json")
    (home / "odev" / "playground").mkdir(parents=True)  # a bind point, empty once unmounted
    before = Sandbox._snapshot_files(home)

    # Made by the agent.
    write("odoo/custom-addons/pos_receipt_compact/__manifest__.py")
    write("pos_receipt_compact_zip/pos_receipt_compact.zip")
    write("odoo/custom-addons/pos_receipt_compact/__pycache__/models.cpython-312.pyc")
    write("tool/node_modules/left-pad/index.js")
    write(".claude/history.jsonl")
    write(".bash_history")
    write("mcp/mcp-config.json")  # rewritten, still the sandbox's own
    (home / "link").symlink_to(home / "mcp" / "mcp-config.json")

    sandbox = SimpleNamespace(odev=SimpleNamespace(home_path=odev_home))
    target = Sandbox._rescue_home_files(sandbox, home, before)
    return SimpleNamespace(home=home, target=target, odev_home=odev_home)


def test_rescue_goes_to_the_playground(rescued: SimpleNamespace) -> None:
    assert rescued.target.parent == rescued.odev_home / "playground"


def test_rescue_moves_a_module(rescued: SimpleNamespace) -> None:
    assert (rescued.target / "odoo/custom-addons/pos_receipt_compact/__manifest__.py").is_file()


def test_rescue_moves_a_file_at_the_top(rescued: SimpleNamespace) -> None:
    assert (rescued.target / "pos_receipt_compact_zip/pos_receipt_compact.zip").is_file()


def test_rescue_moves_rather_than_copies(rescued: SimpleNamespace) -> None:
    assert not (rescued.home / "odoo/custom-addons/pos_receipt_compact/__manifest__.py").exists()


def test_rescue_keeps_exactly_what_the_agent_made(rescued: SimpleNamespace) -> None:
    kept = sorted(str(p.relative_to(rescued.target)) for p in rescued.target.rglob("*") if p.is_file())
    assert kept == [
        "odoo/custom-addons/pos_receipt_compact/__manifest__.py",
        "pos_receipt_compact_zip/pos_receipt_compact.zip",
    ]


def test_nothing_to_rescue(tmp_path: Path) -> None:
    home = tmp_path / "sandbox-home"
    (home / ".cache").mkdir(parents=True)
    (home / ".cache" / "x").write_text("x")
    sandbox = SimpleNamespace(odev=SimpleNamespace(home_path=tmp_path / "odev"))
    assert Sandbox._rescue_home_files(sandbox, home, set()) is None
    assert not (tmp_path / "odev").exists()
