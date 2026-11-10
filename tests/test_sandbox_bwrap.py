"""Which host paths the bwrap sandbox binds for the agent program itself, in `common/sandbox/bwrap.py`.

Each test lays out a fake installation on `tmp_path`, standing for the root of the host, with
the agent's command on a PATH of its own.
"""

from pathlib import Path

import pytest

from odev.plugins.odev_plugin_ai.common.sandbox import bwrap
from odev.plugins.odev_plugin_ai.common.sandbox.bwrap import agent_install_paths


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host root with an empty PATH directory and a home, `/usr` being the one already bound."""
    (tmp_path / "usr" / "bin").mkdir(parents=True)
    (tmp_path / "home" / "user").mkdir(parents=True)
    monkeypatch.setenv("PATH", str(tmp_path / "usr" / "bin"))
    monkeypatch.setattr(bwrap, "SANDBOX_VISIBLE_ROOTS", (tmp_path / "usr",))
    return tmp_path


def install(path: Path, content: str = "\x7fELF") -> Path:
    """Create an executable file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)
    return path


def test_launcher_script_exposes_the_installation(host):
    """Arch's `claude-code`: `/usr/bin/claude` is a script handing over to `/opt/claude-code/bin/claude`."""
    install(host / "opt" / "claude-code" / "bin" / "claude")
    install(
        host / "usr" / "bin" / "claude",
        f'#!/bin/sh\nexport DISABLE_UPDATES=1\nexec {host}/opt/claude-code/bin/claude "$@"\n',
    )

    assert agent_install_paths("claude", host / "home" / "user") == [host / "opt" / "claude-code"]


def test_symbolic_link_exposes_the_installation(host):
    """A link on PATH into a directory that is not bound exposes the directory the program sits in."""
    target = install(host / "opt" / "agent" / "agent")
    (host / "usr" / "bin" / "agent").symlink_to(target)

    assert agent_install_paths(str(host / "usr" / "bin" / "agent"), host / "home" / "user") == [host / "opt" / "agent"]


def test_program_already_visible(host):
    """Nothing more is bound for a program the sandbox exposes already."""
    install(host / "usr" / "bin" / "agent")

    assert agent_install_paths("agent", host / "home" / "user") == []


def test_program_in_the_home(host, monkeypatch):
    """The home is the playground's: an installation under it is never bound from here."""
    install(host / "home" / "user" / ".local" / "bin" / "agent")
    monkeypatch.setenv("PATH", str(host / "home" / "user" / ".local" / "bin"))

    assert agent_install_paths("agent", host / "home" / "user") == []


def test_missing_program(host):
    """An agent that is not installed binds nothing, the sandbox reports it on its own."""
    assert agent_install_paths("agent", host / "home" / "user") == []
