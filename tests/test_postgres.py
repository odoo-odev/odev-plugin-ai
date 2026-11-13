"""Orphan reaping in `common/postgres.py` must spare the dirs of a session that is still running."""

import os
import time
from pathlib import Path

import pytest

from odev.plugins.odev_plugin_ai.common.postgres import _ORPHAN_MIN_AGE_SECONDS, PostgresSandbox


def _old_dir(root: Path, name: str, *children: str) -> Path:
    """Create an `odev-ai-*` dir, with its `children`, old enough to be taken for an orphan."""
    directory = root / name
    directory.mkdir()
    for child in children:
        (directory / child).mkdir()
    past = time.time() - _ORPHAN_MIN_AGE_SECONDS - 60
    os.utime(directory, (past, past))
    return directory


@pytest.fixture
def scan_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the reaper at `tmp_path` only, never at the host's real /tmp."""
    monkeypatch.setattr(PostgresSandbox, "_scan_dirs", staticmethod(lambda: [tmp_path]))
    return tmp_path


def test_reaps_old_unclaimed_dir(scan_root: Path):
    orphan = _old_dir(scan_root, "odev-ai-claude-orphan")
    PostgresSandbox.cleanup_orphans()
    assert not orphan.exists()


def test_spares_old_dir_of_live_session(scan_root: Path):
    home = _old_dir(scan_root, "odev-ai-claude-live", ".claude")
    with PostgresSandbox.claim_dirs(home):
        PostgresSandbox.cleanup_orphans()
        assert (home / ".claude").is_dir()


def test_reaps_dir_once_its_session_ended(scan_root: Path):
    home = _old_dir(scan_root, "odev-ai-claude-ended")
    with PostgresSandbox.claim_dirs(home):
        pass
    PostgresSandbox.cleanup_orphans()
    assert not home.exists()


def test_spares_young_dir(scan_root: Path):
    young = scan_root / "odev-ai-claude-young"
    young.mkdir()
    PostgresSandbox.cleanup_orphans()
    assert young.is_dir()
