"""Tests for ``common/handlers/config_dir.py`` Claude config-dir resolution.

Loads the module by file path (importlib) so the test runs without importing the
plugin package (whose ``__init__.py`` pulls in ``odev``). The module under test
imports only the stdlib, so this works cleanly.
"""

import importlib.util
from pathlib import Path


def _plugin_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "__manifest__.py").exists():
            return parent
    raise RuntimeError("Could not locate the plugin root (no __manifest__.py found).")


def _load_config_dir_module():
    path = _plugin_root() / "common" / "handlers" / "config_dir.py"
    spec = importlib.util.spec_from_file_location("claude_config_dir_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


cfg = _load_config_dir_module()


def test_resolve_defaults_to_dot_claude(monkeypatch):
    monkeypatch.delenv("ODEV_CLAUDE_CONFIG_DIR", raising=False)
    assert cfg.resolve_claude_config_dir("") == ".claude"
    assert cfg.resolve_claude_config_dir(None) == ".claude"
    assert cfg.resolve_claude_config_dir("   ") == ".claude"


def test_resolve_uses_configured_when_env_unset(monkeypatch):
    monkeypatch.delenv("ODEV_CLAUDE_CONFIG_DIR", raising=False)
    assert cfg.resolve_claude_config_dir(".claude-odev") == ".claude-odev"


def test_resolve_env_wins_over_configured(monkeypatch):
    monkeypatch.setenv("ODEV_CLAUDE_CONFIG_DIR", ".claude-env")
    assert cfg.resolve_claude_config_dir(".claude-odev") == ".claude-env"


def test_resolve_blank_env_is_ignored(monkeypatch):
    monkeypatch.setenv("ODEV_CLAUDE_CONFIG_DIR", "   ")
    assert cfg.resolve_claude_config_dir(".claude-odev") == ".claude-odev"


def test_is_override():
    assert cfg.is_override(".claude") is False
    assert cfg.is_override(".claude-odev") is True


def test_extra_env_empty_for_default():
    assert cfg.claude_extra_env(Path("/home/u"), ".claude") == {}


def test_extra_env_absolute_path_for_override():
    assert cfg.claude_extra_env(Path("/home/u"), ".claude-odev") == {
        "CLAUDE_CONFIG_DIR": "/home/u/.claude-odev"
    }


def test_global_config_name_default_is_home_root():
    assert cfg.global_config_name(".claude") == ".claude.json"


def test_global_config_name_override_is_inside_dir():
    assert cfg.global_config_name(".claude-odev") == ".claude-odev/.claude.json"


def test_global_config_path():
    assert cfg.global_config_path(Path("/home/u"), ".claude") == Path("/home/u/.claude.json")
    assert cfg.global_config_path(Path("/home/u"), ".claude-odev") == Path(
        "/home/u/.claude-odev/.claude.json"
    )


def test_config_files_default_mounts_claude_json():
    assert cfg.config_files(".claude") == [".claude.json"]


def test_config_files_override_is_empty():
    assert cfg.config_files(".claude-odev") == []
