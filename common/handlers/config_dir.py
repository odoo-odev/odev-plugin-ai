"""Resolution of the Claude config-directory override (separate-account support).

Kept dependency-free (stdlib only — no ``odev`` imports, no relative imports) so
the precedence + path-derivation logic is unit-testable in isolation by loading
this file directly via ``importlib``.

Layout asymmetry handled here (verified against ``claude`` v2.1.162):
  * Default (no override): the global config is ``~/.claude.json`` (HOME root).
  * Override (CLAUDE_CONFIG_DIR set): the global config is ``$DIR/.claude.json``
    (inside the dir), alongside ``$DIR/{projects,sessions,backups}/``.
"""

import os
from pathlib import Path


#: Env var that overrides the configured value for a single invocation.
ENV_VAR = "ODEV_CLAUDE_CONFIG_DIR"
#: Default Claude config directory name (relative to $HOME).
DEFAULT_CONFIG_DIR = ".claude"
#: Claude's global per-user config/state file name.
GLOBAL_CONFIG_FILE = ".claude.json"


def resolve_claude_config_dir(configured):
    """Return the Claude config dir NAME (relative to $HOME).

    Precedence: ``$ODEV_CLAUDE_CONFIG_DIR`` env var > ``configured`` (odev config
    value) > ``.claude``. Empty/whitespace values are treated as unset.
    """
    env_val = (os.environ.get(ENV_VAR) or "").strip()
    if env_val:
        return env_val
    configured = (configured or "").strip()
    return configured or DEFAULT_CONFIG_DIR


def is_override(config_dir):
    """True when ``config_dir`` is a non-default (separate-account) directory."""
    return config_dir != DEFAULT_CONFIG_DIR


def claude_extra_env(host_home, config_dir):
    """Extra sandbox env for the dir. ``{}`` for the default (no behavior change);
    ``{"CLAUDE_CONFIG_DIR": "<abs path>"}`` for an override so the sandboxed
    ``claude`` reads that dir and its namespaced credentials."""
    if not is_override(config_dir):
        return {}
    return {"CLAUDE_CONFIG_DIR": str(Path(host_home) / config_dir)}


def global_config_name(config_dir):
    """Relative name of the global config, accounting for the layout asymmetry.

    Default → ``.claude.json`` (HOME root). Override → ``<dir>/.claude.json``
    (inside the dir), which makes the sandbox's ``_prepare_agent_config`` treat
    it as already covered by the persistent bind-mount and skip copying the
    default account's file.
    """
    if not is_override(config_dir):
        return GLOBAL_CONFIG_FILE
    return f"{config_dir}/{GLOBAL_CONFIG_FILE}"


def global_config_path(host_home, config_dir):
    """Absolute path to the global config file for ``config_dir``."""
    return Path(host_home) / global_config_name(config_dir)


def config_files(config_dir):
    """Host config files to bind-mount. Default mounts ``~/.claude.json``;
    an override returns ``[]`` because the file lives inside the bind-mounted
    config dir already (mounting the default account's file would be wrong)."""
    return [] if is_override(config_dir) else [GLOBAL_CONFIG_FILE]
