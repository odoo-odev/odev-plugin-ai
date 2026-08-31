import json

from odev.common.logging import logging


logger = logging.getLogger(__name__)


class BaseAgentHandler:
    resolves_latest_natively: bool = False
    """Whether the CLI resolves "the latest session" on its own.

    An agent that does gets the ask passed straight through to it, and is the better
    answer: it reads its own store, and knows which of its sessions is resumable.
    """

    def __init__(self, cli, host_home, odev):
        self.cli = cli
        self.host_home = host_home
        self.odev = odev

    def get_config_dirs(self):
        """Return relative paths of config directories for this agent."""
        return []

    def get_persistent_dirs(self):
        """Return which config directories should be persistent (bind-mounted)."""
        return []

    def get_config_files(self):
        """Return relative paths of host files that should be bind-mounted."""
        return []

    def get_creds_files(self):
        """Return names of credential files to copy if not persistent."""
        return []

    def get_agent_config_rel_path(self):
        """Return the relative path to the agent's main configuration directory."""
        return

    def get_global_skills_dir(self):
        """Return the directory this agent reads its global skills from.

        Only meaningful for agents the skills CLI does not install to. It keeps
        the shared ~/.agents/skills store up to date and symlinks it into
        ~/.claude/skills, so Claude Code needs nothing extra and returns None.
        """
        return

    def get_global_config_name(self):
        """Return name of global config file (e.g. .claude.json)."""
        return

    def inject_trust(self, target_dir, trusted_paths):
        """Inject trusted paths into the agent's config."""
        try:
            # Standard trustedFolders.json supported by many odev-compatible agents
            trust_file = target_dir / "trustedFolders.json"
            trust_data = json.loads(trust_file.read_text()) if trust_file.exists() else {}
            for path in trusted_paths:
                trust_data[path] = "TRUST_FOLDER"
            trust_file.write_text(json.dumps(trust_data, indent=2))
        except (OSError, ValueError, AttributeError) as e:
            logger.debug(f"Failed to inject generic trust: {e}")

    def cleanup_junk(self, target_dir):
        """Clean up junk files that might cause leakage or crashes."""

    @classmethod
    def ensure_skills_discoverable(cls) -> None:
        """Reconcile where the `skills` CLI installs skills with where this agent looks for them.

        Called before suggesting a `skills add` command; override when this agent's global
        skills directory differs from what the `skills` npm package targets for it.
        """

    def get_latest_session_id(self, cwd=None):
        """Return the id of the most recent session of this agent, or None.

        Each CLI keeps its conversations in a store of its own shape, so the answer
        belongs to the handler rather than to :class:`AgentCLI`: a lookup written for
        one agent and applied to all of them is a lookup that finds nothing for the
        others, and reports it as "no previous session" rather than as "I do not know
        where this agent keeps them".

        :param cwd: The directory the resumed run works in. Sessions are per-directory
            for most agents, and the one to resume is the last one of *this* place.
        """
        return

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_server_names=()):  # noqa: PLR0913 - every agent needs the full invocation context
        """Build the command line for the agent."""
        raise NotImplementedError

    def get_mcp_config_args(self, mcp_config_path: str | None) -> list[str]:
        """Return extra CLI args wiring up an MCP config file.

        Default: this agent CLI has no known MCP flag, so requested servers are
        dropped with a warning instead of silently changing what the agent can do.
        """
        if mcp_config_path:
            logger.warning(f"The {self.cli!r} CLI does not support MCP servers; ignoring the ones configured for it.")
        return []

    def _guest_paths(self, all_candidate_paths: list[str]) -> list[str]:
        return all_candidate_paths
