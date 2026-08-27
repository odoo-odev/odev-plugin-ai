import json

from odev.common.logging import logging


logger = logging.getLogger(__name__)


class BaseAgentHandler:
    supports_mcp = False
    """Whether this agent CLI can be handed MCP servers through :meth:`get_command`."""

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
        except Exception as e:
            logger.debug(f"Failed to inject generic trust: {e}")

    def cleanup_junk(self, target_dir):
        """Clean up junk files that might cause leakage or crashes."""

    @classmethod
    def ensure_skills_discoverable(cls) -> None:
        """Reconcile where the `skills` CLI installs skills with where this agent looks for them.

        Called before suggesting a `skills add` command; override when this agent's global
        skills directory differs from what the `skills` npm package targets for it.
        """

    def get_command(
        self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_config=None, mcp_server_names=()
    ):
        """Build the command line for the agent.

        ``mcp_config`` is the guest path of an MCP server config file and
        ``mcp_server_names`` the servers it declares, needed to allow their tools. The
        path is only valid inside the sandbox, so it cannot be read back here. Handlers
        that leave :attr:`supports_mcp` false may ignore both.
        """
        raise NotImplementedError

    def _guest_paths(self, all_candidate_paths: list[str]) -> list[str]:
        return all_candidate_paths
