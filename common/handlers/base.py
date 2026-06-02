import json

from odev.common.logging import logging


logger = logging.getLogger(__name__)


class BaseAgentHandler:
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
        return None

    def get_global_config_name(self):
        """Return name of global config file (e.g. .claude.json)."""
        return None

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

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):
        """Build the command line for the agent."""
        raise NotImplementedError()

    def _guest_paths(self, all_candidate_paths: list[str]) -> list[str]:
        return all_candidate_paths
