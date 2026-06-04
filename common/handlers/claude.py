import json

from odev.common.logging import logging

from . import config_dir
from .base import BaseAgentHandler


logger = logging.getLogger(__name__)


class ClaudeHandler(BaseAgentHandler):
    def _config_dir(self) -> str:
        """Resolve the Claude config dir name. The separate-account override
        applies to the 'claude' CLI only; opencode-cli (a subclass) stays on
        '.claude'."""
        if self.cli != "claude":
            return config_dir.DEFAULT_CONFIG_DIR
        try:
            configured = self.odev.config.ai.claude_config_dir
        except Exception:
            configured = ""
        return config_dir.resolve_claude_config_dir(configured)

    def get_config_dirs(self):
        return [self._config_dir(), ".config/claude"]

    def get_persistent_dirs(self):
        return [self._config_dir(), ".config/claude", ".opencode"]

    def get_config_files(self):
        return config_dir.config_files(self._config_dir())

    def get_creds_files(self):
        return [
            "claude-credentials.json",
            ".credentials.json",
            "hosts.json",
            "hosts.yml",
            "config.yml",
            "settings.json",
            "policy-limits.json",
        ]

    def get_global_config_name(self):
        return config_dir.global_config_name(self._config_dir())

    def get_agent_config_rel_path(self):
        return self._config_dir()

    def get_extra_env(self):
        return config_dir.claude_extra_env(self.host_home, self._config_dir())

    def inject_trust(self, target_dir, trusted_paths):
        super().inject_trust(target_dir, trusted_paths)
        try:
            # Official Claude Code trust
            settings_file = target_dir / "settings.json"
            settings_data = json.loads(settings_file.read_text()) if settings_file.exists() else {}
            trusted_dirs = settings_data.get("trustedDirectories", [])
            if not isinstance(trusted_dirs, list):
                trusted_dirs = []
            for path in trusted_paths:
                if path not in trusted_dirs:
                    trusted_dirs.append(path)
            settings_data["trustedDirectories"] = trusted_dirs
            settings_file.write_text(json.dumps(settings_data, indent=2))

            # .claude.json trust (project-specific). Default: ~/.claude.json;
            # override: ~/<dir>/.claude.json (the override account's own file).
            claude_json_file = config_dir.global_config_path(self.host_home, self._config_dir())
            if claude_json_file.exists():
                try:
                    claude_data = json.loads(claude_json_file.read_text())
                    projects = claude_data.setdefault("projects", {})
                    for path in trusted_paths:
                        project = projects.setdefault(path, {})
                        project["hasTrustDialogAccepted"] = True
                    claude_json_file.write_text(json.dumps(claude_data, indent=2))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to inject Claude trust: {e}")

    def cleanup_junk(self, target_dir):
        structures = {
            "projects.json": {"projects": {}},
            "state.json": {},
            "sessions.json": {"sessions": []},
        }
        for junk, structure in structures.items():
            junk_file = target_dir / junk
            if not junk_file.exists():
                junk_file.write_text(json.dumps(structure))

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):
        cmd = ["claude"]
        if prompt:
            if headless:
                cmd.extend(["-p", prompt])
            else:
                cmd.append(prompt)
        if resume:
            cmd.extend(["--session-id", resume])
        if yolo:
            cmd.append("--dangerously-skip-permissions")
        else:
            cmd.extend(
                [
                    "--permission-mode",
                    "acceptEdits",
                    "--allowedTools",
                    "Bash(rtk:*),Bash(odev:*),Bash(git:*),Bash(pre-commit:*),Read,Edit",
                ]
            )
        if model and model != "auto":
            cmd.extend(["--model", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
