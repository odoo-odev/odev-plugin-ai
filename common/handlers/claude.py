import json
import shutil
from pathlib import Path

from odev.common.logging import logging

from .base import BaseAgentHandler


logger = logging.getLogger(__name__)


class ClaudeHandler(BaseAgentHandler):
    def get_config_dirs(self):
        return [".claude", ".config/claude"]

    def get_persistent_dirs(self):
        return [".claude", ".config/claude", ".opencode"]

    def get_config_files(self):
        return [".claude.json"]

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
        return ".claude.json"

    def get_skill_target(self):
        return (".claude", "CLAUDE.md")

    def inject_skills(self, target_dir):
        skill_target = self.get_skill_target()
        if not skill_target:
            return

        rel_dir, md_filename = skill_target
        skills_dest = target_dir / "skills"
        skill_refs = []

        for plugin in self.odev.plugins:
            skills_dir = Path(plugin.path) / "skills"
            if not skills_dir.is_dir():
                continue
            for skill_pkg in skills_dir.iterdir():
                if not skill_pkg.is_dir() or not (skill_pkg / "SKILL.md").exists():
                    continue
                skills_dest.mkdir(exist_ok=True)
                # Claude / opencode-cli: inject via @import in the MD file
                shutil.copy2(skill_pkg / "SKILL.md", skills_dest / f"{skill_pkg.name}.md")
                skill_refs.append(f"@skills/{skill_pkg.name}.md")

        if skill_refs:
            md_file = target_dir / md_filename
            existing = md_file.read_text() if md_file.exists() else ""
            with open(md_file, "a") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write("\n".join(skill_refs) + "\n")

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

            # .claude.json trust (project-specific)
            claude_json_file = self.host_home / ".claude.json"
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
