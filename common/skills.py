"""Skill management for various AI CLI tools."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from odev.common.logging import logging


if TYPE_CHECKING:
    from odev.common.odev import Odev


logger = logging.getLogger(__name__)


class Skill(NamedTuple):
    name: str
    path: Path
    description: str


class SkillManager:
    """Manages agent skills across multiple AI CLI tools."""

    TOOL_PATHS = {
        "claude": "~/.claude/skills",
        "gemini": "~/.gemini/skills",
        "antigravity": "~/.gemini/antigravity/skills",
        "opencode": "~/.config/opencode/skills",
        "copilot": "~/.copilot/skills",
    }

    def __init__(self, odev: "Odev"):
        self.odev = odev

    def list_available_skills(self) -> list[Skill]:
        """Discover all skills available in enabled plugins."""
        skills = {}
        for plugin in self.odev.plugins:
            skills_dir = plugin.path / "skills"
            if not skills_dir.is_dir():
                continue

            for skill_path in skills_dir.iterdir():
                if not skill_path.is_dir():
                    continue

                if skill_path.name in skills:
                    continue

                skill_md = skill_path / "SKILL.md"
                if not skill_md.exists():
                    continue

                # Try to extract a description from the first few lines of SKILL.md
                description = ""
                try:
                    content = skill_md.read_text()
                    # Check for YAML frontmatter
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            yaml_content = parts[1]
                            match = re.search(r"^description:\s*(.*)$", yaml_content, re.MULTILINE | re.IGNORECASE)
                            if match:
                                desc = match.group(1).strip()
                                # Handle quoted strings and multiline
                                if desc.startswith('"') or desc.startswith("'"):
                                    # Very basic extraction for now, enough for most cases
                                    description = desc.strip("\"'")
                                else:
                                    description = desc

                    if not description:
                        # Fallback to first non-header line
                        lines = content.splitlines()
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith("#") and not line.startswith("-"):
                                description = line
                                break
                except Exception:
                    pass

                skills[skill_path.name] = Skill(skill_path.name, skill_path, description)

        return sorted(skills.values(), key=lambda s: s.name)

    def sync_skills(self, info: bool = False) -> None:
        """Sync enabled skills to all supported AI tools."""
        available_skills = {s.name: s for s in self.list_available_skills()}
        disabled_names = self.odev.config.skills.disabled

        for tool, path_str in self.TOOL_PATHS.items():
            tool_skills_dir = Path(path_str).expanduser()

            if not tool_skills_dir.parent.exists() and tool != "antigravity":
                continue

            if not tool_skills_dir.exists():
                tool_skills_dir.mkdir(parents=True, exist_ok=True)

            # 1. Remove dead symlinks or skills that are now disabled
            for existing in tool_skills_dir.iterdir():
                if existing.is_symlink():
                    target = existing.resolve()
                    # If it points to one of our plugins, and it's disabled, remove it
                    if any(str(target).startswith(str(p.path)) for p in self.odev.plugins):
                        if existing.name in disabled_names or existing.name not in available_skills:
                            existing.unlink()
                            if info:
                                logger.info(f"Removed skill '{existing.name}' from {tool}")

            # 2. Add symlinks for skills that are NOT disabled
            for name, skill in available_skills.items():
                if name in disabled_names:
                    continue

                target_link = tool_skills_dir / name

                if not target_link.exists():
                    try:
                        target_link.symlink_to(skill.path, target_is_directory=True)
                        if info:
                            logger.info(f"Added skill '{name}' to {tool}")
                    except Exception as e:
                        logger.warning(f"Could not link skill '{name}' to {tool}: {e}")
                elif target_link.is_symlink() and target_link.resolve() != skill.path:
                    # Update existing symlink if it points elsewhere
                    target_link.unlink()
                    target_link.symlink_to(skill.path, target_is_directory=True)
                    if info:
                        logger.info(f"Updated skill '{name}' in {tool}")
