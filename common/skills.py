"""Install and update the PS agent skills from git, without npm or npx.

Replaces `npx skills add odoo-ps/ps-ai-skills -g`: the repository is cloned once in the
odev home directory, then every skill it contains is symlinked into the global skills
directory of each AI CLI agent supported by this plugin. Links left behind by a previous
npm installation are removed so skills keep being updated by the weekly git pull.
"""
# ruff: noqa: S101  # the self-check at the bottom of this module is assert-based on purpose

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from odev.common.logging import logging


logger = logging.getLogger(__name__)


SKILLS_REPO = "odoo-ps/ps-ai-skills"
"""Repository holding the PS agent skills."""

SKILL_DIRS = (
    ".claude/skills",
    ".gemini/antigravity/skills",  # Antigravity IDE
    ".gemini/antigravity-cli/skills",  # 'agy', the Antigravity CLI
    ".copilot/skills",
    ".config/opencode/skills",
)
"""Global skills directories of the supported AI CLIs, relative to the user home."""

NPM_LOCK = ".agents/.skill-lock.json"
"""State file of the `skills` npm CLI, listing the skills it installed and their source."""


def _link_target(link: Path) -> Path:
    """Return the path a symlink points to, without resolving it (may be broken)."""
    return Path(os.readlink(link))


def _is_managed(link: Path, skills_root: Path) -> bool:
    """Whether a symlink points inside the skills repository we manage."""
    return link.is_symlink() and skills_root in _link_target(link).parents


def _skill_name(path: Path) -> str:
    """Return the name a skill declares in its `SKILL.md`, defaulting to its directory name.

    Agents identify a skill by that name, and it does not always match the directory name.
    """
    for line in (path / "SKILL.md").read_text(errors="replace").splitlines()[:20]:
        if line.startswith("name:"):
            return line.removeprefix("name:").strip().strip("\"'") or path.name

    return path.name


def _unlink(path: Path) -> bool:
    """Remove a symlink, even a broken one. False if there was none."""
    if not path.is_symlink():
        return False

    path.unlink()
    return True


def _repository_skills(skills_root: Path, disabled: set[str]) -> dict[str, Path]:
    """Map the name of every skill found in the repository to its directory.

    A skill can be disabled by its declared name as well as by its directory name.
    """
    skills: dict[str, Path] = {}

    for path in sorted(skills_root.iterdir()):
        if not (path / "SKILL.md").exists() or path.name in disabled:
            continue

        name = _skill_name(path)

        if name not in disabled:
            skills[name] = path

    return skills


def prune_npm_install(skills: set[str], home: Path, skills_root: Path) -> list[str]:
    """Unlink from the agents the skills a previous `npx skills add` installation left behind.

    They shadow ours under the same name, which would freeze them to the version installed
    back then. Only skills the npm lock file attributes to our repository and that we are
    about to install ourselves are unlinked, so skills coming from another source or
    managed by the user are never touched.

    Only symlinks are removed, and the copies the npm CLI keeps in `~/.agents/skills` are
    deliberately left alone: users do edit them, and nothing else reads that directory.
    Deleting them is `npx skills remove`'s job, not ours.

    :param skills: Names of the skills found in the repository.
    :param home: The user home directory in which agent directories live.
    :param skills_root: The `skills` directory of the cloned repository.
    :return: The names of the unlinked skills.
    """
    lock = home / NPM_LOCK

    if not lock.exists():
        return []

    try:
        entries = json.loads(lock.read_text())["skills"]
    except (ValueError, KeyError) as e:
        logger.debug(f"Ignoring unreadable {lock.as_posix()}: {e}")
        return []

    removed: list[str] = []

    for name, entry in sorted(entries.items()):
        if not isinstance(entry, dict) or entry.get("source") != SKILLS_REPO or name not in skills:
            continue

        unlinked = False

        for relative_dir in SKILL_DIRS:
            path = home / relative_dir / name

            if _is_managed(path, skills_root):
                continue  # a link we already own, not a leftover

            if path.exists() and not path.is_symlink():
                logger.warning(
                    f"Skill {name!r} in {relative_dir!r} is a copy, not a link: it will not be updated. "
                    f"Delete {path.as_posix()} to get the version managed by git."
                )
                continue

            unlinked = _unlink(path) or unlinked

        # Only report what was actually there: the lock file keeps listing the skills long
        # after we unlinked them, and this runs before every agent execution.
        if unlinked:
            removed.append(name)

    return removed


def sync_links(skills_root: Path, home: Path, disabled: set[str] | None = None) -> None:
    """Symlink every skill of the repository into the agents' global skills directories.

    Agents the user does not have are skipped, as `npx skills` does. Existing directories
    and symlinks that we do not own are never overwritten, and our own links are removed
    once their skill disappears from the repository or is disabled.

    :param skills_root: The `skills` directory of the cloned repository.
    :param home: The user home directory in which agent directories live.
    :param disabled: Names of the skills that must not be installed.
    """
    skills = _repository_skills(skills_root, disabled or set())

    if pruned := prune_npm_install(set(skills), home, skills_root):
        logger.info(
            f"Unlinked npm-installed skills, now managed by git: {', '.join(pruned)} "
            f"(their former copies remain in {(home / '.agents/skills').as_posix()})"
        )

    for relative_dir in SKILL_DIRS:
        target_dir = home / relative_dir

        # An agent is considered installed if its configuration directory exists, the very
        # check the `skills` npm CLI does before installing anything for it.
        if not target_dir.parent.is_dir():
            continue

        target_dir.mkdir(exist_ok=True)

        for link in target_dir.iterdir():
            if link.name not in skills and _is_managed(link, skills_root):
                logger.debug(f"Removing outdated skill {link.name!r} from {relative_dir!r}")
                link.unlink()

        for name, path in skills.items():
            link = target_dir / name

            if link.is_symlink():
                if not _is_managed(link, skills_root):
                    logger.debug(f"Skipping skill {name!r} in {relative_dir!r}: linked to another location")
                elif _link_target(link) != path:
                    link.unlink()
                    link.symlink_to(path, target_is_directory=True)
                continue

            if link.exists():
                logger.debug(f"Skipping skill {name!r} in {relative_dir!r}: {link.as_posix()} already exists")
                continue

            logger.debug(f"Linking skill {name!r} in {relative_dir!r}")
            link.symlink_to(path, target_is_directory=True)


def ensure_skills(odev, config) -> None:
    """Clone or update the skills repository and link its skills to the supported agents.

    Failures are logged and swallowed: a missing network or SSH key must never prevent an
    AI agent from starting.

    :param odev: The odev framework instance.
    :param config: The odev configuration.
    """
    # Imported here so that plain odev commands do not pay for the git connector.
    from odev.common.connectors.git import GitConnector  # noqa: PLC0415

    try:
        git = GitConnector(SKILLS_REPO, odev.home_path / "skills")

        if not git.exists:
            git.clone()
            config.skills.date = datetime.now()
        elif config.skills.is_pull_needed():
            git.fetch(detached=False)
            git.pull(force=True)
            config.skills.date = datetime.now()

        sync_links(git.path / "skills", Path.home(), set(config.skills.disabled))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not install the {SKILLS_REPO!r} skills: {e}")


def demo():
    """Self-check of the linking and migration logic on a temporary file tree."""
    root = Path(tempfile.mkdtemp(prefix="odev-skills-demo-"))
    skills_root, home = root / "repo" / "skills", root / "home"
    claude_dir, copilot_dir = home / SKILL_DIRS[0], home / SKILL_DIRS[3]

    # Every agent but the last one is installed, hence its configuration directory exists.
    installed = SKILL_DIRS[:-1]
    for relative_dir in installed:
        (home / relative_dir).parent.mkdir(parents=True, exist_ok=True)

    for name in ("odev", "test_skill", "gone"):
        (skills_root / name).mkdir(parents=True)
        (skills_root / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    (skills_root / "not_a_skill").mkdir()
    # A skill whose declared name differs from its directory name, as agents see it.
    (skills_root / "guidelines_dir").mkdir()
    (skills_root / "guidelines_dir" / "SKILL.md").write_text("---\nname: 'guidelines'\ndescription: x\n---\n")

    # An 'npx skills add' installation of two of our skills, plus one from another source
    # and one of the user's own: only ours may be unlinked.
    npm_dir = home / ".agents/skills"
    npm_dir.mkdir(parents=True)
    for name in ("odev", "guidelines", "find-skills", "mine"):
        (npm_dir / name).mkdir()
    claude_dir.mkdir()
    for name in ("odev", "guidelines", "find-skills", "mine"):
        (claude_dir / name).symlink_to(npm_dir / name, target_is_directory=True)
    # Where symlinking failed, the npm CLI copies the skill instead: never delete a copy.
    copilot_dir.mkdir()
    (copilot_dir / "odev").mkdir()
    (home / NPM_LOCK).write_text(
        json.dumps(
            {
                "version": 3,
                "skills": {
                    "odev": {"source": SKILLS_REPO},
                    "guidelines": {"source": SKILLS_REPO},
                    "find-skills": {"source": "obra/superpowers"},
                    "mine": {"source": SKILLS_REPO},  # not in the repository (anymore)
                },
            }
        )
    )

    # A skill the user manages himself, and a real directory: both must survive untouched.
    foreign = root / "elsewhere" / "test_skill"
    foreign.mkdir(parents=True)
    (claude_dir / "test_skill").symlink_to(foreign, target_is_directory=True)
    (claude_dir / "keep_me").mkdir()

    sync_links(skills_root, home)
    assert _link_target(claude_dir / "test_skill") == foreign, "foreign link was overwritten"
    assert (claude_dir / "keep_me").is_dir() and not (claude_dir / "keep_me").is_symlink(), "directory was replaced"
    assert _link_target(claude_dir / "odev") == skills_root / "odev", "npm-installed skill was not replaced"
    assert (npm_dir / "odev").is_dir(), "npm-installed copy was deleted instead of unlinked"
    assert _link_target(claude_dir / "guidelines") == skills_root / "guidelines_dir", "declared name was not used"
    assert not (claude_dir / "guidelines_dir").exists(), "skill was linked under its directory name"
    assert _link_target(claude_dir / "find-skills") == npm_dir / "find-skills", "skill of another source was pruned"
    assert (claude_dir / "mine").is_symlink(), "npm-installed skill absent from the repository was unlinked"
    assert not (claude_dir / "not_a_skill").exists(), "directory without SKILL.md was linked"
    assert (copilot_dir / "odev").is_dir() and not (copilot_dir / "odev").is_symlink(), "npm copy was deleted"
    for relative_dir in installed:
        assert (home / relative_dir / "gone").is_symlink(), f"skill missing in {relative_dir}"
    assert not (home / SKILL_DIRS[-1]).exists(), "skills installed for an agent the user does not have"

    sync_links(skills_root, home)  # idempotent
    assert _link_target(claude_dir / "odev") == skills_root / "odev", "link lost on second run"
    assert not prune_npm_install({"odev", "guidelines"}, home, skills_root), "own links reported as leftovers"

    # Removed from the repository and disabled: our links must be pruned, the rest kept.
    for path in (skills_root / "gone").iterdir():
        path.unlink()
    (skills_root / "gone").rmdir()
    sync_links(skills_root, home, disabled={"odev", "guidelines_dir"})
    assert not (claude_dir / "gone").exists(), "orphan link was kept"
    assert not (claude_dir / "odev").is_symlink(), "disabled skill was kept"
    assert not (claude_dir / "guidelines").is_symlink(), "skill disabled by directory name was kept"
    assert _link_target(claude_dir / "test_skill") == foreign, "foreign link was pruned"

    print(f"OK: skills linking self-check passed ({root})")  # noqa: T201


if __name__ == "__main__":
    demo()
