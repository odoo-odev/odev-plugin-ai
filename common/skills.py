"""Install and update the PS agent skills from git, without npm or npx.

Replaces `npx skills add odoo-ps/ps-ai-skills -g`: the repository is cloned once in the odev
home directory, then the skills a command asks for are brought into the global skills directory
of every supported AI CLI. Agents that resolve a symlinked skill directory get a symlink, so a
weekly `git pull` updates them for free; the ones that do not (Antigravity / `agy`) get a copy.

On top of the repository, the skills shipped with the Odoo source of the version the agent runs
against are installed too: they live under `<worktrees>/<version>/odoo/skills`, change from one
version to the next, and are picked up from whichever worktree the run selected.

Links left behind by a previous npm installation are removed so skills keep being updated here.
"""
# ruff: noqa: S101  # the self-check at the bottom of this module is assert-based on purpose

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from odev.common.logging import logging


logger = logging.getLogger(__name__)


SKILLS_REPO = "odoo-ps/ps-ai-skills"
"""Repository holding the PS agent skills."""

# Global skills directory of each supported AI CLI, and whether that agent resolves a symlinked
# skill directory. Antigravity (its IDE and the 'agy' CLI) reads the files directly and does not
# follow symlinks, so its skills are copied in rather than linked.
SKILL_TARGETS = (
    (".claude/skills", True),
    (".gemini/antigravity/skills", False),  # Antigravity IDE
    (".gemini/antigravity-cli/skills", False),  # 'agy', the Antigravity CLI
    (".copilot/skills", True),
    (".config/opencode/skills", True),
)
"""Where skills go per agent, relative to the user home, and whether they are linked or copied."""

NPM_LOCK = ".agents/.skill-lock.json"
"""State file of the `skills` npm CLI, listing the skills it installed and their source."""

MANAGED_MARKER = ".odev-skill-source"
"""Marks a copied skill as managed here, and records the source mtime it was copied at.

Only copies carrying this file are refreshed or pruned: a skill a user dropped in by hand, or
one another tool copied, has no marker and is left untouched.
"""


def _link_target(link: Path) -> Path:
    """Return the path a symlink points to, without resolving it (may be broken)."""
    return Path(os.readlink(link))


def _is_managed(link: Path, roots: list[Path]) -> bool:
    """Whether a symlink points inside one of the directories we install skills from."""
    return link.is_symlink() and any(root in _link_target(link).parents for root in roots)


def _newest_mtime(directory: Path) -> float:
    """Return the most recent mtime found anywhere under the given directory."""
    return max((p.stat().st_mtime for p in directory.rglob("*")), default=0.0)


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


def _skills_in(skills_root: Path, disabled: set[str]) -> dict[str, Path]:
    """Map the declared name of every skill found under `skills_root` to its directory.

    A skill can be disabled by its declared name as well as by its directory name.
    """
    skills: dict[str, Path] = {}

    if not skills_root.is_dir():
        return skills

    for path in sorted(skills_root.iterdir()):
        if not (path / "SKILL.md").exists() or path.name in disabled:
            continue

        name = _skill_name(path)

        if name not in disabled:
            skills[name] = path

    return skills


def _version_skills_root(odev, version: str) -> Path | None:
    """Return the `skills` directory shipped with the Odoo source of `version`, if present.

    Odoo ships version-specific skills under `<worktrees>/<version>/odoo/skills`. The version is
    normalized the way the rest of the plugin resolves a worktree, but the raw name is tried too
    so a worktree created under an exact string is still found.
    """
    from odev.common.version import OdooVersion  # noqa: PLC0415

    worktrees = odev.home_path / "worktrees"
    names = [version]

    try:
        names.append(str(OdooVersion(version)))
    except Exception as e:  # noqa: BLE001 - a free-form version is not worth failing a run over
        logger.debug(f"Could not normalize version {version!r}: {e}")

    for name in dict.fromkeys(names):
        root = worktrees / name / "odoo" / "skills"
        if root.is_dir():
            return root

    return None


def prune_npm_install(skills: set[str], home: Path, managed_roots: list[Path]) -> list[str]:
    """Unlink from the agents the skills a previous `npx skills add` installation left behind.

    They shadow ours under the same name, which would freeze them to the version installed back
    then. Only skills the npm lock file attributes to our repository and that we are about to
    install ourselves are unlinked, so skills coming from another source or managed by the user
    are never touched.

    Only symlinks are removed, and the copies the npm CLI keeps in `~/.agents/skills` are
    deliberately left alone: users do edit them, and nothing else reads that directory. Deleting
    them is `npx skills remove`'s job, not ours.

    :param skills: Names of the skills we are installing.
    :param home: The user home directory in which agent directories live.
    :param managed_roots: The directories our own links point into, to tell them from leftovers.
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

        for relative_dir, follows_symlinks in SKILL_TARGETS:
            # npm leaves symlinks behind; the copy-only agents keep their own marked copies.
            if not follows_symlinks:
                continue

            path = home / relative_dir / name

            if _is_managed(path, managed_roots):
                continue  # a link we already own, not a leftover

            if path.exists() and not path.is_symlink():
                logger.warning(
                    f"Skill {name!r} in {relative_dir!r} is a copy, not a link: it will not be updated. "
                    f"Delete {path.as_posix()} to get the version managed by git."
                )
                continue

            unlinked = _unlink(path) or unlinked

        # Only report what was actually there: the lock file keeps listing the skills long after
        # we unlinked them, and this runs before every agent execution.
        if unlinked:
            removed.append(name)

    return removed


def _prune_managed(target_dir: Path, wanted: dict[str, Path], managed_roots: list[Path], follows_symlinks: bool) -> None:
    """Remove our own entries whose skill disappeared from the sources or was disabled."""
    for entry in target_dir.iterdir():
        if entry.name in wanted:
            continue

        if follows_symlinks:
            if _is_managed(entry, managed_roots):
                logger.debug(f"Removing outdated skill {entry.name!r} from {target_dir.as_posix()}")
                entry.unlink()
        elif entry.is_dir() and (entry / MANAGED_MARKER).is_file():
            logger.debug(f"Removing outdated skill {entry.name!r} from {target_dir.as_posix()}")
            shutil.rmtree(entry, ignore_errors=True)


def _install_link(name: str, path: Path, target_dir: Path, managed_roots: list[Path]) -> None:
    """Symlink one skill into an agent that resolves symlinks, never over something we do not own."""
    link = target_dir / name

    if link.is_symlink():
        if not _is_managed(link, managed_roots):
            logger.debug(f"Skipping skill {name!r} in {target_dir.as_posix()}: linked to another location")
        elif _link_target(link) != path:
            link.unlink()
            link.symlink_to(path, target_is_directory=True)
        return

    if link.exists():
        logger.debug(f"Skipping skill {name!r} in {target_dir.as_posix()}: {link.as_posix()} already exists")
        return

    link.symlink_to(path, target_is_directory=True)


def _install_copy(name: str, src: Path, target_dir: Path) -> None:
    """Copy one skill into an agent that does not follow symlinks, refreshing only our own copies."""
    dst = target_dir / name
    marker = dst / MANAGED_MARKER

    if dst.exists() and not (dst.is_dir() and marker.is_file()):
        # A directory or file we did not create: a user's own skill, left untouched.
        logger.debug(f"Skipping skill {name!r} in {target_dir.as_posix()}: {dst.as_posix()} is not managed here")
        return

    source_mtime = _newest_mtime(src)

    if marker.is_file():
        try:
            if float(marker.read_text().strip() or "0") >= source_mtime:
                return  # already up to date
        except ValueError:
            pass  # unreadable marker: refresh anyway
        shutil.rmtree(dst, ignore_errors=True)

    shutil.copytree(src, dst)
    (dst / MANAGED_MARKER).write_text(str(source_mtime))
    logger.debug(f"Copied skill {name!r} into {target_dir.as_posix()}")


def install_skills(skills: dict[str, Path], home: Path, managed_roots: list[Path]) -> None:
    """Install the given `{name: directory}` skills into every supported, installed agent.

    Agents the user does not have are skipped, as `npx skills` does. Existing directories and
    symlinks that we do not own are never overwritten, our own entries are removed once their
    skill disappears from the sources or is disabled, and links a previous npm installation left
    behind are pruned so they stop shadowing ours.

    :param skills: The skills to install, declared name mapped to source directory.
    :param home: The user home directory in which agent directories live.
    :param managed_roots: The directories the sources live under, to tell our links from others'.
    """
    if pruned := prune_npm_install(set(skills), home, managed_roots):
        logger.info(
            f"Unlinked npm-installed skills, now managed by git: {', '.join(pruned)} "
            f"(their former copies remain in {(home / '.agents/skills').as_posix()})"
        )

    for relative_dir, follows_symlinks in SKILL_TARGETS:
        target_dir = home / relative_dir

        # An agent is considered installed if its configuration directory exists, the very check
        # the `skills` npm CLI does before installing anything for it.
        if not target_dir.parent.is_dir():
            continue

        target_dir.mkdir(exist_ok=True)

        _prune_managed(target_dir, skills, managed_roots, follows_symlinks)

        for name, path in skills.items():
            if follows_symlinks:
                _install_link(name, path, target_dir, managed_roots)
            else:
                _install_copy(name, path, target_dir)


def _refresh_repository(odev, config):
    """Clone the skills repository, or pull it when it has not been refreshed recently.

    :return: A :class:`GitConnector` to the local clone.
    """
    from odev.common.connectors.git import GitConnector  # noqa: PLC0415

    git = GitConnector(SKILLS_REPO, odev.home_path / "skills")

    if not git.exists:
        git.clone()
        config.skills.date = datetime.now()
    elif config.skills.is_refresh_needed():
        git.fetch(detached=False)
        git.pull(force=True)
        config.skills.date = datetime.now()

    return git


def ensure_skills(odev, config, required: list[str], version: str | None = None) -> None:
    """Install the skills a command needs into the supported agents, from git and the Odoo source.

    The requested skills are taken from the PS repository, cloned and pulled here rather than from
    npm. The skills shipped with the Odoo source of `version` are added on top, so an agent working
    on a checkout gets that version's own guidelines next to the shared ones.

    Failures are logged and swallowed: a missing network or SSH key must never prevent an AI agent
    from starting.

    :param odev: The odev framework instance.
    :param config: The odev configuration.
    :param required: Declared names of the skills the command needs from the PS repository.
    :param version: The Odoo version the agent runs against, whose source skills to add, if any.
    """
    try:
        git = _refresh_repository(odev, config)

        disabled = set(config.skills.disabled)
        wanted = set(required)

        skills = {name: path for name, path in _skills_in(git.path / "skills", disabled).items() if name in wanted}

        if version and (version_root := _version_skills_root(odev, version)):
            skills.update(_skills_in(version_root, disabled))

        # The worktrees root is always a managed root, not only when a version is given: a run
        # without one must still prune the version skills a previous '-V' run linked in.
        install_skills(skills, Path.home(), [git.path / "skills", odev.home_path / "worktrees"])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not install the {SKILLS_REPO!r} skills: {e}")


def demo():
    """Self-check of the linking, copying and migration logic on a temporary file tree."""
    import tempfile  # noqa: PLC0415

    root = Path(tempfile.mkdtemp(prefix="odev-skills-demo-"))
    clone_root, worktrees_root, home = root / "repo" / "skills", root / "worktrees", root / "home"
    managed_roots = [clone_root, worktrees_root]
    claude_dir = home / ".claude/skills"
    agy_dir = home / ".gemini/antigravity-cli/skills"

    # Every agent but copilot is installed, hence its configuration directory exists.
    for relative_dir, _ in SKILL_TARGETS:
        if not relative_dir.startswith(".copilot"):
            (home / relative_dir).parent.mkdir(parents=True, exist_ok=True)

    for name in ("odev", "test_skill", "gone"):
        (clone_root / name).mkdir(parents=True)
        (clone_root / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    (clone_root / "not_a_skill").mkdir()
    # A skill whose declared name differs from its directory name, as agents see it.
    (clone_root / "guidelines_dir").mkdir()
    (clone_root / "guidelines_dir" / "SKILL.md").write_text("---\nname: 'guidelines'\ndescription: x\n---\n")

    # A version-specific skill shipped with the Odoo 20.0 source.
    version_root = worktrees_root / "20.0" / "odoo" / "skills"
    (version_root / "odoo-guidelines").mkdir(parents=True)
    (version_root / "odoo-guidelines" / "SKILL.md").write_text("---\nname: odoo-guidelines\n---\n")

    # An 'npx skills add' installation of two of our skills, plus one from another source and one
    # of the user's own: only ours may be unlinked.
    npm_dir = home / ".agents/skills"
    npm_dir.mkdir(parents=True)
    for name in ("odev", "guidelines", "find-skills", "mine"):
        (npm_dir / name).mkdir()
    claude_dir.mkdir()
    for name in ("odev", "guidelines", "find-skills", "mine"):
        (claude_dir / name).symlink_to(npm_dir / name, target_is_directory=True)
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

    # A user's own copy sitting in a copy-only (agy) agent: it has no marker, so it must survive.
    agy_dir.mkdir()
    (agy_dir / "mine").mkdir()
    (agy_dir / "mine" / "SKILL.md").write_text("user's own\n")

    skills = {name: path for name, path in _skills_in(clone_root, set()).items() if name in {"odev", "guidelines"}}
    skills.update(_skills_in(version_root, set()))
    install_skills(skills, home, managed_roots)

    assert _link_target(claude_dir / "test_skill") == foreign, "foreign link was overwritten"
    assert (claude_dir / "keep_me").is_dir() and not (claude_dir / "keep_me").is_symlink(), "directory was replaced"
    assert _link_target(claude_dir / "odev") == clone_root / "odev", "npm-installed skill was not replaced"
    assert (npm_dir / "odev").is_dir(), "npm-installed copy was deleted instead of unlinked"
    assert _link_target(claude_dir / "guidelines") == clone_root / "guidelines_dir", "declared name was not used"
    assert not (claude_dir / "guidelines_dir").exists(), "skill was linked under its directory name"
    assert _link_target(claude_dir / "find-skills") == npm_dir / "find-skills", "skill of another source was pruned"
    assert not (claude_dir / "not_a_skill").exists(), "directory without SKILL.md was linked"
    assert _link_target(claude_dir / "odoo-guidelines") == version_root / "odoo-guidelines", "version skill not linked"
    # agy gets copies, not links, and its directory-based skills carry our marker.
    assert (agy_dir / "odev").is_dir() and not (agy_dir / "odev").is_symlink(), "agy skill was linked, not copied"
    assert (agy_dir / "odev" / MANAGED_MARKER).is_file(), "copied skill was not marked as managed"
    assert (agy_dir / "odoo-guidelines").is_dir(), "version skill was not copied into agy"
    assert (agy_dir / "mine" / "SKILL.md").read_text() == "user's own\n", "user's own agy copy was overwritten"
    assert not (home / SKILL_TARGETS[3][0]).exists(), "skills installed for an agent the user does not have"

    install_skills(skills, home, managed_roots)  # idempotent
    assert _link_target(claude_dir / "odev") == clone_root / "odev", "link lost on second run"
    assert not prune_npm_install({"odev", "guidelines"}, home, managed_roots), "own links reported as leftovers"

    # Removed from the repository and disabled: our entries must be pruned, the rest kept.
    for path in (clone_root / "odev").iterdir():
        path.unlink()
    (clone_root / "odev").rmdir()
    skills = {name: path for name, path in _skills_in(clone_root, {"guidelines_dir"}).items() if name == "guidelines"}
    install_skills(skills, home, managed_roots)
    assert not (claude_dir / "odev").exists(), "orphan link was kept"
    assert not (agy_dir / "odev").exists(), "orphan copy was kept"
    assert not (claude_dir / "guidelines").is_symlink(), "disabled skill was kept"
    assert not (claude_dir / "odoo-guidelines").exists(), "version skill was kept after it left the install set"
    assert _link_target(claude_dir / "test_skill") == foreign, "foreign link was pruned"
    assert (agy_dir / "mine" / "SKILL.md").exists(), "user's own agy copy was pruned"

    print(f"OK: skills install self-check passed ({root})")  # noqa: T201


if __name__ == "__main__":
    demo()
