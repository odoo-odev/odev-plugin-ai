"""Behaviours of the git-based skills installation in `common/skills.py`.

One test per behaviour, each asserting a single property of the install so the first failure
never hides the rest. They share two fixtures that build a representative file tree on `tmp_path`:

- :func:`installed` installs a set of skills once.
- :func:`pruned` then removes one skill from the sources and disables another, and installs again.

The tree deliberately mixes, in a single home:

- skills from the PS repository, one whose declared name differs from its directory name;
- a version-specific skill shipped under a worktree's Odoo source;
- an `npx skills add` installation (ours, another source's, and the user's own);
- a skill the user symlinks himself, and a plain directory, that must both survive;
- a copy-only agent (`agy`) holding one of the user's own skills.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from odev.plugins.odev_plugin_ai.common.skills import (
    MANAGED_MARKER,
    NPM_LOCK,
    SKILLS_REPO,
    SKILL_TARGETS,
    _skills_in,
    install_skills,
    prune_npm_install,
)


def _target(link: Path) -> Path:
    """Return the path a symlink points to, without resolving it."""
    return link.readlink()


def _make_skill(directory: Path, name: str | None = None) -> None:
    """Create a skill directory with a `SKILL.md` declaring `name` (its directory name by default)."""
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name or directory.name}\n---\n")


@pytest.fixture
def tree(tmp_path: Path) -> SimpleNamespace:
    """Build the representative skills tree and return the paths the tests assert on."""
    clone_root = tmp_path / "repo" / "skills"
    worktrees_root = tmp_path / "worktrees"
    home = tmp_path / "home"
    managed_roots = [clone_root, worktrees_root]
    claude_dir = home / ".claude/skills"
    agy_dir = home / ".gemini/antigravity-cli/skills"

    # Every agent but copilot is installed, hence its configuration directory exists.
    for relative_dir, _ in SKILL_TARGETS:
        if not relative_dir.startswith(".copilot"):
            (home / relative_dir).parent.mkdir(parents=True, exist_ok=True)

    for name in ("odev", "test_skill", "gone"):
        _make_skill(clone_root / name)
    (clone_root / "not_a_skill").mkdir()
    # A skill whose declared name differs from its directory name, as agents see it.
    _make_skill(clone_root / "guidelines_dir", name="'guidelines'")

    # A version-specific skill shipped with the Odoo 20.0 source.
    version_root = worktrees_root / "20.0" / "odoo" / "skills"
    _make_skill(version_root / "odoo-guidelines")

    # An 'npx skills add' installation of two of our skills, plus one from another source and one
    # of the user's own: only ours may be unlinked.
    npm_dir = home / ".agents/skills"
    npm_dir.mkdir(parents=True)
    claude_dir.mkdir()
    for name in ("odev", "guidelines", "find-skills", "mine"):
        (npm_dir / name).mkdir()
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
    foreign = tmp_path / "elsewhere" / "test_skill"
    foreign.mkdir(parents=True)
    (claude_dir / "test_skill").symlink_to(foreign, target_is_directory=True)
    (claude_dir / "keep_me").mkdir()

    # A user's own copy sitting in a copy-only (agy) agent: it has no marker, so it must survive.
    agy_dir.mkdir()
    (agy_dir / "mine").mkdir()
    (agy_dir / "mine" / "SKILL.md").write_text("user's own\n")

    skills = {name: path for name, path in _skills_in(clone_root, set()).items() if name in {"odev", "guidelines"}}
    skills.update(_skills_in(version_root, set()))

    return SimpleNamespace(
        home=home,
        claude_dir=claude_dir,
        agy_dir=agy_dir,
        clone_root=clone_root,
        version_root=version_root,
        npm_dir=npm_dir,
        foreign=foreign,
        managed_roots=managed_roots,
        skills=skills,
    )


@pytest.fixture
def installed(tree: SimpleNamespace) -> SimpleNamespace:
    """Install the skills once into the tree."""
    install_skills(tree.skills, tree.home, tree.managed_roots)
    return tree


@pytest.fixture
def pruned(installed: SimpleNamespace) -> SimpleNamespace:
    """Remove `odev` from the repository, disable `guidelines`, and install again."""
    for path in (installed.clone_root / "odev").iterdir():
        path.unlink()
    (installed.clone_root / "odev").rmdir()

    skills = {
        name: path
        for name, path in _skills_in(installed.clone_root, {"guidelines_dir"}).items()
        if name == "guidelines"
    }
    install_skills(skills, installed.home, installed.managed_roots)
    return installed


# --- Links and copies left alone -------------------------------------------------------------


def test_foreign_link_is_kept(installed: SimpleNamespace) -> None:
    """A symlink the user points elsewhere is never overwritten, even under a skill's name."""
    assert _target(installed.claude_dir / "test_skill") == installed.foreign


def test_plain_directory_is_kept(installed: SimpleNamespace) -> None:
    """A real directory the user dropped in is not replaced by a link."""
    keep_me = installed.claude_dir / "keep_me"
    assert keep_me.is_dir() and not keep_me.is_symlink()


def test_other_source_skill_is_not_pruned(installed: SimpleNamespace) -> None:
    """A skill the npm lock attributes to another source keeps its own link."""
    assert _target(installed.claude_dir / "find-skills") == installed.npm_dir / "find-skills"


def test_directory_without_skill_md_is_not_linked(installed: SimpleNamespace) -> None:
    """A directory carrying no `SKILL.md` is not a skill and is not linked."""
    assert not (installed.claude_dir / "not_a_skill").exists()


# --- npm leftovers taken over ----------------------------------------------------------------


def test_npm_leftover_is_relinked_to_git(installed: SimpleNamespace) -> None:
    """One of our skills installed by npm is relinked to the git clone so it updates from now on."""
    assert _target(installed.claude_dir / "odev") == installed.clone_root / "odev"


def test_npm_copy_is_unlinked_not_deleted(installed: SimpleNamespace) -> None:
    """Only the leftover link is removed; the copy npm keeps in `~/.agents/skills` is left alone."""
    assert (installed.npm_dir / "odev").is_dir()


# --- Declared name honoured ------------------------------------------------------------------


def test_declared_name_is_used(installed: SimpleNamespace) -> None:
    """A skill is linked under the name it declares in `SKILL.md`, not its directory name."""
    assert _target(installed.claude_dir / "guidelines") == installed.clone_root / "guidelines_dir"


def test_skill_not_linked_under_directory_name(installed: SimpleNamespace) -> None:
    """The directory name is not used when a declared name differs from it."""
    assert not (installed.claude_dir / "guidelines_dir").exists()


# --- Version skills --------------------------------------------------------------------------


def test_version_skill_is_linked(installed: SimpleNamespace) -> None:
    """A skill shipped with the Odoo source of the version is linked next to the shared ones."""
    assert _target(installed.claude_dir / "odoo-guidelines") == installed.version_root / "odoo-guidelines"


# --- Copy-only agents (agy) ------------------------------------------------------------------


def test_agy_gets_a_copy_not_a_link(installed: SimpleNamespace) -> None:
    """An agent that does not follow symlinks gets a real copy of the skill."""
    odev = installed.agy_dir / "odev"
    assert odev.is_dir() and not odev.is_symlink()


def test_agy_copy_is_marked_managed(installed: SimpleNamespace) -> None:
    """A copy we create carries the marker that lets a later run refresh or prune it."""
    assert (installed.agy_dir / "odev" / MANAGED_MARKER).is_file()


def test_agy_version_skill_is_copied(installed: SimpleNamespace) -> None:
    """Version skills reach copy-only agents too."""
    assert (installed.agy_dir / "odoo-guidelines").is_dir()


def test_agy_user_copy_is_not_overwritten(installed: SimpleNamespace) -> None:
    """An unmarked copy in a copy-only agent is the user's own and is never overwritten."""
    assert (installed.agy_dir / "mine" / "SKILL.md").read_text() == "user's own\n"


# --- Installed-agent detection ---------------------------------------------------------------


def test_uninstalled_agent_is_skipped(installed: SimpleNamespace) -> None:
    """No skills directory is created for an agent whose configuration directory is absent."""
    copilot_dir = next(relative for relative, _ in SKILL_TARGETS if relative.startswith(".copilot"))
    assert not (installed.home / copilot_dir).exists()


# --- Idempotence -----------------------------------------------------------------------------


def test_install_is_idempotent(installed: SimpleNamespace) -> None:
    """A second install leaves the links untouched."""
    install_skills(installed.skills, installed.home, installed.managed_roots)
    assert _target(installed.claude_dir / "odev") == installed.clone_root / "odev"


def test_own_links_are_not_reported_as_leftovers(installed: SimpleNamespace) -> None:
    """The npm-leftover pruning does not report the links we already own."""
    assert not prune_npm_install({"odev", "guidelines"}, installed.home, installed.managed_roots)


# --- Pruning on removal and disabling --------------------------------------------------------


def test_orphan_link_is_pruned(pruned: SimpleNamespace) -> None:
    """A link is removed once its skill disappears from the sources."""
    assert not (pruned.claude_dir / "odev").exists()


def test_orphan_copy_is_pruned(pruned: SimpleNamespace) -> None:
    """A managed copy is removed once its skill disappears from the sources."""
    assert not (pruned.agy_dir / "odev").exists()


def test_disabled_skill_is_removed(pruned: SimpleNamespace) -> None:
    """A skill disabled by its directory name is no longer linked."""
    assert not (pruned.claude_dir / "guidelines").is_symlink()


def test_version_skill_removed_when_out_of_set(pruned: SimpleNamespace) -> None:
    """A version skill is removed once a run no longer asks for it (e.g. a run without '-V')."""
    assert not (pruned.claude_dir / "odoo-guidelines").exists()


def test_foreign_link_survives_pruning(pruned: SimpleNamespace) -> None:
    """Pruning never touches a link the user points elsewhere."""
    assert _target(pruned.claude_dir / "test_skill") == pruned.foreign


def test_agy_user_copy_survives_pruning(pruned: SimpleNamespace) -> None:
    """Pruning never touches an unmarked copy the user owns in a copy-only agent."""
    assert (pruned.agy_dir / "mine" / "SKILL.md").exists()
