"""Plugin setup/teardown hooks, run by odev when the plugin is (un)installed.

On Ubuntu 24.04+ (and Debian-like systems that ship the same policy),
unprivileged user namespaces are restricted by AppArmor, which prevents the AI
bwrap sandbox from starting (see `common/sandbox/bwrap.py`). Rather than
globally disabling the restriction via
`kernel.apparmor_restrict_unprivileged_userns=0`, we install a targeted
AppArmor profile that grants the `userns` capability to the bwrap binary only.

Recent AppArmor releases (4.x) ship their own `bwrap-userns-restrict` profile
attached to `/usr/bin/bwrap`, which already grants bwrap the user namespaces the
AI sandbox needs. When that profile is present we defer to it - loading it if the
distro left it unparsed - and never install ours alongside it, since two profiles
attached to the same binary do not compose and leave bwrap under the catch-all
restriction. If an earlier version of this plugin installed its own profile next
to the distro one, setup removes it again to clear that conflict.

The profile lives in a file this plugin owns (`/etc/apparmor.d/odev-bwrap`): we
never overwrite a file we did not write, and `teardown()` removes it again when
the plugin is uninstalled. Nothing touches the host without asking first.
"""

import platform
import shutil
import stat
import subprocess
from pathlib import Path


try:
    from odev.common.logging import logging

    logger = logging.getLogger(__name__)
except ImportError:  # pragma: no cover - odev is always present when the hook runs
    import logging as _logging

    logger = _logging.getLogger(__name__)


BWRAP_PATH = Path("/usr/bin/bwrap")
"""Hardcoded location of the bwrap binary the profile is attached to."""

PROFILE_PATH = Path("/etc/apparmor.d/odev-bwrap")
"""AppArmor profile file this plugin owns; never a path shipped by the distro."""

SYSTEM_BWRAP_PROFILES = (
    Path("/etc/apparmor.d/bwrap-userns-restrict"),
    Path("/etc/apparmor.d/bwrap"),
)
"""Distro-shipped bwrap profiles, newest naming first.

AppArmor 4.x (Ubuntu 24.04.2+, 25.04, 26.04) ships the profile as
`bwrap-userns-restrict`; older or alternative packaging used the bare `bwrap`
name. Either attaches a profile to `/usr/bin/bwrap`, so a second profile of ours
on the same binary would not compose with it.
"""

RESTRICT_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
"""Sysctl flag Ubuntu 24.04+ sets to `1` to block unprivileged user namespaces."""

OS_RELEASE_PATHS = (Path("/etc/os-release"), Path("/usr/lib/os-release"))
"""Standard locations of the os-release file, checked in order."""

APPARMOR_PARSER_CANDIDATES = ("/sbin/apparmor_parser", "/usr/sbin/apparmor_parser")
"""Fallback locations for apparmor_parser, which is usually not on the user's PATH."""

MANAGED_MARKER = "Managed by odev-plugin-ai"
"""Marker in the profile file that identifies it as written (and ownable) by us."""

PROFILE_TEMPLATE = f"""abi <abi/4.0>,
include <tunables/global>

# {MANAGED_MARKER}: installed by its setup hook, removed on uninstall.
# Grants the bwrap binary permission to create unprivileged user namespaces,
# which the AI sandbox requires where AppArmor restricts them.
profile odev-bwrap {{bwrap_path}} flags=(unconfined) {{{{
  userns,
  include if exists <local/odev-bwrap>
}}}}
"""
"""AppArmor profile granting the bwrap binary the right to create user namespaces."""


def _os_release() -> dict[str, str]:
    """Parse the os-release file into a plain mapping of its fields."""
    data: dict[str, str] = {}

    for candidate in OS_RELEASE_PATHS:
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue

        for line in text.splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip().strip('"').strip("'")
        break

    return data


def _is_ubuntu_like() -> bool:
    """Whether the current machine is Ubuntu or a Debian-like derivative."""
    fields = _os_release()
    haystack = f"{fields.get('ID', '')} {fields.get('ID_LIKE', '')}".lower()
    return "ubuntu" in haystack or "debian" in haystack


def _find_apparmor_parser() -> str | None:
    """Locate the apparmor_parser binary, which is often outside the user's PATH."""
    parser = shutil.which("apparmor_parser")
    if parser:
        return parser

    for candidate in APPARMOR_PARSER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    return None


def _restriction_active() -> bool:
    """Whether AppArmor is currently blocking unprivileged user namespaces."""
    try:
        return RESTRICT_SYSCTL.read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def _distro_bwrap_profile() -> Path | None:
    """Return the distro-shipped bwrap AppArmor profile file, if one is present."""
    for candidate in SYSTEM_BWRAP_PROFILES:
        if candidate.exists():
            return candidate
    return None


def _bwrap_userns_works() -> bool:
    """Whether bwrap can already create an unprivileged user namespace on this host.

    This mirrors the sandbox's own support probe (`common/sandbox/bwrap.py`): the
    sysctl flag only says the restriction is switched on, not whether an AppArmor
    profile is granting bwrap the exception it needs, so the one reliable answer is
    to ask bwrap to make a namespace and see whether it is allowed to.
    """
    if not shutil.which("bwrap"):
        return False
    try:
        subprocess.run(  # noqa: S607 - bwrap resolved on PATH, fixed arguments
            ["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def _trusted_bwrap_binary() -> Path | None:
    """Resolve `/usr/bin/bwrap` and confirm it is safe to grant extra capabilities.

    Returns the resolved path only if it is a regular file owned by root and not
    writable by group or others; otherwise returns ``None`` so the caller refuses
    to loosen confinement on a binary a non-root user could have tampered with.
    """
    if not BWRAP_PATH.exists():
        return None

    resolved = BWRAP_PATH.resolve()

    try:
        info = resolved.stat()
    except OSError:
        return None

    if not stat.S_ISREG(info.st_mode):
        logger.warning(f"{resolved} is not a regular file; refusing to grant it user-namespace access.")
        return None

    if info.st_uid != 0:
        logger.warning(f"{resolved} is not owned by root; refusing to grant it user-namespace access.")
        return None

    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        logger.warning(f"{resolved} is writable by group or others; refusing to grant it user-namespace access.")
        return None

    return resolved


def _profile_installed() -> bool:
    """Whether the profile file this plugin owns is already in place."""
    try:
        return MANAGED_MARKER in PROFILE_PATH.read_text(encoding="utf-8")
    except OSError:
        return False


def _profile_path_is_foreign() -> bool:
    """Whether our profile path exists but was written by someone other than us."""
    try:
        return MANAGED_MARKER not in PROFILE_PATH.read_text(encoding="utf-8")
    except OSError:
        return False


def _install_profile(bwrap_path: Path, parser: str) -> None:
    """Write the AppArmor profile and load it, using sudo for the privileged steps."""
    profile = PROFILE_TEMPLATE.format(bwrap_path=bwrap_path)

    logger.info(
        "Installing an AppArmor profile so the AI sandbox (bwrap) can use user namespaces. "
        "You may be prompted for your sudo password."
    )

    # `sudo tee` writes the profile as root; stdout is discarded so the profile
    # is not echoed back to the terminal.
    subprocess.run(  # noqa: S603 - fixed argv, profile content passed via stdin
        ["sudo", "tee", str(PROFILE_PATH)],
        input=profile.encode(),
        stdout=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(  # noqa: S603 - fixed argv built from a validated parser path
        ["sudo", parser, "-r", str(PROFILE_PATH)],
        check=True,
    )


def _remove_profile(parser: str | None) -> None:
    """Unload the profile from the kernel and delete the file we installed."""
    logger.info("Removing the AppArmor profile installed for the AI sandbox (bwrap). You may be prompted for sudo.")

    if parser:
        # Unload before deleting: apparmor_parser needs the file to read the
        # profile name. A failure here is non-fatal; we still remove the file.
        subprocess.run(  # noqa: S603 - fixed argv built from a validated parser path
            ["sudo", parser, "-R", str(PROFILE_PATH)],
            check=False,
        )

    subprocess.run(  # noqa: S603 - fixed argv, hardcoded path we own
        ["sudo", "rm", "-f", str(PROFILE_PATH)],
        check=True,
    )


def _defer_to_distro_profile(odev, distro_profile: Path) -> None:
    """Rely on the distro's own bwrap profile rather than installing a rival one.

    The distro profile (e.g. `bwrap-userns-restrict`) already grants bwrap the
    userns capability the AI sandbox needs. We never add ours next to it - and we
    remove ours if an earlier run put it there, since the two do not compose. When
    bwrap still cannot make a namespace, the distro profile shipped but was not
    loaded, so we load it. All of this only happens when something actually needs
    doing, so a host where the distro profile already works is never touched.
    """
    parser = _find_apparmor_parser()
    has_stale_profile = _profile_installed()
    # Probed before removing ours: with a rival profile loaded the result is False
    # anyway, so this only reads True when nothing needs fixing and we can bail out.
    already_works = not has_stale_profile and _bwrap_userns_works()

    if already_works:
        logger.debug(f"Distro bwrap profile {distro_profile.name} is active; nothing to do.")
        return

    actions = []
    if has_stale_profile:
        actions.append(f"remove the conflicting profile this plugin installed at {PROFILE_PATH}")
    actions.append(f"load the distro bwrap profile ({distro_profile})")

    if not parser and not has_stale_profile:
        logger.warning(
            f"A distro bwrap profile is present ({distro_profile}) but bwrap cannot create a "
            "user namespace and apparmor_parser was not found to load it. Load it manually with:\n"
            f"  sudo apparmor_parser -r {distro_profile}"
        )
        return

    if not _confirm(
        odev,
        f"The AI sandbox needs AppArmor to {' and '.join(actions)}. "
        "This runs sudo to update a kernel security policy. Do it now?",
    ):
        logger.info(
            "Skipped. To enable the AI sandbox later, run:\n"
            + (f"  sudo apparmor_parser -R {PROFILE_PATH} && sudo rm {PROFILE_PATH}\n" if has_stale_profile else "")
            + f"  sudo apparmor_parser -r {distro_profile}"
        )
        return

    if has_stale_profile:
        logger.info(
            f"A distro bwrap profile is present ({distro_profile.name}); removing the competing "
            "profile this plugin installed, which does not compose with it."
        )
        try:
            _remove_profile(parser)
        except (subprocess.CalledProcessError, OSError) as error:
            logger.warning(
                f"Could not remove {PROFILE_PATH} automatically ({error}); remove it with 'sudo rm {PROFILE_PATH}'."
            )

    if parser:
        logger.info(f"Loading the distro bwrap AppArmor profile ({distro_profile}) so the AI sandbox can start.")
        subprocess.run(  # noqa: S603 - fixed argv built from a validated parser path and a distro-owned file
            ["sudo", parser, "-r", str(distro_profile)],
            check=False,
        )


def _manual_instructions(bwrap_path: Path) -> str:
    """Steps for a user to install the profile by hand if the automatic run fails."""
    profile = PROFILE_TEMPLATE.format(bwrap_path=bwrap_path)
    return (
        f"  sudo tee {PROFILE_PATH} > /dev/null <<'EOF'\n"
        f"{profile}EOF\n"
        f"  sudo apparmor_parser -r {PROFILE_PATH}"
    )


def _confirm(odev, message: str) -> bool:
    """Ask the user to confirm a privileged host change, defaulting to no.

    Falls back to declining when no interactive console is available so the hook
    never changes the host silently.
    """
    console = getattr(odev, "console", None)
    if console is None:
        return False
    return console.confirm(message, default=False)


def setup(odev) -> None:
    """Enable the AI bwrap sandbox on Ubuntu-like systems at plugin install time."""
    if platform.system() != "Linux":
        return

    if not _is_ubuntu_like():
        logger.debug("Not an Ubuntu-like system, skipping bwrap AppArmor setup.")
        return

    if not _restriction_active():
        logger.debug("Unprivileged user namespaces are not restricted, skipping bwrap AppArmor setup.")
        return

    distro_profile = _distro_bwrap_profile()
    if distro_profile is not None:
        _defer_to_distro_profile(odev, distro_profile)
        return

    bwrap_path = _trusted_bwrap_binary()
    if not bwrap_path:
        logger.warning(
            "bwrap is missing or not a trusted root-owned binary; the AI sandbox will not start. "
            "Install it with 'sudo apt install bubblewrap' and re-enable this plugin."
        )
        return

    if _profile_installed():
        logger.debug("AppArmor odev-bwrap profile already present, skipping.")
        return

    if _profile_path_is_foreign():
        logger.warning(
            f"{PROFILE_PATH} already exists but was not created by this plugin; leaving it untouched. "
            "Remove it yourself if you want the plugin to manage the bwrap profile."
        )
        return

    parser = _find_apparmor_parser()
    if not parser:
        logger.warning(
            "AppArmor restricts user namespaces but apparmor_parser was not found. "
            "As a fallback, run:\n"
            "  echo 'kernel.apparmor_restrict_unprivileged_userns = 0' | "
            "sudo tee /etc/sysctl.d/60-apparmor-namespace.conf && "
            "sudo sysctl --system"
        )
        return

    if not _confirm(
        odev,
        f"The AI sandbox needs an AppArmor profile at {PROFILE_PATH} to run. "
        "This writes to /etc and loads a kernel security policy with sudo. Install it now?",
    ):
        logger.info(f"Skipped. To enable the AI sandbox later, run:\n{_manual_instructions(bwrap_path)}")
        return

    try:
        _install_profile(bwrap_path, parser)
    except (subprocess.CalledProcessError, OSError) as error:
        logger.warning(
            f"Could not install the AppArmor profile for bwrap automatically ({error}).\n"
            f"To enable the AI sandbox manually, run:\n{_manual_instructions(bwrap_path)}"
        )
        return

    logger.info("AppArmor profile installed: the AI sandbox (bwrap) can now start on this system.")


def teardown(odev) -> None:
    """Remove the AppArmor profile installed by :func:`setup`, at uninstall time."""
    if platform.system() != "Linux":
        return

    if not _profile_installed():
        return

    if not _confirm(
        odev,
        f"Remove the AppArmor profile this plugin installed at {PROFILE_PATH}? "
        "This runs sudo to unload a kernel security policy and delete the file.",
    ):
        logger.info(f"Leaving {PROFILE_PATH} in place. Remove it manually with:\n  sudo rm {PROFILE_PATH}")
        return

    try:
        _remove_profile(_find_apparmor_parser())
    except (subprocess.CalledProcessError, OSError) as error:
        logger.warning(
            f"Could not remove the AppArmor profile for bwrap automatically ({error}).\n"
            f"Remove it manually with:\n  sudo rm {PROFILE_PATH}"
        )
        return

    logger.info(f"Removed the AppArmor profile at {PROFILE_PATH}.")
