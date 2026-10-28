"""Plugin setup/teardown hooks, run by odev when the plugin is (un)installed.

On Ubuntu 24.04+ (and Debian-like systems that ship the same policy),
unprivileged user namespaces are restricted by AppArmor, which prevents the AI
bwrap sandbox from starting (see `common/sandbox/bwrap.py`). Rather than
globally disabling the restriction via
`kernel.apparmor_restrict_unprivileged_userns=0`, we install a targeted
AppArmor profile that grants the `userns` capability to the bwrap binary only.

Recent AppArmor releases (4.1+) ship their own `bwrap-userns-restrict` profile
attached to `/usr/bin/bwrap`; when that is present the system already solves the
problem and we leave it alone, as a second profile attached to the same binary
would conflict with it.

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

SYSTEM_BWRAP_PROFILE = Path("/etc/apparmor.d/bwrap")
"""Distro-shipped bwrap profile (e.g. `bwrap-userns-restrict` on AppArmor 4.1+)."""

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

    if SYSTEM_BWRAP_PROFILE.exists():
        logger.debug(
            f"A distro-provided bwrap profile is present ({SYSTEM_BWRAP_PROFILE}); "
            "leaving it in place instead of installing a conflicting one."
        )
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
