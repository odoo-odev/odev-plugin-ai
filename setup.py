"""Plugin setup script, run once by odev when the plugin is installed.

On Ubuntu 24.04+ (and Debian-like systems that ship the same policy),
unprivileged user namespaces are restricted by AppArmor, which prevents the AI
bwrap sandbox from starting (see `common/sandbox/bwrap.py`). Rather than
globally disabling the restriction via
`kernel.apparmor_restrict_unprivileged_userns=0`, we install a targeted
AppArmor profile that grants the `userns` capability to the bwrap binary only.

This mirrors the manual `Bwrap_Profil` script but runs automatically at install
time, and only on Ubuntu-like machines where the restriction is actually active.
"""

import platform
import shutil
import subprocess
from pathlib import Path


try:
    from odev.common.logging import logging

    logger = logging.getLogger(__name__)
except ImportError:  # pragma: no cover - odev is always present when the hook runs
    import logging as _logging

    logger = _logging.getLogger(__name__)


APPARMOR_PROFILE_PATH = Path("/etc/apparmor.d/bwrap")
"""Where the AppArmor profile for the bwrap binary is installed."""

RESTRICT_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
"""Sysctl flag Ubuntu 24.04+ sets to `1` to block unprivileged user namespaces."""

OS_RELEASE_PATHS = (Path("/etc/os-release"), Path("/usr/lib/os-release"))
"""Standard locations of the os-release file, checked in order."""

APPARMOR_PARSER_CANDIDATES = ("/sbin/apparmor_parser", "/usr/sbin/apparmor_parser")
"""Fallback locations for apparmor_parser, which is usually not on the user's PATH."""

PROFILE_TEMPLATE = """abi <abi/4.0>,
include <tunables/global>

profile bwrap {bwrap_path} flags=(unconfined) {{
  userns,
  include if exists <local/bwrap>
}}
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


def _profile_already_installed() -> bool:
    """Whether a bwrap profile granting `userns` is already in place."""
    try:
        return "userns" in APPARMOR_PROFILE_PATH.read_text(encoding="utf-8")
    except OSError:
        return False


def _install_profile(bwrap_path: str, parser: str) -> None:
    """Write the AppArmor profile and load it, using sudo for the privileged steps."""
    profile = PROFILE_TEMPLATE.format(bwrap_path=bwrap_path)

    logger.info(
        "Installing an AppArmor profile so the AI sandbox (bwrap) can use user namespaces. "
        "You may be prompted for your sudo password."
    )

    # `sudo tee` writes the profile as root; stdout is discarded so the profile
    # is not echoed back to the terminal.
    subprocess.run(  # noqa: S603 - fixed argv, profile content passed via stdin
        ["sudo", "tee", str(APPARMOR_PROFILE_PATH)],
        input=profile.encode(),
        stdout=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(  # noqa: S603 - fixed argv built from a validated parser path
        ["sudo", parser, "-r", str(APPARMOR_PROFILE_PATH)],
        check=True,
    )


def _manual_instructions(bwrap_path: str) -> str:
    """Steps for a user to install the profile by hand if the automatic run fails."""
    profile = PROFILE_TEMPLATE.format(bwrap_path=bwrap_path)
    return (
        f"  sudo tee {APPARMOR_PROFILE_PATH} > /dev/null <<'EOF'\n"
        f"{profile}EOF\n"
        f"  sudo apparmor_parser -r {APPARMOR_PROFILE_PATH}"
    )


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

    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        logger.warning(
            "bwrap is not installed; the AI sandbox will not start. "
            "Install it with 'sudo apt install bubblewrap' and re-enable this plugin."
        )
        return

    if _profile_already_installed():
        logger.debug("AppArmor bwrap profile already present, skipping.")
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

    try:
        _install_profile(bwrap_path, parser)
    except (subprocess.CalledProcessError, OSError) as error:
        logger.warning(
            f"Could not install the AppArmor profile for bwrap automatically ({error}).\n"
            f"To enable the AI sandbox manually, run:\n{_manual_instructions(bwrap_path)}"
        )
        return

    logger.info("AppArmor profile installed: the AI sandbox (bwrap) can now start on this system.")
