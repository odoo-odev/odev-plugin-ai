import shutil
from pathlib import Path

from odev.common.logging import logging

from .base import BaseAgentHandler


logger = logging.getLogger(__name__)


class AgyHandler(BaseAgentHandler):
    # `agy --continue` reads Antigravity's own conversation store, which is the only
    # thing that knows which of its conversations is resumable.
    resolves_latest_natively = True

    def get_config_dirs(self):
        return [".antigravity", ".config/antigravity", ".gemini", ".config/gemini"]

    def get_persistent_dirs(self):
        return [".antigravity", ".config/antigravity", ".gemini", ".config/gemini"]

    def get_global_config_name(self):
        return ".antigravity.json"

    def get_config_files(self):
        return [".antigravity.json"]

    def get_creds_files(self):
        return ["antigravity-credentials.json", "gemini-credentials.json", "google_accounts.json", "oauth_creds.json"]

    def get_agent_config_rel_path(self):
        return ".antigravity"

    @classmethod
    def ensure_skills_discoverable(cls) -> None:
        """Make `~/.gemini/config/skills` resolve to the canonical Antigravity skills dir.

        The `skills` CLI installs Antigravity skills under `~/.gemini/antigravity/skills`,
        but Antigravity itself discovers global skills under `~/.gemini/config/skills`.
        Symlinking the latter to the former lets a single `skills add -g -a antigravity`
        satisfy both. If `~/.gemini/config/skills` already exists as a real directory
        (e.g. skills copied there manually), its contents are migrated into the
        canonical directory first.
        """
        home = Path.home()
        target = home / ".gemini" / "antigravity" / "skills"
        link = home / ".gemini" / "config" / "skills"

        try:
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    logger.warning(f"{link} is a symlink to an unexpected location, leaving it as-is.")
                return

            if link.exists():
                target.mkdir(parents=True, exist_ok=True)
                for item in link.iterdir():
                    dest = target / item.name
                    if dest.exists():
                        logger.warning(f"Skipping migration of {item}, {dest} already exists.")
                        continue
                    shutil.move(str(item), str(dest))
                link.rmdir()

            link.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target, target_is_directory=True)
            logger.info(f"Linked {link} -> {target} so Antigravity can discover installed skills.")
        except OSError as e:
            logger.warning(f"Could not set up the Antigravity skills symlink: {e}")

    def get_command(
        self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_config=None, mcp_server_names=()
    ):
        home = Path.home()
        agy_creds = home / ".antigravity" / "oauth_creds.json"
        gemini_creds = home / ".gemini" / "oauth_creds.json"

        if not agy_creds.exists() and gemini_creds.exists():
            try:
                (home / ".antigravity").mkdir(parents=True, exist_ok=True)
                shutil.copy2(gemini_creds, agy_creds)
                gemini_accts = home / ".gemini" / "google_accounts.json"
                if gemini_accts.exists():
                    shutil.copy2(gemini_accts, home / ".antigravity" / "google_accounts.json")
            except Exception:
                pass

        cmd = ["agy"]
        if prompt:
            cmd.extend(["-p" if headless else "-i", prompt])
        if resume:
            if resume == "latest":
                cmd.append("--continue")
            else:
                cmd.extend(["--conversation", resume])
        if yolo:
            cmd.append("--dangerously-skip-permissions")
        if model and model != "auto":
            cmd.extend(["--model", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
