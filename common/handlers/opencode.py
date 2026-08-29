import shutil
from pathlib import Path

from odev.common.logging import logging

from .claude import ClaudeHandler


logger = logging.getLogger(__name__)


class OpenCodeHandler(ClaudeHandler):
    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):
        # 1. Look in system PATH first
        bin_path = shutil.which("opencode") or shutil.which("opencode-ai")
        print("1", bin_path, type(bin_path))

        # 2. Fall back to standard standalone curl installer path (~/.opencode/bin/opencode)
        if not bin_path:
            fallback = self.host_home / ".opencode" / "bin" / "opencode"
            if fallback.exists():
                bin_path = str(fallback)

        # 3. Fall back to NVM global bin path dynamically (supports any Node version, since it is the recom. installer)
        if not bin_path:
            nvm_node_dir = self.host_home / ".nvm" / "versions" / "node"
            if nvm_node_dir.exists():
                for node_ver in nvm_node_dir.glob("*"):
                    candidate = node_ver / "bin" / "opencode"
                    if candidate.exists():
                        bin_path = str(candidate)
                        break

        # 4. Default to binary name if nothing found
        if not bin_path:
            bin_path = "opencode"
        print("2", bin_path, type(bin_path))

        # Convert to Path object to safely check .exists()
        opencode_bin = Path(bin_path)
        print("1", opencode_bin, type(opencode_bin))

        # Handle relative/system binary names like "opencode" that don't have an absolute path
        if not opencode_bin.is_absolute():
            resolved = shutil.which(str(opencode_bin))
            if resolved:
                opencode_bin = Path(resolved)

        if not opencode_bin.exists():
            logger.error(f"opencode binary not found at {opencode_bin}")
            return []
        cmd = [str(opencode_bin), "run"]
        if prompt:
            cmd.append(prompt)
        if yolo:
            cmd.append("--auto")
        if resume:
            cmd.extend(["--session", resume])
        if model and model != "auto":
            cmd.extend(["-m", model])
        # Target directory / workspace paths
        guest_paths = list(self._guest_paths(all_candidate_paths))
        if guest_paths:
            # Set primary workspace directory
            cmd.extend(["--dir", str(guest_paths[0])])
            # Attach additional candidate paths/files if any
            for extra_path in guest_paths[1:]:
                cmd.extend(["-f", str(extra_path)])
        return cmd
