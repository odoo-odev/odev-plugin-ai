from odev.common.logging import logging

from .claude import ClaudeHandler


logger = logging.getLogger(__name__)


class OpenCodeHandler(ClaudeHandler):
    def get_latest_session_id(self, cwd=None):
        """Return nothing: opencode keeps its sessions somewhere else.

        The Claude config layout is inherited here, its session store is not - and
        answering with a Claude session id would resume the wrong agent's conversation.
        """
        return

    def get_command(
        self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_config=None, mcp_server_names=()
    ):
        opencode_bin = self.host_home / ".opencode/bin/opencode"
        if not opencode_bin.exists():
            logger.error(f"opencode binary not found at {opencode_bin}")
            return []
        cmd = [str(opencode_bin), "run"]
        if prompt:
            cmd.append(prompt)
        if resume:
            cmd.extend(["--session", resume])
        if model and model != "auto":
            cmd.extend(["-m", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
