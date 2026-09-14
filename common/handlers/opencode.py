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

    def _find_session_transcript(self, session_id, cwd=None):
        """Return None: opencode's transcripts are not in Claude's project store.

        Same reason as :meth:`get_latest_session_id` - the Claude layout is inherited but
        not its session store, so the inherited lookup would read the wrong agent's
        transcripts. Until opencode's own store is wired up, its sessions list in
        ``odev ai --sessions`` without a title or token counts.
        """
        return None

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_server_names=()):  # noqa: PLR0913 - signature set by BaseAgentHandler
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
