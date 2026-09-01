import json
import re
from pathlib import Path

from odev.common.logging import logging

from .base import BaseAgentHandler


logger = logging.getLogger(__name__)

PROJECTS_DIR = Path(".claude") / "projects"
"""Where Claude Code keeps its conversations, one directory per working directory.

Each directory is named after the path it belongs to with the separators replaced, and
holds one ``<session-id>.jsonl`` transcript per conversation held there. There is no
index file: the store *is* the listing, and the file name is the session id.
"""

SESSION_CWD_LOOKAHEAD = 50
"""How many entries of a transcript are read looking for the directory it was held in.

The first entries of a transcript are session metadata that carries no path; the
directory shows up with the first message. A ceiling rather than a whole-file scan: a
transcript runs to megabytes, and this only decorates a log line.
"""

NON_SLUG_CHARACTERS = re.compile(r"[^a-zA-Z0-9]+")
"""What a path has replaced by a dash to become the name of its project directory.

Applied to both sides of a comparison rather than used to rebuild the name: which
characters Claude Code folds has changed between its versions - a directory written by
an older one keeps underscores where a newer one dashes them - so a name built here is
compared against names that were built by something else.
"""


class ClaudeHandler(BaseAgentHandler):
    supports_mcp = True

    def get_config_dirs(self):
        return [".claude", ".config/claude"]

    def get_persistent_dirs(self):
        return [".claude", ".config/claude", ".opencode"]

    def get_config_files(self):
        return [".claude.json"]

    def get_creds_files(self):
        return [
            "claude-credentials.json",
            ".credentials.json",
            "hosts.json",
            "hosts.yml",
            "config.yml",
            "settings.json",
            "policy-limits.json",
        ]

    def get_global_config_name(self):
        return ".claude.json"

    def get_agent_config_rel_path(self):
        return ".claude"

    def inject_trust(self, target_dir, trusted_paths):
        super().inject_trust(target_dir, trusted_paths)
        try:
            # Official Claude Code trust
            settings_file = target_dir / "settings.json"
            settings_data = json.loads(settings_file.read_text()) if settings_file.exists() else {}
            trusted_dirs = settings_data.get("trustedDirectories", [])
            if not isinstance(trusted_dirs, list):
                trusted_dirs = []
            for path in trusted_paths:
                if path not in trusted_dirs:
                    trusted_dirs.append(path)
            settings_data["trustedDirectories"] = trusted_dirs
            settings_file.write_text(json.dumps(settings_data, indent=2))

            # .claude.json trust (project-specific)
            claude_json_file = self.host_home / ".claude.json"
            if claude_json_file.exists():
                try:
                    claude_data = json.loads(claude_json_file.read_text())
                    projects = claude_data.setdefault("projects", {})
                    for path in trusted_paths:
                        project = projects.setdefault(path, {})
                        project["hasTrustDialogAccepted"] = True
                    claude_json_file.write_text(json.dumps(claude_data, indent=2))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to inject Claude trust: {e}")

    def cleanup_junk(self, target_dir):
        structures = {
            "projects.json": {"projects": {}},
            "state.json": {},
            "sessions.json": {"sessions": []},
        }
        for junk, structure in structures.items():
            junk_file = target_dir / junk
            if not junk_file.exists():
                junk_file.write_text(json.dumps(structure))

<<<<<<< Updated upstream
    def get_command(
        self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_config=None, mcp_server_names=()
    ):
=======
    def get_latest_session_id(self, cwd=None):
        """Return the id of the last Claude Code conversation, of ``cwd`` for choice.

        Read off the transcripts themselves. What used to be read was
        ``~/.claude/sessions.json``, which Claude Code does not write and
        :meth:`cleanup_junk` here does - as an empty ``{"sessions": []}``, so the lookup
        was reading odev's own placeholder and every ``--resume latest`` ended on "No
        previous session found to resume."

        The sandbox binds a working directory at the same path inside as outside, so the
        project directory of a sandboxed run is the project directory of that path: a
        session held in the sandbox is findable from the host, and the other way round.
        """
        projects_dir = self.host_home / PROJECTS_DIR

        if not projects_dir.is_dir():
            return None

        try:
            # Empty transcripts are sessions that were started and never spoken to;
            # resuming one gives the agent nothing and loses the session that had it.
            transcripts = sorted(
                (path for path in projects_dir.glob("*/*.jsonl") if path.stat().st_size),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError as e:
            logger.debug(f"Could not list the Claude Code sessions in {projects_dir}: {e}")
            return None

        if not transcripts:
            return None

        if cwd:
            slug = self._project_slug(cwd)
            here = [path for path in transcripts if self._project_slug(path.parent.name) == slug]

            if here:
                return here[0].stem

            logger.warning(
                f"No previous Claude Code session was held in {cwd}; "
                "resuming the most recent one from anywhere instead."
            )

        latest = transcripts[0]
        logger.info(f"Resuming the last session of {self._session_cwd(latest) or latest.parent.name}.")
        return latest.stem

    @staticmethod
    def _project_slug(path) -> str:
        """Return a path, or the name of a project directory, in comparable form."""
        return NON_SLUG_CHARACTERS.sub("-", str(path)).strip("-")

    @staticmethod
    def _session_cwd(transcript: Path) -> str | None:
        """Return the directory a session was held in, as the transcript records it.

        Read out of the transcript rather than off the name of the directory holding it:
        that name has the separators of the path replaced by dashes, and a dash in it
        was either a dash or a separator - ``odoo-odev`` and ``odoo/odev`` are written
        the same way. Only used to say where a resumed session comes from, so failing to
        find it costs the log line its detail and nothing else.
        """
        try:
            with transcript.open() as lines:
                for line, _ in zip(lines, range(SESSION_CWD_LOOKAHEAD), strict=False):
                    entry = json.loads(line)

                    if cwd := entry.get("cwd"):
                        return str(cwd)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not read the working directory of {transcript}: {e}")

        return None

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):  # noqa: PLR0913 - signature set by BaseAgentHandler
>>>>>>> Stashed changes
        cmd = ["claude"]
        if prompt:
            if headless:
                cmd.extend(["-p", prompt])
            else:
                cmd.append(prompt)
        if resume:
<<<<<<< Updated upstream
            cmd.extend(["--session-id", resume])

        allowed_tools = ["Bash(rtk:*)", "Bash(odev:*)", "Bash(git:*)", "Bash(pre-commit:*)", "Read", "Edit"]

        if mcp_config:
            # --strict-mcp-config keeps the agent to the servers odev declared, ignoring
            # any config the host user happens to have.
            cmd.extend(["--mcp-config", mcp_config, "--strict-mcp-config"])
            # Tools of a server are named mcp__<server>__<tool>; the bare server name
            # allows all of its tools. Without this the allowlist below blocks them.
            allowed_tools.extend(f"mcp__{server}" for server in mcp_server_names)

        if yolo:
            cmd.append("--dangerously-skip-permissions")
        else:
            cmd.extend(["--permission-mode", "acceptEdits", "--allowedTools", ",".join(allowed_tools)])
=======
            # --resume, not --session-id: the latter *names a new* session, and handed
            # the id of one that already exists it refuses to start at all.
            cmd.extend(["--resume", resume])
        if yolo:
            cmd.append("--dangerously-skip-permissions")
        else:
            cmd.extend(
                [
                    "--permission-mode",
                    "acceptEdits",
                    "--allowedTools",
                    # ``mcp__ps_tools`` covers every tool the Ps-Tools MCP server exposes. Prompting
                    # per call adds no check the server does not already make: the connection is
                    # bound to a single task by a header written host-side, out of the agent's reach.
                    "Bash(rtk:*),Bash(odev:*),Bash(git:*),Bash(pre-commit:*),Read,Edit,mcp__ps_tools",
                ]
            )
>>>>>>> Stashed changes
        if model and model != "auto":
            cmd.extend(["--model", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
