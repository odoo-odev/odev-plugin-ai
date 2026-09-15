import hashlib
import json
import shutil
from pathlib import Path

from odev.common.logging import logging

from .base import BaseAgentHandler


logger = logging.getLogger(__name__)

GEMINI_SESSIONS_DIR = Path(".gemini") / "tmp"
"""Where Antigravity keeps its conversations, inherited from Gemini CLI.

One directory per project - named after a hash of the project's root path - each holding
a ``chats`` directory of one file per session. The store is per-project on purpose:
Gemini CLI switches history when the working directory changes, so a session belongs to
the place it was held in, and the ``chats`` file name is the session id.
"""

CHATS_SUBDIR = "chats"
"""The per-project sub-directory the session files live in, under ``tmp/<project_hash>``."""

MESSAGE_LIST_KEYS = ("messages", "history", "turns", "conversation", "chat", "entries")
"""Keys a session file might carry its list of messages under when it is a JSON object.

Read defensively rather than assumed: the on-disk shape is Gemini CLI's own detail and
not a promised-stable API, so the list is looked for under any of these and a file that
carries none simply lists without a title.
"""

SESSION_TITLE_KEYS = ("title", "summary", "description", "name")
"""Keys a stored session title might live under - the one ``--list-sessions`` shows."""


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

    def get_latest_session_id(self, cwd=None):
        """Return the id of the last Antigravity conversation, of ``cwd`` for choice.

        Read off the ``chats`` files themselves, newest first: the file name is the
        session id and its mtime is when the session was last spoken to. Empty files are
        skipped - a session started and never used gives a resumed run nothing.

        ``resolves_latest_natively`` sends a bare ``--resume`` straight to ``agy``, so this
        is not what resolves "latest" for a resume; it is what lets a *fresh* run be
        recorded, since :meth:`AgentCLI._record_session` asks the handler which session the
        run just created.
        """
        tmp_dir = self.host_home / GEMINI_SESSIONS_DIR

        if not tmp_dir.is_dir():
            return None

        try:
            chats = sorted(
                (
                    path
                    for path in tmp_dir.glob(f"*/{CHATS_SUBDIR}/*")
                    if path.is_file() and path.stat().st_size
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError as e:
            logger.debug(f"Could not list the Antigravity sessions in {tmp_dir}: {e}")
            return None

        if not chats:
            return None

        if cwd:
            project_chats = self._project_chats_dir(cwd)
            here = [path for path in chats if project_chats and path.parent == project_chats]

            if here:
                return here[0].stem

            logger.warning(
                f"No previous Antigravity session was held in {cwd}; "
                "resuming the most recent one from anywhere instead."
            )

        return chats[0].stem

    def _project_chats_dir(self, cwd) -> Path | None:
        """Return the ``chats`` directory of ``cwd``'s project, or None if there is none.

        Gemini CLI names a project's directory after a hash of its root path; that hash is
        its own detail, so a guess that does not resolve to a real directory just costs the
        cwd match its precision - the caller falls back to the newest session anywhere.
        """
        digest = hashlib.sha256(str(Path(cwd).resolve()).encode()).hexdigest()
        candidate = self.host_home / GEMINI_SESSIONS_DIR / digest / CHATS_SUBDIR
        return candidate if candidate.is_dir() else None

    def _find_session_transcript(self, session_id, cwd=None):
        """Return the ``chats`` file for ``session_id``, across every project, or None.

        Looked up by id rather than under ``cwd``: the id is unique, so a session recorded
        from one directory is still found if its store now sits under another project hash.
        See :meth:`BaseAgentHandler.get_session_info`.
        """
        if not session_id:
            return None

        matches = [
            path
            for path in (self.host_home / GEMINI_SESSIONS_DIR).glob(f"*/{CHATS_SUBDIR}/*")
            if path.is_file() and (path.stem == session_id or session_id in path.name)
        ]
        return matches[0] if matches else None

    def _parse_session_transcript(self, transcript):
        """Read a title, a fallback prompt and token counts out of a session file.

        The file is Gemini CLI's own conversation store: a list of messages, each a role
        and its content, sometimes wrapped in a JSON object that also carries a title and
        session-level token totals. The listing prefers that stored title - the one
        ``--list-sessions`` shows - and falls back to the first user prompt; the last user
        prompt is the further fallback. Input and output tokens are summed apart, the input
        side folding in cache reads, so a resumed session's cache does not read as spend.

        Every field is read with a fallback: the on-disk shape is not a promised-stable
        API, so a message the parser does not recognise costs the listing a detail and
        never the run.
        """
        first_prompt = last_prompt = title = model = None
        tokens_in = tokens_out = 0

        messages, meta = self._load_session(transcript)

        for key in SESSION_TITLE_KEYS:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                title = value.strip()
                break

        for message in messages:
            if not isinstance(message, dict):
                continue

            if self._message_role(message) == "user":
                text = self._message_text(message)
                if text:
                    last_prompt = text
                    if first_prompt is None:
                        first_prompt = text

            if isinstance(message.get("model"), str):
                model = message["model"]

            got_in, got_out = self._message_tokens(message)
            tokens_in += got_in
            tokens_out += got_out

        # Some stores keep the token totals and the model on the wrapper rather than on the
        # messages; only read them there when the messages carried none, to avoid a double.
        if not tokens_in and not tokens_out:
            tokens_in, tokens_out = self._message_tokens(meta)
        if not model and isinstance(meta.get("model"), str):
            model = meta["model"]

        return {
            "title": title or first_prompt,
            "last_prompt": last_prompt,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": model,
        }

    @staticmethod
    def _load_session(transcript: Path) -> tuple[list, dict]:
        """Return ``(messages, meta)`` read out of a session file, both possibly empty.

        The file is JSON when the whole conversation is one object or list, JSONL when it
        is one message per line - both are tried, and a file that is neither yields no
        messages rather than an error.
        """
        text = transcript.read_text()

        try:
            data = json.loads(text)
        except ValueError:
            messages = []
            for line in text.splitlines():
                try:
                    messages.append(json.loads(line))
                except ValueError:
                    continue
            return messages, {}

        if isinstance(data, list):
            return data, {}

        if isinstance(data, dict):
            for key in MESSAGE_LIST_KEYS:
                if isinstance(data.get(key), list):
                    return data[key], data
            return [], data

        return [], {}

    @staticmethod
    def _message_role(message: dict) -> str:
        """Return a message's role folded to ``user`` for a human turn, or its raw role."""
        role = message.get("role") or message.get("type") or message.get("sender") or ""
        return "user" if str(role).lower() in ("user", "human") else str(role).lower()

    @staticmethod
    def _message_text(message: dict) -> str:
        """Return a message's text, joining Gemini's ``parts`` when that is its shape."""
        parts = message.get("parts")
        if isinstance(parts, list):
            pieces = [
                part["text"].strip()
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
            ]
            if pieces:
                return " ".join(pieces)

        for key in ("content", "text", "message", "prompt"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _message_tokens(self, obj: dict) -> tuple[int, int]:
        """Return ``(input, output)`` token counts read off a message or a wrapper.

        The counts hide under one of a few block names - Gemini's ``usageMetadata`` among
        them - and the input side folds in cache reads. A shape the parser does not know
        yields zero, which reads as "no counts" and not as a spend of nothing.
        """
        if not isinstance(obj, dict):
            return (0, 0)

        for key in ("usageMetadata", "usage", "tokens"):
            block = obj.get(key)
            if isinstance(block, dict):
                got_in = self._first_int(block, "input_tokens", "promptTokenCount", "prompt_tokens", "input")
                got_in += self._first_int(block, "cachedContentTokenCount", "cache_read_input_tokens", "cached_tokens")
                got_out = self._first_int(block, "output_tokens", "candidatesTokenCount", "completion_tokens", "output")
                if got_in or got_out:
                    return (got_in, got_out)

        return (0, 0)

    @staticmethod
    def _first_int(block: dict, *keys: str) -> int:
        """Return the first integer ``block`` carries under any of ``keys``, or 0."""
        for key in keys:
            value = block.get(key)
            if isinstance(value, int):
                return value
        return 0

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_server_names=()):  # noqa: PLR0913 - signature set by BaseAgentHandler
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
            except OSError as e:
                # Not fatal - agy asks for a login of its own. Said out loud all the same:
                # silently skipping the copy leaves the agent at a login prompt inside the
                # sandbox with nothing on screen to say why.
                logger.warning(f"Could not seed the Antigravity credentials from Gemini's: {e}")

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
