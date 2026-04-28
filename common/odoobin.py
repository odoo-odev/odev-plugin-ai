"""AI-specific OdoobinProcess that filters log output when running inside a sandbox."""

import os
import re
from collections.abc import Callable
from typing import ClassVar

from odev.common import string
from odev.common.odoobin import OdoobinProcess


class AI_OdoobinProcess(OdoobinProcess):
    """OdoobinProcess variant used inside AI sandboxes.

    Overrides get_stream_filter() to reduce token consumption by filtering out
    noisy Odoo log lines (e.g., werkzeug 200/304 requests, timestamps, PIDs).
    """

    LOG_REGEX_SHORT: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<time>\d{2}:\d{2}:\d{2},\d{3}) (?P<level>\w+) (?P<database>[\w?]+) (?P<logger>[\w.]+): (?P<description>.*)$"
    )

    def _print_run_info(self, info_message: str, formatted_command: str) -> None:
        """Suppress the run-info echo inside the AI sandbox to save tokens."""

    def get_stream_filter(self) -> Callable[[str], str | None] | None:
        """Return the AI sandbox log filter when running inside a sandbox."""
        if os.environ.get("AI_SANDBOX") == "1":
            return self._ai_sandbox_filter
        return None

    def _ai_sandbox_filter(self, line: str) -> str | None:
        """Filter Odoo logs to reduce token consumption when running in an AI sandbox."""
        line = string.strip_ansi_colors(line).replace("\r", "")

        match = self.LOG_REGEX.match(line) or self.LOG_REGEX_SHORT.match(line)

        if match:
            description = match.group("description")
            logger_name = match.group("logger")

            # Suppress noisy werkzeug logs for successful requests
            if logger_name == "werkzeug":
                w_match = self.LOG_WERKZEUG_REGEX.match(description)
                if w_match and w_match.group("code") in ("200", "304"):
                    return None

            # Return a cleaned version without date, time, pid, and database name
            return f"{match.group('level')} {logger_name}: {description}"

        return line
