from datetime import datetime, timedelta

from odev.common.config import Section


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
"""How a refresh date is written to the config file, matching odev's own sections."""


class AiSection(Section):
    _name = "ai"

    @property
    def cli(self) -> str:
        """The default AI CLI agent to use."""
        return self.get("cli", "")

    @cli.setter
    def cli(self, value: str):
        self.set("cli", value)

    @property
    def favorite_cli(self) -> str:
        """The default AI CLI agent to use."""
        return self.get("favorite_cli", self.cli or "claude")

    @favorite_cli.setter
    def favorite_cli(self, value: str):
        self.set("favorite_cli", value)

    @property
    def offer_github_access(self) -> str:
        """Whether to offer giving the AI agent a GitHub token, for `gh` to open pull requests.

        'ask' offers it the next time an agent starts, 'never' stops offering it. Set
        back to 'ask' to be offered again. Only offered when `gh` is installed.
        """
        return self.get("offer_github_access", "ask")

    @offer_github_access.setter
    def offer_github_access(self, value: str):
        if value not in ("ask", "never"):
            raise ValueError(f"'ai.offer_github_access' must be 'ask' or 'never', got {value!r}")

        self.set("offer_github_access", value)

    @property
    def gh_tip_shown(self) -> str:
        """Whether the tip about installing `gh` was shown already ('true' or 'false')."""
        return self.get("gh_tip_shown", "false")

    @gh_tip_shown.setter
    def gh_tip_shown(self, value: str):
        self.set("gh_tip_shown", str(value).lower())

    def get_favorite_model(self, cli: str) -> str | None:
        """Return the favorite model specifically for the given CLI agent."""
        return self.get(f"favorite_model_{cli}")

    def set_favorite_model(self, cli: str, model: str):
        """Set the favorite model specifically for the given CLI agent."""
        self.set(f"favorite_model_{cli}", model)


class SkillsSection(Section):
    _name = "skills"

    @property
    def disabled(self) -> list[str]:
        """List of disabled skills."""
        return [s for s in self.get("disabled", "").split(",") if s]

    @disabled.setter
    def disabled(self, value: list[str]):
        self.set("disabled", ",".join(value))

    @property
    def date(self) -> datetime:
        """Last time the installed skills were refreshed from their repository.

        You should not have to modify this value as it is updated automatically.
        Missing, it reads as the epoch: a store that has never been refreshed is stale,
        which is the case of every store installed before this was tracked.
        """
        value = self.get("date")

        if not value:
            return datetime.fromtimestamp(0)

        return datetime.strptime(value, DATETIME_FORMAT)

    @date.setter
    def date(self, value: str | datetime):
        self.set("date", value.strftime(DATETIME_FORMAT) if isinstance(value, datetime) else value)

    @property
    def interval(self) -> int:
        """Interval between skill refreshes, in days. Defaults to 1, as odev's own updates.

        Set it to 0 to refresh on every run, which costs a clone of the skills repository
        each time a command starts an agent.
        """
        return int(self.get("interval", "1"))

    @interval.setter
    def interval(self, value: str | int):
        if not str(value).isdigit() or int(value) < 0:
            raise ValueError(f"'skills.interval' must be a positive integer, got {value!r}")

        self.set("interval", str(value))

    def is_refresh_needed(self) -> bool:
        """Return whether the installed skills are old enough to be fetched again.

        A method and not a property: ``Config.fill_defaults`` writes every property of a
        section back through its setter to materialise defaults, so a read-only property
        here crashes odev on the first run after this section changed.
        """
        return datetime.now() >= self.date + timedelta(days=self.interval)
