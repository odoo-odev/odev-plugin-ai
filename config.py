from datetime import datetime, timedelta
from typing import cast

from odev.common.config import DATETIME_FORMAT, Section


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
        """Last time the skills repository was pulled from GitHub.
        You should not have to modify this value as it is updated automatically.
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
        """Interval in days between two pulls of the skills repository. Defaults to 7 days."""
        return int(cast(str, self.get("interval", "7")))

    @interval.setter
    def interval(self, value: str | int):
        if not str(value).isdigit() or int(value) < 0:
            raise ValueError(f"'skills.interval' must be a positive integer, got {value!r}")

        self.set("interval", str(value))

    def is_pull_needed(self) -> bool:
        """Whether the skills repository should be pulled again.

        Not a property: read-only properties break `Config.fill_defaults()`.
        """
        return datetime.now() >= self.date + timedelta(days=self.interval)
