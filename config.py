from odev.common.config import Section


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
    def claude_config_dir(self) -> str:
        """Config dir name under $HOME for the Claude account `odev ai` uses (e.g. .claude-odev). Empty = default ~/.claude."""
        return self.get("claude_config_dir", "")

    @claude_config_dir.setter
    def claude_config_dir(self, value: str):
        self.set("claude_config_dir", value)

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
