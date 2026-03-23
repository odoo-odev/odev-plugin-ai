from odev.common.config import Section


class AiSection(Section):
    _name = "ai"

    @property
    def favorite_cli(self) -> str:
        """The favorite AI CLI agent to use."""
        return self.get("favorite_cli", "")

    @favorite_cli.setter
    def favorite_cli(self, value: str):
        self.set("favorite_cli", value)
