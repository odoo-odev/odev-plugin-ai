from collections.abc import Iterable
from typing import cast

from odev.common.config import Section


class AiSection(Section):
    _name = "ai"

    @property
    def llm_order(self) -> list[str]:
        """Default LLM."""
        return [llm for llm in cast(str, self.get("llm_order", "")).split(",") if llm]

    @llm_order.setter
    def llm_order(self, value: str | Iterable[str]):
        self.set("llm_order", value if isinstance(value, str) else ",".join(list(value)))
