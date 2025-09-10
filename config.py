from odev.common.config import Section


class AiSection(Section):
    _name = "ai"

    @property
    def default_llm(self) -> str:
        """Test config parameter."""
        return self.get("default_llm", "Gemini")

    @default_llm.setter
    def default_llm(self, value: str):
        self.set("default_llm", value)

    @property
    def llm_api_key(self) -> str:
        """Test config parameter."""
        return self.get("llm_api_key", None)

    @llm_api_key.setter
    def llm_api_key(self, value: str):
        self.set("llm_api_key", value)
