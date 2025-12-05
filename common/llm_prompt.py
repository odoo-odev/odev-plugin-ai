import base64
from typing import Any, Union


class LLMFile:
    """Represents a file to be included in the LLM prompt."""

    __slots__ = ["path", "content"]

    def __init__(self, path: str, content: str):
        self.path = path
        self.content = content


class LLMPrompt:
    """A class to build and format prompts for LLMs, handling system messages, user text, and files."""

    def __init__(self) -> None:
        self._system: str | None = None
        self._user_parts: list[Union[dict[str, str], LLMFile]] = []

    def set_system(self, content: str) -> None:
        """Set the system message."""
        self._system = content

    def add_user(
        self,
        content: Union[str, list[str], LLMFile, list[LLMFile], list[Union[str, LLMFile]]],
    ) -> None:
        """Add user content. Can be a string, a list of strings, an LLMFile, or a list of LLMFiles."""
        if isinstance(content, list):
            for item in content:
                self.add_user(item)
        elif isinstance(content, str):
            self._user_parts.append({"type": "text", "text": content})
        elif isinstance(content, LLMFile):
            self._user_parts.append(content)
        elif isinstance(content, dict):
            self._user_parts.append(content)
        else:
            # Fallback for other types if necessary, or raise error
            pass

    def add_file(self, path: str, content: str) -> None:
        """Add a file to the user message."""
        self.add_user(LLMFile(path, content))

    def to_messages(self, model_name: str) -> list[dict[str, Any]]:
        """Render the prompt to a list of messages formatted for the specific LLM model."""
        messages: list[dict[str, Any]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})

        if self._user_parts:
            user_content: list[dict[str, Any]] = []
            is_gemini = "gemini" in model_name.lower()

            for part in self._user_parts:
                if isinstance(part, dict):
                    user_content.append(part)
                elif isinstance(part, LLMFile):
                    if is_gemini:
                        encoded_content = base64.b64encode(part.content.encode("utf-8")).decode("utf-8")
                        user_content.append(
                            {
                                "type": "text",
                                "text": f"File: {part.path}",
                            }
                        )
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:text/plain;base64,{encoded_content}"},
                            }
                        )
                    else:
                        user_content.append(
                            {
                                "type": "text",
                                "text": f"File: {part.path}",
                            }
                        )
                        user_content.append(
                            {
                                "type": "text",
                                "text": part.content,
                            }
                        )

            messages.append({"role": "user", "content": user_content})

        return messages
