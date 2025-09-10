from typing import Dict, List, Optional

import litellm
from litellm import InternalServerError, ModelResponse

from odev.common.logging import logging


logger = logging.getLogger(__name__)
# Reduce litellm's default logging verbosity
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


# A mapping of provider names to a list of model names to try in order of preference.
# Note: These may be custom model names/aliases specific to a litellm proxy setup.
LLM_LIST: Dict[str, List[str]] = {
    "Gemini": ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"],
    "ChatGPT": ["chatgpt/gpt-5", "chatgpt/gpt-4.5"],
    "Claude": ["claude/claude-4", "claude/claude-3.7"],
    "Grok": ["grok/grok-4", "grok/grok-3"],
}


class LLM:
    """A client for interacting with Large Language Models via litellm."""

    provider: str
    api_key: str

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initializes the LLM client.

        :param provider: The name of the LLM provider (e.g., "Gemini", "ChatGPT").
                         This must match a key in the `LLM_LIST`.
        :param api_key: The API key for the specified provider.
        :raises ValueError: If the provider or API key is not provided.
        """
        if not provider:
            raise ValueError("An LLM provider must be configured.")
        if not api_key:
            raise ValueError("An LLM API key must be configured.")

        self.provider = provider
        self.api_key = api_key

    def completion(self, messages: List[Dict[str, str]], response_format: Optional[type] = None) -> Optional[str]:
        """
        Sends a completion request to the configured LLM and expects a structured response.

        This method iterates through a list of models for the configured provider,
        attempting each one until a successful structured response is obtained.

        :param messages: A list of messages forming the conversation history for the prompt.
        :return: The string content of the response, or `None` if all attempts fail.
        """
        model_list: List[str] = LLM_LIST.get(self.provider, [])
        if not model_list:
            logger.error(f"No models are configured for the provider '{self.provider}'.")
            return None

        for model_name in model_list:
            try:
                logger.debug(f"Attempting completion with model: {model_name}")
                litellm.suppress_debug_info = True

                response: ModelResponse = litellm.completion(  # type: ignore
                    model=model_name,
                    messages=messages,
                    api_key=self.api_key,
                    response_format=response_format,
                    verbose=False,
                )
                logger.info(f"Successfully received a response from {model_name}.")

                # Callers expect the string content of the message.
                if response.choices and response.choices[0].message.content:  # type: ignore
                    return response.choices[0].message.content  # type: ignore
                return None
            except InternalServerError as e:
                logger.warning(f"Model '{model_name}' failed with an internal server error: {e}")
                # Continue to the next model in the list
            except Exception as e:
                # Catch other potential exceptions from litellm (e.g., validation, connection errors)
                logger.error(f"An unexpected error occurred with model '{model_name}': {e}")
                # Continue to the next model in the list

        logger.error(
            f"All configured models for provider '{self.provider}' failed. " f"Attempted: {', '.join(model_list)}"
        )
        return None
