from odev.common.logging import logging
from odev.common.progress import spinner


logger = logging.getLogger(__name__)


# A mapping of provider names to a list of model names to try in order of preference.
# Note: These may be custom model names/aliases specific to a litellm proxy setup.
LLM_LIST: dict[str, list[str]] = {
    "Gemini": ["gemini/gemini-3-pro-preview", "gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"],
    "ChatGPT": ["chatgpt/gpt-5", "chatgpt/gpt-4.5"],
    "Claude": ["claude/claude-4", "claude/claude-3.7"],
    "Grok": ["grok/grok-4", "grok/grok-3"],
}


class LLM:
    """A client for interacting with Large Language Models via litellm."""

    provider: str
    api_key: str
    model: str | None = None

    def __init__(self, model_identifier: str | None = None, api_key: str | None = None):
        """Initialize the LLM client.

        :param model_identifier: The name of the LLM provider (e.g., "Gemini", "ChatGPT") or a specific model
                                 identifier (e.g., "gemini/gemini-1.5-pro"). If a provider is given, it must
                                 match a key in `LLM_LIST`.
        :param api_key: The API key for the specified provider.
        :raises ValueError: If the provider or API key is not provided.
        """
        if not model_identifier:
            raise ValueError("An LLM provider must be configured.")
        if not api_key:
            raise ValueError("An LLM API key must be configured.")

        if "/" in model_identifier:
            self.provider = model_identifier.split("/")[0].capitalize()
            self.model = model_identifier
        else:
            self.provider = model_identifier

        self.api_key = api_key

    def completion(
        self, messages: list[dict[str, str]], response_format: type | None = None, progress: spinner = None
    ) -> str | None:
        """Send a completion request to the configured LLM and expect a structured response.

        If a specific model was provided during initialization, it will use that model.
        Otherwise, it iterates through a list of models for the configured provider,
        attempting each one until a successful structured response is obtained.

        :param messages: A list of messages forming the conversation history for the prompt.
        :return: The string content of the response, or `None` if all attempts fail.
        """
        import litellm  # noqa: PLC0415
        from litellm import (  # noqa: PLC0415
            ContextWindowExceededError,
            InternalServerError,
            ModelResponse,
            RateLimitError,
            token_counter,
        )

        # Reduce litellm's default logging verbosity
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        model_list: list[str] = [self.model] if self.model else LLM_LIST.get(self.provider, [])
        if not model_list:
            logger.error(f"No models are configured for the provider '{self.provider}'.")
            return None

        for model_name in model_list:
            if progress:
                progress.update(f"Scaffolding analysis with {model_name} ..")
            logger.debug(f"Token counter : {token_counter(model=model_name, messages=messages)} tokens")
            try:
                logger.debug(f"Attempting completion with model: '{model_name}'")
                litellm.suppress_debug_info = True

                response: ModelResponse = litellm.completion(
                    model=model_name,
                    messages=messages,
                    api_key=self.api_key,
                    response_format=response_format,
                    verbose=False,
                )
            except RateLimitError:
                logger.error(f"{model_name} is currently rate limited.")
            except ContextWindowExceededError:
                logger.error(f"The context windows is to big for {model_name}.")
            except InternalServerError as e:
                logger.warning(f"Model '{model_name}' failed with an internal server error: {e}")
                # Continue to the next model in the list
            except Exception as e:  # noqa: BLE001
                # Catch other potential exceptions from litellm (e.g., validation, connection errors)
                logger.error(f"An unexpected error occurred with model '{model_name}': {e}")
                # Continue to the next model in the list
            else:
                logger.info(f"Successfully received a response from {model_name}.")

                if response.usage:
                    logger.debug(
                        f"LLM Usage: {response.usage.prompt_tokens} prompt, "
                        f"{response.usage.completion_tokens} completion, "
                        f"{response.usage.total_tokens} total tokens."
                    )

                # Callers expect the string content of the message.
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                return None

        logger.error(f"All configured models for provider '{self.provider}' failed. Attempted: {', '.join(model_list)}")
        return None
