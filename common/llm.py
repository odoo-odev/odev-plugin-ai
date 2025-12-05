import itertools

from odev.common import progress
from odev.common.logging import logging
from odev.common.mixins.framework import OdevFrameworkMixin

from odev.plugins.odev_plugin_ai.common.llm_prompt import LLMPrompt


logger = logging.getLogger(__name__)


# A mapping of provider names to a list of model names to try in order of preference.
# Note: These may be custom model names/aliases specific to a litellm proxy setup.
LLM_PROVIDER_LIST: dict = {
    "Gemini": {
        "flagship": "gemini/gemini-3-pro-preview",
        "stable": "gemini/gemini-2.5-pro",
        "fast": "gemini/gemini-2.5-flash",
    },
    "Anthropic": {
        "flagship": "anthropic/claude-4-5-opus",
        "stable": "anthropic/claude-4-5-sonnet",
        "fast": "anthropic/claude-3-5-haiku",
    },
    "OpenAI": {"flagship": "gpt-4.1", "stable": "gpt-4.1-mini", "fast": "gpt-4.1-nano"},
    "xAI": {
        "flagship": "xai/grok-4-1",
        "stable": "xai/grok-4",
        "fast": "xai/grok-4-fast",
    },
}


class LLM(OdevFrameworkMixin):
    """A client for interacting with Large Language Models via litellm."""

    provider: str
    model: str | None = None
    llm_order: list = []

    def __init__(self, model_identifier: str | None = None, llm_order: list | None = None):
        """Initialize the LLM client.

        :param model_identifier: The name of the LLM provider (e.g., "Gemini", "ChatGPT") or a specific model
                                 identifier (e.g., "gemini/gemini-1.5-pro"). If a provider is given, it must
                                 match a key in `LLM_LIST`.
        :param api_key: The API key for the specified provider.
        :raises ValueError: If the provider or API key is not provided.
        """
        self.llm_order = [] if llm_order is None else llm_order

        if model_identifier and "/" in model_identifier:
            self.provider = model_identifier.split("/")[0].capitalize()
            self.model = model_identifier
        else:
            self.provider = model_identifier

    def _get_model_list(self) -> list[str]:
        """Get the list of models to attempt based on configuration.

        :return: A list of model names to try, in order of preference.
        """
        model_list = []

        if self.model:
            model_list = [self.model]
        elif self.llm_order:
            # Get models for each provider in order
            providers_models = [
                list(LLM_PROVIDER_LIST[provider].values())
                for provider in self.llm_order
                if provider in LLM_PROVIDER_LIST
            ]

            # Interleave models: P1_M1, P2_M1, P1_M2, P2_M2, ...
            for models in itertools.zip_longest(*providers_models):
                for model in models:
                    if model:
                        model_list.append(model)
        elif self.provider and self.provider in LLM_PROVIDER_LIST:
            model_list = list(LLM_PROVIDER_LIST[self.provider].values())

        return model_list

    def _try_model_completion(
        self,
        model_name: str,
        prompt: LLMPrompt,
        response_format: type | None,
    ) -> str | None:
        """Attempt completion with a single model.

        :param model_name: The name of the model to try.
        :param prompt: The LLMPrompt object containing the conversation.
        :param response_format: Optional response format specification.
        :return: The response content as a string, or None if the attempt failed.
        """
        import litellm  # noqa: PLC0415
        from litellm import (  # noqa: PLC0415
            ContextWindowExceededError,
            InternalServerError,
            ModelResponse,
            RateLimitError,
            token_counter,
        )

        # Convert LLMPrompt to messages for the specific model
        messages = prompt.to_messages(model_name)

        logger.debug(f"Token counter : {token_counter(model=model_name, messages=messages)} tokens")

        try:
            logger.debug(f"Attempting completion with model: '{model_name}'")
            litellm.suppress_debug_info = True

            response: ModelResponse = litellm.completion(
                model=model_name,
                messages=messages,
                response_format=response_format,
                verbose=False,
            )
        except RateLimitError:
            logger.error(f"{model_name} is currently rate limited.")
            return None
        except ContextWindowExceededError:
            logger.error(f"The context windows is to big for {model_name}.")
            return None
        except InternalServerError as e:
            logger.warning(f"Model '{model_name}' failed with an internal server error: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"An unexpected error occurred with model '{model_name}': {e}")
            return None

        # Success case
        logger.info(f"Successfully received a response from {model_name}.")

        if response.usage:
            logger.debug(
                f"LLM Usage: {response.usage.prompt_tokens} prompt, "
                f"{response.usage.completion_tokens} completion, "
                f"{response.usage.total_tokens} total tokens."
            )

        # Extract and return content
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None

    def completion(self, prompt: LLMPrompt, response_format: type | None = None) -> str | None:
        """Send a completion request to the configured LLM and expect a structured response.

        If a specific model was provided during initialization, it will use that model.
        Otherwise, it iterates through a list of models for the configured provider,
        attempting each one until a successful structured response is obtained.

        :param prompt: An LLMPrompt object containing the conversation.
        :return: The string content of the response, or `None` if all attempts fail.
        """
        # Reduce litellm's default logging verbosity
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        model_list = self._get_model_list()
        if not model_list:
            logger.error("No models configured to try.")
            return None

        for model_name in model_list:
            with progress.spinner(f"Calling {model_name} ..."):
                result = self._try_model_completion(
                    model_name,
                    prompt,
                    response_format,
                )

            if result is not None:
                return result

        logger.error(f"All configured models for provider '{self.provider}' failed. Attempted: {', '.join(model_list)}")
        return None
