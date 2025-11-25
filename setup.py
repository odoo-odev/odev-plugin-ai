from odev.common import string
from odev.common.console import console
from odev.common.logging import logging
from odev.common.odev import Odev

from odev.plugins.odev_plugin_ai.common.llm import LLM_PROVIDER_LIST


logger = logging.getLogger(__name__)


def setup(odev: Odev) -> None:
    """Set up the AI plugin by configuring the default LLM and API key."""
    choices = [(name, name) for name in LLM_PROVIDER_LIST]

    providers = console.checkbox(
        "Which LLM do you want to use ? (You will be prompted next for the API key)",
        choices,
        LLM_PROVIDER_LIST["Gemini"][0],
    )

    if providers is None:
        raise ValueError("No LLM selected. Please select a valid LLM provider.")

    odev.config.ai.llm_order = providers

    logger.info(
        string.normalize_indent(
            """
            You can find your API key at the following URLs:
             - Gemini: https://aistudio.google.com/app/apikey
             - ChatGPT: https://platform.openai.com/account/api-keys
             - Claude: https://console.anthropic.com/
             - Grok: https://x.ai/
            """
        )
    )

    for provider in providers:
        odev.store.secrets.get(
            f"{provider.lower()}_api_key",
            scope="api",
            fields=["password"],
            prompt_format=f"Enter your {provider} API KEY:",
        )

    logger.info("AI plugin configured successfully.")
