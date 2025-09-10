from odev.common import string
from odev.common.console import console
from odev.common.logging import logging
from odev.common.odev import Odev

from odev.plugins.odev_plugin_ai.common.llm import LLM_LIST


logger = logging.getLogger(__name__)


def setup(odev: Odev) -> None:
    """Set up the AI plugin by configuring the default LLM and API key."""
    # The `console.select` returns the value from the (value, display_name) tuple.
    # The LLM class expects the provider name to be capitalized (e.g., "Gemini").
    choices = sorted((name, name) for name in LLM_LIST.keys())

    llm_name = console.select(
        "Which LLM do you want to use?",
        choices,
        list(LLM_LIST.keys())[0],
    )

    if llm_name is None:
        raise ValueError("No LLM selected. Please select a valid LLM provider.")

    odev.config.set("ai", "default_llm", llm_name)

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

    odev.store.secrets.get("llm_api_key", scope="api", fields=["password"], prompt_format="Enter your LLM API Key:")

    logger.info("AI plugin configured successfully.")
