from odev.common.logging import logging
from odev.common.odev import Odev


logger = logging.getLogger(__name__)


def setup(odev: Odev) -> None:
    """Set up the AI plugin."""
    logger.info("AI plugin configured successfully (CLI execution mode).")
