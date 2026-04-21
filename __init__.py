"""AI plugin initialization."""

from odev.common.odev import Odev
from odev.plugins.odev_plugin_ai.common.skills import SkillManager

original_upgrade = Odev.upgrade


def patched_upgrade(self):
    """Override standard update/upgrade to sync skills."""
    original_upgrade(self)
    try:
        # Sync skills automatically after an upgrade
        SkillManager(self).sync_skills(info=True)
    except Exception as e:
        from odev.common.logging import logging

        logging.getLogger(__name__).warning(f"Could not automatically sync skills after upgrade: {e}")


Odev.upgrade = patched_upgrade
