"""odev-plugin-ai: AI agent integration for odev.

On load, substitutes the default OdoobinProcess with AI_OdoobinProcess so that
log filtering and other AI-sandbox-specific behaviors are applied automatically
whenever Odoo is started inside an AI bwrap sandbox (i.e., when AI_SANDBOX=1).
"""

import os

if os.environ.get("AI_SANDBOX") == "1" or os.environ.get("ANTIGRAVITY_AGENT") == "1":
    from odev.common.odev import Odev
    from .common.odoobin import AI_OdoobinProcess

    # Set at class level so it applies even if framework was created before this runs
    Odev._odoobin_process_class = AI_OdoobinProcess

    # Also patch the existing instance for update-hook overrides
    from odev import common as _common

    if _common.framework is not None:
        _common.framework.odoobin_process_class = AI_OdoobinProcess
        _common.framework.check_release = lambda: None
        _common.framework._should_update_now = lambda: False
