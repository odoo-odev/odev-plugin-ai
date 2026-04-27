"""odev-plugin-ai: AI agent integration for odev.

On load, substitutes the default OdoobinProcess with AI_OdoobinProcess so that
log filtering and other AI-sandbox-specific behaviors are applied automatically
whenever Odoo is started inside an AI bwrap sandbox (i.e., when AI_SANDBOX=1).
"""

import os

if os.environ.get("AI_SANDBOX") == "1":
    from odev.common import framework as _framework
    from odev.plugins.odev_plugin_ai.common.odoobin import AI_OdoobinProcess

    # framework may be None if accessed before init_framework(); use _framework
    # module-level variable which is populated by the time plugins are loaded.
    if _framework is not None:
        _framework.odoobin_process_class = AI_OdoobinProcess

        # Override update checks to skip them in the sandbox
        _framework.check_release = lambda: None
        _framework._should_update_now = lambda: False
