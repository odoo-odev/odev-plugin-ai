from .base import BaseAgentHandler
from .claude import ClaudeHandler
from .agy import AgyHandler
from .copilot import CopilotHandler
from .opencode import OpenCodeHandler


def get_agent_handler(cli, host_home, odev):
    handlers = {
        "claude": ClaudeHandler,
        "agy": AgyHandler,
        "copilot": CopilotHandler,
        "opencode": OpenCodeHandler,
    }
    handler_cls = handlers.get(cli, BaseAgentHandler)
    return handler_cls(cli, host_home, odev)
