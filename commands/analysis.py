"""`odev analysis`: analyze an Odoo task with an AI agent, read from the clipboard.

An alias of `odev ai`: it reads the clipboard by default, installs the task-analysis
skill, and tells the agent to work by it before handing it the pasted task. It replaces
the heavier `analyze` command of the scaffold plugin, which pulls the task from a task
database and resolves the client's code - here the task is whatever was copied.
"""

from odev.plugins.odev_plugin_ai.common.aliases import PasteAliasCommand
from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin


METHOD_SKILL = "odoo_task_analysis"
"""Skill carrying the task-analysis method and the shape the written analysis must take."""


class AnalysisCommand(PasteAliasCommand):
    """Analyze an Odoo task with an AI agent, reading the task from the clipboard."""

    _name = "analysis"

    # Imported base plus the method skill, rather than named from scratch: the preamble
    # sends the agent to METHOD_SKILL, and the skill installed and the skill the prompt
    # names have to be the same one.
    required_skills = [*AICommandMixin.required_skills, METHOD_SKILL]

    prompt_preamble = (
        "You are an expert Odoo developer analyzing a task. "
        f"Load the `{METHOD_SKILL}` skill and work by it: it holds the analysis method and the "
        "shape the result must take. Write the analysis in your response. The task to analyze follows."
    )
