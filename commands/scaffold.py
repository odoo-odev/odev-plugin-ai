"""`odev scaffold`: scaffold an Odoo module from a task, read from the clipboard.

An alias of `odev ai`: it reads the clipboard by default, installs the coding-guidelines
and SaaS-development skills, and tells the agent to scaffold a module by them before
handing it the pasted task. It replaces the heavier `scaffold` command of the scaffold
plugin, which pulls the task from a task database and resolves the client's code - here
the task is whatever was copied.
"""

from odev.plugins.odev_plugin_ai.common.aliases import PasteAliasCommand
from odev.plugins.odev_plugin_ai.common.mixins import GUIDELINES_SKILL, AICommandMixin


SAAS_SKILL = "odoo_saas_development"
"""Skill carrying what a SaaS module may be: importable data, no custom Python."""


class ScaffoldCommand(PasteAliasCommand):
    """Scaffold an Odoo module from a task, reading the task from the clipboard."""

    _name = "scaffold"

    # The base already carries GUIDELINES_SKILL; the SaaS skill is added so an agent that
    # finds the module has to be importable has read what that allows. Both are named in
    # the preamble, so what is installed and what the prompt points at stay the same set.
    required_skills = [*AICommandMixin.required_skills, SAAS_SKILL]

    prompt_preamble = (
        "You are an expert Odoo developer. Scaffold an Odoo module from the task below. "
        f"Work by the `{GUIDELINES_SKILL}` skill for module layout, file naming and the conventions of "
        "each language; when the module targets Odoo Online (SaaS) it has to be importable data, so work "
        f"by the `{SAAS_SKILL}` skill too - no Python, everything expressed as XML records. Read the "
        "modules already in the working directory before deciding whether to extend one or create a new "
        "one. The task to implement follows."
    )
