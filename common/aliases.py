"""Shared base for the `scaffold` and `analysis` aliases of `odev ai`.

Each alias is `odev ai` with the skills and the framing of one kind of task baked in,
and the clipboard read on by default: a developer copies a task out of the browser and
runs `odev scaffold` or `odev analysis` without retyping `--paste` or naming the skill
the agent should work by. Everything else - the sandbox, the agent, the paste parsing -
is `ai`'s, so the aliases carry nothing of their own but a preamble and a skill list.
"""

from odev.common import args

from odev.plugins.odev_plugin_ai.commands.ai import AICommand


class PasteAliasCommand(AICommand):
    """`odev ai` with the clipboard read on by default and a kind of task baked in.

    A subclass sets :attr:`prompt_preamble` (what the task is and which skill to work by)
    and :attr:`required_skills` (what to install before the agent starts). The reading of
    the clipboard, the sandbox and the agent are all inherited from :class:`AICommand`.
    """

    no_paste = args.Flag(
        aliases=["--no-paste"],
        description="Type the prompt instead of reading it from the clipboard. These aliases read "
        "the clipboard by default; pass this to turn that off and type the task yourself.",
        default=False,
    )

    def run(self) -> None:
        # Paste on unless turned off: sending what was copied is the whole point of the
        # alias. A prompt typed on the command line is still kept - `ai` appends the
        # pasted content after it - so `--no-paste` is only for the no-clipboard case.
        if not self.args.no_paste:
            self.args.paste = True

        super().run()
