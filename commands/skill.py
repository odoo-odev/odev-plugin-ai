"""Manage agent skills for various AI CLI tools."""

from odev.common import args
from odev.common.commands import Command

from odev.plugins.odev_plugin_ai.common.skills import SkillManager


class SkillCommand(Command):
    """Manage agent skills across AI CLI tools."""

    _name = "skill"
    _aliases = ["skills"]

    show = args.Flag(
        aliases=["-s", "--show"],
        description="Show available skills and choose which ones to enable.",
        default=False,
    )

    info = args.Flag(
        aliases=["-i", "--info"],
        description="Show detailed information about synced skills.",
        default=False,
    )

    def run(self) -> None:
        manager = SkillManager(self.odev)

        if self.args.show:
            available = manager.list_available_skills()
            if not available:
                self.console.print("No skills found in enabled plugins.")
                return

            choices = [(s.name, f"{s.name}: {s.description}" if s.description else s.name) for s in available]
            disabled = self.config.skills.disabled
            defaults = [s.name for s in available if s.name not in disabled]

            enabled = self.console.checkbox(
                "Select skills to enable\n  SPACE to toggle, ENTER to confirm",
                choices=choices,
                defaults=defaults,
            )

            # Store the ones that are NOT enabled as disabled
            self.config.skills.disabled = [s.name for s in available if s.name not in enabled]
            self.console.print(
                f"Updated skill configuration ({len(enabled)} enabled, {len(self.config.skills.disabled)} disabled)."
            )
            manager.sync_skills(info=True)
        else:
            manager.sync_skills(info=self.args.info)
            if not self.args.info:
                self.console.print("Skills synchronized successfully. Use [bold]--info[/bold] for details.")
