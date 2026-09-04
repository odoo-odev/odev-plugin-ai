from .base import BaseAgentHandler


class CopilotHandler(BaseAgentHandler):
    def get_config_dirs(self):
        return [".copilot", ".config/github-copilot", ".config/gh"]

    def get_persistent_dirs(self):
        return [".copilot", ".config/github-copilot", ".config/gh"]

    def get_creds_files(self):
        return ["config.json", "hosts.json", "hosts.yml", "config.yml"]

    def get_agent_config_rel_path(self):
        return ".copilot"

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo, mcp_server_names=()):  # noqa: PLR0913 - signature set by BaseAgentHandler
        cmd = ["copilot"]
        if prompt:
            cmd.extend(["-p" if headless else "-i", prompt])
        if resume:
            cmd.append(f"--resume={resume}")
        if yolo:
            cmd.append("--yolo")
        else:
            cmd.extend(
                [
                    "--allow-tool=read",
                    "--allow-tool=write",
                    "--allow-tool=shell(rtk:*)",
                    "--allow-tool=shell(odev:*)",
                    "--allow-tool=shell(git:*)",
                    "--allow-tool=shell(pre-commit:*)",
                ]
            )
        if model and model != "auto":
            cmd.extend(["-m", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
