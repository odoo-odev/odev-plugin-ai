from .base import BaseAgentHandler


class AgyHandler(BaseAgentHandler):
    def get_config_dirs(self):
        return [".antigravity", ".config/antigravity", ".gemini", ".config/gemini"]

    def get_persistent_dirs(self):
        return [".antigravity", ".config/antigravity", ".gemini", ".config/gemini"]

    def get_global_config_name(self):
        return ".antigravity.json"

    def get_config_files(self):
        return [".antigravity.json"]

    def get_creds_files(self):
        return ["antigravity-credentials.json", "gemini-credentials.json", "google_accounts.json", "oauth_creds.json"]

    def get_agent_config_rel_path(self):
        return ".antigravity"

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):
        cmd = ["agy"]
        if prompt:
            cmd.extend(["-p" if headless else "-i", prompt])
        if resume:
            if resume == "latest":
                cmd.append("--continue")
            else:
                cmd.extend(["--conversation", resume])
        if yolo:
            cmd.append("--dangerously-skip-permissions")
        if model and model != "auto":
            cmd.extend(["--model", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--add-dir", path])
        return cmd
