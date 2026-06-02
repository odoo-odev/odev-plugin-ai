from .base import BaseAgentHandler


class GeminiHandler(BaseAgentHandler):
    def get_config_dirs(self):
        return [".gemini", ".config/gemini"]

    def get_persistent_dirs(self):
        return [".gemini", ".config/gemini"]

    def get_global_config_name(self):
        return ".gemini.json"

    def get_config_files(self):
        return [".gemini.json"]

    def get_creds_files(self):
        return ["gemini-credentials.json", "google_accounts.json", "oauth_creds.json"]

    def get_agent_config_rel_path(self):
        return ".gemini"

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):
        cmd = ["gemini"]
        if prompt:
            cmd.extend(["-p" if headless else "-i", prompt])
        if resume:
            cmd.extend(["--resume", resume])
        cmd.append("--approval-mode")
        cmd.append("yolo" if yolo else "auto_edit")
        if model and model != "auto":
            cmd.extend(["-m", model])
        for path in self._guest_paths(all_candidate_paths):
            cmd.extend(["--include-directories", path])
        return cmd
