from .base import BaseAgentHandler


class AgyHandler(BaseAgentHandler):
    def get_config_dirs(self):
        return [".antigravity", ".config/antigravity"]

    def get_persistent_dirs(self):
        return [".antigravity", ".config/antigravity"]

    def get_global_config_name(self):
        return ".antigravity.json"

    def get_config_files(self):
        return [".antigravity.json"]

    def get_creds_files(self):
        return ["antigravity-credentials.json", "google_accounts.json", "oauth_creds.json"]

    def get_agent_config_rel_path(self):
        return ".antigravity"

    def get_command(self, prompt, resume, all_candidate_paths, model, headless, yolo):
        cmd = ["agy"]
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
