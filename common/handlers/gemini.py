import shutil

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

    def get_skill_target(self):
        return (".gemini", "GEMINI.md")

    def inject_skills(self, target_dir):
        skills_dest = target_dir / "skills"
        for skill in self.get_enabled_skills():
            skills_dest.mkdir(parents=True, exist_ok=True)
            dest = skills_dest / skill.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(skill.path, dest)

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
