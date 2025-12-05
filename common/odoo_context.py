import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import networkx as nx
from defusedxml import ElementTree as SafeET

from odev.common.logging import logging
from odev.common.odoobin import OdoobinProcess

from odev.plugins.odev_plugin_ai.common import graph
from odev.plugins.odev_plugin_ai.common.llm_prompt import LLMPrompt


logger = logging.getLogger(__name__)


class OdooContext:
    """Gathers and holds the context of an Odoo project based on a list of modules and a development analysis.

    It inspects module files to retrieve relevant source code for models, views, controllers, etc.
    """

    def __init__(self, process: OdoobinProcess) -> None:
        """Initialize the OdooContext with the given process."""
        self.process: OdoobinProcess = process

    def gather_po_context(self, po_content: str) -> LLMPrompt:
        """Gathers context from files referenced in PO file content.

        It looks for lines like `#: code:path/to/file:line_number`,
        extracts the unique file paths, reads their content, and returns
        it in a LLMPrompt object.

        :param po_content: The string content of a PO file.
        :return: A LLMPrompt object containing the gathered files.
        """
        module_names = []
        context: LLMPrompt = LLMPrompt()

        # Regex to find file paths in lines like `#: code:path/to/file:line_number`
        file_paths = re.findall(r"#: code:(.*?):\d+", po_content)
        for file_path_str in set(file_paths):
            full_path = None
            module_name = None
            module_path = None
            if file_path_str.startswith("addons/"):
                path_parts = Path(file_path_str).parts
                if len(path_parts) > 1:
                    module_name = path_parts[1]
                    module_names.append(module_name)
                    module_path = graph._get_module_path(self.process, module_name)
                    if module_path:
                        relative_file_path = Path(*path_parts[2:])
                        full_path = module_path / relative_file_path

            if full_path and full_path.exists() and module_name and module_path:
                context.add_file(f"{module_name}/{relative_file_path}", full_path.read_text())
            else:
                logger.warning(f"Could not find context file: {file_path_str}")

        logger.info(f"Gathered context from files in modules: {', '.join(set(module_names))}")

        return context

    def _build_dependency_info(
        self, initial_depends: list[str], dependency_level: int
    ) -> tuple[list[str], dict[str, Path]]:
        """Build dependency graph, sort modules topologically, and find their paths."""
        logger.info("Building dependency tree for Odoo context...")
        dependency_graph = graph.build_dependency_tree(self.process, initial_depends, max_level=dependency_level)
        sorted_modules: list[str] = []
        module_paths: dict[str, Path] = {}
        try:
            if dependency_graph:
                sorted_modules = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible:
            logger.error("Circular dependency detected. Context may be incomplete.")
            sorted_modules = list(dependency_graph.nodes())

        for module_name in sorted_modules:
            path = graph._get_module_path(self.process, module_name)
            if path:
                module_paths[module_name] = path
            else:
                logger.warning(f"Could not find path for module '{module_name}'. It will be skipped.")
        return sorted_modules, module_paths

    def gather_context(
        self,
        depends: list[str] | None = None,
        analysis: dict[str, Any] | None = None,
        override_module_name: str = "",
        dependency_level: int = 0,
    ) -> LLMPrompt:
        """Iterate over modules and gather context based on the analysis."""
        if not depends:
            logger.warning("No depends found on the analysis or provided as arguments. No context will be provided.")

        if analysis is None:
            analysis = {}

        sorted_modules: list[str] = []
        module_paths: dict[str, Path] = {}
        if depends:
            sorted_modules, module_paths = self._build_dependency_info(depends, dependency_level)

        context: LLMPrompt = LLMPrompt()

        for module_name in sorted_modules:
            if module_name in ["base", "web", "mail", "utm"]:
                continue

            module_path = module_paths.get(module_name)
            if not module_path:
                continue

            self._gather_manifest(context, module_name, module_path)
            self._gather_models(context, module_name, module_path, analysis, override_module_name)
            self._gather_views(context, module_name, module_path, analysis)
            self._gather_controllers(context, module_name, module_path, analysis)
            self._gather_assets(context, module_name, module_path, analysis)
            self._gather_security(context, module_name, module_path)
            self._gather_reports(context, module_name, module_path, analysis)
            self._gather_website_templates(context, module_name, module_path, analysis)
            self._gather_data(context, module_name, module_path)

        logger.info(f"Gathered context from files in modules: {', '.join(set(sorted_modules))}")

        return context

    def _gather_manifest(self, context: LLMPrompt, module_name: str, module_path: Path) -> None:
        """Gathers manifest file content."""
        manifest_path = module_path / "__manifest__.py"
        if manifest_path.exists():
            context.add_file(f"{module_name}/__manifest__.py", manifest_path.read_text())

    def _gather_models(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        analysis: dict[str, Any],
        override_module_name: str = "",
    ) -> None:
        """Gathers model files if they are loaded in __init__.py and present in the analysis."""
        models_dir = module_path / "models"
        init_py = models_dir / "__init__.py"
        analysis_models: set[str] = {m["name"] for m in analysis.get("models", [])}

        if not models_dir.is_dir() or not init_py.exists():
            return

        try:
            init_content = init_py.read_text()
            context.add_file(f"{module_name}/{init_py.relative_to(module_path)}", init_content)
            loaded_py_files: list[Path] = []
            for line in init_content.splitlines():
                match = re.match(r"^\s*from\s+\.\s+import\s+([\w, ]+)", line)
                if match:
                    for mod in match.group(1).split(","):
                        py_file = models_dir / f"{mod.strip()}.py"
                        if py_file.exists():
                            loaded_py_files.append(py_file)

            for py_file in loaded_py_files:
                self._process_model_file(
                    context,
                    module_name,
                    module_path,
                    py_file,
                    analysis_models,
                    override_module_name,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not process models for {module_name}: {e}")

    def _process_model_file(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        py_file: Path,
        analysis_models: set[str],
        override_module_name: str,
    ) -> None:
        """Process a single model file to extract relevant classes."""
        file_content = py_file.read_text()
        content_lines = file_content.splitlines()
        i = 0
        while i < len(content_lines):
            line = content_lines[i]
            class_match = re.match(r"^(\s*)class\s+.*:", line)
            if not class_match:
                i += 1
                continue

            class_indentation = len(class_match.group(1))
            class_block_lines: list[str] = [line]

            j = i + 1
            while j < len(content_lines):
                next_line = content_lines[j]
                if next_line.strip() and (len(next_line) - len(next_line.lstrip(" "))) <= class_indentation:
                    break
                class_block_lines.append(next_line)
                j += 1

            class_content = "\n".join(class_block_lines)
            declared: list[str] = re.findall(r"_name\s*=\s*['\"]([^'\"]+)['\"]", class_content)
            inherited: list[str] = re.findall(r"_inherit\s*=\s*['\"]([^'\"]+)['\"]", class_content)
            inherited_list_str = re.search(r"_inherit\s*=\s*\[([^\]]+)\]", class_content)

            if inherited_list_str:
                inherited.extend(re.findall(r"['\"]([^'\"]+)['\"]", inherited_list_str.group(1)))

            if analysis_models.intersection(set(declared + inherited)) or (
                not analysis_models and module_name == override_module_name
            ):
                content = file_content if module_name == override_module_name else class_content
                context.add_file(f"{module_name}/{py_file.relative_to(module_path)}", content)
            i = j

    def _gather_views(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        analysis: dict[str, Any],
    ) -> None:
        """Gathers view files based on models in the analysis."""
        analysis_views: list[dict[str, Any]] = analysis.get("views", [])
        if not analysis_views:
            return
        analysis_view_models: set[str] = {v["model"] for v in analysis_views if "model" in v}

        for xml_file in module_path.rglob("*.xml"):
            try:
                content = xml_file.read_text()
                tree = SafeET.fromstring(content)
                for record in tree.findall(".//record[@model='ir.ui.view']"):
                    model_field = record.find("./field[@name='model']")
                    if model_field is not None and model_field.text in analysis_view_models:
                        context.add_file(
                            f"{module_name}/{xml_file.relative_to(module_path)}",
                            content,
                        )
                        break
            except (ET.ParseError, FileNotFoundError):
                continue

    def _gather_controllers(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        analysis: dict[str, Any],
    ) -> None:
        """Gathers controller files by matching routes from the analysis."""
        analysis_controllers: list[dict[str, Any]] = analysis.get("controller", [])
        if not analysis_controllers:
            return
        analysis_routes: set[str] = {c["action_name"] for c in analysis_controllers if "action_name" in c}

        controllers_dir = module_path / "controllers"
        if not controllers_dir.is_dir():
            return

        for py_file in controllers_dir.rglob("*.py"):
            content = py_file.read_text()
            routes_in_file: list[str] = re.findall(r"@http\.route\(\s*['\"]([^'\"]+)['\"]", content)
            routes_in_file_list: list[str] = re.findall(r"@http\.route\(\s*\[([^\]]+)\]", content)
            for route_list in routes_in_file_list:
                routes_in_file.extend(re.findall(r"['\"]([^'\"]+)['\"]", route_list))

            if analysis_routes.intersection(routes_in_file):
                context.add_file(f"{module_name}/{py_file.relative_to(module_path)}", content)

    def _gather_assets(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        analysis: dict[str, Any],
    ) -> None:
        """Gathers asset files by matching paths from the analysis."""
        for asset in analysis.get("assets", []):
            file_path_str: str | None = asset.get("file_path")
            if not file_path_str:
                continue

            if file_path_str.startswith(f"/{module_name}/"):
                file_path_str = file_path_str[len(f"/{module_name}/") :]

            potential_path = module_path / file_path_str
            if potential_path.exists():
                context.add_file(
                    f"{module_name}/{potential_path.relative_to(module_path)}",
                    potential_path.read_text(),
                )
            else:
                # Fallback to search by filename
                filename = Path(file_path_str).name
                for f in module_path.rglob(filename):
                    context.add_file(f"{module_name}/{f.relative_to(module_path)}", f.read_text())
                    break

    def _gather_security(self, context: LLMPrompt, module_name: str, module_path: Path) -> None:
        """Gathers all files from the 'security' directory."""
        security_dir = module_path / "security"
        if security_dir.is_dir():
            for sec_file in security_dir.iterdir():
                if sec_file.is_file() and (sec_file.name.endswith(".csv") or sec_file.name.endswith(".xml")):
                    context.add_file(f"{module_name}/security/{sec_file.name}", sec_file.read_text())

    def _gather_reports(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        analysis: dict[str, Any],
    ) -> None:
        """Gathers report definition files based on models in the analysis."""
        report_models: set[str] = {r["model"] for r in analysis.get("reports", []) if "model" in r}
        if not report_models:
            return

        for xml_file in module_path.rglob("*.xml"):
            try:
                content = xml_file.read_text()
                tree = SafeET.fromstring(content)
                for record in tree.findall(".//record[@model='ir.actions.report']"):
                    model_field = record.find("./field[@name='model']")
                    if model_field is not None and model_field.text in report_models:
                        context.add_file(
                            f"{module_name}/{xml_file.relative_to(module_path)}",
                            content,
                        )
                        break
            except (ET.ParseError, FileNotFoundError):
                continue

    def _gather_website_templates(
        self,
        context: LLMPrompt,
        module_name: str,
        module_path: Path,
        analysis: dict[str, Any],
    ) -> None:
        """Gathers website template files by matching template IDs from the analysis."""
        view_ids: set[str] = {v["view"] for v in analysis.get("website_views", []) if "view" in v}
        if not view_ids:
            return

        for xml_file in module_path.rglob("*.xml"):
            try:
                content = xml_file.read_text()
                tree = SafeET.fromstring(content)
                for template in tree.findall(".//template"):
                    template_id = template.get("id")
                    if template_id and template_id in view_ids:
                        context.add_file(
                            f"{module_name}/{xml_file.relative_to(module_path)}",
                            content,
                        )
                        break
            except (ET.ParseError, FileNotFoundError):
                continue

    def _gather_data(self, context: LLMPrompt, module_name: str, module_path: Path) -> None:
        """Gathers all XML data files from the 'data' directory."""
        data_dir = module_path / "data"
        if data_dir.is_dir():
            for data_file in data_dir.iterdir():
                if data_file.is_file() and (data_file.name.endswith(".csv") or data_file.name.endswith(".xml")):
                    context.add_file(f"{module_name}/data/{data_file.name}", data_file.read_text())
