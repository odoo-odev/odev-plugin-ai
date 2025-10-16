import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx

from odev.common.logging import logging
from odev.common.odoobin import OdoobinProcess

from odev.plugins.odev_plugin_ai.common import graph


logger = logging.getLogger(__name__)


class OdooContext:
    """
    Gathers and holds the context of an Odoo project based on a list of modules
    and a development analysis. It inspects module files to retrieve relevant
    source code for models, views, controllers, etc.
    """

    def __init__(self, process: OdoobinProcess) -> None:
        """
        :param process: The OdoobinProcess instance.
        """
        self.process: OdoobinProcess = process

    def gather_po_context(self, po_content: str) -> dict:
        """
        Gathers context from files referenced in PO file content.

        It looks for lines like `#: code:path/to/file:line_number`,
        extracts the unique file paths, reads their content, and returns
        it in a dictionary.

        :param po_content: The string content of a PO file.
        :return: A dictionary mapping file paths to their content.
        """
        module_names = []
        context_files = {}
        # Regex to find file paths in lines like `#: code:path/to/file:line_number`
        file_paths = re.findall(r"#: code:(.*?):\d+", po_content)
        for file_path_str in set(file_paths):
            full_path = None
            if file_path_str.startswith("addons/"):
                path_parts = Path(file_path_str).parts
                if len(path_parts) > 1:
                    module_name = path_parts[1]
                    module_names.append(module_name)
                    module_path = graph._get_module_path(self.process, module_name)
                    if module_path:
                        relative_file_path = Path(*path_parts[2:])
                        full_path = module_path / relative_file_path

            if full_path and full_path.exists():
                context_files[file_path_str] = full_path.read_text()
            else:
                logger.warning(f"Could not find context file: {file_path_str}")

        logger.info(f"Gathered context from {len(context_files)} files in modules: {', '.join(set(module_names))}")

        return context_files

    def _build_dependency_info(
        self, initial_depends: List[str], dependency_level: int
    ) -> Tuple[List[str], Dict[str, Path]]:
        """Builds dependency graph, sorts modules topologically, and finds their paths."""
        logger.info("Building dependency tree for Odoo context...")
        dependency_graph = graph.build_dependency_tree(self.process, initial_depends, max_level=dependency_level)
        sorted_modules: List[str] = []
        module_paths: Dict[str, Path] = {}
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
        self, depends: Optional[List[str]] = None, analysis: Optional[Dict[str, Any]] = None, dependency_level: int = 0
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Iterates over modules and gathers context based on the analysis."""
        if analysis is None:
            analysis = {}

        sorted_modules: List[str] = []
        module_paths: Dict[str, Path] = {}
        if depends:
            sorted_modules, module_paths = self._build_dependency_info(depends, dependency_level)

        logger.info(f"Gathering Odoo context from modules: {', '.join(sorted_modules)}")

        context: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for module_name in sorted_modules:
            if module_name in ["base", "web", "mail", "utm"]:
                continue

            module_path = module_paths.get(module_name)
            if not module_path:
                continue

            context[module_name] = {
                "models": [],
                "views": [],
                "controllers": [],
                "assets": [],
                "security": [],
                "reports": [],
                "website": [],
                "data": [],
                "manifests": [],
            }

            self._gather_manifest(context, module_name, module_path)
            self._gather_models(context, module_name, module_path, analysis)
            self._gather_views(context, module_name, module_path, analysis)
            self._gather_controllers(context, module_name, module_path, analysis)
            self._gather_assets(context, module_name, module_path, analysis)
            self._gather_security(context, module_name, module_path)
            self._gather_reports(context, module_name, module_path, analysis)
            self._gather_website_templates(context, module_name, module_path, analysis)
            self._gather_data(context, module_name, module_path)

        summary_message = "Odoo Context Summary:\n"
        grand_total_items = grand_total_lines = grand_total_chars = 0

        for module_name, categories in context.items():
            module_total_items = sum(len(items) for items in categories.values())
            if module_total_items > 0:
                summary_message += f"  - {module_name}:\n"
                for category, items in sorted(categories.items()):
                    if items:
                        num_items = len(items)
                        num_lines = sum(len(item.get("content", "").splitlines()) for item in items)
                        num_chars = sum(len(item.get("content", "")) for item in items)
                        grand_total_items += num_items
                        grand_total_lines += num_lines
                        grand_total_chars += num_chars
                        summary_message += f"    - {category.capitalize()}: "
                        summary_message += f"{num_items} item(s), {num_lines} line(s)"
                        summary_message += f", {num_chars} char(s)\n"
        summary_message += (
            f"\nTotal: {grand_total_items} item(s), {grand_total_lines} line(s), {grand_total_chars} char(s)"
        )
        logger.debug(summary_message)

        return context

    def _gather_manifest(self, context: dict, module_name: str, module_path: Path) -> None:
        """Gathers manifest file content."""
        manifest_path = module_path / "__manifest__.py"
        if manifest_path.exists():
            context[module_name]["manifests"].append(
                {
                    "path": str(manifest_path),
                    "content": manifest_path.read_text(),
                }
            )

    def _gather_models(self, context: dict, module_name: str, module_path: Path, analysis: Dict[str, Any]) -> None:
        """Gathers model files if they are loaded in __init__.py and present in the analysis."""

        models_dir = module_path / "models"
        init_py = models_dir / "__init__.py"
        analysis_models: Set[str] = {m["name"] for m in analysis.get("models", [])}

        if not models_dir.is_dir() or not init_py.exists() or not analysis_models:
            return

        try:
            init_content = init_py.read_text()
            loaded_py_files: List[Path] = []
            for line in init_content.splitlines():
                match = re.match(r"^\s*from\s+\.\s+import\s+([\w, ]+)", line)
                if match:
                    for mod in match.group(1).split(","):
                        py_file = models_dir / f"{mod.strip()}.py"
                        if py_file.exists():
                            loaded_py_files.append(py_file)

            for py_file in loaded_py_files:
                content_lines = py_file.read_text().splitlines()
                i = 0
                while i < len(content_lines):
                    line = content_lines[i]
                    class_match = re.match(r"^(\s*)class\s+.*:", line)
                    if not class_match:
                        i += 1
                        continue

                    class_indentation = len(class_match.group(1))
                    class_block_lines: List[str] = [line]

                    j = i + 1
                    while j < len(content_lines):
                        next_line = content_lines[j]
                        if next_line.strip() and (len(next_line) - len(next_line.lstrip(" "))) <= class_indentation:
                            break
                        class_block_lines.append(next_line)
                        j += 1

                    class_content = "\n".join(class_block_lines)
                    declared: List[str] = re.findall(r"_name\s*=\s*['\"]([^'\"]+)['\"]", class_content)
                    inherited: List[str] = re.findall(r"_inherit\s*=\s*['\"]([^'\"]+)['\"]", class_content)
                    inherited_list_str = re.search(r"_inherit\s*=\s*\[([^\]]+)\]", class_content)

                    if inherited_list_str:
                        inherited.extend(re.findall(r"['\"]([^'\"]+)['\"]", inherited_list_str.group(1)))

                    if analysis_models.intersection(set(declared + inherited)):
                        context[module_name]["models"].append({"path": str(py_file), "content": class_content})
                    i = j
        except Exception as e:
            logger.warning(f"Could not process models for {module_name}: {e}")

    def _gather_views(self, context: dict, module_name: str, module_path: Path, analysis: Dict[str, Any]) -> None:
        """Gathers view files based on models in the analysis."""
        analysis_views: List[Dict[str, Any]] = analysis.get("views", [])
        if not analysis_views:
            return
        analysis_view_models: Set[str] = {v["model"] for v in analysis_views if "model" in v}

        for xml_file in module_path.rglob("*.xml"):
            try:
                content = xml_file.read_text()
                tree = ET.fromstring(content)
                for record in tree.findall(".//record[@model='ir.ui.view']"):
                    model_field = record.find("./field[@name='model']")
                    if model_field is not None and model_field.text in analysis_view_models:
                        context[module_name]["views"].append({"path": str(xml_file), "content": content})
                        break
            except (ET.ParseError, FileNotFoundError):
                continue

    def _gather_controllers(self, context: dict, module_name: str, module_path: Path, analysis: Dict[str, Any]) -> None:
        """Gathers controller files by matching routes from the analysis."""
        analysis_controllers: List[Dict[str, Any]] = analysis.get("controller", [])
        if not analysis_controllers:
            return
        analysis_routes: Set[str] = {c["action_name"] for c in analysis_controllers if "action_name" in c}

        controllers_dir = module_path / "controllers"
        if not controllers_dir.is_dir():
            return

        for py_file in controllers_dir.rglob("*.py"):
            content = py_file.read_text()
            routes_in_file: List[str] = re.findall(r"@http\.route\(\s*['\"]([^'\"]+)['\"]", content)
            routes_in_file_list: List[str] = re.findall(r"@http\.route\(\s*\[([^\]]+)\]", content)
            for route_list in routes_in_file_list:
                routes_in_file.extend(re.findall(r"['\"]([^'\"]+)['\"]", route_list))

            if analysis_routes.intersection(routes_in_file):
                context[module_name]["controllers"].append({"path": str(py_file), "content": content})

    def _gather_assets(self, context: dict, module_name: str, module_path: Path, analysis: Dict[str, Any]) -> None:
        """Gathers asset files by matching paths from the analysis."""
        for asset in analysis.get("assets", []):
            file_path_str: Optional[str] = asset.get("file_path")
            if not file_path_str:
                continue

            if file_path_str.startswith(f"/{module_name}/"):
                file_path_str = file_path_str[len(f"/{module_name}/") :]

            potential_path = module_path / file_path_str
            if potential_path.exists():
                context[module_name]["assets"].append(
                    {
                        "path": str(potential_path),
                        "content": potential_path.read_text(),
                    }
                )
            else:
                # Fallback to search by filename
                filename = Path(file_path_str).name
                for f in module_path.rglob(filename):
                    context[module_name]["assets"].append({"path": str(f), "content": f.read_text()})
                    break

    def _gather_security(self, context: dict, module_name: str, module_path: Path) -> None:
        """Gathers all files from the 'security' directory."""
        security_dir = module_path / "security"
        if security_dir.is_dir():
            for sec_file in security_dir.iterdir():
                if sec_file.is_file() and (sec_file.name.endswith(".csv") or sec_file.name.endswith(".xml")):
                    context[module_name]["security"].append({"path": str(sec_file), "content": sec_file.read_text()})

    def _gather_reports(self, context: dict, module_name: str, module_path: Path, analysis: Dict[str, Any]) -> None:
        """Gathers report definition files based on models in the analysis."""
        report_models: Set[str] = {r["model"] for r in analysis.get("reports", []) if "model" in r}
        if not report_models:
            return

        for xml_file in module_path.rglob("*.xml"):
            try:
                content = xml_file.read_text()
                tree = ET.fromstring(content)
                for record in tree.findall(".//record[@model='ir.actions.report']"):
                    model_field = record.find("./field[@name='model']")
                    if model_field is not None and model_field.text in report_models:
                        context[module_name]["reports"].append({"path": str(xml_file), "content": content})
                        break
            except (ET.ParseError, FileNotFoundError):
                continue

    def _gather_website_templates(
        self, context: dict, module_name: str, module_path: Path, analysis: Dict[str, Any]
    ) -> None:
        """Gathers website template files by matching template IDs from the analysis."""
        view_ids: Set[str] = {v["view"] for v in analysis.get("website_views", []) if "view" in v}
        if not view_ids:
            return

        for xml_file in module_path.rglob("*.xml"):
            try:
                content = xml_file.read_text()
                tree = ET.fromstring(content)
                for template in tree.findall(".//template"):
                    template_id = template.get("id")
                    if template_id and template_id in view_ids:
                        context[module_name]["website"].append({"path": str(xml_file), "content": content})
                        break
            except (ET.ParseError, FileNotFoundError):
                continue

    def _gather_data(self, context: dict, module_name: str, module_path: Path) -> None:
        """Gathers all XML data files from the 'data' directory."""
        data_dir = module_path / "data"
        if data_dir.is_dir():
            for data_file in data_dir.glob("*.xml"):
                if data_file.is_file():
                    context[module_name]["data"].append({"path": str(data_file), "content": data_file.read_text()})
