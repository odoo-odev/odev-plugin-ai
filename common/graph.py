import ast
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

import networkx as nx

from odev.common.console import console
from odev.common.logging import logging
from odev.common.odoobin import OdoobinProcess


logger = logging.getLogger(__name__)


def _read_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """
    Reads an Odoo manifest file and returns its content as a dictionary.

    This function safely parses the `__manifest__.py` file, which is a Python
    file containing a dictionary, and extracts this dictionary.

    :param manifest_path: The path to the __manifest__.py file.
    :return: A dictionary with the manifest content, or None if the file
             cannot be read or parsed.
    """
    if not manifest_path.is_file():
        return None
    try:
        content: str = manifest_path.read_text(encoding="utf-8")
        # An Odoo manifest is a Python file that should contain a dictionary literal.
        # We parse the file content into an Abstract Syntax Tree (AST)
        # and look for the first dictionary node.
        tree: ast.AST = ast.parse(content, filename=manifest_path.name)
        dict_node = next((node for node in ast.walk(tree) if isinstance(node, ast.Dict)), None)
        if dict_node:
            # ast.literal_eval can safely evaluate a node if it's a literal structure.
            return ast.literal_eval(dict_node)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Could not parse manifest file {manifest_path}: {e}")
    return None


def _get_module_path(process: OdoobinProcess, module_name: str) -> Optional[Path]:
    """
    Find the path of a module within the Odoo addons paths.

    :param process: The OdoobinProcess instance which knows about addon paths.
    :param module_name: The name of the module to find.
    :return: The path to the module, or None if not found.
    """
    # The OdoobinProcess provides the necessary addons paths.
    # `update_worktrees` was called here, but it does not exist on OdoobinProcess.
    # Any necessary updates should be performed before calling this graph builder.
    for addons_path in process.odoo_addons_paths:
        module_path: Path = addons_path / module_name
        if module_path.is_dir() and (module_path / "__manifest__.py").exists():
            return module_path
    return None


def build_dependency_tree(process: OdoobinProcess, modules: List[str]) -> nx.DiGraph:
    """
    Build a dependency tree for a list of Odoo modules.

    This method parses modules from the standard Odoo repositories (odoo,
    enterprise, design-themes), reads their manifests to find dependencies,
    and recursively builds a dependency graph.

    :param process: The OdoobinProcess instance to use for finding modules.
    :param modules: A list of module names to build the dependency tree from.
    :return: A networkx.DiGraph representing the dependency tree.
    """
    graph: nx.DiGraph = nx.DiGraph()
    to_process: List[str] = list(modules)
    processed: Set[str] = set()

    while to_process:
        module_name: str = to_process.pop(0)
        if module_name in processed:
            continue

        processed.add(module_name)
        graph.add_node(module_name)

        module_path: Optional[Path] = _get_module_path(process, module_name)
        if not module_path:
            logger.warning(f"Module '{module_name}' not found in standard Odoo repositories.")
            continue

        manifest: Optional[Dict[str, Any]] = _read_manifest(module_path / "__manifest__.py")
        if manifest and "depends" in manifest:
            dependencies: List[str] = manifest.get("depends", [])
            for dependency in dependencies:
                graph.add_edge(dependency, module_name)
                if dependency not in processed:
                    to_process.append(dependency)

    return graph


def print_dependency_tree(graph: nx.DiGraph, modules: List[str]) -> None:
    """
    Prints the dependency tree and installation order for a given graph.

    :param graph: The dependency graph, as returned by `build_dependency_tree`.
    :param modules: The list of initial modules to highlight in the output.
    """
    console.print(f"\n[bold underline]Dependency Tree for: {', '.join(modules)}[/bold underline]\n")

    sorted_modules: List[str] = sorted(graph.nodes())

    for module in sorted_modules:
        dependencies: List[str] = sorted(graph.predecessors(module))
        if dependencies:
            console.print(f"  [bold cyan]{module}[/bold cyan] -> {', '.join(dependencies)}")

    try:
        installation_order: List[str] = list(nx.topological_sort(graph))
        console.print("\n[bold underline]Installation Order (Topological Sort):[/bold underline]\n")
        for module in installation_order:
            console.print(f"  - {module}")
    except nx.NetworkXUnfeasible:
        console.print(
            "\n[bold red]Error: Circular dependency detected, cannot determine installation order.[/bold red]"
        )
