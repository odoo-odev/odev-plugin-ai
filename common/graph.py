from pathlib import Path
from typing import Any

import networkx as nx

from odev.common import string
from odev.common.console import console
from odev.common.logging import logging
from odev.common.odoobin import OdoobinProcess


logger = logging.getLogger(__name__)


def _get_module_path(process: OdoobinProcess, module_name: str) -> Path | None:
    """Find the path of a module within the Odoo addons paths.

    :param process: The OdoobinProcess instance which knows about addon paths.
    :param module_name: The name of the module to find.
    :return: The path to the module, or None if not found.
    """
    # The OdoobinProcess provides the necessary addons paths.
    # `update_worktrees` was called here, but it does not exist on OdoobinProcess.
    # Any necessary updates should be performed before calling this graph builder.
    for addons_path in process.addons_paths:
        module_path: Path = addons_path / module_name
        if OdoobinProcess.check_addon_path(module_path):
            return module_path
    return None


def build_dependency_tree(process: OdoobinProcess, modules: list[str], max_level: int = 1) -> nx.DiGraph:
    """Build a dependency tree for a list of Odoo modules.

    This method parses modules from the standard Odoo repositories (odoo,
    enterprise, design-themes), reads their manifests to find dependencies,
    and recursively builds a dependency graph up to a certain level.

    :param process: The OdoobinProcess instance to use for finding modules.
    :param modules: A list of module names to build the dependency tree from.
    :param max_level: The maximum depth of dependencies to traverse. Defaults to 1.
    :return: A networkx.DiGraph representing the dependency tree.
    """
    graph: nx.DiGraph = nx.DiGraph()
    to_process: list[tuple[str, int]] = [(m, 0) for m in modules]
    processed: set[str] = set()

    while to_process:
        module_name, level = to_process.pop(0)
        if module_name in processed:
            continue

        processed.add(module_name)
        graph.add_node(module_name)

        if max_level is not None and level >= max_level:
            continue

        module_path: Path | None = _get_module_path(process, module_name)
        if not module_path:
            logger.warning(f"Module '{module_name}' not found in standard Odoo repositories.")
            continue

        manifest: dict[str, Any] | None = process.read_manifest(module_path / "__manifest__.py")
        if manifest and "depends" in manifest:
            dependencies: list[str] = manifest.get("depends", [])
            for dependency in dependencies:
                graph.add_edge(dependency, module_name)
                if dependency not in processed:
                    to_process.append((dependency, level + 1))

    return graph


def print_dependency_tree(graph: nx.DiGraph, modules: list[str]) -> None:
    """Print the dependency tree and installation order for a given graph.

    :param graph: The dependency graph, as returned by `build_dependency_tree`.
    :param modules: The list of initial modules to highlight in the output.
    """
    console.print(string.stylize(f"\nDependency Tree for: {', '.join(modules)}\n", "bold underline"))

    sorted_modules: list[str] = sorted(graph.nodes())

    for module in sorted_modules:
        dependencies: list[str] = sorted(graph.predecessors(module))
        if dependencies:
            console.print(f"  {string.stylize(module, 'bold cyan')} -> {', '.join(dependencies)}")

    try:
        installation_order: list[str] = list(nx.topological_sort(graph))
        console.print(f"\n{string.stylize('Installation Order (Topological Sort):', 'bold underline')}\n")
        for module in installation_order:
            console.print(f"  - {module}")
    except nx.NetworkXUnfeasible:
        console.print(
            f"\n{string.stylize('Error: Circular dependency detected, cannot determine installation order.', 'bold red')}"
        )
