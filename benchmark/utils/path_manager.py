# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""System path management utilities for external dependencies.

This module provides centralized management of sys.path manipulation needed
for importing external model implementations (CLIP-ReID, TransReID, etc.).
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

EXTERNAL_DIR = Path(__file__).parent.parent.parent / "external"

EXTERNAL_MODULES = {
    "clipreid": EXTERNAL_DIR / "CLIP-ReID",
    "transreid": EXTERNAL_DIR / "TransReID",
    "pecore": EXTERNAL_DIR / "perception_models",
}


def add_external_path(module_name: str, suppress_warnings: bool = False) -> bool:
    """
    Add an external module directory to sys.path.

    Args:
        module_name: Name of the external module ('clipreid', 'transreid', 'pecore')
        suppress_warnings: If True, don't warn if module path doesn't exist

    Returns:
        True if path was added successfully, False otherwise
    """
    if module_name not in EXTERNAL_MODULES:
        if not suppress_warnings:
            print(f"Warning: Unknown external module '{module_name}'")
            print(f"Known modules: {', '.join(EXTERNAL_MODULES.keys())}")
        return False

    module_path = EXTERNAL_MODULES[module_name]

    if not module_path.exists():
        if not suppress_warnings:
            print(f"Warning: External module path does not exist: {module_path}")
        return False

    module_path_str = str(module_path)

    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

    return True


def remove_external_path(module_name: str) -> bool:
    """
    Remove an external module directory from sys.path.

    Args:
        module_name: Name of the external module to remove

    Returns:
        True if path was removed, False if it wasn't in sys.path
    """
    if module_name not in EXTERNAL_MODULES:
        return False

    module_path_str = str(EXTERNAL_MODULES[module_name])

    removed = False
    while module_path_str in sys.path:
        sys.path.remove(module_path_str)
        removed = True

    return removed


def clear_module_cache(module_names: List[str]) -> None:
    """
    Clear cached imports to avoid conflicts between modules.

    Args:
        module_names: List of module names to clear from sys.modules
    """
    for module_name in module_names:
        if module_name in sys.modules:
            del sys.modules[module_name]


@contextmanager
def isolated_external_path(
    module_name: str, exclude_modules: Optional[List[str]] = None
):
    """
    Context manager for isolated sys.path manipulation.

    This ensures that external module paths are properly added and removed,
    and conflicts with other modules are avoided.

    Args:
        module_name: Name of the external module to add
        exclude_modules: List of module names to temporarily remove from sys.path

    Example:
        with isolated_external_path('clipreid', exclude_modules=['transreid']):
            from model.make_model import build_transformer
    """
    original_path = sys.path.copy()

    try:
        if exclude_modules:
            for exclude_module in exclude_modules:
                remove_external_path(exclude_module)

        add_external_path(module_name)

        yield

    finally:
        sys.path = original_path


def setup_clipreid():
    """Setup sys.path for CLIP-ReID imports."""
    remove_external_path("transreid")
    clear_module_cache(["model", "model.make_model"])
    add_external_path("clipreid")


def setup_transreid():
    """Setup sys.path for TransReID imports."""
    remove_external_path("clipreid")
    clear_module_cache(["model", "model.make_model"])
    add_external_path("transreid")


def setup_pecore():
    """Setup sys.path for PECore imports."""
    add_external_path("pecore")
