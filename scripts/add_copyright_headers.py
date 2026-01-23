#!/usr/bin/env python
"""
Script to add copyright headers to Python files.

Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional


DEFAULT_HEADER = """# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""

HEADER_WITH_MODULE_DOC = '''# Copyright (c) 2026 MoiiAi Inc. All rights reserved.

"""{}"""
'''


def has_copyright(content: str) -> bool:
    """Check if file already has a copyright header."""
    return "MoiiAi Inc" in content or "Copyright" in content[:500]


def add_header_to_file(
    filepath: Path,
    header: str = DEFAULT_HEADER,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Add copyright header to a Python file.

    Args:
        filepath: Path to the Python file
        header: Copyright header to add
        dry_run: If True, only print what would be done
        force: If True, add header even if copyright already exists

    Returns:
        True if header was added (or would be added in dry run), False otherwise
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    if not force and has_copyright(content):
        print(f"⏭️  Skipping {filepath} (already has copyright)")
        return False

    new_content = None

    if content.startswith("#!/usr/bin/env python"):
        lines = content.split("\n", 1)
        shebang = lines[0]
        rest = lines[1] if len(lines) > 1 else ""

        rest_stripped = rest.lstrip("\n")
        if rest_stripped.startswith('"""') or rest_stripped.startswith("'''"):
            quote = '"""' if rest_stripped.startswith('"""') else "'''"
            try:
                end_idx = rest_stripped.index(quote, 3) + 3
                docstring = rest_stripped[:end_idx]
                after_docstring = rest_stripped[end_idx:]

                new_content = f"{shebang}\n{header}{docstring}{after_docstring}"
            except ValueError:
                new_content = f"{shebang}\n{header}\n{rest}"
        else:
            new_content = f"{shebang}\n{header}\n{rest}"

    elif content.startswith('"""') or content.startswith("'''"):
        quote = '"""' if content.startswith('"""') else "'''"
        try:
            end_idx = content.index(quote, 3) + 3
            docstring = content[:end_idx]
            after_docstring = content[end_idx:]

            new_content = f"{header}{docstring}{after_docstring}"
        except ValueError:
            new_content = f"{header}\n{content}"
    else:
        new_content = f"{header}\n{content}"

    if dry_run:
        print(f"✓ Would add header to {filepath}")
        return True

    # Write the modified content
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✓ Added header to {filepath}")
        return True
    except Exception as e:
        print(f"Error writing {filepath}: {e}")
        return False


def find_python_files(
    directories: List[Path], exclude_patterns: Optional[List[str]] = None
) -> List[Path]:
    """
    Find all Python files in the given directories.

    Args:
        directories: List of directories to search
        exclude_patterns: Patterns to exclude (e.g., '__pycache__', 'external')

    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", ".egg-info", "external", ".venv", "venv"]

    python_files = []

    for directory in directories:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [
                d for d in dirs if not any(pattern in d for pattern in exclude_patterns)
            ]

            if any(pattern in str(root) for pattern in exclude_patterns):
                continue

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

    return sorted(python_files)


def main():
    parser = argparse.ArgumentParser(
        description="Add copyright headers to Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on benchmark directory
  python scripts/add_copyright_headers.py --dry-run benchmark/

  # Add headers to specific directories
  python scripts/add_copyright_headers.py benchmark/ reid/

  # Add headers including already copyrighted files
  python scripts/add_copyright_headers.py --force benchmark/

  # Add headers to all project files
  python scripts/add_copyright_headers.py benchmark/ reid/ scripts/ tests/
        """,
    )

    parser.add_argument(
        "directories", nargs="+", type=Path, help="Directories to process"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Add header even if copyright already exists",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["__pycache__", ".egg-info", "external", ".venv", "venv"],
        help="Patterns to exclude from processing",
    )

    args = parser.parse_args()

    for directory in args.directories:
        if not directory.exists():
            print(f"Error: Directory {directory} does not exist")
            return 1
        if not directory.is_dir():
            print(f"Error: {directory} is not a directory")
            return 1

    print(f"{'=' * 60}")
    print(f"MoiiAi Inc. Copyright Header Tool")
    print(f"{'=' * 60}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Force: {args.force}")
    print(f"Directories: {', '.join(str(d) for d in args.directories)}")
    print(f"Excluding: {', '.join(args.exclude)}")
    print(f"{'=' * 60}\n")

    python_files = find_python_files(args.directories, args.exclude)

    print(f"Found {len(python_files)} Python files\n")

    if not python_files:
        print("No Python files found!")
        return 0

    added_count = 0
    for filepath in python_files:
        if add_header_to_file(filepath, DEFAULT_HEADER, args.dry_run, args.force):
            added_count += 1

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Total files: {len(python_files)}")
    print(f"  Headers {'would be added' if args.dry_run else 'added'}: {added_count}")
    print(f"  Skipped: {len(python_files) - added_count}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
