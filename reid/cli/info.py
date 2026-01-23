# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""System information and version commands."""

import platform
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .utils import safe_import

console = Console()


@click.command()
def info():
    """
    Display system and environment information.

    Shows Python version, CUDA availability, installed packages,
    and project configuration.
    """
    try:
        console.print("\n[bold cyan]System Information[/bold cyan]\n")

        # System info table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        # Python info
        table.add_row(
            "Python Version",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
        table.add_row("Python Path", sys.executable)
        table.add_row("Platform", platform.platform())

        # CUDA info
        if safe_import("torch"):
            import torch

            if torch.cuda.is_available():
                table.add_row("CUDA Available", "Yes")
                cuda_version = torch.version.cuda or "Unknown"
                table.add_row("CUDA Version", cuda_version)
                table.add_row("GPU Count", str(torch.cuda.device_count()))

                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    table.add_row(f"  GPU {i}", gpu_name)

                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    if mem_allocated > 0 or mem_reserved > 0:
                        table.add_row(
                            "    Memory",
                            f"{mem_allocated:.2f}GB / {mem_reserved:.2f}GB reserved",
                        )
            else:
                table.add_row("CUDA Available", "No (CPU only)")

            # PyTorch info
            table.add_row("PyTorch Version", torch.__version__)
        else:
            table.add_row("PyTorch", "Not installed")

        # Project info
        project_root = Path.cwd()
        table.add_row("Project Root", str(project_root))

        data_dir = project_root / "reid-data"
        if data_dir.exists():
            dataset_count = sum(1 for d in data_dir.iterdir() if d.is_dir())
            table.add_row("Data Directory", f"{data_dir} ({dataset_count} datasets)")
        else:
            table.add_row("Data Directory", f"Not found ({data_dir})")

        models_dir = project_root / "pretrained_models"
        if models_dir.exists():
            model_count = sum(1 for _ in models_dir.rglob("*.pth"))
            table.add_row("Pretrained Models", f"{models_dir} ({model_count} weights)")
        else:
            table.add_row("Pretrained Models", f"Not found ({models_dir})")

        results_dir = project_root / "results"
        if results_dir.exists():
            result_count = len(list(results_dir.glob("*.csv")))
            table.add_row("Results Directory", f"{results_dir} ({result_count} files)")

        console.print(table)
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]ERROR: {e}[/red]\n")
        sys.exit(1)


@click.command()
def version():
    """Display version information."""
    try:
        from reid import __version__

        console.print(
            f"\n[bold cyan]reid[/bold cyan] version [bold green]{__version__}[/bold green]\n"
        )
    except Exception as e:
        console.print(f"\n[red]ERROR: {e}[/red]\n")
        sys.exit(1)
