# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Dataset management commands."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .utils import error_exit, warning

console = Console()


@click.group()
def dataset():
    """Manage and query datasets."""
    pass


@dataset.command("list")
@click.option(
    "--custom",
    is_flag=True,
    help="Show only custom datasets",
)
@click.option(
    "--standard",
    is_flag=True,
    help="Show only standard torchreid datasets",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "simple"]),
    default="table",
    help="Output format",
)
def list_datasets(custom: bool, standard: bool, format: str) -> None:
    """
    List all available datasets.

    \b
    Examples:
      reid dataset list
      reid dataset list --custom
      reid dataset list --format json
    """
    try:
        from benchmark.datasets.registry import DATASET_REGISTRY
    except ImportError as e:
        error_exit(f"Failed to import dataset registry: {e}")
        return

    if not DATASET_REGISTRY:
        warning("No datasets available in registry")
        return

    datasets_to_show = list(DATASET_REGISTRY.items())

    if custom and not standard:
        datasets_to_show = [(n, c) for n, c in datasets_to_show if c.is_custom]
    elif standard and not custom:
        datasets_to_show = [(n, c) for n, c in datasets_to_show if not c.is_custom]

    if format == "json":
        data = [
            {
                "name": name,
                "height": config.height,
                "width": config.width,
                "batch_size": config.batch_size,
                "is_custom": config.is_custom,
            }
            for name, config in sorted(datasets_to_show)
        ]
        console.print_json(json.dumps(data, indent=2))
    elif format == "simple":
        for name, _ in sorted(datasets_to_show):
            console.print(name)
    else:  # table
        if not datasets_to_show:
            warning("No datasets to display")
            return
        table = Table(title="Available Datasets", show_header=True)
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Size", style="yellow")
        table.add_column("Batch", style="magenta", justify="right")
        table.add_column("Type", style="green")

        for name, config in sorted(datasets_to_show):
            dataset_type = "Custom" if config.is_custom else "Standard"
            table.add_row(
                name,
                f"{config.height}×{config.width}",
                str(config.batch_size),
                dataset_type,
            )

        console.print()
        console.print(table)
        console.print(f"\n[dim]Total: {len(datasets_to_show)} datasets[/dim]\n")


@dataset.command("info")
@click.argument("name")
def dataset_info(name: str) -> None:
    """
    Show detailed information about a dataset.

    \b
    Example:
      reid dataset info market1501
    """
    from benchmark.datasets.registry import DATASET_REGISTRY

    if name not in DATASET_REGISTRY:
        console.print(f"[red]ERROR: Dataset '{name}' not found[/red]")
        console.print(
            "\n[dim]Use 'reid dataset list' to see available datasets[/dim]\n"
        )
        sys.exit(1)

    config = DATASET_REGISTRY[name]

    console.print(f"\n[bold cyan]Dataset: {name}[/bold cyan]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Name", config.name)
    table.add_row("Source", config.source)
    table.add_row("Target", config.target)
    table.add_row("Image Size", f"{config.height} × {config.width}")
    table.add_row("Batch Size", str(config.batch_size))
    table.add_row("Split ID", str(config.split_id))
    table.add_row("Type", "Custom" if config.is_custom else "Standard (torchreid)")

    if config.is_custom and config.custom_class:
        table.add_row("Custom Class", config.custom_class.__name__)

    if config.use_cuhk03_metric:
        table.add_row("CUHK03 Metric", "Enabled")
        table.add_row("CUHK03 Labeled", str(config.cuhk03_labeled))
        table.add_row("CUHK03 Classic Split", str(config.cuhk03_classic_split))

    console.print(table)
    console.print()


@dataset.command("check")
@click.argument("name", required=False)
def check_dataset(name: Optional[str]) -> None:
    """
    Verify dataset files exist and are accessible.

    \b
    Examples:
      reid dataset check market1501
      reid dataset check             # Check all datasets
    """
    try:
        from benchmark.datasets.registry import DATASET_REGISTRY
    except ImportError as e:
        error_exit(f"Failed to import dataset registry: {e}")
        return

    data_root = Path("reid-data")

    if not data_root.exists():
        console.print(f"[red]ERROR: Data directory not found: {data_root}[/red]\n")
        console.print(
            "[dim]Tip: Create reid-data/ directory and download datasets[/dim]\n"
        )
        sys.exit(1)

    datasets_to_check = [name] if name else list(DATASET_REGISTRY.keys())

    console.print(f"\n[bold cyan]Checking datasets in {data_root}[/bold cyan]\n")

    results = []
    for dataset_name in datasets_to_check:
        if dataset_name not in DATASET_REGISTRY:
            console.print(f"[yellow]WARNING: Unknown dataset: {dataset_name}[/yellow]")
            continue

        config = DATASET_REGISTRY[dataset_name]
        dataset_path = data_root / config.source

        if dataset_path.exists():
            status = "[green]Available[/green]"
            status_text = "Found"
            style = "green"
        else:
            status = "[red]Not Available[/red]"
            status_text = "Not found"
            style = "red"

        results.append((dataset_name, status, status_text, style))
        console.print(f"{status} [{style}]{dataset_name:<20}[/{style}] - {status_text}")

    found_count = sum(1 for _, status, _, _ in results if "Available" in status)
    total_count = len(results)

    console.print(f"\n[dim]Found {found_count}/{total_count} datasets[/dim]\n")


@dataset.command("stats")
@click.argument("name")
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
def dataset_stats(name: str, detailed: bool) -> None:
    """
    Show statistics for a specific dataset.

    Requires the dataset to be available in reid-data directory.

    \b
    Example:
      reid dataset stats market1501
      reid dataset stats market1501 --detailed
    """
    try:
        from benchmark.datasets.registry import DATASET_REGISTRY
        from benchmark.datasets import load_dataset
    except ImportError as e:
        error_exit(f"Failed to import required modules: {e}")
        return

    if not name or not name.strip():
        error_exit("Dataset name cannot be empty")
        return

    if name not in DATASET_REGISTRY:
        console.print(f"[red]ERROR: Dataset '{name}' not found[/red]\n")
        sys.exit(1)

    try:
        console.print(f"\n[bold cyan]Loading dataset: {name}[/bold cyan]\n")

        with console.status("[yellow]Loading dataset...[/yellow]", spinner="dots"):
            test_loader, num_classes, dataset_config = load_dataset(
                name, root="reid-data"
            )

        console.print("[green]Dataset loaded successfully[/green]\n")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")

        table.add_row("Training Identities", str(num_classes))
        table.add_row("Query Images", str(len(test_loader["query"].dataset)))
        table.add_row("Gallery Images", str(len(test_loader["gallery"].dataset)))
        table.add_row(
            "Total Test Images",
            str(
                len(test_loader["query"].dataset) + len(test_loader["gallery"].dataset)
            ),
        )

        console.print(table)
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]\n")
        sys.exit(130)
    except FileNotFoundError as e:
        console.print(f"[red]ERROR: Dataset files not found: {e}[/red]\n")
        console.print("[dim]Tip: Check if dataset is downloaded in reid-data/[/dim]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]ERROR: Error loading dataset: {e}[/red]\n")
        sys.exit(1)
