# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Results viewing and analysis commands."""

import datetime
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from .utils import error_exit, success

console = Console()


@click.group()
def results():
    """View and analyze benchmark results."""
    pass


@results.command("list")
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Number of recent results to show",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show all results",
)
def list_results(limit: int, show_all: bool) -> None:
    """
    List recent benchmark results.

    \b
    Examples:
      reid results list
      reid results list --limit 20
      reid results list --all
    """
    # Look for result CSV files
    result_files = []

    # Check common locations
    search_paths = [
        Path("."),
        Path("results"),
        Path("outputs"),
    ]

    for search_path in search_paths:
        if search_path.exists():
            result_files.extend(search_path.glob("*.csv"))
            result_files.extend(search_path.glob("**/*.csv"))

    result_files = sorted(
        set(result_files), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not result_files:
        console.print("[yellow]WARNING: No result files found[/yellow]\n")
        return

    if not show_all:
        result_files = result_files[:limit]

    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Modified", style="yellow")
    table.add_column("Size", style="magenta", justify="right")

    for result_file in result_files:
        mtime = datetime.datetime.fromtimestamp(result_file.stat().st_mtime)
        size = result_file.stat().st_size

        # Format size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"

        table.add_row(
            str(result_file),
            mtime.strftime("%Y-%m-%d %H:%M:%S"),
            size_str,
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Showing {len(result_files)} result file(s)[/dim]\n")


@results.command("show")
@click.argument("result_file", type=click.Path(exists=True))
@click.option(
    "--format",
    type=click.Choice(["table", "json", "summary"]),
    default="table",
    help="Output format",
)
@click.option(
    "--sort-by",
    type=click.Choice(["map", "rank1", "model", "dataset"]),
    default="map",
    help="Sort results by column",
)
def show_results(result_file: str, format: str, sort_by: str) -> None:
    """
    Display benchmark results from a CSV file.

    \b
    Examples:
      reid results show unified_benchmark_results.csv
      reid results show results/run_001.csv --format json
      reid results show results/run_001.csv --sort-by rank1
    """
    try:
        df = pd.read_csv(result_file)

        if df.empty:
            console.print("[yellow]WARNING: Result file is empty[/yellow]\n")
            return

        sort_column_map = {
            "map": "mAP",
            "rank1": "Rank-1",
            "model": "model",
            "dataset": "dataset",
        }

        if sort_by in sort_column_map and sort_column_map[sort_by] in df.columns:
            ascending = sort_by in ["model", "dataset"]
            df = df.sort_values(by=sort_column_map[sort_by], ascending=ascending)

        if format == "json":
            console.print_json(df.to_json(orient="records", indent=2))

        elif format == "summary":
            console.print("\n[bold cyan]Benchmark Results Summary[/bold cyan]\n")
            console.print(f"Total runs: {len(df)}")
            console.print(f"Unique datasets: {df['dataset'].nunique()}")
            console.print(f"Unique models: {df['model'].nunique()}")

            if "mAP" in df.columns:
                console.print("\nmAP statistics:")
                console.print(f"  Mean: {df['mAP'].mean():.2f}%")
                console.print(f"  Max: {df['mAP'].max():.2f}%")
                console.print(f"  Min: {df['mAP'].min():.2f}%")

            console.print()

        else:  # table
            table = Table(title=f"Results: {Path(result_file).name}", show_header=True)

            # Add columns
            columns_to_show = ["model", "dataset", "mAP", "Rank-1", "Rank-5", "Rank-10"]
            for col in columns_to_show:
                if col in df.columns:
                    justify = "right" if col not in ["model", "dataset"] else "left"
                    style = "cyan" if col in ["model", "dataset"] else "yellow"
                    table.add_column(col, style=style, justify=justify)

            # Add rows
            for _, row in df.iterrows():
                row_data = []
                for col in columns_to_show:
                    if col in df.columns:
                        value = row[col]
                        # Format percentages
                        if col in ["mAP", "Rank-1", "Rank-5", "Rank-10"]:
                            row_data.append(f"{value:.1f}%")
                        else:
                            row_data.append(str(value))
                table.add_row(*row_data)

            console.print()
            console.print(table)
            console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]\n")
        sys.exit(130)
    except FileNotFoundError:
        error_exit(f"Result file not found: {result_file}")
    except pd.errors.EmptyDataError:
        error_exit(f"Result file is empty or invalid: {result_file}")
    except Exception as e:
        error_exit(f"Error reading results: {e}")


@results.command("compare")
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option(
    "--metric",
    type=click.Choice(["map", "rank1", "rank5", "rank10"]),
    default="map",
    help="Metric to compare",
)
def compare_results(file1: str, file2: str, metric: str) -> None:
    """
    Compare results from two benchmark runs.

    \b
    Example:
      reid results compare run1.csv run2.csv
      reid results compare run1.csv run2.csv --metric rank1
    """
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        metric_col_map = {
            "map": "mAP",
            "rank1": "Rank-1",
            "rank5": "Rank-5",
            "rank10": "Rank-10",
        }

        metric_col = metric_col_map[metric]

        if metric_col not in df1.columns or metric_col not in df2.columns:
            error_exit(f"Metric '{metric_col}' not found in both files")

        console.print(f"\n[bold cyan]Comparing {metric_col}[/bold cyan]\n")

        df1["key"] = (
            df1["model"].astype(str).str.cat(df1["dataset"].astype(str), sep="_")
        )
        df2["key"] = (
            df2["model"].astype(str).str.cat(df2["dataset"].astype(str), sep="_")
        )

        # Merge
        merged = pd.merge(
            df1[["key", "model", "dataset", metric_col]],
            df2[["key", metric_col]],
            on="key",
            suffixes=("_1", "_2"),
            how="outer",
        )

        merged["diff"] = merged[f"{metric_col}_2"].fillna(0) - merged[
            f"{metric_col}_1"
        ].fillna(0)

        table = Table(title="Comparison", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Dataset", style="cyan")
        table.add_column(
            f"File 1 ({Path(file1).name})", style="yellow", justify="right"
        )
        table.add_column(
            f"File 2 ({Path(file2).name})", style="yellow", justify="right"
        )
        table.add_column("Difference", style="green", justify="right")

        for _, row in merged.iterrows():
            val1 = (
                f"{row[f'{metric_col}_1']:.1f}%"
                if pd.notna(row[f"{metric_col}_1"])
                else "N/A"
            )
            val2 = (
                f"{row[f'{metric_col}_2']:.1f}%"
                if pd.notna(row[f"{metric_col}_2"])
                else "N/A"
            )

            diff = row["diff"]
            if pd.isna(diff):
                diff_str = "N/A"
                style = "white"
            elif diff > 0:
                diff_str = f"+{diff:.1f}%"
                style = "green"
            elif diff < 0:
                diff_str = f"{diff:.1f}%"
                style = "red"
            else:
                diff_str = "0.0%"
                style = "dim"

            table.add_row(
                row["model"],
                row["dataset"],
                val1,
                val2,
                f"[{style}]{diff_str}[/{style}]",
            )

        console.print(table)
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        error_exit(f"Error comparing results: {e}")


@results.command("export")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "markdown"]),
    default="json",
    help="Export format",
)
def export_results(input_file: str, output_file: str, format: str) -> None:
    """
    Export results to different formats.

    \b
    Examples:
      reid results export results.csv output.json
      reid results export results.csv output.md --format markdown
    """
    try:
        df = pd.read_csv(input_file)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "markdown":
            with open(output_path, "w") as f:
                f.write("# Benchmark Results\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n")

        success(f"Results exported to {output_path}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]\n")
        sys.exit(130)
    except PermissionError:
        error_exit(f"Permission denied writing to: {output_file}")
    except Exception as e:
        error_exit(f"Error exporting results: {e}")
