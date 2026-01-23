# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Results management and display with Rich formatting."""

from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .logger import console


def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save benchmark results to CSV.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    console.print(f"\n[success]Results saved to[/success] [bold]{output_file}[/bold]")
    return df


def print_summary(results: List[Dict[str, Any]]):
    """
    Print comprehensive summary of benchmark results with Rich formatting.

    Args:
        results: List of result dictionaries
    """
    if not results:
        console.print("[warning]No results to display.[/warning]")
        return

    df = pd.DataFrame(results)

    # Main results table
    table = Table(
        title="Unified Benchmark Summary - All Models & Datasets",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_blue",
        title_style="bold cyan",
    )

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="blue")
    table.add_column("Params", justify="right", style="green")
    table.add_column("mAP", justify="right", style="yellow")
    table.add_column("Rank-1", justify="right", style="yellow")
    table.add_column("Rank-5", justify="right", style="dim yellow")
    table.add_column("Rank-10", justify="right", style="dim yellow")
    table.add_column("Rank-20", justify="right", style="dim yellow")
    table.add_column("Time(s)", justify="right", style="magenta")

    for result in results:
        model = result["model"]
        dataset = result["dataset"]
        params = f"{result['params'] / 1e6:.1f}M"
        mAP = f"{result.get('mAP', 0):.1f}%"
        rank1 = f"{result.get('Rank-1', 0):.1f}%"
        rank5 = f"{result.get('Rank-5', 0):.1f}%"
        rank10 = f"{result.get('Rank-10', 0):.1f}%"
        rank20 = f"{result.get('Rank-20', 0):.1f}%"
        time_s = f"{result['time']:.1f}"

        table.add_row(model, dataset, params, mAP, rank1, rank5, rank10, rank20, time_s)

    console.print("\n")
    console.print(table)

    # Comparison tables if multiple datasets
    if df["dataset"].nunique() > 1:
        # mAP comparison
        console.print("\n")
        map_table = Table(
            title="mAP Comparison (Higher is better)",
            show_header=True,
            header_style="bold yellow",
            border_style="yellow",
        )

        pivot_map = df.pivot(index="model", columns="dataset", values="mAP")
        map_table.add_column("Model", style="cyan")
        for col in pivot_map.columns:
            map_table.add_column(col, justify="right", style="yellow")

        for model in pivot_map.index:
            row_data = [model]
            for col in pivot_map.columns:
                val = pivot_map.loc[model, col]
                row_data.append(f"{val:.1f}%" if pd.notna(val) else "-")
            map_table.add_row(*row_data)

        console.print(map_table)

        # Rank-1 comparison
        console.print("\n")
        rank1_table = Table(
            title="Rank-1 Comparison (Higher is better)",
            show_header=True,
            header_style="bold green",
            border_style="green",
        )

        pivot_rank1 = df.pivot(index="model", columns="dataset", values="Rank-1")
        rank1_table.add_column("Model", style="cyan")
        for col in pivot_rank1.columns:
            rank1_table.add_column(col, justify="right", style="green")

        for model in pivot_rank1.index:
            row_data = [model]
            for col in pivot_rank1.columns:
                val = pivot_rank1.loc[model, col]
                row_data.append(f"{val:.1f}%" if pd.notna(val) else "-")
            rank1_table.add_row(*row_data)

        console.print(rank1_table)
        console.print("\n")
        print("INFERENCE TIME COMPARISON (Lower is better)")
        print("=" * 80)
        pivot_time = df.pivot(index="model", columns="dataset", values="time")
        print(pivot_time.to_string())
