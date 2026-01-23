# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Benchmark execution commands."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel


console = Console()


@click.group()
def benchmark():
    """Run benchmarks and experiments."""
    pass


@benchmark.command("run")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to Hydra config file",
)
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="Dataset(s) to benchmark (can be specified multiple times)",
)
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="Model(s) to benchmark (can be specified multiple times)",
)
@click.option(
    "--gpu",
    "-g",
    type=int,
    default=0,
    help="GPU ID to use",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be run without executing",
)
def run_benchmark(
    config: Optional[str],
    dataset: Tuple[str, ...],
    model: Tuple[str, ...],
    gpu: int,
    output: Optional[str],
    dry_run: bool,
) -> None:
    """
    Run benchmarks with specified configuration.

    \b
    Examples:
      reid benchmark run --dataset market1501 --model osnet
      reid benchmark run --config benchmark/configs/quick_test.yaml
      reid benchmark run -d market1501 -d dukemtmcreid -m clip --gpu 1
      reid benchmark run --dry-run
    """
    import torch

    if not torch.cuda.is_available():
        console.print(
            "[red]ERROR: CUDA is not available. This command requires a GPU.[/red]\n"
        )
        return

    if gpu >= torch.cuda.device_count():
        console.print(
            f"[red]ERROR: GPU {gpu} not available. Found {torch.cuda.device_count()} GPUs.[/red]\n"
        )
        return

    cmd_args = []

    if config:
        # Hydra uses --config-path and --config-name, not --config
        # We'll pass the config file directly as an override
        cmd_args.extend(["--config-path", str(Path(config).parent), "--config-name", Path(config).stem])

    if dataset:
        dataset_list = ",".join(dataset)
        cmd_args.append(f"datasets=[{dataset_list}]")

    if model:
        # This would require config modification - show warning
        console.print(
            "[yellow]WARNING: Model filtering via CLI not yet implemented[/yellow]"
        )
        console.print("[dim]Please use --config to specify models[/dim]\n")

    if gpu is not None:
        cmd_args.append(f"gpu_id={gpu}")

    if output:
        cmd_args.append(f"output.results_dir={output}")

    console.print(
        Panel(
            f"[bold cyan]Benchmark Configuration[/bold cyan]\n\n"
            f"GPU: {gpu} ({torch.cuda.get_device_name(gpu)})\n"
            f"Config: {config or 'default'}\n"
            f"Datasets: {', '.join(dataset) if dataset else 'from config'}\n"
            f"Models: {', '.join(model) if model else 'from config'}",
            title="Benchmark Run",
            border_style="cyan",
        )
    )

    if dry_run:
        console.print("\n[yellow]Dry run - command that would be executed:[/yellow]")
        console.print(f"python run_benchmark.py {' '.join(cmd_args)}\n")
        return

    console.print("\n[bold green]▶️  Starting benchmark...[/bold green]\n")

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    sys.argv = ["run_benchmark.py"] + cmd_args

    try:
        import run_benchmark
        run_benchmark.main()
    except KeyboardInterrupt:
        console.print("\n[yellow]WARNING: Benchmark interrupted by user[/yellow]\n")
    except Exception as e:
        console.print(f"\n[red]ERROR: Error running benchmark: {e}[/red]\n")
        raise


@benchmark.command("quick")
@click.option(
    "--gpu",
    "-g",
    type=int,
    default=0,
    help="GPU ID to use",
)
def quick_benchmark(gpu):
    """
    Run a quick benchmark test.

    Uses the quick_test.yaml configuration with Market-1501 dataset
    and a subset of models for rapid validation.

    \b
    Example:
      reid benchmark quick
      reid benchmark quick --gpu 1
    """
    import torch

    if not torch.cuda.is_available():
        console.print(
            "[red]ERROR: CUDA is not available. This command requires a GPU.[/red]\n"
        )
        return

    if gpu >= torch.cuda.device_count():
        console.print(
            f"[red]ERROR: GPU {gpu} not available. Found {torch.cuda.device_count()} GPUs.[/red]\n"
        )
        return

    config_path = Path("benchmark/configs/quick_test.yaml")

    if not config_path.exists():
        console.print(f"[red]ERROR: Quick test config not found: {config_path}[/red]\n")
        return

    console.print(
        Panel(
            f"[bold cyan]Quick Benchmark Test[/bold cyan]\n\n"
            f"GPU: {gpu} ({torch.cuda.get_device_name(gpu)})\n"
            f"Config: {config_path}\n"
            f"Duration: ~5-10 minutes",
            title="⚡ Quick Test",
            border_style="cyan",
        )
    )

    console.print("\n[bold green]▶️  Starting quick benchmark...[/bold green]\n")

    sys.argv = [
        "run_benchmark.py",
        "--config",
        str(config_path),
        f"gpu_id={gpu}",
    ]

    try:
        from run_benchmark import main

        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]WARNING: Benchmark interrupted by user[/yellow]\n")
    except Exception as e:
        console.print(f"\n[red]ERROR: Error running quick benchmark: {e}[/red]\n")
        raise
