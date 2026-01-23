# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Main CLI entry point for Object Re-ID Benchmark."""

import click
from rich.console import Console

from . import benchmark, config, dataset, info, model, results

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="reid")
@click.pass_context
def cli(ctx):
    """
    Object Re-Identification Benchmark CLI

    A comprehensive toolkit for person re-identification benchmarking.

    \b
    Common commands:
      reid info                    # Show system information
      reid dataset list            # List available datasets
      reid model list              # List available models
      reid benchmark quick         # Run quick benchmark test
      reid benchmark run           # Run full benchmark

    \b
    For more help on any command:
      reid <command> --help
    """
    ctx.ensure_object(dict)


# Register command groups
cli.add_command(info.info)
cli.add_command(dataset.dataset)
cli.add_command(model.model)
cli.add_command(benchmark.benchmark)
cli.add_command(config.config)
cli.add_command(results.results)


if __name__ == "__main__":
    cli()
