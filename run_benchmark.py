#!/usr/bin/env python
# Copyright (c) 2026 MoiiAi Inc. All rights reserved.

"""
Unified Person Re-Identification Benchmark Runner

Config-driven benchmark framework using OmegaConf.

Usage:
    python run_benchmark.py
    python run_benchmark.py --config benchmark/configs/quick_test.yaml
    python run_benchmark.py gpu_id=0 datasets=[market1501]
    python run_benchmark.py --config benchmark/configs/default.yaml gpu_id=0
"""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from benchmark.datasets import load_dataset
from benchmark.models import create_model
from benchmark.utils import benchmark_model, print_summary, save_results

# Add perception_models to Python path for PE-Core imports
perception_models_path = Path(__file__).parent / "external" / "perception_models"
if perception_models_path.exists() and str(perception_models_path) not in sys.path:
    sys.path.insert(0, str(perception_models_path))

console = Console()


def run_benchmarks(cfg: DictConfig):
    """
    Run benchmarks based on configuration.

    Args:
        cfg: OmegaConf configuration object
    """
    if not torch.cuda.is_available():
        console.print(
            "[bold red]ERROR: CUDA is not available. This script requires a GPU.[/bold red]"
        )
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    gpu_id = cfg.gpu_id
    if gpu_id >= torch.cuda.device_count():
        console.print(
            f"[bold red]ERROR: GPU {gpu_id} not available. Found {torch.cuda.device_count()} GPUs.[/bold red]"
        )
        raise ValueError(
            f"GPU {gpu_id} not available. Found {torch.cuda.device_count()} GPUs."
        )

    gpu_info = Text()
    gpu_info.append("GPU: ", style="bold cyan")
    gpu_info.append(f"{torch.cuda.get_device_name(gpu_id)}", style="bold green")
    gpu_info.append(f" (Device {gpu_id})", style="dim")
    console.print(Panel(gpu_info, border_style="cyan", padding=(0, 2)))

    config_yaml = OmegaConf.to_yaml(cfg)
    syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False)
    console.print(
        Panel(syntax, title="Configuration", border_style="blue", padding=(1, 2))
    )

    device = torch.device(f"cuda:{gpu_id}")
    all_results = []

    cross_domain_mode = cfg.get("cross_domain_mode", False)
    if cross_domain_mode:
        info_text = Text()
        info_text.append("CROSS-DOMAIN EVALUATION MODE\n\n", style="bold magenta")
        info_text.append("Source Domain: ", style="cyan")
        info_text.append(f"{cfg.get('source_domain', 'N/A')}\n", style="bold yellow")
        info_text.append("Target Domains: ", style="cyan")
        info_text.append(
            f"{', '.join(cfg.get('target_domains', cfg.get('datasets', [])))}",
            style="bold yellow",
        )
        console.print(Panel(info_text, border_style="magenta", padding=(1, 2)))
        datasets_to_eval = cfg.get("target_domains", cfg.get("datasets", []))
    else:
        datasets_to_eval = cfg.datasets

    for dataset_name in datasets_to_eval:
        dataset_header = Text()
        dataset_header.append("BENCHMARKING DATASET: ", style="bold cyan")
        dataset_header.append(dataset_name.upper(), style="bold yellow")
        console.print(Panel(dataset_header, border_style="yellow", padding=(0, 2)))

        try:
            test_loader, num_classes, dataset_config = load_dataset(
                dataset_name, root=cfg.data.root
            )

            for model_cfg in cfg.models:
                try:
                    # In cross-domain mode, use source_domain for model initialization
                    # but evaluate on current dataset_name
                    if cross_domain_mode:
                        source_domain = model_cfg.get(
                            "source_domain", cfg.get("source_domain", dataset_name)
                        )
                        model = create_model(
                            model_cfg,
                            num_classes,
                            device,
                            source_domain,
                            cross_domain=True,
                            target_dataset=dataset_name,
                        )
                        console.print(
                            f"  [cyan]Cross-domain:[/cyan] trained on [yellow]{source_domain}[/yellow], "
                            f"evaluating on [yellow]{dataset_name}[/yellow]"
                        )
                    else:
                        model = create_model(
                            model_cfg, num_classes, device, dataset_name
                        )

                    result = benchmark_model(
                        model, test_loader, dataset_config, gpu_id=gpu_id
                    )
                    all_results.append(result)

                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    console.print(
                        f"[bold red]ERROR benchmarking {model_cfg.name} on {dataset_name}:[/bold red] {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        except Exception as e:
            console.print(
                f"[bold red]ERROR loading dataset {dataset_name}:[/bold red] {e}"
            )
            import traceback

            traceback.print_exc()
            continue

    if all_results:
        output_dir = Path(cfg.output.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / cfg.output.csv_filename
        save_results(all_results, str(output_file))

        print_summary(all_results)
    else:
        console.print(
            "\n[bold yellow]WARNING: No results generated. Please check for errors above.[/bold yellow]"
        )


@hydra.main(version_base=None, config_path="benchmark/configs", config_name="default")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration management."""
    run_benchmarks(cfg)


if __name__ == "__main__":
    main()
