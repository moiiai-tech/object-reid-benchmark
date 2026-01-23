# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Configuration management commands."""

import os
from pathlib import Path
from typing import List

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console()


@click.group()
def config():
    """Manage benchmark configurations."""
    pass


@config.command("list")
def list_configs():
    """
    List all available configuration files.

    \b
    Example:
      reid config list
    """
    config_dir = Path("benchmark/configs")

    if not config_dir.exists():
        console.print(f"[red]ERROR: Config directory not found: {config_dir}[/red]\n")
        return

    config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        console.print("[yellow]WARNING: No configuration files found[/yellow]\n")
        return

    table = Table(title="Available Configurations", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim")

    for config_file in config_files:
        name = config_file.stem
        table.add_row(name, str(config_file))

    console.print()
    console.print(table)
    console.print(f"\n[dim]Total: {len(config_files)} configurations[/dim]\n")


@config.command("show")
@click.argument("name")
@click.option(
    "--full-path",
    is_flag=True,
    help="Use full path instead of name",
)
def show_config(name, full_path):
    """
    Display the contents of a configuration file.

    \b
    Examples:
      reid config show default
      reid config show quick_test
      reid config show /path/to/custom.yaml --full-path
    """
    if full_path:
        config_path = Path(name)
    else:
        config_path = Path(f"benchmark/configs/{name}.yaml")
        if not config_path.exists():
            config_path = Path(f"benchmark/configs/{name}")

    if not config_path.exists():
        console.print(f"[red]ERROR: Config file not found: {config_path}[/red]\n")
        return

    try:
        with open(config_path, "r") as f:
            content = f.read()

        syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)

        console.print(f"\n[bold cyan]Configuration: {config_path.name}[/bold cyan]\n")
        console.print(syntax)
        console.print()

    except Exception as e:
        console.print(f"[red]ERROR: Error reading config file: {e}[/red]\n")


@config.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Enable strict validation (check datasets/models exist)")
def validate_config(config_file, strict):
    """
    Validate a configuration file.
    Checks syntax, required fields, and optionally verifies datasets/models.

    \b
    Examples:
      reid config validate benchmark/configs/quick_validation.yaml
      reid config validate my_config.yaml --strict
    """
    console.print(f"\n[bold cyan]Validating: {config_file}[/bold cyan]\n")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Validation results
        errors = []
        warnings = []
        info = []

        required_keys = ["datasets", "models"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: '{key}'")

        if "datasets" in config:
            if not isinstance(config["datasets"], list):
                errors.append("'datasets' must be a list")
            elif len(config["datasets"]) == 0:
                warnings.append("'datasets' list is empty")
            else:
                # Check dataset names
                if strict:
                    try:
                        from benchmark.datasets.registry import DATASET_REGISTRY
                        for ds in config["datasets"]:
                            if ds not in DATASET_REGISTRY:
                                errors.append(f"Unknown dataset: '{ds}' (not in registry)")
                            else:
                                data_root = config.get("data", {}).get("root", "reid-data")
                                ds_path = Path(data_root) / ds
                                if not ds_path.exists():
                                    warnings.append(f"Dataset not downloaded: '{ds}' (path: {ds_path})")
                    except ImportError:
                        warnings.append("Could not import dataset registry for strict validation")

        if "models" in config:
            if not isinstance(config["models"], list):
                errors.append("'models' must be a list")
            elif len(config["models"]) == 0:
                warnings.append("'models' list is empty")
            else:
                valid_model_types = ["osnet", "clip", "clipreid", "transreid", "pecore", "dinov2", "dinov3", "siglip2"]

                for i, model in enumerate(config["models"], 1):
                    if not isinstance(model, dict):
                        errors.append(f"Model #{i} must be a dictionary")
                        continue

                    # Required fields
                    if "type" not in model:
                        errors.append(f"Model #{i}: missing 'type' field")
                    elif strict and model["type"] not in valid_model_types:
                        errors.append(f"Model #{i}: unknown type '{model['type']}' (valid: {', '.join(valid_model_types)})")

                    if "name" not in model:
                        errors.append(f"Model #{i}: missing 'name' field")

                    model_type = model.get("type")

                    if model_type == "clipreid":
                        # Check for common CLIP-ReID parameters
                        if "stride_size" in model:
                            stride = model["stride_size"]
                            if not isinstance(stride, list) or len(stride) != 2:
                                errors.append(f"Model #{i}: 'stride_size' must be a list of 2 integers")
                            elif stride not in [[12, 12], [16, 16]]:
                                warnings.append(f"Model #{i}: unusual stride_size {stride} (common: [12,12] or [16,16])")

                        if "sie_camera" in model and not isinstance(model["sie_camera"], bool):
                            errors.append(f"Model #{i}: 'sie_camera' must be boolean")

                        # Warn about redundant parameters (auto-detected)
                        if "camera_num" in model:
                            info.append(f"Model #{i}: 'camera_num' is auto-detected from dataset (can be omitted)")
                        if "num_classes" in model:
                            info.append(f"Model #{i}: 'num_classes' is auto-detected from dataset (can be omitted)")

                    elif model_type == "transreid":
                        # Check TransReID parameters
                        if "sie_camera" in model and not isinstance(model["sie_camera"], bool):
                            errors.append(f"Model #{i}: 'sie_camera' must be boolean")

                        if "jpm" in model and not isinstance(model["jpm"], bool):
                            errors.append(f"Model #{i}: 'jpm' must be boolean")

                        # Warn about redundant parameters
                        if "camera_num" in model:
                            info.append(f"Model #{i}: 'camera_num' is auto-detected from dataset (can be omitted)")
                        if "num_classes" in model:
                            info.append(f"Model #{i}: 'num_classes' is auto-detected from dataset (can be omitted)")

        if "gpu_id" in config:
            if not isinstance(config["gpu_id"], int):
                errors.append("'gpu_id' must be an integer")
            elif config["gpu_id"] < 0:
                errors.append(f"'gpu_id' must be non-negative (got {config['gpu_id']})")
            elif strict:
                import torch
                if config["gpu_id"] >= torch.cuda.device_count():
                    errors.append(f"GPU {config['gpu_id']} not available (only {torch.cuda.device_count()} GPUs detected)")

        if "data" in config:
            if not isinstance(config["data"], dict):
                errors.append("'data' must be a dictionary")
            else:
                if "root" in config["data"]:
                    root_path = Path(config["data"]["root"])
                    if strict and not root_path.exists():
                        warnings.append(f"Data root directory does not exist: {root_path}")

        if config.get("cross_domain_mode"):
            if "source_domain" not in config:
                errors.append("'source_domain' required when cross_domain_mode is enabled")
            if "target_domains" not in config:
                errors.append("'target_domains' required when cross_domain_mode is enabled")
            elif not isinstance(config["target_domains"], list):
                errors.append("'target_domains' must be a list")

        has_issues = errors or warnings or info

        if errors:
            console.print("[red]✗ Validation failed with errors:[/red]\n")
            for error in errors:
                console.print(f"  [red]ERROR:[/red] {error}")
            console.print()

        if warnings:
            console.print("[yellow]⚠ Warnings:[/yellow]\n")
            for warning in warnings:
                console.print(f"  [yellow]WARN:[/yellow] {warning}")
            console.print()

        if info:
            console.print("[cyan]ℹ Information:[/cyan]\n")
            for i in info:
                console.print(f"  [cyan]INFO:[/cyan] {i}")
            console.print()

        if not has_issues:
            console.print("[green]✓ Configuration is valid[/green]\n")
        elif not errors:
            console.print("[green]✓ Configuration is valid (with warnings/info)[/green]\n")

        if not errors:
            table = Table(title="Configuration Summary", show_header=True, box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            if "datasets" in config:
                table.add_row("Datasets", str(len(config["datasets"])))
            if "models" in config:
                table.add_row("Models", str(len(config["models"])))
            if "gpu_id" in config:
                table.add_row("GPU ID", str(config["gpu_id"]))
            if config.get("cross_domain_mode"):
                table.add_row("Mode", "Cross-Domain")
            else:
                table.add_row("Mode", "Standard")

            if "datasets" in config and "models" in config:
                total_runs = len(config["datasets"]) * len(config["models"])
                if config.get("cross_domain_mode"):
                    total_runs = len(config.get("target_domains", [])) * len(config["models"])
                table.add_row("Total Runs", str(total_runs))

            console.print(table)
            console.print()

    except yaml.YAMLError as e:
        console.print(f"[red]✗ YAML parsing error:[/red]\n")
        console.print(f"  {e}\n")
    except Exception as e:
        console.print(f"[red]✗ Error validating config:[/red]\n")
        console.print(f"  {e}\n")


@config.command("preview")
@click.argument("config_file", type=click.Path(exists=True))
def preview_config(config_file):
    """
    Preview what will run with this configuration.
    Shows resolved parameters including auto-detected values.

    \b
    Example:
      reid config preview benchmark/configs/quick_validation.yaml
    """
    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)

        console.print(f"\n[bold cyan]Preview: {Path(config_file).name}[/bold cyan]\n")

        # GPU Info
        console.print(Panel(
            f"[bold]GPU:[/bold] {cfg.get('gpu_id', 0)}\n"
            f"[bold]Data Root:[/bold] {cfg.get('data', {}).get('root', 'reid-data')}\n"
            f"[bold]Output Dir:[/bold] {cfg.get('output', {}).get('results_dir', '.')}\n"
            f"[bold]CSV File:[/bold] {cfg.get('output', {}).get('csv_filename', 'results.csv')}",
            title="[bold]Configuration[/bold]",
            border_style="cyan"
        ))

        datasets = cfg.get("datasets", [])
        if datasets:
            console.print(f"\n[bold yellow]Datasets ({len(datasets)}):[/bold yellow]")
            for ds in datasets:
                dataset_path = Path(cfg.get('data', {}).get('root', 'reid-data')) / ds
                status = "[green]✓[/green]" if dataset_path.exists() else "[red]✗ (not downloaded)[/red]"
                console.print(f"  {status} {ds}")

        models = cfg.get("models", [])
        if models:
            console.print(f"\n[bold yellow]Models ({len(models)}):[/bold yellow]")
            for i, model in enumerate(models, 1):
                model_type = model.get("type", "unknown")
                model_name = model.get("name", "unknown")
                console.print(f"  [{i}] {model_type} - {model_name}")

                if model_type in ["clipreid", "transreid"]:
                    sie = model.get("sie_camera", False)
                    stride = model.get("stride_size", [16, 16])
                    console.print(f"      [dim]SIE: {sie}, Stride: {stride}[/dim]")

        # Cross-domain mode
        if cfg.get("cross_domain_mode"):
            console.print(f"\n[bold magenta]Cross-Domain Mode:[/bold magenta]")
            console.print(f"  Source: {cfg.get('source_domain', 'N/A')}")
            console.print(f"  Targets: {', '.join(cfg.get('target_domains', []))}")

        # Execution summary
        total_runs = len(datasets) * len(models)
        if cfg.get("cross_domain_mode"):
            total_runs = len(cfg.get("target_domains", [])) * len(models)

        console.print(f"\n[bold green]Total benchmark runs: {total_runs}[/bold green]\n")

    except yaml.YAMLError as e:
        console.print(f"[red]ERROR: YAML parsing error: {e}[/red]\n")
    except Exception as e:
        console.print(f"[red]ERROR: Error previewing config: {e}[/red]\n")


@config.command("create")
@click.option("--output", "-o", default=None, help="Output file path")
def create_config(output):
    """
    Interactively create a new benchmark configuration.
    Guides you through selecting datasets, models, and parameters.

    \b
    Example:
      reid config create
      reid config create --output my_config.yaml
    """
    console.print(Panel(
        "[bold]Interactive Configuration Builder[/bold]\n\n"
        "This wizard will help you create a benchmark configuration.\n"
        "Dataset-specific parameters (camera_num, num_classes) are auto-detected.",
        border_style="cyan"
    ))

    try:
        from benchmark.datasets.registry import DATASET_REGISTRY
        from benchmark.utils.dataset_params import DATASET_PARAMS
    except ImportError:
        console.print("[red]ERROR: Could not import dataset registry[/red]\n")
        return

    console.print("\n[bold cyan]Step 1: GPU Configuration[/bold cyan]")
    gpu_id = IntPrompt.ask("Which GPU to use?", default=0)

    console.print("\n[bold cyan]Step 2: Data Location[/bold cyan]")
    data_root = Prompt.ask("Dataset root directory", default="reid-data")

    console.print("\n[bold cyan]Step 3: Select Datasets[/bold cyan]")
    available_datasets = sorted(DATASET_REGISTRY.keys())

    table = Table(title="Available Datasets", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")

    for idx, ds_name in enumerate(available_datasets, 1):
        ds_path = Path(data_root) / ds_name
        status = "[green]Downloaded[/green]" if ds_path.exists() else "[yellow]Not downloaded[/yellow]"

        details = ""
        if ds_name in DATASET_PARAMS:
            params = DATASET_PARAMS[ds_name]
            details = f"{params['num_classes']} IDs, {params['camera_num']} cams"

        table.add_row(str(idx), ds_name, status, details)

    console.print(table)

    dataset_input = Prompt.ask(
        "\nEnter dataset numbers (comma-separated) or names",
        default="1"
    )

    selected_datasets = []
    for item in dataset_input.split(","):
        item = item.strip()
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(available_datasets):
                selected_datasets.append(available_datasets[idx])
        elif item in available_datasets:
            selected_datasets.append(item)

    if not selected_datasets:
        console.print("[red]No valid datasets selected![/red]\n")
        return

    console.print(f"\n[green]Selected: {', '.join(selected_datasets)}[/green]")

    console.print("\n[bold cyan]Step 4: Select Models[/bold cyan]")

    model_types = {
        "1": ("osnet", "OSNet - Lightweight CNN baseline (fast, good accuracy)"),
        "2": ("clip", "CLIP - Zero-shot vision-language model"),
        "3": ("clipreid", "CLIP-ReID - Fine-tuned CLIP for person ReID"),
        "4": ("transreid", "TransReID - Transformer-based ReID (state-of-the-art)"),
        "5": ("pecore", "PE-Core - Large pretrained vision model"),
        "6": ("dinov2", "DINOv2 - Self-supervised ViT"),
        "7": ("dinov3", "DINOv3 - Latest DINO version"),
        "8": ("siglip2", "SigLIP2 - Sigmoid loss CLIP"),
    }

    table = Table(show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="white")

    for num, (model_type, desc) in model_types.items():
        table.add_row(num, model_type, desc)

    console.print(table)

    model_input = Prompt.ask(
        "\nEnter model numbers (comma-separated)",
        default="1,2"
    )

    selected_models = []
    for item in model_input.split(","):
        item = item.strip()
        if item in model_types:
            model_type, _ = model_types[item]
            selected_models.append(model_type)

    if not selected_models:
        console.print("[red]No valid models selected![/red]\n")
        return

    models = []
    for model_type in selected_models:
        if model_type == "osnet":
            models.append({
                "type": "osnet",
                "name": "osnet_x1_0",
                "pretrained_path": None
            })
        elif model_type == "clip":
            models.append({
                "type": "clip",
                "name": "ViT-B/32"
            })
        elif model_type == "clipreid":
            console.print(f"\n[bold]Configure {model_type}:[/bold]")
            use_sie = Confirm.ask("  Use Side Information Embedding (camera ID)?", default=False)
            use_olp = Confirm.ask("  Use Overlapping Local Patches (stride 12x12)?", default=False) if use_sie else False

            models.append({
                "type": "clipreid",
                "name": "ViT-B-16",
                "pretrained_path": None,
                "view_num": 1,
                "stride_size": [12, 12] if use_olp else [16, 16],
                "input_size": [256, 128],
                "sie_camera": use_sie,
                "sie_coe": 1.0
            })
        elif model_type == "transreid":
            console.print(f"\n[bold]Configure {model_type}:[/bold]")
            use_sie = Confirm.ask("  Use Side Information Embedding?", default=True)
            use_jpm = Confirm.ask("  Use Jigsaw Patch Module?", default=False)

            models.append({
                "type": "transreid",
                "name": "vit_base_patch16_224_TransReID",
                "pretrained_path": None,
                "view_num": 1,
                "stride_size": [16, 16],
                "input_size": [256, 128],
                "sie_camera": use_sie,
                "sie_view": False,
                "sie_coe": 3.0,
                "jpm": use_jpm,
                "drop_path": 0.1,
                "drop_out": 0.0,
                "att_drop_rate": 0.0
            })
        elif model_type == "pecore":
            models.append({
                "type": "pecore",
                "name": "PE-Core-L14-336"
            })
        elif model_type == "dinov2":
            variant = Prompt.ask("  Variant", choices=["vitb14", "vitl14", "vitg14"], default="vitb14")
            models.append({
                "type": "dinov2",
                "name": f"dinov2_{variant}",
                "pretrained_path": None,
                "input_size": [256, 256]
            })
        elif model_type == "dinov3":
            variant = Prompt.ask("  Variant", choices=["vitb14", "vitl14"], default="vitb14")
            models.append({
                "type": "dinov3",
                "name": f"dinov3_{variant}",
                "pretrained_path": None,
                "input_size": [256, 256]
            })
        elif model_type == "siglip2":
            variant = Prompt.ask(
                "  Variant",
                choices=["base-patch16-256", "base-patch16-384", "so400m-patch14-384"],
                default="base-patch16-256"
            )
            input_size = [384, 384] if "384" in variant else [256, 256]
            models.append({
                "type": "siglip2",
                "name": f"google/siglip2-{variant}",
                "pretrained_path": None,
                "input_size": input_size
            })

    console.print("\n[bold cyan]Step 5: Output Configuration[/bold cyan]")
    results_dir = Prompt.ask("Results directory", default=".")
    csv_filename = Prompt.ask("CSV filename", default="benchmark_results.csv")

    config = {
        "gpu_id": gpu_id,
        "data": {
            "root": data_root
        },
        "datasets": selected_datasets,
        "models": models,
        "output": {
            "results_dir": results_dir,
            "csv_filename": csv_filename
        }
    }

    if output is None:
        default_name = f"{'_'.join(selected_datasets[:2])}_config.yaml"
        output = Prompt.ask("\nSave configuration as", default=default_name)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Auto-generated configuration\n")
        f.write(f"# Datasets: {', '.join(selected_datasets)}\n")
        f.write(f"# Models: {', '.join(selected_models)}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]✓ Configuration saved to: {output_path}[/green]")

    if Confirm.ask("\nPreview configuration?", default=True):
        with open(output_path, "r") as f:
            content = f.read()
        syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)

    console.print(f"\n[bold]Run with:[/bold] reid benchmark run --config {output_path}\n")


@config.command("templates")
def list_templates():
    """
    List available configuration templates with descriptions.

    \b
    Example:
      reid config templates
    """
    templates = {
        "quick_validation.yaml": "Fast smoke test - single dataset, simple model",
        "comprehensive_benchmark.yaml": "Full benchmark - multiple datasets and models",
        "advanced_models.yaml": "Test advanced models (CLIP-ReID, DINOv2, SigLIP2, etc.)",
        "cross_domain_test.yaml": "Cross-domain evaluation - test generalization",
        "minimal_template.yaml": "Minimal template - copy and customize",
    }

    console.print("\n[bold cyan]Available Configuration Templates[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Template", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Location", style="dim")

    config_dir = Path("benchmark/configs")
    for template, description in templates.items():
        template_path = config_dir / template
        if template_path.exists():
            table.add_row(template, description, str(template_path))

    console.print(table)

    console.print("\n[bold]Usage:[/bold]")
    console.print("  • View template: [cyan]reid config show <template_name>[/cyan]")
    console.print("  • Use template: [cyan]reid benchmark run --config benchmark/configs/<template>.yaml[/cyan]")
    console.print("  • Copy template: [cyan]cp benchmark/configs/<template>.yaml my_config.yaml[/cyan]\n")
