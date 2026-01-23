# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Model management commands."""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def model():
    """Manage and query models."""
    pass


@model.command("list")
@click.option(
    "--format",
    type=click.Choice(["table", "json", "simple"]),
    default="table",
    help="Output format",
)
def list_models(format):
    """
    List all available model types.

    \b
    Examples:
      reid model list
      reid model list --format json
    """
    # Define available models with their info
    models = [
        {
            "type": "osnet",
            "name": "OSNet",
            "variants": ["osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25"],
            "description": "Omni-Scale Network for Re-ID",
        },
        {
            "type": "clip",
            "name": "CLIP",
            "variants": ["ViT-B/32", "ViT-B/16", "ViT-L/14"],
            "description": "OpenAI's Contrastive Language-Image Pre-training",
        },
        {
            "type": "clipreid",
            "name": "CLIP-ReID",
            "variants": ["ViT-B-16"],
            "description": "CLIP adapted for person Re-ID",
        },
        {
            "type": "transreid",
            "name": "TransReID",
            "variants": [
                "vit_base_patch16_224_TransReID",
                "vit_small_patch16_224_TransReID",
                "deit_small_patch16_224_TransReID",
            ],
            "description": "Vision Transformer for person Re-ID",
        },
        {
            "type": "pecore",
            "name": "PE-Core",
            "variants": ["PE-Core-L14-336"],
            "description": "Perception-focused encoder",
        },
        {
            "type": "dinov2",
            "name": "DINOv2",
            "variants": ["dinov2_vitb14", "dinov2_vitl14"],
            "description": "Self-supervised vision transformer",
        },
        {
            "type": "dinov3",
            "name": "DINOv3",
            "variants": ["dinov3_vitb14", "dinov3_vitl14"],
            "description": "DINOv3 vision transformer",
        },
        {
            "type": "siglip2",
            "name": "SigLIP2",
            "variants": ["siglip2_so400m_patch14_384"],
            "description": "Sigmoid loss for image pre-training",
        },
    ]

    if format == "json":
        console.print_json(json.dumps(models, indent=2))
    elif format == "simple":
        for model_info in models:
            console.print(model_info["type"])
    else:  # table
        table = Table(title="Available Models", show_header=True)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Name", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Variants", style="dim")

        for model_info in models:
            variant_count = len(model_info["variants"])
            variant_text = f"{variant_count} variant{'s' if variant_count > 1 else ''}"

            table.add_row(
                model_info["type"],
                model_info["name"],
                model_info["description"],
                variant_text,
            )

        console.print()
        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} model types[/dim]\n")


@model.command("info")
@click.argument("model_type")
def model_info(model_type):
    """
    Show detailed information about a model type.

    \b
    Example:
      reid model info transreid
    """
    # Model information database
    models_db = {
        "osnet": {
            "name": "OSNet",
            "full_name": "Omni-Scale Network",
            "description": "Lightweight CNN designed specifically for person Re-ID",
            "variants": [
                "osnet_x1_0 (2.2M params)",
                "osnet_x0_75 (1.4M params)",
                "osnet_x0_5 (0.9M params)",
                "osnet_x0_25 (0.4M params)",
            ],
            "pretrained": "Market-1501, DukeMTMC-reID",
            "input_size": "256×128",
        },
        "clip": {
            "name": "CLIP",
            "full_name": "Contrastive Language-Image Pre-training",
            "description": "OpenAI's vision-language model for zero-shot Re-ID",
            "variants": [
                "ViT-B/32 (151M params)",
                "ViT-B/16 (149M params)",
                "ViT-L/14 (428M params)",
            ],
            "pretrained": "400M image-text pairs",
            "input_size": "224×224 (varies by variant)",
        },
        "clipreid": {
            "name": "CLIP-ReID",
            "full_name": "CLIP for Person Re-Identification",
            "description": "CLIP fine-tuned with SIE and other Re-ID specific modifications",
            "variants": ["ViT-B-16"],
            "pretrained": "Market-1501, DukeMTMC-reID, MSMT17",
            "input_size": "256×128",
        },
        "transreid": {
            "name": "TransReID",
            "full_name": "Transformer-based person Re-ID",
            "description": "Vision Transformer with SIE and JPM for person Re-ID",
            "variants": [
                "vit_base_patch16_224_TransReID (86M params)",
                "vit_small_patch16_224_TransReID (22M params)",
                "deit_small_patch16_224_TransReID (22M params)",
            ],
            "pretrained": "Market-1501, DukeMTMC-reID, MSMT17",
            "input_size": "256×256",
        },
        "pecore": {
            "name": "PE-Core",
            "full_name": "Perception Encoder Core",
            "description": "Perception-focused CLIP variant",
            "variants": ["PE-Core-L14-336"],
            "pretrained": "Large-scale perception data",
            "input_size": "336×336",
        },
        "dinov2": {
            "name": "DINOv2",
            "full_name": "Self-Distillation with No Labels v2",
            "description": "Self-supervised vision transformer",
            "variants": [
                "dinov2_vitb14 (86M params)",
                "dinov2_vitl14 (304M params)",
            ],
            "pretrained": "LVD-142M dataset",
            "input_size": "256×256",
        },
        "dinov3": {
            "name": "DINOv3",
            "full_name": "Self-Distillation with No Labels v3",
            "description": "Latest DINO vision transformer",
            "variants": [
                "dinov3_vitb14",
                "dinov3_vitl14",
            ],
            "pretrained": "Large-scale self-supervised",
            "input_size": "256×256",
        },
        "siglip2": {
            "name": "SigLIP2",
            "full_name": "Sigmoid Loss for Language-Image Pre-training",
            "description": "CLIP-like model with sigmoid loss",
            "variants": ["siglip2_so400m_patch14_384"],
            "pretrained": "WebLI dataset",
            "input_size": "384×384",
        },
    }

    if model_type not in models_db:
        console.print(f"[red]ERROR: Model type '{model_type}' not found[/red]")
        console.print("\n[dim]Use 'reid model list' to see available models[/dim]\n")
        return

    info = models_db[model_type]

    console.print(f"\n[bold cyan]{info['name']}[/bold cyan] - {info['full_name']}\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Description", info["description"])
    table.add_row("Input Size", info["input_size"])
    table.add_row("Pretrained On", info["pretrained"])

    console.print(table)
    console.print("\n[cyan]Available Variants:[/cyan]")
    for variant in info["variants"]:
        console.print(f"  • {variant}")

    console.print()


@model.command("weights")
@click.argument("model_type")
@click.option("--dataset", help="Dataset name for dataset-specific weights")
def model_weights(model_type, dataset):
    """
    Check available pretrained weights for a model.

    \b
    Examples:
      reid model weights clipreid
      reid model weights transreid --dataset market1501
    """
    if model_type in ["clipreid", "transreid"]:
        from benchmark.utils.weight_resolver import CLIPREID_WEIGHTS, TRANSREID_WEIGHTS

        weights_db = CLIPREID_WEIGHTS if model_type == "clipreid" else TRANSREID_WEIGHTS

        console.print(
            f"\n[bold cyan]Available weights for {model_type.upper()}[/bold cyan]\n"
        )

        if dataset:
            if dataset in weights_db:
                table = Table(show_header=True)
                table.add_column("Variant", style="cyan")
                table.add_column("Google Drive ID", style="yellow")

                for variant, file_id in weights_db[dataset].items():
                    table.add_row(variant, file_id)

                console.print(f"[bold]Dataset: {dataset}[/bold]\n")
                console.print(table)
            else:
                console.print(
                    f"[yellow]WARNING: No weights available for dataset '{dataset}'[/yellow]"
                )
                console.print(
                    f"\n[dim]Available datasets: {', '.join(weights_db.keys())}[/dim]"
                )
        else:
            for ds_name, variants in weights_db.items():
                console.print(f"[bold]{ds_name}[/bold]: {len(variants)} variants")

        console.print()
    else:
        console.print(
            f"[yellow]WARNING: Weight management not yet implemented for '{model_type}'[/yellow]\n"
        )
