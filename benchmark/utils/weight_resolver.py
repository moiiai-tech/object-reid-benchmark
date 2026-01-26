# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Automatic pretrained weight path resolver for dataset-specific models."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .logger import console
from ..exceptions import WeightResolutionError, ValidationError
from .validation import validate_file_readable, validate_checkpoint_file
from .logging import BenchmarkLogger

CLIPREID_WEIGHTS = {
    "market1501": {
        "vit_clipreid_sie_olp": "1K32xrosw0gPrxYCWXER81mhWObEW5-d4",
        "vit_clipreid": "1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7",
        "vit_baseline": "1XKUcP4LEpWr4Ah6sVdXNveUo4bAsVyjt",
    },
    "dukemtmcreid": {
        "vit_clipreid_sie_olp": "1zkHLrLy3z9lP0cR2MVQtr4ujoC6eQLKP",
        "vit_clipreid": "1ldjSkj-7pXAWmx8on5x0EftlCaolU4dY",
        "vit_baseline": "13qSSyi87Bkj3Qq-UKy3646vuIyIaE7Mt",
    },
    "msmt17": {
        "vit_clipreid_sie_olp": "1sPZbWTv2_stXBGutjHMvE87pAbSAgVaz",
        "vit_clipreid": "1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO",
        "vit_baseline": "1I715ZWacRvEGLiju1bZ9xcmUhhFx0aN6",
    },
}

TRANSREID_WEIGHTS = {
    "market1501": {
        "transreid_vit": "11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO",
        "baseline_vit": "1crYsKRrW4eUq6abT4KK8_atMLFsbq56W",
        "transreid_deit": "1cbUK2KozdPSoewzvF0ucFQnZ0yfZiu_H",
    },
    "dukemtmcreid": {
        "transreid_vit": "1BipxoqyThefQviJzuJIKtFJvNblIlPGN",
        "baseline_vit": "17GQqFuTleAZWLD92AtEd1c_dnTyZHl4k",
        "transreid_deit": "1ltaX9zGFO31Wwwu47K9c4WTTBZVLdzLw",
    },
    "msmt17": {
        "transreid_vit": "1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK",
        "baseline_vit": "1iF5JNPw9xi-rLY3Ri9EY-PFAkK6Vg_Pf",
        "transreid_deit": "1WSUD0gKjGIG_gzTc2izH_y-EuDzweN95",
    },
}


def download_from_google_drive(file_id: str, output_path: Path) -> bool:
    """
    Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        output_path: Path where to save the file

    Returns:
        True if download successful, False otherwise
    """
    try:
        try:
            import gdown
        except ImportError:
            console.print(
                "[cyan]Installing gdown for automatic weight download...[/cyan]"
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "gdown"]
            )
            import gdown

        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print(
            f"[cyan]Downloading weights to[/cyan] [bold]{output_path}[/bold]..."
        )
        console.print("[dim]   This may take a few minutes...[/dim]")

        gdown.download(id=file_id, output=str(output_path), quiet=False)

        if output_path.exists():
            console.print(
                f"[green]Download complete:[/green] [bold]{output_path}[/bold]"
            )
            return True
        else:
            console.print(
                f"[red]ERROR: Download failed:[/red] [bold]{output_path}[/bold]"
            )
            return False

    except Exception as e:
        console.print(f"[red]ERROR downloading weights:[/red] {e}")
        return False


class WeightResolver:
    """
    Resolves pretrained weight paths based on model type, variant, and dataset.

    Handles dataset-specific weight loading with fallback to MSMT17 weights
    when dataset-specific weights are not available.
    """

    def __init__(self, weights_root: str = "pretrained_models"):
        """
        Initialize weight resolver.

        Args:
            weights_root: Root directory containing pretrained weights
        """
        self.weights_root = Path(weights_root)

    def resolve_clipreid_weights(
        self,
        dataset_name: str,
        variant: str = "vit_clipreid_sie_olp",
        fallback_to_msmt17: bool = True,
        auto_download: bool = True,
    ) -> Optional[str]:
        """
        Resolve CLIP-ReID weight path for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'market1501', 'dukemtmcreid')
            variant: Model variant:
                - 'vit_clipreid': Standard ViT-CLIP-ReID
                - 'vit_clipreid_sie_olp': ViT-CLIP-ReID with SIE and Overlapping Local Patches (Best)
                - 'cnn_clipreid': CNN-based CLIP-ReID
                - 'cnn_baseline': CNN baseline without CLIP
                - 'vit_baseline': ViT baseline without CLIP
            fallback_to_msmt17: If True, use MSMT17 weights when dataset-specific weights not found
            auto_download: If True, automatically download weights if not found locally

        Returns:
            Path to weight file, or None if not found and download failed
        """
        weight_path = self.weights_root / "clipreid" / dataset_name / f"{variant}.pth"

        if weight_path.exists():
            console.print(
                f"[green]Found CLIP-ReID weights for {dataset_name}:[/green] [bold]{weight_path}[/bold]"
            )
            return str(weight_path)

        if auto_download and dataset_name in CLIPREID_WEIGHTS:
            if variant in CLIPREID_WEIGHTS[dataset_name]:
                file_id = CLIPREID_WEIGHTS[dataset_name][variant]
                console.print(
                    f"[yellow]WARNING: CLIP-ReID weights not found locally for {dataset_name}[/yellow]"
                )
                console.print(
                    "[cyan]   Attempting automatic download from Google Drive...[/cyan]"
                )

                if download_from_google_drive(file_id, weight_path):
                    return str(weight_path)

        if fallback_to_msmt17:
            fallback_path = self.weights_root / "clipreid" / "msmt17" / f"{variant}.pth"

            # Check if MSMT17 weights exist
            if fallback_path.exists():
                print(
                    f"WARNING: No CLIP-ReID weights for '{dataset_name}', using MSMT17 fallback: {fallback_path}"
                )
                return str(fallback_path)

            # Try to download MSMT17 fallback weights
            if auto_download and "msmt17" in CLIPREID_WEIGHTS:
                if variant in CLIPREID_WEIGHTS["msmt17"]:
                    file_id = CLIPREID_WEIGHTS["msmt17"][variant]
                    print("WARNING: MSMT17 fallback weights not found locally")
                    print("   Downloading MSMT17 weights as fallback...")

                    if download_from_google_drive(file_id, fallback_path):
                        return str(fallback_path)

        console.print(
            f"[red]ERROR: No CLIP-ReID weights found for {dataset_name} (variant: {variant})[/red]"
        )
        return None

    def resolve_transreid_weights(
        self,
        dataset_name: str,
        variant: str = "transreid_vit",
        fallback_to_msmt17: bool = True,
        auto_download: bool = True,
    ) -> Optional[str]:
        """
        Resolve TransReID weight path for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'market1501', 'dukemtmcreid')
            variant: Model variant:
                - 'baseline_vit': ViT baseline (no SIE, no JPM)
                - 'transreid_vit': TransReID with ViT (SIE + JPM) - Best
                - 'transreid_deit': TransReID with DeiT (SIE + JPM)
                - 'transreid_vit_stride': TransReID with ViT and stride size [12,12]
            fallback_to_msmt17: If True, use MSMT17 weights when dataset-specific weights not found
            auto_download: If True, automatically download weights if not found locally

        Returns:
            Path to weight file, or None if not found and download failed
        """
        # Try dataset-specific weights first
        weight_path = self.weights_root / "transreid" / dataset_name / f"{variant}.pth"

        if weight_path.exists():
            console.print(
                f"[green]Found TransReID weights for {dataset_name}:[/green] [bold]{weight_path}[/bold]"
            )
            return str(weight_path)

        # Try to download if auto_download is enabled and we have the file ID
        if auto_download and dataset_name in TRANSREID_WEIGHTS:
            if variant in TRANSREID_WEIGHTS[dataset_name]:
                file_id = TRANSREID_WEIGHTS[dataset_name][variant]
                console.print(
                    f"[yellow]WARNING: TransReID weights not found locally for {dataset_name}[/yellow]"
                )
                console.print(
                    "[cyan]   Attempting automatic download from Google Drive...[/cyan]"
                )

                if download_from_google_drive(file_id, weight_path):
                    return str(weight_path)

        # Fallback to MSMT17
        if fallback_to_msmt17:
            fallback_path = (
                self.weights_root / "transreid" / "msmt17" / f"{variant}.pth"
            )

            # Check if MSMT17 weights exist
            if fallback_path.exists():
                print(
                    f"WARNING: No TransReID weights for '{dataset_name}', using MSMT17 fallback: {fallback_path}"
                )
                return str(fallback_path)

            # Try to download MSMT17 fallback weights
            if auto_download and "msmt17" in TRANSREID_WEIGHTS:
                if variant in TRANSREID_WEIGHTS["msmt17"]:
                    file_id = TRANSREID_WEIGHTS["msmt17"][variant]
                    print("WARNING: MSMT17 fallback weights not found locally")
                    print("   Downloading MSMT17 weights as fallback...")

                    if download_from_google_drive(file_id, fallback_path):
                        return str(fallback_path)

        console.print(
            f"[red]ERROR: No TransReID weights found for {dataset_name} (variant: {variant})[/red]"
        )
        return None

    def get_clipreid_variant_from_config(self, model_cfg) -> str:
        """
        Determine CLIP-ReID variant from model configuration.

        Args:
            model_cfg: OmegaConf model configuration

        Returns:
            Variant name string
        """
        stride_size = model_cfg.get("stride_size", [16, 16])
        sie_camera = model_cfg.get("sie_camera", False)
        model_arch = model_cfg.get("name", "ViT-B-16")

        # Determine variant based on configuration
        if "RN" in model_arch or "ResNet" in model_arch:
            # CNN-based
            if sie_camera:
                return "cnn_clipreid"
            else:
                return "cnn_baseline"
        else:
            # ViT-based
            if sie_camera and stride_size[0] == 12:
                return "vit_clipreid_sie_olp"  # Best variant
            elif sie_camera:
                return "vit_clipreid"
            else:
                return "vit_baseline"

    def get_transreid_variant_from_config(self, model_cfg) -> str:
        """
        Determine TransReID variant from model configuration.

        Args:
            model_cfg: OmegaConf model configuration

        Returns:
            Variant name string
        """
        sie_camera = model_cfg.get("sie_camera", False)
        jpm = model_cfg.get("jpm", False)
        model_arch = model_cfg.get("name", "vit_base_patch16_224_TransReID")

        # Determine variant based on configuration
        if "deit" in model_arch.lower():
            # DeiT-based
            if sie_camera and jpm:
                return "transreid_deit"
            else:
                return "baseline_deit"  # Not common
        else:
            # ViT-based
            if sie_camera and jpm:
                # The publicly available weights are named "transreid_vit"
                # and were trained with stride [12,12] (the best variant)
                return "transreid_vit"
            else:
                return "baseline_vit"


# Global instance
_weight_resolver = WeightResolver()


def resolve_model_weights(
    model_type: str, dataset_name: str, model_cfg
) -> Optional[str]:
    """
    Convenience function to resolve weights for any model type.

    Args:
        model_type: Type of model ('clipreid' or 'transreid')
        dataset_name: Dataset name (target dataset for evaluation)
        model_cfg: OmegaConf model configuration

    Returns:
        Path to weight file, or None if not found
    """
    # If pretrained_path is explicitly set in config, use it
    if model_cfg.get("pretrained_path") not in [None, "null", ""]:
        return model_cfg.get("pretrained_path")

    # Check if source_domain is specified (for cross-domain evaluation)
    # If so, use source_domain instead of dataset_name for weight resolution
    source_domain = model_cfg.get("source_domain", None)
    weight_dataset = source_domain if source_domain else dataset_name

    # Disable fallback to MSMT17 when source_domain is explicitly specified
    # (we don't want to fall back when we're already using a specific source)
    use_fallback = not bool(source_domain)

    if source_domain:
        console.print(
            f"[cyan]Cross-domain evaluation:[/cyan] Loading {model_type} weights from "
            f"[bold]{source_domain}[/bold], evaluating on [bold]{dataset_name}[/bold]"
        )

    # Otherwise, auto-resolve based on model type
    if model_type == "clipreid":
        variant = _weight_resolver.get_clipreid_variant_from_config(model_cfg)
        return _weight_resolver.resolve_clipreid_weights(
            weight_dataset, variant, fallback_to_msmt17=use_fallback
        )
    elif model_type == "transreid":
        variant = _weight_resolver.get_transreid_variant_from_config(model_cfg)
        return _weight_resolver.resolve_transreid_weights(
            weight_dataset, variant, fallback_to_msmt17=use_fallback
        )
    else:
        return None
