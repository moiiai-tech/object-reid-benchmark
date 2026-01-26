# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""OSNet model wrapper."""

import os

import torch
import torchreid

from .base import BaseModelWrapper


class OSNetWrapper(BaseModelWrapper):
    """Wrapper for OSNet model."""

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        num_classes: int = 751,
        pretrained_path: str | None = None,
        device: str = "cuda",
    ):
        """
        Initialize OSNet model.

        Args:
            model_name: OSNet variant (e.g., 'osnet_x1_0', 'osnet_x0_75')
            num_classes: Number of training classes
            pretrained_path: Path to pretrained weights
            device: Device to load model on
        """
        super().__init__(device)
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.load_model()

    def load_model(self):
        """Load and initialize OSNet model."""
        print(f"Loading {self.model_name}...")

        # Build model
        self.model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=self.num_classes,
            loss="softmax",
            pretrained=False,
        )

        if self.pretrained_path is None:
            filename = f"{self.model_name}_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
            if os.path.exists(f"pretrained_models/osnet/{filename}"):
                self.pretrained_path = f"pretrained_models/osnet/{filename}"
            elif os.path.exists(f"models/{filename}"):
                self.pretrained_path = f"models/{filename}"
            else:
                self.pretrained_path = filename

        if os.path.exists(self.pretrained_path):
            print(f"Loading pretrained weights from {self.pretrained_path}")
            checkpoint = torch.load(self.pretrained_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            model_state = self.model.state_dict()
            filtered_checkpoint = {}

            for k, v in checkpoint.items():
                if k in model_state:
                    if model_state[k].shape == v.shape:
                        filtered_checkpoint[k] = v
                    else:
                        print(
                            f"Skipping {k} due to shape mismatch: "
                            f"checkpoint {v.shape} vs model {model_state[k].shape}"
                        )
                else:
                    print(f"Skipping {k}: not found in model")

            self.model.load_state_dict(filtered_checkpoint, strict=False)
            print(
                "Successfully loaded re-ID pretrained weights (feature extractor only)"
            )
        else:
            raise FileNotFoundError(
                f"Pretrained weights not found: {self.pretrained_path}\n"
                f"Cannot benchmark model without pretrained weights. "
                f"Random initialization would produce meaningless results.\n"
                f"Please download the weights or remove this model from the benchmark config."
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features(
        self, imgs: torch.Tensor, cam_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract features from tensor images.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
            cam_labels: Ignored - OSNet does not use camera labels
        """
        with torch.no_grad():
            features = self.model(imgs)
        return features
