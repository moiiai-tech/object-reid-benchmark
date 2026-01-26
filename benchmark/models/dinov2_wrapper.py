# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""DINOv2 model wrapper for person re-identification."""

from typing import Any

import torch
import torch.nn.functional as F

from ..utils import get_model_logger
from .base import BaseModelWrapper

logger = get_model_logger("DINOv2Wrapper")


class DINOv2Wrapper(BaseModelWrapper):
    """Wrapper for DINOv2 (self-supervised Vision Transformer) model."""

    def __init__(
        self,
        model_name: str = "vitb14",
        pretrained_path: str = None,
        device: str = "cuda",
        input_size: tuple = (256, 256),
    ):
        """
        Initialize DINOv2 model.

        Args:
            model_name: DINOv2 variant short name (vits14, vitb14, vitl14, vitg14)
                       or full name (dinov2_vits14, dinov2_vitb14, etc.)
            pretrained_path: Path to pretrained weights (optional)
            device: Device to load model on
            input_size: Input image size (H, W)
        """
        super().__init__(device)
        # Normalize model name to full format
        if not model_name.startswith('dinov2_'):
            self.dinov2_model_name = f"dinov2_{model_name}"
        else:
            self.dinov2_model_name = model_name
        self.model_name = f"DINOv2-{model_name}"
        self.pretrained_path = pretrained_path
        self.input_size = input_size
        self.load_model()

    def load_model(self):
        """Load and initialize DINOv2 model."""
        logger.info(f"Loading DINOv2 {self.dinov2_model_name}...")

        try:
            self.model: Any = torch.hub.load(
                "facebookresearch/dinov2", self.dinov2_model_name
            )

            if self.pretrained_path:
                logger.info(f"Loading pretrained weights from {self.pretrained_path}")
                state_dict = torch.load(self.pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded DINOv2 {self.dinov2_model_name}")
            logger.info(f"Model parameters: {self.get_num_params():,}")

        except Exception as e:
            logger.error(f"Error loading DINOv2 model: {e}")
            raise

    def extract_features(
        self, imgs: torch.Tensor, cam_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extract features from images using DINOv2.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
                  Expected to be normalized with ImageNet mean/std
            cam_labels: Ignored - DINOv2 does not use camera labels

        Returns:
            features: Tensor of shape (B, embed_dim)
        """
        with torch.no_grad():
            imgs = imgs.to(self.device)

            current_size = imgs.shape[-2:]
            if current_size != (224, 224):
                imgs = F.interpolate(
                    imgs, size=(224, 224), mode="bilinear", align_corners=False
                )

            features = self.model(imgs)
            features = F.normalize(features, p=2, dim=1)

        return features
