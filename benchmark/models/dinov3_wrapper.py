# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""DINOv3 model wrapper for person re-identification."""

import torch
import torch.nn.functional as F

from ..utils import get_model_logger
from .base import BaseModelWrapper

logger = get_model_logger("DINOv3Wrapper")


class DINOv3Wrapper(BaseModelWrapper):
    """Wrapper for DINOv3 (next generation self-supervised Vision Transformer) model."""

    def __init__(
        self,
        model_name: str = "vitb14",
        pretrained_path: str = None,
        device: str = "cuda",
        input_size: tuple = (256, 256),
    ):
        """
        Initialize DINOv3 model.

        NOTE: DINOv3 is not yet released. This wrapper falls back to DINOv2.

        Args:
            model_name: DINOv3/v2 variant short name (vits14, vitb14, vitl14, vitg14)
                       or full name (dinov2_vits14, dinov2_vitb14, etc.)
            pretrained_path: Path to pretrained weights (optional)
            device: Device to load model on
            input_size: Input image size (H, W)
        """
        super().__init__(device)
        # Normalize model name - try dinov3 first, fall back to dinov2
        if not model_name.startswith('dinov'):
            self.dinov3_model_name = f"dinov3_{model_name}"
            self.fallback_model_name = f"dinov2_{model_name}"
        else:
            self.dinov3_model_name = model_name
            self.fallback_model_name = model_name.replace('dinov3', 'dinov2')
        self.model_name = f"DINOv3-{model_name}"
        self.pretrained_path = pretrained_path
        self.input_size = input_size
        self.load_model()

    def load_model(self):
        """Load and initialize DINOv3 model (falls back to DINOv2)."""
        logger.info(f"Loading DINOv3 {self.dinov3_model_name}...")

        try:
            # Note: DINOv3 is not yet publicly released
            # This implementation falls back to DINOv2

            # Try to load from torch.hub (adjust repo name when DINOv3 is released)
            try:
                self.model = torch.hub.load('facebookresearch/dinov3', self.dinov3_model_name)
                logger.info(f"Successfully loaded DINOv3 from torch.hub")
            except Exception:
                # Fallback: Use DINOv2 since DINOv3 is not available
                logger.warning(f"DINOv3 not available. Using DINOv2 {self.fallback_model_name} as fallback.")
                self.model = torch.hub.load('facebookresearch/dinov2', self.fallback_model_name)

            # If custom pretrained weights are provided, load them
            if self.pretrained_path:
                logger.info(f"Loading pretrained weights from {self.pretrained_path}")
                state_dict = torch.load(self.pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded DINOv3 {self.dinov3_model_name}")
            logger.info(f"Model parameters: {self.get_num_params():,}")

        except Exception as e:
            logger.error(f"Error loading DINOv3 model: {e}")
            raise

    def extract_features(
        self, imgs: torch.Tensor, cam_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extract features from images using DINOv3.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
                  Expected to be normalized with ImageNet mean/std
            cam_labels: Ignored - DINOv3 does not use camera labels

        Returns:
            features: Tensor of shape (B, embed_dim)
        """
        with torch.no_grad():
            imgs = imgs.to(self.device)

            # Resize if needed to match DINOv3 expected input
            # DINOv3 expects 224x224 or multiples of patch_size
            current_size = imgs.shape[-2:]
            if current_size != (224, 224):
                imgs = F.interpolate(
                    imgs,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )

            # DINOv3 returns class token embeddings by default
            features = self.model(imgs)

            # Normalize features (L2 normalization)
            features = F.normalize(features, p=2, dim=1)

        return features
