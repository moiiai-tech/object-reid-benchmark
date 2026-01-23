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
        model_name: str = "dinov3_vitb14",
        pretrained_path: str = None,
        device: str = "cuda",
        input_size: tuple = (256, 256),
    ):
        """
        Initialize DINOv3 model.

        Args:
            model_name: DINOv3 variant (dinov3_vits14, dinov3_vitb14, dinov3_vitl14, dinov3_vitg14)
            pretrained_path: Path to pretrained weights (optional)
            device: Device to load model on
            input_size: Input image size (H, W)
        """
        super().__init__(device)
        self.dinov3_model_name = model_name
        self.model_name = f"DINOv3-{model_name}"
        self.pretrained_path = pretrained_path
        self.input_size = input_size
        self.load_model()

    def load_model(self):
        """Load and initialize DINOv3 model."""
        logger.info(f"Loading DINOv3 {self.dinov3_model_name}...")

        try:
            # Note: DINOv3 might not be available in torch.hub yet
            # This implementation assumes it will follow similar API to DINOv2
            # You may need to adjust based on the actual DINOv3 release

            # Try to load from torch.hub (adjust repo name when DINOv3 is released)
            try:
                self.model = torch.hub.load('facebookresearch/dinov3', self.dinov3_model_name)
            except Exception:
                # Fallback: If DINOv3 is not available, try using DINOv2 as placeholder
                logger.warning(f"DINOv3 not found in torch.hub. Using DINOv2 {self.dinov3_model_name.replace('dinov3', 'dinov2')} as fallback.")
                fallback_name = self.dinov3_model_name.replace('dinov3', 'dinov2')
                self.model = torch.hub.load('facebookresearch/dinov2', fallback_name)

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

    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using DINOv3.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
                  Expected to be normalized with ImageNet mean/std

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
