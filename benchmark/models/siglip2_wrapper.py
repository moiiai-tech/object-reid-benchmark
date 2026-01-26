# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""SigLIP2 model wrapper for person re-identification."""

import torch
import torch.nn.functional as F

from ..utils import get_model_logger
from .base import BaseModelWrapper

logger = get_model_logger("SigLIP2Wrapper")


class SigLIP2Wrapper(BaseModelWrapper):
    """Wrapper for SigLIP2 (Sigmoid Loss for Language-Image Pre-training v2) model."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-256",
        pretrained_path: str = None,
        num_classes: int = None,
        device: str = "cuda",
        input_size: tuple = (256, 256),
    ):
        """
        Initialize SigLIP2 model.

        Args:
            model_name: SigLIP2 variant from HuggingFace
                       - google/siglip2-base-patch16-256 (86M params)
                       - google/siglip2-base-patch16-384 (86M params)
                       - google/siglip2-large-patch16-256 (303M params)
                       - google/siglip2-large-patch16-384 (303M params)
                       - google/siglip2-so400m-patch14-384 (400M params)
                       - google/siglip2-so400m-patch14-512 (400M params)
                       - google/siglip2-base-patch16-naflex (dynamic resolution)
            pretrained_path: Path to custom pretrained weights (optional)
            num_classes: Number of classes (not used for SigLIP2, kept for compatibility)
            device: Device to load model on
            input_size: Input image size (H, W)
        """
        super().__init__(device)
        self.siglip2_model_name = model_name
        self.model_name = f"SigLIP2-{model_name.split('/')[-1]}"
        self.pretrained_path = pretrained_path
        self.num_classes = num_classes
        self.input_size = input_size
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load and initialize SigLIP2 model."""
        logger.info(f"Loading SigLIP2 {self.siglip2_model_name}...")

        try:
            from transformers import AutoModel, AutoProcessor

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.siglip2_model_name)
            self.model = AutoModel.from_pretrained(
                self.siglip2_model_name,
                device_map=self.device
            )

            # If custom pretrained weights are provided, load them
            if self.pretrained_path:
                logger.info(f"Loading pretrained weights from {self.pretrained_path}")
                state_dict = torch.load(self.pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)

            self.model.eval()

            logger.info(f"Successfully loaded SigLIP2 {self.siglip2_model_name}")
            logger.info(f"Model parameters: {self.get_num_params():,}")

        except Exception as e:
            logger.error(f"Error loading SigLIP2 model: {e}")
            logger.error("Note: SigLIP2 requires transformers >= 4.49.0")
            logger.error("Install with: pip install git+https://github.com/huggingface/transformers@main")
            raise

    def extract_features(
        self, imgs: torch.Tensor, cam_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extract features from images using SigLIP2.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
                  Expected to be normalized with ImageNet mean/std
            cam_labels: Ignored - SigLIP2 does not use camera labels

        Returns:
            features: Tensor of shape (B, embed_dim)
        """
        with torch.no_grad():
            # Denormalize from ImageNet normalization
            mean_imagenet = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
            std_imagenet = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs.device)
            imgs_denorm = imgs * std_imagenet + mean_imagenet

            # Convert to PIL images for processor
            from torchvision import transforms as T
            pil_images = []
            for img in imgs_denorm:
                img = img.cpu()
                img = T.ToPILImage()(img)
                pil_images.append(img)


            inputs = self.processor(
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)

            # Extract image features
            # SigLIP2 provides get_image_features method
            features = self.model.get_image_features(**inputs)

            # Normalize features (L2 normalization)
            features = F.normalize(features, p=2, dim=1)

        return features
