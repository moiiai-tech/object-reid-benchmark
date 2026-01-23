# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""PE-Core model wrapper."""

import torch
from torchvision import transforms

from ..utils import get_model_logger, setup_pecore
from .base import BaseModelWrapper

logger = get_model_logger("PECoreWrapper")


class PECoreWrapper(BaseModelWrapper):
    """Wrapper for PE-Core model."""

    def __init__(self, model_config: str = "PE-Core-L14-336", device: str = "cuda"):
        """
        Initialize PE-Core model.

        Args:
            model_config: PE-Core configuration name
            device: Device to load model on
        """
        super().__init__(device)
        self.model_config = model_config
        self.model_name = model_config
        self.load_model()

    def load_model(self):
        """Load and initialize PE-Core model."""
        logger.info(f"Loading {self.model_config}...")

        setup_pecore()

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        self.model = pe.CLIP.from_config(self.model_config, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.device_type = (
            self.device.split(":")[0] if isinstance(self.device, str) else self.device.type
        )
        logger.info(f"Successfully loaded {self.model_config}")

    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract features from tensor images (converts to PIL internally)."""
        with torch.no_grad(), torch.autocast(self.device_type):
            # Convert tensor images to PIL
            pil_images = []
            for img in imgs:
                img = img.cpu()
                img = transforms.ToPILImage()(img)
                pil_images.append(img)

            # Preprocess and encode
            processed_images = torch.stack(
                [self.preprocess(img) for img in pil_images]
            )
            processed_images = processed_images.to(self.device)

            # Get image features
            image_features, _, _ = self.model(processed_images, None)

        return image_features
