# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""PE-Spatial model wrapper."""

import torch
from torchvision import transforms

from ..utils import get_model_logger, setup_pecore
from .base import BaseModelWrapper

logger = get_model_logger("PESpatialWrapper")


class PESpatialWrapper(BaseModelWrapper):
    """Wrapper for PE-Spatial model."""

    def __init__(self, model_config: str = "PE-Spatial-G14-448", device: str = "cuda"):
        """
        Initialize PE-Spatial model.

        Args:
            model_config: PE-Spatial configuration name (e.g., 'PE-Spatial-G14-448')
            device: Device to load model on
        """
        super().__init__(device)
        self.model_config = model_config
        self.model_name = model_config
        self.load_model()

    def load_model(self):
        """Load and initialize PE-Spatial model."""
        logger.info(f"Loading {self.model_config}...")

        # Setup sys.path for PE models (same as PE-Core)
        setup_pecore()

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        # PE-Spatial uses VisionTransformer (vision-only), not CLIP (vision+text)
        self.model = pe.VisionTransformer.from_config(
            self.model_config, pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.device_type = (
            self.device.split(":")[0]
            if isinstance(self.device, str)
            else self.device.type
        )
        logger.info(f"Successfully loaded {self.model_config}")

    def extract_features(
        self, imgs: torch.Tensor, cam_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract features from tensor images (converts to PIL internally).

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
            cam_labels: Ignored - PESpatial does not use camera labels
        """
        with torch.no_grad(), torch.autocast(self.device_type):
            # Convert tensor images to PIL
            pil_images = []
            for img in imgs:
                img = img.cpu()
                img = transforms.ToPILImage()(img)
                pil_images.append(img)

            # Preprocess and encode
            processed_images = torch.stack([self.preprocess(img) for img in pil_images])
            processed_images = processed_images.to(self.device)

            # Get image features - VisionTransformer returns [B, num_tokens, hidden_dim]
            image_features = self.model(processed_images)

            # Mean pool spatial tokens for ReID (reduces 786K dim to 768 dim)
            image_features = image_features.mean(dim=1)

        return image_features
