# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""CLIP model wrapper."""

import torch
import torch.nn.functional as F
from torchvision import transforms

from .base import BaseModelWrapper


class CLIPWrapper(BaseModelWrapper):
    """Wrapper for CLIP model."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP variant (e.g., 'ViT-B/32', 'ViT-B/16', 'ViT-L/14')
            device: Device to load model on
        """
        super().__init__(device)
        self.clip_model_name = model_name
        self.model_name = f"CLIP-{model_name.replace('/', '-')}"
        self.load_model()

    def load_model(self):
        """Load and initialize CLIP model."""
        print(f"Loading CLIP {self.clip_model_name}...")

        import clip

        self.model, self.preprocess = clip.load(
            self.clip_model_name, device=self.device
        )
        self.model.eval()
        print(f"Successfully loaded CLIP {self.clip_model_name}")

    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract features from tensor images (converts to PIL internally)."""
        with torch.no_grad():
            # Convert tensor images to PIL for CLIP preprocessing
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
            image_features = self.model.encode_image(processed_images)
            image_features = F.normalize(image_features, p=2, dim=1)

        return image_features
