# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Base model wrapper interface."""

from abc import ABC, abstractmethod

import torch


class BaseModelWrapper(ABC):
    """Base class for all model wrappers. Implement this to add new models."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.model_name = None

    @abstractmethod
    def load_model(self):
        """Load and initialize the model."""
        pass

    @abstractmethod
    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)

        Returns:
            features: Tensor of shape (B, feature_dim)
        """
        pass

    def get_num_params(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.model.parameters())
