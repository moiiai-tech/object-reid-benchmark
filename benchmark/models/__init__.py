# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Model wrappers for person re-identification benchmarking."""

from .base import BaseModelWrapper
from .clip_wrapper import CLIPWrapper
from .clipreid_wrapper import CLIPReIDWrapper
from .dinov2_wrapper import DINOv2Wrapper
from .dinov3_wrapper import DINOv3Wrapper
from .factory import create_model
from .osnet import OSNetWrapper
from .pecore import PECoreWrapper
from .pespatial import PESpatialWrapper
from .siglip2_wrapper import SigLIP2Wrapper
from .transreid_wrapper import TransReIDWrapper

__all__ = [
    "BaseModelWrapper",
    "OSNetWrapper",
    "CLIPWrapper",
    "PECoreWrapper",
    "PESpatialWrapper",
    "CLIPReIDWrapper",
    "TransReIDWrapper",
    "DINOv2Wrapper",
    "DINOv3Wrapper",
    "SigLIP2Wrapper",
    "create_model",
]
