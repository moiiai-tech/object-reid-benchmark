# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Image normalization utilities for person re-identification models.

This module provides functions to convert between different normalization schemes
commonly used in ReID models (ImageNet, CLIP, TransReID, etc.).
"""

import torch


def denormalize_imagenet(imgs: torch.Tensor) -> torch.Tensor:
    """
    Denormalize images from ImageNet normalization.

    Args:
        imgs: Batch of images normalized with ImageNet statistics (B, C, H, W)

    Returns:
        Denormalized images in [0, 1] range
    """
    mean_imagenet = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
    std_imagenet = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs.device)
    return imgs * std_imagenet + mean_imagenet


def normalize_clip(imgs: torch.Tensor) -> torch.Tensor:
    """
    Normalize images for CLIP models (mean=0.5, std=0.5).

    Args:
        imgs: Batch of images in [0, 1] range (B, C, H, W)

    Returns:
        Images normalized for CLIP
    """
    mean_clip = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(imgs.device)
    std_clip = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(imgs.device)
    return (imgs - mean_clip) / std_clip


def normalize_transreid(imgs: torch.Tensor) -> torch.Tensor:
    """
    Normalize images for TransReID models (mean=0.5, std=0.5).

    Note: TransReID uses the same normalization as CLIP.

    Args:
        imgs: Batch of images in [0, 1] range (B, C, H, W)

    Returns:
        Images normalized for TransReID
    """
    mean_transreid = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(imgs.device)
    std_transreid = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(imgs.device)
    return (imgs - mean_transreid) / std_transreid


def imagenet_to_clip(imgs: torch.Tensor) -> torch.Tensor:
    """
    Convert images from ImageNet normalization to CLIP normalization.

    This is a convenience function that combines denormalize_imagenet and normalize_clip.

    Args:
        imgs: Batch of images with ImageNet normalization (B, C, H, W)

    Returns:
        Images with CLIP normalization
    """
    imgs_denorm = denormalize_imagenet(imgs)
    return normalize_clip(imgs_denorm)


def imagenet_to_transreid(imgs: torch.Tensor) -> torch.Tensor:
    """
    Convert images from ImageNet normalization to TransReID normalization.

    This is a convenience function that combines denormalize_imagenet and normalize_transreid.

    Args:
        imgs: Batch of images with ImageNet normalization (B, C, H, W)

    Returns:
        Images with TransReID normalization
    """
    imgs_denorm = denormalize_imagenet(imgs)
    return normalize_transreid(imgs_denorm)
