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
    Normalize images for CLIP models using official OpenAI normalization values.

    These values are from the official OpenAI CLIP implementation:
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L79

    Args:
        imgs: Batch of images in [0, 1] range (B, C, H, W)

    Returns:
        Images normalized for CLIP with mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    """
    mean_clip = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(imgs.device)
    std_clip = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(imgs.device)
    return (imgs - mean_clip) / std_clip


def normalize_clipreid(imgs: torch.Tensor) -> torch.Tensor:
    """
    Normalize images for CLIP-ReID models (mean=0.5, std=0.5).

    IMPORTANT: CLIP-ReID uses [0.5, 0.5, 0.5] normalization for training/inference,
    NOT the standard CLIP normalization [0.481..., 0.457..., 0.408...].

    This is despite using CLIP's visual encoder, because the pretrained weights were
    fine-tuned with [0.5, 0.5, 0.5] normalization.

    Source: https://github.com/Syliz/CLIP-ReID/blob/main/configs/person/vit_clipreid.yml#L20-L21

    Args:
        imgs: Batch of images in [0, 1] range (B, C, H, W)

    Returns:
        Images normalized for CLIP-ReID with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    """
    mean_clipreid = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(imgs.device)
    std_clipreid = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(imgs.device)
    return (imgs - mean_clipreid) / std_clipreid


def normalize_transreid(imgs: torch.Tensor) -> torch.Tensor:
    """
    Normalize images for TransReID models (mean=0.5, std=0.5).

    TransReID uses simplified normalization values compared to CLIP's official values.
    This is consistent across all TransReID config files in the official repository.

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


def imagenet_to_clipreid(imgs: torch.Tensor) -> torch.Tensor:
    """
    Convert images from ImageNet normalization to CLIP-ReID normalization.

    This is a convenience function that combines denormalize_imagenet and normalize_clipreid.

    Use this for CLIP-ReID models that were trained with [0.5, 0.5, 0.5] normalization.

    Args:
        imgs: Batch of images with ImageNet normalization (B, C, H, W)

    Returns:
        Images with CLIP-ReID normalization
    """
    imgs_denorm = denormalize_imagenet(imgs)
    return normalize_clipreid(imgs_denorm)


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
