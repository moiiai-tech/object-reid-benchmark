# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""CLIP-ReID model wrapper for benchmarking."""

import torch
import torch.nn.functional as F

from ..utils import (
    DEFAULT_DATASET,
    get_dataset_params,
    get_model_logger,
    imagenet_to_clipreid,
    setup_clipreid,
)
from .base import BaseModelWrapper

logger = get_model_logger("CLIPReIDWrapper")


class CLIPReIDWrapper(BaseModelWrapper):
    """Wrapper for CLIP-ReID model variants."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained_path: str | None = None,
        device: str = "cuda",
        dataset_name: str | None = None,
        num_classes: int | None = None,
        camera_num: int | None = None,
        view_num: int = 1,
        stride_size: tuple = (16, 16),
        input_size: tuple = (256, 128),
        sie_camera: bool = False,
        sie_coe: float = 1.0,
        load_classifier: bool = False,
    ):
        """Initialize CLIP-ReID model."""
        super().__init__(device)
        self.model_name = f"CLIP-ReID-{model_name}"
        self.arch_name = model_name
        self.pretrained_path = pretrained_path

        if dataset_name:
            try:
                dataset_params = get_dataset_params(dataset_name, fallback=True)
            except KeyError:
                logger.warning(f"No dataset-specific parameters for '{dataset_name}'")
                logger.warning(
                    f"Falling back to {DEFAULT_DATASET.upper()} model parameters"
                )
                dataset_params = get_dataset_params(DEFAULT_DATASET, fallback=False)

            # Use dataset params as defaults, allow manual override
            self.num_classes = (
                num_classes
                if num_classes is not None
                else dataset_params["num_classes"]
            )
            self.camera_num = (
                camera_num if camera_num is not None else dataset_params["camera_num"]
            )
            logger.info(
                f"Using CLIP-ReID for '{dataset_name}': num_classes={self.num_classes}, camera_num={self.camera_num}"
            )
        else:
            if num_classes is None:
                logger.warning(
                    f"No dataset_name or num_classes specified, defaulting to {DEFAULT_DATASET.upper()} parameters"
                )
                dataset_params = get_dataset_params(DEFAULT_DATASET, fallback=False)
                self.num_classes = dataset_params["num_classes"]
                self.camera_num = dataset_params["camera_num"]
            else:
                self.num_classes = num_classes
                self.camera_num = camera_num if camera_num is not None else 6

        self.view_num = view_num
        self.stride_size = stride_size
        self.input_size = input_size
        self.sie_camera = sie_camera
        self.sie_coe = sie_coe
        self.load_classifier = load_classifier

        self.load_model()

    def load_model(self):
        """Load and initialize CLIP-ReID model."""
        logger.info(f"Loading {self.model_name}...")

        try:
            setup_clipreid()

            from config import cfg as clipreid_cfg  # type: ignore
            from model.make_model import build_transformer  # type: ignore

            # Normalize model name for CLIP-ReID (e.g., "ViT-B-16-dense" -> "ViT-B-16")
            # The "-dense" suffix is just a config variant (different stride), not a different model
            normalized_arch_name = self.arch_name.replace("-dense", "")
            clipreid_cfg.MODEL.NAME = normalized_arch_name
            clipreid_cfg.MODEL.STRIDE_SIZE = list(self.stride_size)
            clipreid_cfg.INPUT.SIZE_TRAIN = list(self.input_size)
            clipreid_cfg.INPUT.SIZE_TEST = list(self.input_size)
            clipreid_cfg.MODEL.SIE_CAMERA = self.sie_camera
            clipreid_cfg.MODEL.SIE_VIEW = False  # Not commonly used
            clipreid_cfg.MODEL.SIE_COE = self.sie_coe
            clipreid_cfg.MODEL.NECK = "bnneck"
            clipreid_cfg.MODEL.COS_LAYER = False
            clipreid_cfg.TEST.NECK_FEAT = "before"

            self.model = build_transformer(
                num_classes=self.num_classes,
                camera_num=self.camera_num,
                view_num=self.view_num,
                cfg=clipreid_cfg,
            )

            if self.pretrained_path:
                logger.info(f"Loading pretrained weights from {self.pretrained_path}")
                param_dict = torch.load(self.pretrained_path, map_location="cpu")

                param_dict = {
                    k.replace("module.", ""): v for k, v in param_dict.items()
                }

                # Filter out classifier head if load_classifier=False (cross-domain evaluation)
                if not self.load_classifier:
                    param_dict = {
                        k: v
                        for k, v in param_dict.items()
                        if not any(
                            x in k
                            for x in [
                                "classifier",
                                "classifier_proj",
                                "bottleneck.weight",
                                "bottleneck.bias",
                                "image_encoder.positional_embedding",  # Position embeddings depend on stride
                            ]
                        )
                    }
                    logger.info("Cross-domain mode: Skipping classifier head and position embedding weights")

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    param_dict, strict=False
                )

                if missing_keys:
                    logger.warning(
                        f"Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
                    )
                if unexpected_keys:
                    logger.warning(
                        f"Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
                    )

                logger.info(
                    f"Loaded pretrained weights (matched {len(param_dict) - len(unexpected_keys)}/{len(param_dict)} keys)"
                )

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading CLIP-ReID model: {e}")
            raise

    def extract_features(
        self, imgs: torch.Tensor, cam_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
                  Expected to be normalized with mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
            cam_labels: Optional camera labels (B,) for SIE (Spatial Instance Encoding).
                       If provided and sie_camera=True, uses actual camera IDs.
                       If not provided, falls back to zeros (disables SIE benefit).

        Returns:
            features: Normalized feature tensor (B, feature_dim)
        """
        with torch.no_grad():
            current_size = imgs.shape[-2:]
            if current_size != tuple(self.input_size):
                imgs = F.interpolate(
                    imgs,
                    size=tuple(self.input_size),
                    mode="bilinear",
                    align_corners=False,
                )

            # CLIP-ReID uses [0.5, 0.5, 0.5] normalization, NOT standard CLIP normalization
            imgs_clipreid = imagenet_to_clipreid(imgs)

            batch_size = imgs_clipreid.shape[0]
            if self.sie_camera:
                if cam_labels is not None:
                    # Use actual camera labels for SIE (critical for performance!)
                    cam_label = cam_labels.to(imgs.device)
                else:
                    # Fallback to zeros if no camera labels provided
                    # WARNING: This significantly reduces performance (~8-10% mAP loss)
                    logger.warning(
                        "SIE camera is enabled but no camera labels provided. "
                        "Performance will be degraded. Pass cam_labels to extract_features()."
                    )
                    cam_label = torch.zeros(
                        batch_size, dtype=torch.long, device=imgs.device
                    )
                view_label = torch.zeros(
                    batch_size, dtype=torch.long, device=imgs.device
                )
            else:
                cam_label = None
                view_label = None

            features = self.model(
                imgs_clipreid, cam_label=cam_label, view_label=view_label
            )

            features = F.normalize(features, p=2, dim=1)

        return features

    def get_num_params(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
