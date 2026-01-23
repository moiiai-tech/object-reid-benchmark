# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""TransReID model wrapper for benchmarking."""

import torch
import torch.nn.functional as F

from ..utils import (
    DEFAULT_DATASET,
    get_dataset_params,
    get_model_logger,
    imagenet_to_transreid,
    setup_transreid,
)
from .base import BaseModelWrapper

logger = get_model_logger("TransReIDWrapper")


class TransReIDWrapper(BaseModelWrapper):
    """Wrapper for TransReID (Transformer-based Re-ID) model variants."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224_TransReID",
        pretrained_path: str | None = None,
        device: str = "cuda",
        dataset_name: str | None = None,
        num_classes: int | None = None,
        camera_num: int | None = None,
        view_num: int | None = None,
        stride_size: tuple = (16, 16),
        input_size: tuple = (256, 256),
        sie_camera: bool = False,
        sie_view: bool = False,
        sie_coe: float = 3.0,
        jpm: bool = False,
        drop_path: float = 0.1,
        drop_out: float = 0.0,
        att_drop_rate: float = 0.0,
        load_classifier: bool = False,
    ):
        """
        Initialize TransReID model.

        Args:
            model_name: Transformer type ('vit_base_patch16_224_TransReID',
                       'vit_small_patch16_224_TransReID', or 'deit_small_patch16_224_TransReID')
            pretrained_path: Path to pretrained weights
            device: Device to load model on
            dataset_name: Dataset name for auto-configuration (e.g., 'market1501', 'dukemtmcreid')
                         If not in DATASET_PARAMS, falls back to MSMT17 parameters
            num_classes: Number of person identities for training (overrides dataset_name)
            camera_num: Number of cameras in dataset (overrides dataset_name)
            view_num: Number of views (overrides dataset_name)
            stride_size: Stride size for vision transformer patches
            input_size: Input image size (height, width)
            sie_camera: Enable Spatial Information Enhancement with camera
            sie_view: Enable Spatial Information Enhancement with view
            sie_coe: SIE coefficient weight
            jpm: Enable Jigsaw Patch Module
            drop_path: Drop path rate for transformer
            drop_out: Dropout rate
            att_drop_rate: Attention dropout rate
            load_classifier: If False, skip loading classifier head (for cross-domain evaluation)
        """
        super().__init__(device)
        self.model_name = f"TransReID-{model_name}"
        self.transformer_type = model_name
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
            self.view_num = (
                view_num if view_num is not None else dataset_params["view_num"]
            )
            logger.info(
                f"Using TransReID for '{dataset_name}': num_classes={self.num_classes}, camera_num={self.camera_num}, view_num={self.view_num}"
            )
        else:
            if num_classes is None:
                logger.warning(
                    f"No dataset_name or num_classes specified, defaulting to {DEFAULT_DATASET.upper()} parameters"
                )
                dataset_params = get_dataset_params(DEFAULT_DATASET, fallback=False)
                self.num_classes = dataset_params["num_classes"]
                self.camera_num = dataset_params["camera_num"]
                self.view_num = dataset_params["view_num"]
            else:
                self.num_classes = num_classes
                self.camera_num = camera_num if camera_num is not None else 6
                self.view_num = view_num if view_num is not None else 1

        self.stride_size = stride_size
        self.input_size = input_size
        self.sie_camera = sie_camera
        self.sie_view = sie_view
        self.sie_coe = sie_coe
        self.jpm = jpm
        self.drop_path = drop_path
        self.drop_out = drop_out
        self.att_drop_rate = att_drop_rate
        self.load_classifier = load_classifier

        self.load_model()

    def load_model(self):
        """Load and initialize TransReID model."""
        logger.info(f"Loading {self.model_name}...")

        try:
            setup_transreid()

            from config import cfg as transreid_cfg  # type: ignore
            from model.backbones.vit_pytorch import (  # type: ignore
                deit_small_patch16_224_TransReID,
                vit_base_patch16_224_TransReID,
                vit_small_patch16_224_TransReID,
            )
            from model.make_model import (  # type: ignore
                build_transformer,
                build_transformer_local,
            )

            __factory_T_type = {
                "vit_base_patch16_224_TransReID": vit_base_patch16_224_TransReID,
                "vit_small_patch16_224_TransReID": vit_small_patch16_224_TransReID,
                "deit_small_patch16_224_TransReID": deit_small_patch16_224_TransReID,
            }

            transreid_cfg.MODEL.TRANSFORMER_TYPE = self.transformer_type
            transreid_cfg.MODEL.STRIDE_SIZE = list(self.stride_size)
            transreid_cfg.INPUT.SIZE_TRAIN = list(self.input_size)
            transreid_cfg.INPUT.SIZE_TEST = list(self.input_size)
            transreid_cfg.MODEL.SIE_CAMERA = self.sie_camera
            transreid_cfg.MODEL.SIE_VIEW = self.sie_view
            transreid_cfg.MODEL.SIE_COE = self.sie_coe
            transreid_cfg.MODEL.JPM = self.jpm
            transreid_cfg.MODEL.DROP_PATH = self.drop_path
            transreid_cfg.MODEL.DROP_OUT = self.drop_out
            transreid_cfg.MODEL.ATT_DROP_RATE = self.att_drop_rate
            transreid_cfg.MODEL.NECK = "bnneck"
            transreid_cfg.MODEL.COS_LAYER = False
            transreid_cfg.MODEL.ID_LOSS_TYPE = "softmax"
            transreid_cfg.TEST.NECK_FEAT = "before"
            transreid_cfg.MODEL.PRETRAIN_CHOICE = "self"

            if self.jpm:
                self.model = build_transformer_local(
                    num_classes=self.num_classes,
                    camera_num=self.camera_num,
                    view_num=self.view_num,
                    cfg=transreid_cfg,
                    factory=__factory_T_type,
                    rearrange=shuffle_unit,
                )
            else:
                self.model = build_transformer(
                    num_classes=self.num_classes,
                    camera_num=self.camera_num,
                    view_num=self.view_num,
                    cfg=transreid_cfg,
                    factory=__factory_T_type,
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
                                "bottleneck.weight",
                                "bottleneck.bias",
                            ]
                        )
                    }
                    logger.info("Cross-domain mode: Skipping classifier head weights")

                # Load with strict=False to handle missing/unexpected keys
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
            logger.error(f"Error loading TransReID model: {e}")
            raise

    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            imgs: Batch of images as tensor (B, C, H, W)
                  Expected to be normalized with mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]

        Returns:
            features: Normalized feature tensor (B, feature_dim)
        """
        with torch.no_grad():
            # Resize images to match the model's expected input size
            current_size = imgs.shape[-2:]  # (H, W)
            if current_size != tuple(self.input_size):
                imgs = F.interpolate(
                    imgs,
                    size=tuple(self.input_size),
                    mode="bilinear",
                    align_corners=False,
                )

            # TransReID expects images normalized with mean=0.5, std=0.5
            # Convert from ImageNet normalization to TransReID normalization
            imgs_transreid = imagenet_to_transreid(imgs)

            # During inference, the model returns global features
            # For SIE-enabled models, we need to provide dummy camera/view labels
            # Create dummy labels (all zeros) for inference when actual labels are not available
            batch_size = imgs_transreid.shape[0]
            if self.sie_camera or self.sie_view:
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
                imgs_transreid, cam_label=cam_label, view_label=view_label
            )

            # Normalize features for similarity computation
            features = F.normalize(features, p=2, dim=1)

        return features

    def get_num_params(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())


def shuffle_unit(features, shift, group, begin=1):
    """
    Shuffle unit for Jigsaw Patch Module.

    Args:
        features: Input features
        shift: Shift amount
        group: Group size
        begin: Begin index

    Returns:
        Shuffled features
    """
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat(
        [features[:, begin - 1 + shift :], features[:, begin : begin - 1 + shift]],
        dim=1,
    )
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except Exception:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x
