# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Model factory for creating model instances from config."""

import logging
from omegaconf import DictConfig

from benchmark.exceptions import ModelLoadError, ValidationError, ModelNotFoundError
from benchmark.utils.validation import (
    validate_device,
    validate_model_config,
    validate_positive_int,
    validate_model_name,
    validate_dataset_name,
    validate_model_parameters,
)
from benchmark.utils.logging import BenchmarkLogger
from benchmark.utils.weight_resolver import resolve_model_weights

from .clip_wrapper import CLIPWrapper
from .clipreid_wrapper import CLIPReIDWrapper
from .dinov2_wrapper import DINOv2Wrapper
from .dinov3_wrapper import DINOv3Wrapper
from .osnet import OSNetWrapper
from .pecore import PECoreWrapper
from .siglip2_wrapper import SigLIP2Wrapper
from .transreid_wrapper import TransReIDWrapper

logger = BenchmarkLogger.get_logger(__name__)


def create_model(
    model_cfg: DictConfig,
    num_classes: int,
    device: str,
    dataset_name: str,
    cross_domain: bool = False,
    target_dataset: str | None = None,
):
    """
    Create a model instance from configuration.

    Args:
        model_cfg: OmegaConf configuration for the model
        num_classes: Number of training classes (for classification models)
        device: Device to load model on
        dataset_name: Name of the dataset (for dataset-specific model configuration)
        cross_domain: If True, load weights from dataset_name but evaluate on target_dataset
        target_dataset: Target dataset for cross-domain evaluation

    Returns:
        BaseModelWrapper: Initialized model wrapper

    Raises:
        ValidationError: If configuration is invalid
        ModelLoadError: If model creation fails

    Example config:
        type: "osnet"
        name: "osnet_x1_0"
        pretrained_path: null
    """
    try:
        validate_model_config(model_cfg, required_fields=["type", "name"])
    except ValidationError as e:
        raise ValidationError(
            f"Invalid model configuration:\n{e}\n\n"
            f"Required fields:\n"
            f"  - type: Model type (e.g., 'osnet', 'clip', 'clipreid')\n"
            f"  - name: Model name/architecture\n\n"
            f"Example:\n"
            f"  type: osnet\n"
            f"  name: osnet_x1_0"
        ) from e

    model_type = model_cfg.type
    model_name = model_cfg.name

    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise ValidationError(f"Invalid model name: {e}") from e

    try:
        validate_dataset_name(dataset_name)
    except ValidationError as e:
        raise ValidationError(f"Invalid dataset name: {e}") from e

    if target_dataset:
        try:
            validate_dataset_name(target_dataset)
        except ValidationError as e:
            raise ValidationError(f"Invalid target dataset name: {e}") from e

    try:
        validate_positive_int(num_classes, "num_classes")
    except ValidationError as e:
        raise ValidationError(
            f"Invalid num_classes parameter:\n{e}\n"
            f"This typically indicates a dataset loading issue."
        ) from e

    try:
        device = validate_device(device)
    except ValidationError as e:
        raise ValidationError(
            f"Invalid device parameter:\n{e}\n\n"
            f"Available options:\n"
            f"  - 'cpu' for CPU-only inference\n"
            f"  - 'cuda:0', 'cuda:1', etc. for specific GPU\n\n"
            f"Check GPU availability with: reid info"
        ) from e

    try:
        validate_model_parameters(dict(model_cfg))
    except ValidationError as e:
        raise ValidationError(f"Invalid model parameters: {e}") from e

    SUPPORTED_MODEL_TYPES = [
        "osnet",
        "clip",
        "pecore",
        "clipreid",
        "transreid",
        "dinov2",
        "dinov3",
        "siglip2",
    ]

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ModelNotFoundError(model_type, SUPPORTED_MODEL_TYPES)

    BenchmarkLogger.log_model_creation(
        logger,
        f"{model_type}/{model_name}",
        False,
        f"Attempting creation - Dataset: {dataset_name}, Device: {device}",
    )

    if cross_domain and target_dataset:
        logger.info(
            f"Cross-domain mode: Loading {model_type} weights from {dataset_name}, "
            f"evaluating on {target_dataset}"
        )

    if model_type == "osnet":
        try:
            model = OSNetWrapper(
                model_name=model_name,
                num_classes=num_classes,
                pretrained_path=model_cfg.get("pretrained_path", None),
                device=device,
            )
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                True,
                f"Successfully created OSNet model",
            )
            return model
        except Exception as e:
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                False,
                f"OSNet creation failed: {e}",
            )
            raise ModelLoadError(
                f"Failed to create OSNet model '{model_name}': {e}"
            ) from e
    elif model_type == "clip":
        try:
            model = CLIPWrapper(model_name=model_name, device=device)
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                True,
                f"Successfully created CLIP model",
            )
            return model
        except Exception as e:
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                False,
                f"CLIP creation failed: {e}",
            )
            raise ModelLoadError(
                f"Failed to create CLIP model '{model_name}': {e}"
            ) from e
    elif model_type == "pecore":
        try:
            model = PECoreWrapper(model_config=model_name, device=device)
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                True,
                f"Successfully created PECore model",
            )
            return model
        except Exception as e:
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                False,
                f"PECore creation failed: {e}",
            )
            raise ModelLoadError(
                f"Failed to create PECore model '{model_name}': {e}"
            ) from e
    elif model_type == "clipreid":
        try:
            pretrained_path = resolve_model_weights("clipreid", dataset_name, model_cfg)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to resolve CLIP-ReID weights for dataset '{dataset_name}':\n{e}\n\n"
                f"Possible fixes:\n"
                f"  1. Check if dataset name is correct: {dataset_name}\n"
                f"  2. Ensure pretrained weights exist or can be downloaded\n"
                f"  3. Check weight_resolver.py configuration\n"
                f"  4. Set pretrained_path explicitly in config if using custom weights"
            ) from e

        if pretrained_path is None:
            raise ModelLoadError(
                f"Failed to resolve pretrained weights for CLIP-ReID model '{model_name}' "
                f"on dataset '{dataset_name}'.\n\n"
                f"The weight resolver returned None. This usually means:\n"
                f"  - Weights for this dataset are not available\n"
                f"  - Download failed\n"
                f"  - Network connectivity issues\n\n"
                f"Possible fixes:\n"
                f"  1. Check available datasets: reid dataset list\n"
                f"  2. Download dataset first: reid dataset download {dataset_name}\n"
                f"  3. Check weight_resolver.py for supported datasets\n"
                f"  4. Provide explicit pretrained_path in config"
            )

        try:
            return CLIPReIDWrapper(
                model_name=model_name,
                pretrained_path=pretrained_path,
                dataset_name=dataset_name,
                num_classes=model_cfg.get("num_classes", None),
                camera_num=model_cfg.get("camera_num", None),
                device=device,
                view_num=model_cfg.get("view_num", 1),
                stride_size=tuple(model_cfg.get("stride_size", [16, 16])),
                input_size=tuple(model_cfg.get("input_size", [256, 128])),
                sie_camera=model_cfg.get("sie_camera", False),
                sie_coe=model_cfg.get("sie_coe", 1.0),
                load_classifier=not cross_domain,  # Don't load classifier for cross-domain
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize CLIP-ReID model '{model_name}':\n{e}\n\n"
                f"Model configuration:\n"
                f"  Dataset: {dataset_name}\n"
                f"  Weights: {pretrained_path}\n"
                f"  Device: {device}"
            ) from e
    elif model_type == "transreid":
        try:
            pretrained_path = resolve_model_weights(
                "transreid", dataset_name, model_cfg
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to resolve TransReID weights for dataset '{dataset_name}':\n{e}\n\n"
                f"Possible fixes:\n"
                f"  1. Check if dataset name is correct: {dataset_name}\n"
                f"  2. Ensure pretrained weights exist or can be downloaded\n"
                f"  3. Check weight_resolver.py configuration\n"
                f"  4. Set pretrained_path explicitly in config if using custom weights"
            ) from e

        if pretrained_path is None:
            raise ModelLoadError(
                f"Failed to resolve pretrained weights for TransReID model '{model_name}' "
                f"on dataset '{dataset_name}'.\n\n"
                f"The weight resolver returned None. This usually means:\n"
                f"  - Weights for this dataset are not available\n"
                f"  - Download failed\n"
                f"  - Network connectivity issues\n\n"
                f"Possible fixes:\n"
                f"  1. Check available datasets: reid dataset list\n"
                f"  2. Download dataset first: reid dataset download {dataset_name}\n"
                f"  3. Check weight_resolver.py for supported datasets\n"
                f"  4. Provide explicit pretrained_path in config"
            )

        try:
            return TransReIDWrapper(
                model_name=model_name,
                pretrained_path=pretrained_path,
                dataset_name=dataset_name,
                num_classes=model_cfg.get("num_classes", None),
                camera_num=model_cfg.get("camera_num", None),
                view_num=model_cfg.get("view_num", None),
                device=device,
                stride_size=tuple(model_cfg.get("stride_size", [16, 16])),
                input_size=tuple(model_cfg.get("input_size", [256, 256])),
                sie_camera=model_cfg.get("sie_camera", False),
                sie_view=model_cfg.get("sie_view", False),
                sie_coe=model_cfg.get("sie_coe", 3.0),
                jpm=model_cfg.get("jpm", False),
                drop_path=model_cfg.get("drop_path", 0.1),
                drop_out=model_cfg.get("drop_out", 0.0),
                att_drop_rate=model_cfg.get("att_drop_rate", 0.0),
                load_classifier=not cross_domain,  # Don't load classifier for cross-domain
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize TransReID model '{model_name}':\n{e}\n\n"
                f"Model configuration:\n"
                f"  Dataset: {dataset_name}\n"
                f"  Weights: {pretrained_path}\n"
                f"  Device: {device}"
            ) from e
    elif model_type == "dinov2":
        try:
            model = DINOv2Wrapper(
                model_name=model_name,
                pretrained_path=model_cfg.get("pretrained_path", None),
                device=device,
                input_size=tuple(model_cfg.get("input_size", [256, 256])),
            )
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                True,
                f"Successfully created DINOv2 model",
            )
            return model
        except Exception as e:
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                False,
                f"DINOv2 creation failed: {e}",
            )
            raise ModelLoadError(
                f"Failed to create DINOv2 model '{model_name}': {e}"
            ) from e
    elif model_type == "dinov3":
        try:
            model = DINOv3Wrapper(
                model_name=model_name,
                pretrained_path=model_cfg.get("pretrained_path", None),
                device=device,
                input_size=tuple(model_cfg.get("input_size", [256, 256])),
            )
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                True,
                f"Successfully created DINOv3 model",
            )
            return model
        except Exception as e:
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                False,
                f"DINOv3 creation failed: {e}",
            )
            raise ModelLoadError(
                f"Failed to create DINOv3 model '{model_name}': {e}"
            ) from e
    elif model_type == "siglip2":
        try:
            model = SigLIP2Wrapper(
                model_name=model_name,
                pretrained_path=model_cfg.get("pretrained_path", None),
                num_classes=num_classes,
                device=device,
                input_size=tuple(model_cfg.get("input_size", [256, 256])),
            )
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                True,
                f"Successfully created SigLIP2 model",
            )
            return model
        except Exception as e:
            BenchmarkLogger.log_model_creation(
                logger,
                f"{model_type}/{model_name}",
                False,
                f"SigLIP2 creation failed: {e}",
            )
            raise ModelLoadError(
                f"Failed to create SigLIP2 model '{model_name}': {e}"
            ) from e
    else:
        # This should never happen due to validation above, but keeping as failsafe
        error_msg = (
            f"Unhandled model type: '{model_type}'\n\n"
            f"This is a bug - the model type passed validation but no handler exists.\n"
            f"Please report this issue with your configuration."
        )
        BenchmarkLogger.log_model_creation(
            logger, f"{model_type}/{model_name}", False, f"Bug: Unhandled model type"
        )
        raise ValidationError(error_msg)
