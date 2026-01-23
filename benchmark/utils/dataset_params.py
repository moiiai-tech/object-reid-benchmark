# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Dataset-specific parameters for person re-identification models.

This module centralizes dataset configuration parameters used across different
model wrappers (CLIP-ReID, TransReID, etc.) to avoid code duplication.
"""

DATASET_PARAMS = {
    "market1501": {
        "num_classes": 751,
        "camera_num": 6,
        "view_num": 1,
    },
    "dukemtmcreid": {
        "num_classes": 702,
        "camera_num": 8,
        "view_num": 1,
    },
    "msmt17": {
        "num_classes": 1041,
        "camera_num": 15,
        "view_num": 1,
    },
    "occ_duke": {
        "num_classes": 702,
        "camera_num": 8,
        "view_num": 1,
    },
    "veri": {
        "num_classes": 576,
        "camera_num": 20,
        "view_num": 1,
    },
    "vehicleid": {
        "num_classes": 13164,
        "camera_num": 2,
        "view_num": 1,
    },
}

DEFAULT_DATASET = "msmt17"


def get_dataset_params(dataset_name: str, fallback: bool = True) -> dict:
    """
    Get dataset-specific parameters for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'market1501', 'dukemtmcreid')
        fallback: Whether to fallback to default dataset if not found

    Returns:
        Dictionary containing num_classes, camera_num, and view_num

    Raises:
        KeyError: If dataset not found and fallback=False
    """
    if dataset_name in DATASET_PARAMS:
        return DATASET_PARAMS[dataset_name].copy()

    if not fallback:
        raise KeyError(f"Dataset '{dataset_name}' not found in DATASET_PARAMS")

    return DATASET_PARAMS[DEFAULT_DATASET].copy()
