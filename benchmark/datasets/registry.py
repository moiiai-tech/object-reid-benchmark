# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Dataset registry and loader."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torchreid

from benchmark.exceptions import (
    DatasetLoadError,
    DatasetNotFoundError,
    ImageLoadError,
    ValidationError,
)
from benchmark.utils.validation import validate_dataset_name, validate_directory_exists
from benchmark.utils.logging import BenchmarkLogger

# Import custom dataset classes
try:
    from benchmark.datasets.custom import (
        LASTID,
        MSMT17,
        PKU,
        CelebReID,
        G2AVReID,
        ILIDSVIDCustom,
        IUSTReID,
    )
except ImportError:
    # Fallback if running from different directory
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from benchmark.datasets.custom import (
        LASTID,
        MSMT17,
        PKU,
        CelebReID,
        G2AVReID,
        ILIDSVIDCustom,
        IUSTReID,
    )


@dataclass
class DatasetConfig:
    """Configuration for a dataset. Add new datasets by creating instances of this class."""

    name: str
    source: str
    target: str
    height: int = 256
    width: int = 128
    batch_size: int = 100
    split_id: int = 0
    use_cuhk03_metric: bool = False
    cuhk03_labeled: bool = True
    cuhk03_classic_split: bool = False
    is_custom: bool = False  # True if using custom dataset wrapper
    custom_class: Optional[type] = None  # Custom dataset class if applicable

    def to_datamanager_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for torchreid.data.ImageDataManager."""
        return {
            "sources": self.source,
            "targets": self.target,
            "height": self.height,
            "width": self.width,
            "batch_size_test": self.batch_size,
            "split_id": self.split_id,
            "cuhk03_labeled": self.cuhk03_labeled,
            "cuhk03_classic_split": self.cuhk03_classic_split,
        }


# Dataset registry - add new datasets here
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "market1501": DatasetConfig(
        name="market1501",
        source="market1501",
        target="market1501",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
    ),
    "dukemtmcreid": DatasetConfig(
        name="dukemtmcreid",
        source="dukemtmcreid",
        target="dukemtmcreid",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
    ),
    "cuhk03": DatasetConfig(
        name="cuhk03",
        source="cuhk03",
        target="cuhk03",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        use_cuhk03_metric=True,
        cuhk03_labeled=True,
        cuhk03_classic_split=False,
    ),
    "msmt17": DatasetConfig(
        name="msmt17",
        source="msmt17",
        target="msmt17",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=MSMT17,
    ),
    "grid": DatasetConfig(
        name="grid",
        source="grid",
        target="grid",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
    ),
    # Custom datasets (using custom wrapper classes)
    "celebreid": DatasetConfig(
        name="celebreid",
        source="celebreid",
        target="celebreid",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=CelebReID,
    ),
    "pku": DatasetConfig(
        name="pku",
        source="pku",
        target="pku",
        height=48,  # PKUv1a is 128x48
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=PKU,
    ),
    "lastid": DatasetConfig(
        name="lastid",
        source="lastid",
        target="lastid",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=LASTID,
    ),
    "ilids": DatasetConfig(
        name="ilids",
        source="ilids",
        target="ilids",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
    ),
    "ilidsvid": DatasetConfig(
        name="ilidsvid",
        source="ilidsvid",
        target="ilidsvid",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=ILIDSVIDCustom,
    ),
    "g2a": DatasetConfig(
        name="g2a",
        source="g2a",
        target="g2a",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=G2AVReID,
    ),
    "iustreid": DatasetConfig(
        name="iustreid",
        source="iustreid",
        target="iustreid",
        height=256,
        width=128,
        batch_size=100,
        split_id=0,
        is_custom=True,
        custom_class=IUSTReID,
    ),
}


def load_dataset(
    dataset_name: str, root: str = "reid-data"
) -> Tuple[Dict, int, DatasetConfig]:
    """
    Load dataset with appropriate configuration.

    Args:
        dataset_name: Name of dataset (must be in DATASET_REGISTRY)
        root: Root directory for reid datasets

    Returns:
        test_loader: Dictionary with 'query' and 'gallery' DataLoaders
        num_classes: Number of training identities
        dataset_config: Configuration used for this dataset

    Raises:
        DatasetNotFoundError: If dataset name is not in registry
        DatasetLoadError: If dataset loading fails
        ValidationError: If parameters are invalid
    """
    logger = BenchmarkLogger.get_logger(__name__)

    # Validate inputs
    try:
        validate_dataset_name(dataset_name)
    except ValidationError as e:
        raise DatasetNotFoundError(dataset_name, list(DATASET_REGISTRY.keys())) from e

    if dataset_name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise DatasetNotFoundError(dataset_name, available)

    dataset_config = DATASET_REGISTRY[dataset_name]

    # Validate root directory
    try:
        root_path = validate_directory_exists(root, create=False)
    except ValidationError as e:
        raise DatasetLoadError(
            dataset_name, f"Root directory validation failed: {e}"
        ) from e

    logger.info(f"Initializing {dataset_config.name} data manager from {root_path}")
    BenchmarkLogger.log_dataset_loading(
        logger, dataset_name, False, f"Starting dataset loading from {root_path}"
    )

    # Handle custom datasets differently
    if dataset_config.is_custom and dataset_config.custom_class:
        from torch.utils.data import DataLoader
        from torchvision import transforms as T

        logger.info(f"Loading custom dataset: {dataset_config.name}")

        try:
            # Create dataset instance
            dataset = dataset_config.custom_class(root=root)
        except Exception as e:
            BenchmarkLogger.log_dataset_loading(
                logger, dataset_name, False, f"Custom dataset instantiation failed: {e}"
            )
            raise DatasetLoadError(
                dataset_name, f"Failed to instantiate custom dataset class: {e}"
            ) from e

        # Create transforms
        transform_test = T.Compose(
            [
                T.Resize((dataset_config.height, dataset_config.width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Create test loaders
        from torchreid.reid.data.datasets.dataset import Dataset

        # Manually create query and gallery datasets
        class SimpleDataset(Dataset):
            def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                from PIL import Image

                # Handle both 3-element and 4-element tuples
                data_item = self.data[idx]
                if len(data_item) == 4:
                    img_path, pid, camid, _ = data_item
                else:
                    img_path, pid, camid = data_item

                # Handle corrupt images gracefully
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    # Return a dummy black image as fallback
                    import torch

                    img = torch.zeros(3, dataset_config.height, dataset_config.width)
                    # Convert back to PIL for consistency with transform pipeline
                    from torchvision import transforms

                    to_pil = transforms.ToPILImage()
                    img = to_pil(img)

                if self.transform:
                    img = self.transform(img)
                return img, pid, camid, img_path

        try:
            query_dataset = SimpleDataset(dataset.query, transform_test)
            gallery_dataset = SimpleDataset(dataset.gallery, transform_test)

            query_loader = DataLoader(
                query_dataset,
                batch_size=dataset_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            gallery_loader = DataLoader(
                gallery_dataset,
                batch_size=dataset_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            test_loader = {"query": query_loader, "gallery": gallery_loader}

            # Handle both 3-element (img_path, pid, camid) and 4-element (img_path, pid, camid, dsetid) tuples
            if dataset.train and len(dataset.train[0]) == 4:
                num_classes = len(set([pid for _, pid, _, _ in dataset.train]))
            else:
                num_classes = len(set([pid for _, pid, _ in dataset.train]))

            if num_classes == 0:
                raise DatasetLoadError(
                    dataset_name,
                    "No training identities found - dataset may be empty or corrupted",
                )

        except Exception as e:
            BenchmarkLogger.log_dataset_loading(
                logger,
                dataset_name,
                False,
                f"Custom dataset loader creation failed: {e}",
            )
            raise DatasetLoadError(
                dataset_name, f"Failed to create data loaders for custom dataset: {e}"
            ) from e

    else:
        # Use standard torchreid data manager
        try:
            kwargs = dataset_config.to_datamanager_kwargs()
            datamanager = torchreid.data.ImageDataManager(
                root=root,
                batch_size_train=32,
                **kwargs,
            )

            test_loader = datamanager.test_loader[dataset_config.source]
            num_classes = datamanager.num_train_pids

            if num_classes == 0:
                raise DatasetLoadError(
                    dataset_name,
                    "No training identities found - dataset may be empty or not properly downloaded",
                )

        except Exception as e:
            BenchmarkLogger.log_dataset_loading(
                logger,
                dataset_name,
                False,
                f"Standard dataset loader creation failed: {e}",
            )
            raise DatasetLoadError(
                dataset_name, f"Failed to create torchreid data manager: {e}"
            ) from e

    # Validate loaded data
    try:
        query_count = len(test_loader["query"].dataset)
        gallery_count = len(test_loader["gallery"].dataset)

        if query_count == 0 or gallery_count == 0:
            raise DatasetLoadError(
                dataset_name,
                f"Empty dataset splits - Query: {query_count}, Gallery: {gallery_count}",
            )

    except Exception as e:
        BenchmarkLogger.log_dataset_loading(
            logger, dataset_name, False, f"Dataset validation failed: {e}"
        )
        raise DatasetLoadError(dataset_name, f"Dataset validation failed: {e}") from e

    logger.info(f"Dataset loaded successfully: {num_classes} training identities")
    logger.info(f"Query: {query_count}, Gallery: {gallery_count}")

    BenchmarkLogger.log_dataset_loading(
        logger,
        dataset_name,
        True,
        f"Successfully loaded - Classes: {num_classes}, Query: {query_count}, Gallery: {gallery_count}",
    )

    return test_loader, num_classes, dataset_config
