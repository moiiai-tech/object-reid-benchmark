# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""EntireID dataset wrapper."""

from .generic import GenericImageDataset


class EntireID(GenericImageDataset):
    """
    EntireID: Person Re-identification Dataset

    Dataset structure:
        entireid_blured/
            bounding_box_train/
            query/
            bounding_box_test/
    """

    dataset_dir = 'entireid_blured'

    def __init__(self, root='', **kwargs):
        super(EntireID, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
