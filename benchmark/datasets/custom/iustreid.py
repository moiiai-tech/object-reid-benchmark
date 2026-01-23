# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""IUST-ReID dataset wrapper."""

from .generic import GenericImageDataset


class IUSTReID(GenericImageDataset):
    """
    IUST-ReID: Iran University of Science and Technology Re-identification Dataset

    Dataset structure:
        IUSTPersonReID/
            bounding_box_train/
            query/
            bounding_box_test/

    Reference:
        IUST person re-identification dataset
    """

    dataset_dir = 'IUSTPersonReID'

    def __init__(self, root='', **kwargs):
        super(IUSTReID, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
