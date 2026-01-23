# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Celeb-reID dataset wrapper."""

from .generic import GenericImageDataset


class CelebReID(GenericImageDataset):
    """
    Celeb-reID: Celebrity Re-identification Dataset

    Dataset structure:
        Celeb-reID/
            train/
            query/
            gallery/

    Reference:
        Huang et al. Celebrity Recognition in the Wild: A Large Scale Dataset and Benchmark
    """

    dataset_dir = 'Celeb-reID'

    def __init__(self, root='', **kwargs):
        super(CelebReID, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
