# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""PeP dataset wrapper."""

from .generic import GenericImageDataset


class PeP(GenericImageDataset):
    """
    PeP: Person in Place Dataset

    Dataset structure:
        pep_256x128/
            train/
            query/
            gallery/

    Reference:
        PeP person re-identification dataset
    """

    dataset_dir = 'pep_256x128'

    def __init__(self, root='', **kwargs):
        super(PeP, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
