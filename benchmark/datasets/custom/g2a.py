# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""G2A dataset wrapper."""

from .generic import GenericImageDataset


class G2A(GenericImageDataset):
    """
    G2A: Gallery-to-Gallery Re-identification Dataset

    Dataset structure:
        G2A-VReID.v2/
            train/
            query/
            gallery/

    Reference:
        G2A person re-identification dataset
    """

    dataset_dir = 'G2A-VReID.v2'

    def __init__(self, root='', **kwargs):
        super(G2A, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
