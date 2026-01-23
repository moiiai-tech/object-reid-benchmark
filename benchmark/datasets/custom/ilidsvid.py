# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""iLIDS-VID dataset wrapper."""

from .generic import GenericImageDataset


class ILIDSVID(GenericImageDataset):
    """
    iLIDS-VID: i-LIDS Video Re-identification Dataset

    Dataset structure:
        iLIDS-VID/
            sequences/
                cam1/
                cam2/

    Reference:
        Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    """

    dataset_dir = 'iLIDS-VID'

    def __init__(self, root='', **kwargs):
        super(ILIDSVID, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
