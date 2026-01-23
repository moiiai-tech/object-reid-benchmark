# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Occluded-PoseTrack dataset wrapper."""

from .generic import GenericImageDataset


class OccludedPoseTrack(GenericImageDataset):
    """
    Occluded-PoseTrack: Person Re-identification with Occlusions

    Dataset structure:
        occluded_posetrack_reid/
            train/
            query/
            gallery/

    Reference:
        Occluded person re-identification from PoseTrack dataset
    """

    dataset_dir = 'occluded_posetrack_reid'

    def __init__(self, root='', **kwargs):
        super(OccludedPoseTrack, self).__init__(root=root, dataset_name=self.dataset_dir, **kwargs)
