# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""G2A-VReID.v2 dataset wrapper."""

from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

try:
    from torchreid.reid.data.datasets.dataset import ImageDataset
except ImportError:
    from torchreid.data.datasets.dataset import ImageDataset


class G2AVReID(ImageDataset):
    """
    G2A-VReID.v2: Video-based Person Re-identification Dataset

    Dataset structure:
        G2A-VReID.v2/
            bbox_train/
                {pid}/
                    {pid}C{camid}T{trackid}F{frameid}.jpg
            bbox_test/
                {pid}/
                    {pid}C{camid}T{trackid}F{frameid}.jpg

    Reference:
        G2A person re-identification dataset
    """

    dataset_dir = "G2A-VReID.v2"

    def __init__(self, root="", train_split=0.5, **kwargs):
        """
        Args:
            root: Root directory
            train_split: Proportion of identities to use for training (default: 0.5)
        """
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_split = train_split

        self.train_dir = osp.join(self.dataset_dir, "bbox_train")
        self.test_dir = osp.join(self.dataset_dir, "bbox_test")

        # Process all data
        train_data = self.process_dir(self.train_dir)
        test_data = self.process_dir(self.test_dir)

        # Get all unique person IDs
        all_train_pids = sorted(set([item[1] for item in train_data]))

        # Split train data
        num_train_pids = int(len(all_train_pids) * self.train_split)
        train_pids = set(all_train_pids[:num_train_pids])
        val_pids = set(all_train_pids[num_train_pids:])

        # Create train/query/gallery splits
        train = [item for item in train_data if item[1] in train_pids]

        # Use validation split from train_data for query
        val_data = [item for item in train_data if item[1] in val_pids]

        # Use test_data for gallery
        query = val_data
        gallery = test_data

        # Relabel train data
        train_pid2label = {pid: label for label, pid in enumerate(sorted(train_pids))}
        train = [(path, train_pid2label[pid], camid) for path, pid, camid in train]

        super(G2AVReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        """Process directory with subdirectories for each person"""
        if not osp.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist")
            return []

        data = []
        # Pattern: {pid}C{camid}T{trackid}F{frameid}.jpg
        pattern = re.compile(r"(\d+)C(\d+)T")

        # Each subdirectory is a person ID
        person_dirs = glob.glob(osp.join(dir_path, "*"))

        for person_dir in person_dirs:
            if not osp.isdir(person_dir):
                continue

            pid = int(osp.basename(person_dir))

            # Get all images for this person
            img_paths = glob.glob(osp.join(person_dir, "*.jpg"))

            for img_path in img_paths:
                fname = osp.basename(img_path)
                match = pattern.search(fname)

                if match:
                    camid = int(match.group(2)) - 1  # 0-indexed
                    data.append((img_path, pid, camid))

        return data
