# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""PKU dataset wrapper."""

from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

try:
    from torchreid.reid.data.datasets.dataset import ImageDataset
except ImportError:
    from torchreid.data.datasets.dataset import ImageDataset


class PKU(ImageDataset):
    """
    PKU-Reid Dataset (PKUv1a_128x48)

    Dataset structure:
        PKUv1a_128x48/
            {pid}_{camid}_{imgid}.png

    All images are in a single directory with naming convention: pid_camid_imgid.png
    """

    dataset_dir = 'PKUv1a_128x48'

    def __init__(self, root='', train_split=0.5, **kwargs):
        """
        Args:
            root: Root directory
            train_split: Proportion of identities to use for training (default: 0.5)
        """
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_split = train_split

        # Process all images
        all_data = self.process_dir(self.dataset_dir)

        # Split into train/query/gallery by person ID
        all_pids = sorted(set([item[1] for item in all_data]))
        num_train_pids = int(len(all_pids) * self.train_split)

        train_pids = set(all_pids[:num_train_pids])
        test_pids = set(all_pids[num_train_pids:])

        # Split data
        train = [item for item in all_data if item[1] in train_pids]
        test_data = [item for item in all_data if item[1] in test_pids]

        # Relabel train data
        train_pid2label = {pid: label for label, pid in enumerate(sorted(train_pids))}
        train = [(path, train_pid2label[pid], camid) for path, pid, camid in train]

        # Split test into query and gallery (half each)
        query = []
        gallery = []
        for pid in test_pids:
            pid_data = [item for item in test_data if item[1] == pid]
            mid = len(pid_data) // 2
            query.extend(pid_data[:mid])
            gallery.extend(pid_data[mid:])

        super(PKU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        """Process directory with images named as {pid}_{camid}_{imgid}.png"""
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        # Pattern: pid_camid_imgid.png
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)')

        data = []
        for img_path in img_paths:
            fname = osp.basename(img_path)
            match = pattern.search(fname)

            if match:
                pid, camid, _ = map(int, match.groups())
                data.append((img_path, pid, camid - 1))  # 0-indexed camid

        return data
