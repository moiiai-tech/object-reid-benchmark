# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""LAST-ID dataset wrapper."""

from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

try:
    from torchreid.reid.data.datasets.dataset import ImageDataset
except ImportError:
    from torchreid.data.datasets.dataset import ImageDataset


class LASTID(ImageDataset):
    """
    LAST-ID: Large-Scale Spatio-Temporal Person Re-identification Dataset

    Dataset structure:
        last/
            train/
            val/
            test/

    Images follow naming convention: {pid}_{other_info}.jpg
    """

    dataset_dir = 'last'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        # val and test have query/gallery subdirectories
        self.val_query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.test_query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')

        train = self.process_dir(self.train_dir, relabel=True)
        # Use test set for evaluation (has more identities)
        query = self.process_dir(self.test_query_dir, relabel=False)
        gallery = self.process_dir(self.test_gallery_dir, relabel=False)

        super(LASTID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        """Process directory with images in subdirectories (for train) or directly (for query/gallery)"""
        if not osp.exists(dir_path):
            print(f'Warning: Directory {dir_path} does not exist')
            return []

        img_paths = []
        # First try to find images in subdirectories (for train)
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_paths.extend(glob.glob(osp.join(dir_path, '*', ext)))

        # If no images found in subdirectories, try directly in the directory (for query/gallery)
        if not img_paths:
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                img_paths.extend(glob.glob(osp.join(dir_path, ext)))

        if not img_paths:
            print(f'Warning: No images found in {dir_path}')
            return []

        pid_container = set()
        data_list = []

        # Pattern: {pid}_{cam}_{other}.jpg
        pattern = re.compile(r'(\d+)_(\d+)_')

        for img_path in img_paths:
            # Check if image is in a subdirectory
            parent_dir = osp.dirname(img_path)
            if parent_dir != dir_path:
                # Image is in subdirectory - use directory name as PID
                pid_dir = osp.basename(parent_dir)
                try:
                    pid = int(pid_dir)
                    pid_container.add(pid)
                    data_list.append((img_path, pid, 0))
                except ValueError:
                    continue
            else:
                # Image is directly in directory - parse filename
                fname = osp.basename(img_path)
                match = pattern.search(fname)
                if match:
                    pid = int(match.group(1))
                    camid = int(match.group(2))
                    pid_container.add(pid)
                    data_list.append((img_path, pid, camid))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
            data = [(path, pid2label[pid], camid) for path, pid, camid in data_list]
        else:
            data = data_list

        return data
