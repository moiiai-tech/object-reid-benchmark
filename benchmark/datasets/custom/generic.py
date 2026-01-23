# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Generic dataset wrapper for standard ReID dataset structures."""

from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

try:
    from torchreid.reid.data.datasets.dataset import ImageDataset
except ImportError:
    from torchreid.data.datasets.dataset import ImageDataset


class GenericImageDataset(ImageDataset):
    """
    Generic person re-identification dataset.

    Supports standard dataset structures:
    - Market1501-like: train/query/gallery or bounding_box_train/query/bounding_box_test
    - Simple: train/val/test with images named as {pid}_{camid}_{imgid}.jpg

    Naming convention: {pid}_{camid}_{imgid}.{ext}
    or {pid}_c{camid}_{imgid}.{ext}

    Args:
        root (str): Root path to dataset directory
        dataset_name (str): Name of dataset subdirectory
        **kwargs: Additional arguments for ImageDataset
    """

    def __init__(self, root='', dataset_name='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, dataset_name)

        # Try to find train/query/gallery directories
        train_dir = self._find_dir(['bounding_box_train', 'train'])
        query_dir = self._find_dir(['query'])
        gallery_dir = self._find_dir(['bounding_box_test', 'gallery', 'test'])

        if not all([train_dir, query_dir, gallery_dir]):
            raise RuntimeError(
                f'Required directories not found in {self.dataset_dir}. '
                f'Expected train/query/gallery or bounding_box_train/query/bounding_box_test'
            )

        train = self.process_dir(train_dir, relabel=True)
        query = self.process_dir(query_dir, relabel=False)
        gallery = self.process_dir(gallery_dir, relabel=False)

        super(GenericImageDataset, self).__init__(train, query, gallery, **kwargs)

    def _find_dir(self, possible_names):
        """Find first existing directory from list of possible names."""
        for name in possible_names:
            path = osp.join(self.dataset_dir, name)
            if osp.isdir(path):
                return path
        return None

    def process_dir(self, dir_path, relabel=False):
        """
        Process a directory of images.

        Expected filename format: {pid}_{camid}_{imgid}.{ext}
        or {pid}_c{camid}_{imgid}.{ext}
        """
        # Find all image files
        img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_paths.extend(glob.glob(osp.join(dir_path, ext)))
            img_paths.extend(glob.glob(osp.join(dir_path, ext.upper())))

        if not img_paths:
            print(f'Warning: No images found in {dir_path}')
            return []

        # Try different naming patterns
        # Pattern 1: pid_camid_imgid.ext or pid_c{camid}_imgid.ext
        pattern1 = re.compile(r'([-\d]+)_c?(\d+)')

        pid_container = set()
        valid_paths = []

        for img_path in img_paths:
            fname = osp.basename(img_path)
            match = pattern1.search(fname)

            if match:
                pid, camid = map(int, match.groups())
                if pid == -1:
                    continue  # Skip junk images
                pid_container.add(pid)
                valid_paths.append((img_path, pid, camid))

        if not valid_paths:
            print(f'Warning: No valid images found in {dir_path}')
            return []

        # Create pid to label mapping for relabeling
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        # Process data
        data = []
        for img_path, pid, camid in valid_paths:
            if relabel:
                pid = pid2label[pid]
            # Ensure camid is 0-indexed
            camid = max(0, camid - 1) if camid > 0 else camid
            data.append((img_path, pid, camid))

        return data
