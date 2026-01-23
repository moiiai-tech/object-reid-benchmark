# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""MSMT17 dataset wrapper."""

from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

try:
    from torchreid.reid.data.datasets.dataset import ImageDataset
except ImportError:
    from torchreid.data.datasets.dataset import ImageDataset


class MSMT17(ImageDataset):
    """
    MSMT17: Multi-Scene Multi-Time Person Re-identification Dataset

    Dataset structure:
        MSMT17_V1/
            train/
                {pid}/
                    {pid}_{imgid}_{camid}_{scene}_{trackid}_{frameid}.jpg
            test/
                {pid}/
                    {pid}_{imgid}_{camid}_{scene}_{trackid}_{frameid}.jpg
            list_train.txt
            list_query.txt
            list_gallery.txt
            list_val.txt

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    """

    dataset_dir = 'MSMT17_V1'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        # Process train data
        train = self.process_dir(self.train_dir, self.list_train_path, relabel=True)

        # Process test data
        query = self.process_split(self.test_dir, self.list_query_path, relabel=False)
        gallery = self.process_split(self.test_dir, self.list_gallery_path, relabel=False)

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path, relabel=False):
        """Process directory using list file or by scanning subdirectories"""
        if osp.exists(list_path):
            return self.process_split(dir_path, list_path, relabel)
        else:
            return self.process_from_dir(dir_path, relabel)

    def process_split(self, dir_path, list_path, relabel=False):
        """Process split using list file"""
        if not osp.exists(list_path):
            print(f'Warning: List file {list_path} does not exist')
            return []

        data = []
        pid_container = set()

        # Pattern: {pid}_{imgid}_{camid}_{scene}_{trackid}_{frameid}.jpg
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)_')

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                img_path_rel = parts[0]
                pid = int(parts[1])

                img_path = osp.join(dir_path, img_path_rel)

                if not osp.exists(img_path):
                    continue

                fname = osp.basename(img_path)
                match = pattern.search(fname)

                if match:
                    camid = int(match.group(3)) - 1  # 0-indexed
                else:
                    camid = 0

                pid_container.add(pid)
                data.append((img_path, pid, camid))

        # Relabel if needed
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
            data = [(path, pid2label[pid], camid) for path, pid, camid in data]

        return data

    def process_from_dir(self, dir_path, relabel=False):
        """Process directory by scanning subdirectories (fallback method)"""
        if not osp.exists(dir_path):
            print(f'Warning: Directory {dir_path} does not exist')
            return []

        data = []
        pid_container = set()

        # Pattern: {pid}_{imgid}_{camid}_{scene}_{trackid}_{frameid}.jpg
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)_')

        # Each subdirectory is a person ID
        person_dirs = glob.glob(osp.join(dir_path, '*'))

        for person_dir in person_dirs:
            if not osp.isdir(person_dir):
                continue

            pid = int(osp.basename(person_dir))
            pid_container.add(pid)

            # Get all images for this person
            img_paths = glob.glob(osp.join(person_dir, '*.jpg'))

            for img_path in img_paths:
                fname = osp.basename(img_path)
                match = pattern.search(fname)

                if match:
                    camid = int(match.group(3)) - 1  # 0-indexed
                    data.append((img_path, pid, camid))

        # Relabel if needed
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
            data = [(path, pid2label[pid], camid) for path, pid, camid in data]

        return data
