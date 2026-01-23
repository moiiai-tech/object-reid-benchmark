# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""iLIDS-VID dataset wrapper."""

from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

try:
    from torchreid.reid.data.datasets.dataset import ImageDataset
except ImportError:
    from torchreid.data.datasets.dataset import ImageDataset


class ILIDSVIDCustom(ImageDataset):
    """
    iLIDS-VID: i-LIDS Video Re-identification Dataset

    Dataset structure:
        iLIDS-VID/
            i-LIDS-VID/
                sequences/
                    cam1/
                        person{pid}/
                            *.png
                    cam2/
                        person{pid}/
                            *.png

    Reference:
        Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    """

    dataset_dir = 'iLIDS-VID/i-LIDS-VID'

    def __init__(self, root='', train_split=0.5, **kwargs):
        """
        Args:
            root: Root directory
            train_split: Proportion of identities to use for training (default: 0.5)
        """
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_split = train_split

        self.sequences_dir = osp.join(self.dataset_dir, 'sequences')

        # Process both cameras
        cam1_dir = osp.join(self.sequences_dir, 'cam1')
        cam2_dir = osp.join(self.sequences_dir, 'cam2')

        cam1_data = self.process_dir(cam1_dir, camid=0)
        cam2_data = self.process_dir(cam2_dir, camid=1)

        all_data = cam1_data + cam2_data

        # Get all unique person IDs
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

        # Split test into query and gallery (cam1 vs cam2)
        query = [item for item in test_data if item[2] == 0]  # cam1
        gallery = [item for item in test_data if item[2] == 1]  # cam2

        super(ILIDSVIDCustom, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, camid):
        """Process camera directory with person subdirectories"""
        if not osp.exists(dir_path):
            print(f'Warning: Directory {dir_path} does not exist')
            return []

        data = []
        # Pattern: person{pid}
        pattern = re.compile(r'person(\d+)')

        # Each subdirectory is a person
        person_dirs = glob.glob(osp.join(dir_path, 'person*'))

        for person_dir in person_dirs:
            if not osp.isdir(person_dir):
                continue

            dirname = osp.basename(person_dir)
            match = pattern.search(dirname)

            if match:
                pid = int(match.group(1))

                # Get all images for this person
                img_paths = glob.glob(osp.join(person_dir, '*.png'))
                img_paths.extend(glob.glob(osp.join(person_dir, '*.jpg')))

                for img_path in img_paths:
                    data.append((img_path, pid, camid))

        return data
