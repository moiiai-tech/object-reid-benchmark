# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Custom dataset wrappers for datasets not supported by torchreid."""

from .celebreid import CelebReID
from .entireid import EntireID
from .g2a import G2A
from .g2a_custom import G2AVReID
from .generic import GenericImageDataset
from .ilidsvid import ILIDSVID
from .ilidsvid_custom import ILIDSVIDCustom
from .iustreid import IUSTReID
from .lastid import LASTID
from .msmt17 import MSMT17
from .occluded_posetrack import OccludedPoseTrack
from .pep import PeP
from .pku import PKU

__all__ = [
    "GenericImageDataset",
    "CelebReID",
    "EntireID",
    "LASTID",
    "PKU",
    "ILIDSVID",
    "ILIDSVIDCustom",
    "G2A",
    "G2AVReID",
    "IUSTReID",
    "OccludedPoseTrack",
    "PeP",
    "MSMT17",
]
