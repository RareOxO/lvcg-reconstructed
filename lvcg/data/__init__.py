"""LVCG data processing modules.

This package was missing from the public LVCG release: the ``.gitignore`` entry ``data/``
also matches ``lvcg/data``, so ``lvcg.models`` and ``scripts/train.py`` could not be
imported (upstream issue #1). Every module here is a reconstruction.

* ``angle``            lead direction vectors (paper Table 7) and reorder utilities
* ``beat_segmentation``  R-to-R beat segmentation (paper Appendix A.4)
* ``mimic``            MIMIC-IV-ECG manifest, WFDB reading and preprocessing (Appendix B.1)
* ``pipeline``         the datasets and data loaders ``scripts/train.py`` imports

Function names, signatures and call relations follow the release's own code graph
(``graphify-out``); numerical details the release does not record are stated where
they are chosen.
"""

from .angle import (
    LEAD_DIRECTIONS_MIMIC,
    LEAD_DIRECTIONS_PTBXL,
    LEAD_NAMES,
    LEAD_ORDERS,
    MIMIC_LEAD_NAMES,
    compute_lead_directions,
    get_lead_directions,
    reorder_leads,
)
from .beat_segmentation import BeatSegmenter
from .pipeline import ECGDataset, MimicRawDataset, make_dataloader, make_dataloaders, split_indices

__all__ = [
    "BeatSegmenter",
    "ECGDataset",
    "LEAD_DIRECTIONS_MIMIC",
    "LEAD_DIRECTIONS_PTBXL",
    "LEAD_NAMES",
    "LEAD_ORDERS",
    "MIMIC_LEAD_NAMES",
    "MimicRawDataset",
    "compute_lead_directions",
    "get_lead_directions",
    "make_dataloader",
    "make_dataloaders",
    "reorder_leads",
    "split_indices",
]
