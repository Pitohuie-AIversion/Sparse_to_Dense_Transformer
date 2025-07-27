"""Data module for VIVTransformer

This module provides data loading and processing functionality for various datasets,
including PDEBench integration.
"""

from .dataloader import (
    get_loaders,
    get_pdebench_loaders,
    get_adaptive_loaders,
    CustomSubset,
    collate_fn
)

from .dataset import PressureDataset
from .pdebench_adapter import (
    PDEBenchDataset,
    PDEBenchDataLoader,
    create_pdebench_datasets,
    create_pdebench_loaders
)

# Import transforms if available
try:
    from . import transforms
except ImportError:
    transforms = None

__all__ = [
    'get_loaders',
    'get_pdebench_loaders', 
    'get_adaptive_loaders',
    'CustomSubset',
    'collate_fn',
    'PressureDataset',
    'PDEBenchDataset',
    'PDEBenchDataLoader',
    'create_pdebench_datasets',
    'create_pdebench_loaders',
    'transforms'
]