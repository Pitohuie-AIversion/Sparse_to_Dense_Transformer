#!/usr/bin/env python3
"""
Multi-Scale Super-Resolution Reconstruction Module

This module provides comprehensive support for multi-scale super-resolution reconstruction
using the VIVTransformer architecture. It enables upscaling prediction from local to whole
by training models to reconstruct high-resolution outputs from low-resolution inputs.

Key Features:
- Multi-scale data processing with configurable scale factors (2x, 4x, 8x, 16x)
- Intelligent downsampling methods (average, bilinear, nearest)
- Comprehensive training and testing workflows
- Visualization and analysis tools
- Integration with PDEBench datasets

Modules:
- data: Data processing and loading utilities
- configs: Configuration files for different experiments
- training: Training scripts and utilities
- testing: Testing and validation scripts
- examples: Example usage scripts
- docs: Documentation and guides

Author: VIVTransformer Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VIVTransformer Team"

# Import main components for easy access
try:
    from .data.multiscale_adapter import (
        MultiScaleDataset,
        MultiScaleDataLoader,
        create_multiscale_datasets,
        create_multiscale_loaders,
        analyze_multiscale_data
    )
except ImportError:
    # Handle case where dependencies might not be available
    pass

__all__ = [
    "MultiScaleDataset",
    "MultiScaleDataLoader", 
    "create_multiscale_datasets",
    "create_multiscale_loaders",
    "analyze_multiscale_data"
]