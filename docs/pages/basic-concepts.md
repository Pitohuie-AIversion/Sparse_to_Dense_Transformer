---
layout: default
title: Basic Concepts
nav_order: 1
parent: Getting Started
permalink: /pages/basic-concepts/
---

# Basic Concepts

This page introduces the core ideas behind the VIVTransformer project.

## What is VIVTransformer?
VIVTransformer is a Transformer-based model designed to learn and predict vortex-induced vibration (VIV) behavior from multi-source time-series inputs (e.g., flow field signals and structural responses). It aims to achieve accurate dynamics modeling while improving efficiency via sparse/linear attention variants.

## Key Components
- Multi-branch encoder for heterogeneous inputs (flow/structure)
- Cross-attention fusion between flow and structure streams
- Physics-informed loss terms (e.g., consistency, energy constraints)
- Optional SVD-based regularization for stability and interpretability
- Flexible attention backends: standard, linear, sparse

## Data Modalities
- Flow signals: velocity, pressure, turbulence features
- Structure responses: displacement, velocity, acceleration
- Optional context: sensor metadata, environmental conditions

## Input/Output Shapes
- Input: [batch, seq_len, feature_dim]
  - flow_input_dim, structure_input_dim defined in config
- Output: dictionary of predictions, e.g. {"displacement", "velocity", "force"}

## Training Targets
- Predict structural displacement/velocity and fluid forces
- Maintain temporal consistency across long horizons
- Balance accuracy and computational cost

## Attention Variants
- Standard attention: best accuracy, O(L^2) memory
- Linear attention: kernel tricks for O(L) scalability
- Sparse attention: top-k or block patterns for long sequences

## Loss Design
- MSE for supervised targets
- Physics/consistency constraints to stabilize long-horizon rollout
- Multi-loss weighting with curriculum or dynamic schedules

## Evaluation Overview
- Accuracy: MAE/RMSE for displacement/velocity/force
- Efficiency: FPS, memory, FLOPs
- Stability: drift and error accumulation in rollout
- Interpretability: attention maps and sensitivity

## When to Use Which Attention
- Short sequences/high accuracy: Standard
- Long sequences/limited memory: Linear
- Very long sequences with locality: Sparse

## Minimal Example
```python
from vivtransformer import VIVTransformer, VIVConfig

config = VIVConfig(
    d_model=256,
    num_layers=4,
    num_heads=4,
    flow_input_dim=64,
    structure_input_dim=32,
    output_dim=9
)
model = VIVTransformer(config)
```

## Next Steps
- See Examples for full training/inference pipelines
- Check Implementation Details for architecture internals
- Review Hyperparameter Tuning for practical tips