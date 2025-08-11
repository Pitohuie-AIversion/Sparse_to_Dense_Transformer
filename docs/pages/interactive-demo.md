---
layout: default
title: Interactive Demo
nav_order: 5
parent: Tutorials & Examples
description: "A simple demo to explore VIVTransformer behavior"
permalink: /pages/interactive-demo/
---

# Interactive Demo

This page provides a minimal interactive walkthrough to understand how inputs are transformed and predictions are produced.

## What You Can Explore
- Input preprocessing and normalization
- Step-by-step forward pass through key layers
- Attention maps and their qualitative behavior
- Final predictions and error metrics

## Quick Start
1) Prepare a small sample or toy dataset
2) Load the pretrained or checkpointed model
3) Run the demo notebook/script to visualize each stage

## Example Script (pseudo-code)
```python
from demo import load_model, visualize

# Load model and data
model = load_model(checkpoint="checkpoints/best.ckpt")
inputs, targets = load_toy_batch()

# Forward pass with hooks
outputs, records = model.forward_with_records(inputs)

# Visualize
visualize.attention(records["attn_maps"])       # attention maps
visualize.svd(records.get("sv_spectra", None))  # singular value spectra
visualize.prediction(outputs, targets)           # predictions vs targets
```

## Recommended Visualizations
- Attention heatmaps per head/layer
- Singular value spectra for key weight matrices
- Loss curves and component breakdown over time

## Tips
- Use smaller input sizes for responsiveness
- Enable mixed precision on GPU-capable machines
- Save visualizations for comparisons across runs

## Related Pages
- [Model Design](/pages/model-design/)
- [Loss Functions](/pages/loss-functions/)
- [Convergence Analysis](/pages/convergence-analysis/)