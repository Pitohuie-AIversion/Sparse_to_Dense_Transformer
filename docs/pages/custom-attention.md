---
layout: default
title: Custom Attention
description: Implementation guide for custom attention mechanisms
permalink: /pages/custom-attention/
---

# Custom Attention Mechanisms {#custom-attention-mechanisms}

This document provides a detailed guide on how to create and integrate custom attention mechanisms within the VIVTransformer framework, including design principles, implementation steps, testing methods, and best practices.

## 📋 Table of Contents {#table-of-contents}

- [Design Principles](#design-principles)
- [Base Interface](#base-interface)
- [Implementation Steps](#implementation-steps)
- [Example Implementation](#example-implementation)
- [Registration Mechanism](#registration-mechanism)
- [Testing & Verification](#testing-verification)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

## Design Principles {#design-principles}

### 🎯 Core Principles {#core-principles}

1. Unified interface: All attention mechanisms must follow a unified interface specification
2. Modular design: Each attention mechanism should be an independent and replaceable module
3. Configuration-driven: Select attention types and parameters via configuration files
4. Performance first: Optimize computational efficiency and memory usage
5. Extensibility: Easy to add new attention variants

### 📐 Design Constraints {#design-constraints}

```python
# Input/Output constraints {#input-output-constraints}
class AttentionConstraints:
    """
    Design constraints for attention mechanisms
    
    Input:
        - hidden_states: [batch_size, seq_len, d_model]
        - attention_mask: [batch_size, 1, 1, seq_len] (optional)
        - position_ids: [batch_size, seq_len] (optional)
    
    Output:
        - output: [batch_size, seq_len, d_model]
        - attention_weights: [batch_size, num_heads, seq_len, seq_len] (optional)
    
    Constraints:
        - Input/output dimensions must remain consistent
        - Support variable-length sequences (via attention_mask)
        - Memory usage should be controllable
        - Computational complexity should be explicit
    """
    pass
```

## Base Interface {#base-interface}

### 🔧 Abstract Base Class {#abstract-base-class}

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms
    
    All custom attention mechanisms should inherit from this class and implement the required methods.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super(BaseAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        
        # Validate parameters
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Common components
        self.dropout_layer = nn.Dropout(dropout)
        
        # Subclass-specific initialization
        self._init_parameters(**kwargs)
    
    @abstractmethod
    def _init_parameters(self, **kwargs):
        """Initialize subclass-specific parameters
        
        Subclasses should initialize their own parameters here, such as linear layers, convolutional layers, etc.
        
        Args:
            **kwargs: Additional initialization parameters
        """
        pass
    
    @abstractmethod
    def compute_attention(self, 
                         query: torch.Tensor, 
                         key: torch.Tensor, 
                         value: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            **kwargs: Extra parameters
        
        Returns:
            output: Attention output [batch_size, num_heads, seq_len, head_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len] (optional)
        """
        pass
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False,
                **kwargs) -> torch.Tensor:
        """Forward pass
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            return_attention_weights: Whether to return attention weights
            **kwargs: Extra parameters
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Generate Q, K, V
        query, key, value = self.generate_qkv(hidden_states, position_ids, **kwargs)
        
        # Reshape to multi-head format
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_output, attention_weights = self.compute_attention(
            query, key, value, attention_mask, **kwargs
        )
        
        # Reshape back to original format
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.output_projection(attention_output)
        
        if return_attention_weights:
            return output, attention_weights
        return output
    
    @abstractmethod
    def generate_qkv(self, 
                     hidden_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor] = None,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate query, key, value tensors
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            position_ids: Position IDs [batch_size, seq_len]
            **kwargs: Extra parameters
        
        Returns:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
        """
        pass
    
    @abstractmethod
    def output_projection(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Output projection
        
        Args:
            attention_output: Attention output [batch_size, seq_len, d_model]
        
        Returns:
            Projected output [batch_size, seq_len, d_model]
        """
        pass
    
    def get_attention_info(self) -> Dict[str, Any]:
        """Get attention information
        
        Returns:
            Contains attention information dictionary
        """
        return {
            'type': self.__class__.__name__,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
```

## Implementation Steps {#implementation-steps}

### 📝 Step1: Inherit Base Class {#step1-inherit-base-class}

```python
class CustomAttention(BaseAttention):
    """Custom attention mechanism example
    
    This is a sample implementation that shows how to create custom attention mechanism.
    """
    
    def _init_parameters(self, **kwargs):
        """Initialize parameters"""
        # Linear projection layers
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # Custom parameters
        self.temperature = kwargs.get('temperature', 1.0)
        self.use_bias = kwargs.get('use_bias', True)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
```

### 📝 Step2: Generate QKV {#step2-generate-qkv}

```python
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        """Generate Q, K, V tensors"""
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # You can add positional encoding or other transforms here
        if position_ids is not None:
            # Example: add position-related transform
            pos_encoding = self._get_position_encoding(position_ids)
            query = query + pos_encoding
            key = key + pos_encoding
        
        return query, key, value
    
    def _get_position_encoding(self, position_ids):
        """Get positional encoding (example implementation)"""
        # Various positional encoding schemes can be implemented here
        # e.g., sinusoidal, learnable, rotary position encodings, etc.
        batch_size, seq_len = position_ids.shape
        pos_encoding = torch.zeros(batch_size, seq_len, self.d_model, device=position_ids.device)
        
        # Simple sinusoidal position encoding example
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pos_encoding[:, pos, i] = torch.sin(position_ids[:, pos] / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_encoding[:, pos, i + 1] = torch.cos(position_ids[:, pos] / (10000 ** (i / self.d_model)))
        
        return pos_encoding
```

### 📝 Step3: Compute Attention {#step3-compute-attention}

```python
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        """Compute custom attention"""
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply temperature parameter
        attention_scores = attention_scores / self.temperature
        
        # Add bias (if used)
        if self.use_bias:
            attention_scores = attention_scores + self.bias
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Custom attention logic
        # e.g., different activations, noise injection, sparsification, etc.
        attention_scores = self._apply_custom_logic(attention_scores, **kwargs)
        
        # Softmax normalization
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        # Apply attention weights
        context = torch.matmul(attention_probs, value)
        
        return context, attention_probs
    
    def _apply_custom_logic(self, attention_scores, **kwargs):
        """Apply custom logic"""
        # Example 1: Add Gaussian noise
        if kwargs.get('add_noise', False):
            noise = torch.randn_like(attention_scores) * kwargs.get('noise_std', 0.1)
            attention_scores = attention_scores + noise
        
        # Example 2: Apply sparsification
        if kwargs.get('apply_sparsity', False):
            sparsity_ratio = kwargs.get('sparsity_ratio', 0.1)
            k = int(attention_scores.size(-1) * (1 - sparsity_ratio))
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
            sparse_scores = torch.full_like(attention_scores, float('-inf'))
            sparse_scores.scatter_(-1, topk_indices, topk_values)
            attention_scores = sparse_scores
        
        # Example 3: Apply local attention window
        if kwargs.get('local_window', False):
            window_size = kwargs.get('window_size', 64)
            attention_scores = self._apply_local_window(attention_scores, window_size)
        
        return attention_scores
    
    def _apply_local_window(self, attention_scores, window_size):
        """Apply local attention window"""
        seq_len = attention_scores.size(-1)
        mask = torch.zeros_like(attention_scores)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[..., i, start:end] = 1
        
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        return attention_scores
```

### 📝 Step4: Output Projection {#step4-output-projection}

```python
    def output_projection(self, attention_output):
        """Output projection"""
        return self.out_proj(attention_output)

# Helper classes: Different attention modes {#helper-classes-different-attention-modes}
class GlobalAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class LocalAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, window_size=64):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.window_size = window_size
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply local window mask
        seq_len = attention_scores.size(-1)
        local_mask = torch.zeros_like(attention_scores)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            local_mask[..., i, start:end] = 1
        
        attention_scores = attention_scores.masked_fill(local_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class SparseAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_ratio=0.1):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.sparsity_ratio = sparsity_ratio
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply sparsification
        seq_len = attention_scores.size(-1)
        k = int(seq_len * (1 - self.sparsity_ratio))
        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
        sparse_scores = torch.full_like(attention_scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_values)
        
        if attention_mask is not None:
            sparse_scores = sparse_scores + attention_mask
        
        attention_probs = F.softmax(sparse_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

# Auxiliary Classes: Different Attention Modes {#auxiliary-classes-different-attention-modes}

class DifferentAttentionModes:
    """
    Auxiliary class for different attention modes.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply local window mask
        seq_len = attention_scores.size(-1)
        local_mask = torch.zeros_like(attention_scores)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            local_mask[..., i, start:end] = 1
        
        attention_scores = attention_scores.masked_fill(local_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class SparseAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_ratio=0.1):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.sparsity_ratio = sparsity_ratio
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply sparsification
        seq_len = attention_scores.size(-1)
        k = int(seq_len * (1 - self.sparsity_ratio))
        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
        sparse_scores = torch.full_like(attention_scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_values)
        
        if attention_mask is not None:
            sparse_scores = sparse_scores + attention_mask
        
        attention_probs = F.softmax(sparse_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs
```

### 🌟 Example1: Multi-scale Attention {#example1-multi-scale-attention}

```python
@AttentionFactory.register("multiscale")
class MultiScaleAttention(BaseAttention):
    """Multi-scale attention mechanism
    
    Compute attention at different scales and then fuse results.
    """
    
    def _init_parameters(self, scales=[1, 2, 4], **kwargs):
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create projection layers for each scale
        self.scale_projections = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(self.d_model, self.d_model // self.num_scales, bias=False),
                'k_proj': nn.Linear(self.d_model, self.d_model // self.num_scales, bias=False),
                'v_proj': nn.Linear(self.d_model, self.d_model // self.num_scales, bias=False)
            }) for _ in self.scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        self.scale = (self.d_model // self.num_scales // self.num_heads) ** -0.5
    
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        batch_size, seq_len, d_model = hidden_states.shape
        
        all_queries, all_keys, all_values = [], [], []
        
        for scale_idx, scale in enumerate(self.scales):
            # Downsample if scale > 1
            if scale > 1:
                # Use average pooling to downsample
                pooled_states = F.avg_pool1d(
                    hidden_states.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
            else:
                pooled_states = hidden_states
            
            # Generate Q, K, V
            proj = self.scale_projections[scale_idx]
            q = proj['q_proj'](pooled_states)
            k = proj['k_proj'](pooled_states)
            v = proj['v_proj'](pooled_states)
            
            # Upsample back to original length if needed
            if scale > 1:
                q = F.interpolate(q.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
                k = F.interpolate(k.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
                v = F.interpolate(v.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
            
            all_queries.append(q)
            all_keys.append(k)
            all_values.append(v)
        
        # Concatenate all scales
        query = torch.cat(all_queries, dim=-1)
        key = torch.cat(all_keys, dim=-1)
        value = torch.cat(all_values, dim=-1)
        
        return query, key, value
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        # Standard attention computation
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        return context, attention_probs
    
    def output_projection(self, attention_output):
        # Pass through fusion layer first, then output projection
        fused = self.fusion(attention_output)
        return self.out_proj(fused)
```

### 🌟 Example2: Adaptive Attention {#example2-adaptive-attention}

```python
@AttentionFactory.register("adaptive")
class AdaptiveAttention(BaseAttention):
    """Adaptive attention mechanism
    
    Dynamically adjusts attention patterns based on input.
    """
    
    def _init_parameters(self, **kwargs):
        # Standard projection layers
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # Adaptive controller
        self.controller = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, self.num_heads),
            nn.Sigmoid()
        )
        
        # Multiple attention modes
        self.attention_modes = nn.ModuleDict({
            'global': GlobalAttentionMode(self.d_model, self.num_heads),
            'local': LocalAttentionMode(self.d_model, self.num_heads),
            'sparse': SparseAttentionMode(self.d_model, self.num_heads)
        })
        
        self.scale = self.head_dim ** -0.5
    
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        return query, key, value
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute adaptive weights
        # Use global average pooling to get sequence representation
        global_repr = query.mean(dim=2)  # [batch_size, num_heads, head_dim]
        adaptive_weights = self.controller(global_repr.view(batch_size, -1))  # [batch_size, num_heads]
        adaptive_weights = adaptive_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_heads, 1, 1]
        
        # Compute attention for different modes
        attention_outputs = {}
        attention_weights = {}
        
        for mode_name, mode in self.attention_modes.items():
            output, weights = mode(query, key, value, attention_mask)
            attention_outputs[mode_name] = output
            attention_weights[mode_name] = weights
        
        # Adaptive fusion
        # Simplified as weighted average of three modes here
        mode_weights = F.softmax(adaptive_weights.squeeze(-1).squeeze(-1), dim=-1)  # [batch_size, num_heads]
        
        final_output = torch.zeros_like(attention_outputs['global'])
        final_weights = torch.zeros_like(attention_weights['global'])
        
        for i, (mode_name, output) in enumerate(attention_outputs.items()):
            weight = mode_weights[:, :, None, None] if i == 0 else mode_weights[:, :, None, None]
            final_output += weight * output
            final_weights += weight * attention_weights[mode_name]
        
        return final_output, final_weights
    
    def output_projection(self, attention_output):
        return self.out_proj(attention_output)

# Auxiliary Classes: Different Attention Modes {#auxiliary-classes-different-attention-modes}
class GlobalAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class LocalAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, window_size=64):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.window_size = window_size
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply local window mask
        seq_len = attention_scores.size(-1)
        local_mask = torch.zeros_like(attention_scores)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            local_mask[..., i, start:end] = 1
        
        attention_scores = attention_scores.masked_fill(local_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class SparseAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_ratio=0.1):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.sparsity_ratio = sparsity_ratio
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply sparsification
        seq_len = attention_scores.size(-1)
        k = int(seq_len * (1 - self.sparsity_ratio))
        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
        sparse_scores = torch.full_like(attention_scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_values)
        
        if attention_mask is not None:
            sparse_scores = sparse_scores + attention_mask
        
        attention_probs = F.softmax(sparse_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs
```

## Registration Mechanism {#registration-mechanism}

### 🔧 Register Custom Attention {#register-custom-attention}

```python
# Method 1: Use decorator for registration {#method1-use-decorator-for-registration}
@AttentionFactory.register("my_custom_attention")
class MyCustomAttention(BaseAttention):
    # Implementation details...
    pass

# Method 2: Manual registration {#method2-manual-registration}
class AnotherCustomAttention(BaseAttention):
    # Implementation details...
    pass

AttentionFactory.register_class("another_custom", AnotherCustomAttention)

# Method 3: Batch registration {#method3-batch-registration}
custom_attentions = {
    "attention_a": AttentionA,
    "attention_b": AttentionB,
    "attention_c": AttentionC
}

for name, cls in custom_attentions.items():
    AttentionFactory.register_class(name, cls)
```

### ⚙️ Configuration File Integration {#configuration-file-integration}

```yaml
# config.yaml {#config-yaml}
model:
  attention_config:
    # Use custom attention
    attention_types: ["my_custom_attention", "multiscale", "adaptive"]
    
    # Custom attention parameters
    attention_kwargs:
      my_custom_attention:
        temperature: 1.5
        use_bias: true
        add_noise: false
        noise_std: 0.1
      
      multiscale:
        scales: [1, 2, 4, 8]
      
      adaptive:
        controller_hidden_dim: 256
        num_modes: 3
```

```python
# Use in code {#use-in-code}
config = load_config("config.yaml")
model = VIVTransformer(config)

# Automatically use the attention mechanism specified in config {#automatically-use-attention-mechanism-specified-in-config}
```

## Testing and Validation {#testing-and-validation}

### 🧪 Unit Tests {#unit-tests}

```python
import unittest
import torch

class TestCustomAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 128
        self.d_model = 512
        self.num_heads = 8
        
        self.attention = CustomAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.attention_mask = torch.ones(self.batch_size, 1, 1, self.seq_len)
    
    def test_output_shape(self):
        """Test output shape"""
        output = self.attention(self.hidden_states, self.attention_mask)
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
    
    def test_attention_weights_shape(self):
        """Test attention weights shape"""
        output, weights = self.attention(
            self.hidden_states, 
            self.attention_mask, 
            return_attention_weights=True
        )
        expected_weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(weights.shape, expected_weights_shape)
    
    def test_attention_weights_sum(self):
        """Test attention weights sum to 1"""
        _, weights = self.attention(
            self.hidden_states, 
            self.attention_mask, 
            return_attention_weights=True
        )
        weights_sum = weights.sum(dim=-1)
        expected_sum = torch.ones_like(weights_sum)
        torch.testing.assert_close(weights_sum, expected_sum, atol=1e-6, rtol=1e-6)
    
    def test_gradient_flow(self):
        """Test gradient flow"""
        self.hidden_states.requires_grad_(True)
        output = self.attention(self.hidden_states, self.attention_mask)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        self.assertIsNotNone(self.hidden_states.grad)
        self.assertFalse(torch.isnan(self.hidden_states.grad).any())
    
    def test_mask_effectiveness(self):
        """Test mask effectiveness"""
        # Create partial mask
        masked_attention_mask = self.attention_mask.clone()
        masked_attention_mask[:, :, :, self.seq_len//2:] = 0  # Mask the second half
        
        _, weights = self.attention(
            self.hidden_states, 
            masked_attention_mask, 
            return_attention_weights=True
        )
        
        # Check if attention weights for masked parts are close to 0
        masked_weights = weights[:, :, :, self.seq_len//2:]
        self.assertTrue((masked_weights < 1e-6).all())
    
    def test_parameter_count(self):
        """Test parameter count"""
        info = self.attention.get_attention_info()
        self.assertGreater(info['parameters'], 0)
        self.assertGreater(info['trainable_parameters'], 0)
    
    def test_different_sequence_lengths(self):
        """Test different sequence lengths"""
        for seq_len in [32, 64, 256, 512]:
            hidden_states = torch.randn(self.batch_size, seq_len, self.d_model)
            attention_mask = torch.ones(self.batch_size, 1, 1, seq_len)
            
            output = self.attention(hidden_states, attention_mask)
            expected_shape = (self.batch_size, seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
```

### 📊 Performance Testing {#performance-testing}

```python
class AttentionBenchmark:
    """Attention mechanism performance testing"""
    
    def __init__(self, attention_types, test_configs):
        self.attention_types = attention_types
        self.test_configs = test_configs
        self.results = {}
    
    def benchmark_memory(self, attention_type, config):
        """Memory usage testing"""
        attention = AttentionFactory.create(attention_type, **config)
        
        batch_size, seq_len, d_model = config['batch_size'], config['seq_len'], config['d_model']
        hidden_states = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = attention(hidden_states)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return peak_memory
    
    def benchmark_speed(self, attention_type, config, num_runs=100):
        """Speed testing"""
        attention = AttentionFactory.create(attention_type, **config)
        attention = attention.cuda()
        
        batch_size, seq_len, d_model = config['batch_size'], config['seq_len'], config['d_model']
        hidden_states = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = attention(hidden_states)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = attention(hidden_states)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = batch_size / avg_time
        
        return avg_time, throughput
    
    def run_benchmark(self):
        """Run complete benchmark testing"""
        for attention_type in self.attention_types:
            self.results[attention_type] = {}
            
            for config_name, config in self.test_configs.items():
                print(f"Testing {attention_type} with {config_name}...")
                
                try:
                    # Memory testing
                    memory_usage = self.benchmark_memory(attention_type, config)
                    
                    # Speed testing
                    avg_time, throughput = self.benchmark_speed(attention_type, config)
                    
                    self.results[attention_type][config_name] = {
                        'memory_mb': memory_usage,
                        'avg_time_ms': avg_time * 1000,
                        'throughput': throughput
                    }
                    
                except Exception as e:
                    print(f"Error testing {attention_type} with {config_name}: {e}")
                    self.results[attention_type][config_name] = {'error': str(e)}
        
        return self.results
    
    def print_results(self):
        """Print test results"""
        for attention_type, configs in self.results.items():
            print(f"\n{attention_type}:")
            for config_name, metrics in configs.items():
                if 'error' in metrics:
                    print(f"  {config_name}: ERROR - {metrics['error']}")
                else:
                    print(f"  {config_name}:")
                    print(f"    Memory: {metrics['memory_mb']:.2f} MB")
                    print(f"    Time: {metrics['avg_time_ms']:.2f} ms")
                    print(f"    Throughput: {metrics['throughput']:.2f} samples/sec")

# Usage example {#usage-example}
test_configs = {
    'small': {'d_model': 256, 'num_heads': 8, 'batch_size': 4, 'seq_len': 128},
    'medium': {'d_model': 512, 'num_heads': 8, 'batch_size': 4, 'seq_len': 256},
    'large': {'d_model': 768, 'num_heads': 12, 'batch_size': 2, 'seq_len': 512}
}

benchmark = AttentionBenchmark(
    attention_types=['standard', 'linear', 'sparse', 'my_custom_attention'],
    test_configs=test_configs
)

results = benchmark.run_benchmark()
benchmark.print_results()
```

## Performance Optimization {#performance-optimization}

### ⚡ Computational Optimization {#computational-optimization}

```python
class OptimizedAttention(BaseAttention):
    """Optimized attention implementation"""
    
    def _init_parameters(self, use_flash_attention=True, **kwargs):
        # Standard parameter initialization
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # Optimization options
        self.use_flash_attention = use_flash_attention
        self.chunk_size = kwargs.get('chunk_size', 1024)
        
        # Try importing Flash Attention
        if self.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("Flash Attention not available, falling back to standard implementation")
                self.use_flash_attention = False
        
        self.scale = self.head_dim ** -0.5
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        if self.use_flash_attention and attention_mask is None:
            # Use Flash Attention (more efficient)
            return self._flash_attention(query, key, value)
        elif query.size(-2) > self.chunk_size:
            # Use chunked computation (memory saving)
            return self._chunked_attention(query, key, value, attention_mask)
        else:
            # Standard computation
            return self._standard_attention(query, key, value, attention_mask)
    
    def _flash_attention(self, query, key, value):
        """Flash Attention implementation"""
        # Rearrange dimensions to match Flash Attention API
        q = query.transpose(1, 2)  # [B, L, H, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        # Call Flash Attention
        output = self.flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        # Rearrange back to original format
        output = output.transpose(1, 2)  # [B, H, L, D]
        
        return output, None  # Flash Attention doesn't return weights
    
    def _chunked_attention(self, query, key, value, attention_mask=None):
        """Chunked attention computation"""
        batch_size, num_heads, seq_len, head_dim = query.shape
        chunk_size = self.chunk_size
        
        outputs = []
        attention_weights = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end_i, :]
            
            # Compute attention for current chunk
            chunk_scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, i:end_i, :]
                chunk_scores = chunk_scores + chunk_mask
            
            chunk_probs = F.softmax(chunk_scores, dim=-1)
            chunk_probs = self.dropout_layer(chunk_probs)
            
            chunk_output = torch.matmul(chunk_probs, value)
            
            outputs.append(chunk_output)
            attention_weights.append(chunk_probs)
        
        # Concatenate all chunks
        output = torch.cat(outputs, dim=2)
        weights = torch.cat(attention_weights, dim=2)
        
        return output, weights
    
    def _standard_attention(self, query, key, value, attention_mask=None):
        """Standard attention computation"""
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        return context, attention_probs
```

### 🧠 Memory Optimization {#memory-optimization}

```python
class MemoryEfficientAttention(BaseAttention):
    """Memory-efficient attention implementation"""
    
    def _init_parameters(self, **kwargs):
        # Use fewer parameters
        self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # Gradient checkpointing
        self.use_checkpoint = kwargs.get('use_checkpoint', True)
        
        self.scale = self.head_dim ** -0.5
    
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        # Generate Q, K, V in one go
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        return query, key, value
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing
            return checkpoint(self._attention_forward, query, key, value, attention_mask)
        else:
            return self._attention_forward(query, key, value, attention_mask)
    
    def _attention_forward(self, query, key, value, attention_mask=None):
        """Attention forward pass"""
        # Use more memory-efficient implementation
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Chunked computation to save memory
        chunk_size = min(512, seq_len)
        outputs = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end_i, :]
            
            # Compute attention scores
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, i:end_i, :]
                scores = scores + mask_chunk
            
            # Use numerically stable softmax
            max_scores = scores.max(dim=-1, keepdim=True)[0]
            scores = scores - max_scores
            exp_scores = torch.exp(scores)
            sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
            probs = exp_scores / sum_exp_scores
            
            # Apply dropout
            probs = self.dropout_layer(probs)
            
            # Compute output
            chunk_output = torch.matmul(probs, value)
            outputs.append(chunk_output)
        
        output = torch.cat(outputs, dim=2)
        return output, None  # Don't return attention weights to save memory
    
    def output_projection(self, attention_output):
        return self.out_proj(attention_output)
```

## Best Practices {#best-practices}

### ✅ Design Recommendations {#design-recommendations}

1. **Interface Consistency**: Always follow the BaseAttention interface
2. **Parameter Validation**: Validate parameter validity during initialization
3. **Error Handling**: Gracefully handle exceptional cases
4. **Complete Documentation**: Provide detailed docstrings
5. **Test Coverage**: Write comprehensive unit tests

### 🚀 Performance Recommendations {#performance-recommendations}

1. **Memory Optimization**: Use gradient checkpointing and chunked computation
2. **Computational Optimization**: Leverage efficient implementations like Flash Attention
3. **Numerical Stability**: Use numerically stable algorithms
4. **Cache-Friendly**: Optimize memory access patterns
5. **Parallelization**: Fully utilize GPU parallel computing capabilities

### 🔧 Debugging Tips {#debugging-tips}

```python
class DebuggableAttention(BaseAttention):
    """Debuggable attention implementation"""
    
    def __init__(self, *args, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.debug_info = {}
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        if self.debug:
            # Record debug information
            self.debug_info['input_shapes'] = {
                'query': query.shape,
                'key': key.shape,
                'value': value.shape
            }
            
            # Check numerical stability
            self._check_numerical_stability(query, key, value)
        
        # Normal computation
        output, weights = self._standard_attention(query, key, value, attention_mask)
        
        if self.debug:
            self.debug_info['output_shape'] = output.shape
            if weights is not None:
                self.debug_info['attention_stats'] = {
                    'min': weights.min().item(),
                    'max': weights.max().item(),
                    'mean': weights.mean().item(),
                    'std': weights.std().item()
                }
        
        return output, weights
    
    def _check_numerical_stability(self, query, key, value):
        """Check numerical stability"""
        for name, tensor in [('query', query), ('key', key), ('value', value)]:
            if torch.isnan(tensor).any():
                print(f"Warning: NaN detected in {name}")
            if torch.isinf(tensor).any():
                print(f"Warning: Inf detected in {name}")
            if tensor.abs().max() > 1e6:
                print(f"Warning: Large values detected in {name}: max={tensor.abs().max()}")
    
    def get_debug_info(self):
        """Get debug information"""
        return self.debug_info
```

---

## 📚 Summary {#summary}

Through this document, you have learned how to create custom attention mechanisms in the VIVTransformer framework. Key points include:

### 🎯 Core Elements {#core-elements}

1. **Inherit BaseAttention**: Ensure interface consistency
2. **Implement Required Methods**: `_init_parameters`, `generate_qkv`, `compute_attention`, `output_projection`
3. **Registration Mechanism**: Use AttentionFactory for registration
4. **Configuration Integration**: Control parameters through configuration files
5. **Testing and Validation**: Write comprehensive test cases

### 🚀 Optimization Strategies {#optimization-strategies}

- **Computational Optimization**: Flash Attention, chunked computation
- **Memory Optimization**: Gradient checkpointing, parameter sharing
- **Numerical Stability**: Stable softmax, gradient clipping
- **Debug Support**: Detailed debug information and checks

### 📈 Extension Directions {#extension-directions}

- **New Attention Patterns**: Local, sparse, adaptive
- **Multi-modal Attention**: Cross-modal interactions
- **Dynamic Attention**: Adjust behavior based on input
- **Efficient Implementations**: Leverage latest hardware features

By following these guidelines, you can create efficient, reliable, and maintainable custom attention mechanisms that contribute new functionality to the VIVTransformer project.

---

*Need help? Check [FAQ](faq) or [Troubleshooting](troubleshooting) pages.*
