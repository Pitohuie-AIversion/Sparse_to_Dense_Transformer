from enum import Enum
import torch
import torch.nn as nn
from fightingcv_attention.attention.A2Atttention import DoubleAttention
from fightingcv_attention.attention.AFT import AFT_FULL
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.CoAtNet import CoAtNet
from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.CoTAttention import CoTAttention
from fightingcv_attention.attention.DANet import DAModule
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.ExternalAttention import ExternalAttention
from fightingcv_attention.attention.HaloAttention import HaloAttention
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from fightingcv_attention.attention.ParNetAttention import ParNetAttention
from fightingcv_attention.attention.PolarizedSelfAttention import (
    SequentialPolarizedSelfAttention,
)
from fightingcv_attention.attention.PSA import PSA
from fightingcv_attention.attention.ResidualAttention import ResidualAttention
from fightingcv_attention.attention.S2Attention import S2Attention
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.SGE import SpatialGroupEnhance
from fightingcv_attention.attention.ShuffleAttention import ShuffleAttention
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.TripletAttention import TripletAttention
from fightingcv_attention.attention.ViP import WeightedPermuteMLP


class AdapterType(Enum):
    QKV = "qkv"
    CNN = "cnn"
    SINGLE_INPUT = "single_input"


class AttentionAdapter(nn.Module):
    """Base adapter for attention mechanisms."""

    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class QKVAttentionAdapter(AttentionAdapter):
    """Adapter for attention mechanisms that expect Q, K, V inputs."""

    def forward(self, x, memory=None, return_attention=False):
        if memory is None:
            memory = x
        try:
            return self.attention(x, memory, memory, return_attention=return_attention)
        except TypeError:
            if return_attention:
                return self.attention(x, memory, memory), None
            else:
                return self.attention(x, memory, memory)


class CNNStyleAttentionAdapter(AttentionAdapter):
    """Adapter for CNN-style attention mechanisms that expect a 4D tensor."""

    def forward(self, x, memory=None, return_attention=False):
        batch_size, seq_len, d_model = x.shape
        spatial_dim = int(seq_len**0.5)
        if spatial_dim * spatial_dim != seq_len:
            raise ValueError(
                "Sequence length cannot form square spatial dimensions for CNN attention."
            )

        x_reshaped = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, d_model, spatial_dim, spatial_dim)
        )
        
        try:
            output = self.attention(x_reshaped)
            if isinstance(output, tuple):
                output = output[0]

            if output is None:
                print(f"ERROR: {self.attention.__class__.__name__} returned None (either directly or in a tuple).")
                output = torch.zeros_like(x_reshaped)

        except Exception as e:
            # 记录错误信息但允许程序继续运行，以便调试
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"ERROR: Exception in {self.attention.__class__.__name__}: {e}")
            # 抛出异常而不是静默失败，这样可以更好地调试问题
            raise RuntimeError(f"Attention module {self.attention.__class__.__name__} failed: {e}") from e

        output = output.view(batch_size, d_model, seq_len).transpose(1, 2)

        if return_attention:
            return output, None  # 没有注意力权重可返回
        else:
            return output


class SingleInputAttentionAdapter(AttentionAdapter):
    """Adapter for attention mechanisms that expect a single tensor input."""

    def forward(self, x, memory=None, return_attention=False):
        # Special handling for attentions that expect 4D input (B, H, W, C)
        if isinstance(self.attention, (OutlookAttention, WeightedPermuteMLP)):
            B, N, C = x.shape
            H = W = int(N**0.5)
            if H * W != N:
                raise ValueError(
                    f"{self.attention.__class__.__name__} requires a square input, but got sequence length {N}"
                )
            x_reshaped = x.view(B, H, W, C)
            try:
                output = self.attention(x_reshaped)
                if isinstance(output, tuple):
                    output = output[0]
                if output is None:
                    output = torch.zeros_like(x_reshaped)
                output = output.view(B, N, C)  # Reshape back to (B, N, C)
            except Exception as e:
                import traceback
                print(f"ERROR: Exception in {self.attention.__class__.__name__}: {e}")
                traceback.print_exc()
                output = torch.zeros_like(x)
        else:
            # Default behavior for other single-input attentions
            try:
                output = self.attention(x)
                if isinstance(output, tuple):
                    output = output[0]

                if output is None:
                    output = torch.zeros_like(x)

            except Exception as e:
                import traceback
                print(f"ERROR: Exception in {self.attention.__class__.__name__}: {e}")
                traceback.print_exc()
                output = torch.zeros_like(x)

        if return_attention:
            return output, None
        return output


ADAPTER_CATALOG = {
    AdapterType.QKV: QKVAttentionAdapter,
    AdapterType.CNN: CNNStyleAttentionAdapter,
    AdapterType.SINGLE_INPUT: SingleInputAttentionAdapter,
}

def get_attention_adapter(attention_module, adapter_type: AdapterType):
    """Factory function to get the appropriate attention adapter."""
    adapter_class = ADAPTER_CATALOG.get(adapter_type)
    if adapter_class:
        return adapter_class(attention_module)
    raise ValueError(f"Unknown adapter type: {adapter_type}")
